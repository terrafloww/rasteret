"""
Copyright 2025 Terrafloww Labs, Inc

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Cloud storage configuration and provider classes.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from urllib.parse import urlparse
import re
import boto3
from rasteret.logging import setup_logger

logger = setup_logger()


@dataclass
class CloudConfig:
    """Storage configuration for data source"""

    provider: str
    requester_pays: bool = False
    region: str = "us-west-2"
    url_patterns: Dict[str, str] = None

    @classmethod
    def get_config(cls, data_source: str) -> Optional["CloudConfig"]:
        """Get cloud configuration for data source."""
        CONFIGS = {
            "landsat-c2l2-sr": cls(
                provider="aws",
                requester_pays=True,
                region="us-west-2",
                url_patterns={
                    "https://landsatlook.usgs.gov/data/": "s3://usgs-landsat/"
                },
            ),
            "sentinel-2-l2a": cls(
                provider="aws", requester_pays=False, region="us-west-2"
            ),
        }
        return CONFIGS.get(data_source.lower())


class CloudProvider:
    """Base class for cloud providers"""

    @staticmethod
    def check_aws_credentials() -> bool:
        """Check AWS credentials before any operations"""
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            if credentials is None:
                logger.info("\nAWS credentials not found.\n")
                return False
            return True
        except Exception:
            return False

    def get_url(self, url: str, config: CloudConfig) -> str:
        """Central URL resolution and signing method"""
        raise NotImplementedError


class AWSProvider(CloudProvider):
    def __init__(self, profile: Optional[str] = None, region: str = "us-west-2"):
        # Do not force AWS credentials at construction time; Sentinel-2 (non-requester-pays)
        # should work without credentials. We'll only require creds when signing is needed.
        self._have_creds = self.check_aws_credentials()

        try:
            session = (
                boto3.Session(profile_name=profile) if profile else boto3.Session()
            )
            self.client = session.client("s3", region_name=region)
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {str(e)}")
            # If client can't be created (e.g., boto3 missing), leave as None; signing will be skipped
            self.client = None

    @staticmethod
    def _extract_s3_bucket_key(url: str) -> Optional[Tuple[str, str]]:
        """Extract (bucket, key) from s3:// or HTTPS S3 URL styles.
        Supports virtual-hosted and path-style endpoints.
        """
        try:
            p = urlparse(url)
            if p.scheme == "s3":
                return p.netloc, p.path.lstrip("/")
            if p.scheme == "https":
                host = p.netloc
                path = p.path.lstrip("/")
                # Virtual-hosted-style: <bucket>.s3.amazonaws.com or <bucket>.s3.<region>.amazonaws.com
                m = re.match(
                    r"^([a-z0-9.\-]+)\.s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$", host
                )
                if m:
                    return m.group(1), path
                # Path-style: s3.amazonaws.com/<bucket>/key or s3.<region>.amazonaws.com/<bucket>/key
                if (
                    re.match(r"^s3(?:[.-][a-z0-9-]+)?\.amazonaws\.com$", host)
                    and "/" in path
                ):
                    bucket, key = path.split("/", 1)
                    return bucket, key
        except Exception:
            pass
        return None

    def get_url(self, url: str, config: CloudConfig) -> Optional[str]:
        """Resolve and sign URL based on configuration."""
        # First check for alternate S3 URL in STAC metadata
        if (
            isinstance(url, dict)
            and "alternate" in url
            and "s3" in url.get("alternate", {})
        ):
            s3_url = url["alternate"]["s3"]["href"]
            logger.debug(f"Using alternate S3 URL: {s3_url}")
            url = s3_url
        # Then check URL patterns if defined (e.g., landsatlook -> s3://usgs-landsat)
        elif config.url_patterns:
            for http_pattern, s3_pattern in config.url_patterns.items():
                if isinstance(url, str) and url.startswith(http_pattern):
                    url = url.replace(http_pattern, s3_pattern)
                    logger.debug(f"Converted to S3 URL: {url}")
                    break

        # Attempt to extract bucket/key from either s3:// or HTTPS S3 styles
        bucket_key = self._extract_s3_bucket_key(url) if isinstance(url, str) else None
        if bucket_key:
            # For non-requester-pays (e.g., Sentinel-2), do not sign; allow public/anon access
            if not config.requester_pays:
                return url
            bucket, key = bucket_key
            try:
                # Requester-pays requires signing and RequestPayer header
                if not getattr(self, "_have_creds", False) or not getattr(
                    self, "client", None
                ):
                    logger.error(
                        "AWS credentials/client not available for requester-pays signing"
                    )
                    return None
                params = {"Bucket": bucket, "Key": key, "RequestPayer": "requester"}
                return self.client.generate_presigned_url(
                    "get_object", Params=params, ExpiresIn=3600
                )
            except Exception as e:
                logger.error(f"Failed to sign URL {url}: {str(e)}")
                return None

        # Fallback: return as-is if not an S3 URL
        return url


__all__ = ["CloudConfig", "AWSProvider"]
