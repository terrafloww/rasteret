""" Utilities for cloud storage """

from dataclasses import dataclass
from typing import Optional, Dict
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
                logger.error(
                    "\nAWS credentials not found. To configure:\n"
                    "1. Create ~/.aws/credentials with:\n"
                    "[default]\n"
                    "aws_access_key_id = YOUR_ACCESS_KEY\n"
                    "aws_secret_access_key = YOUR_SECRET_KEY\n"
                    "OR\n"
                    "2. Set environment variables:\n"
                    "export AWS_ACCESS_KEY_ID='your_key'\n"
                    "export AWS_SECRET_ACCESS_KEY='your_secret'"
                )
                return False
            return True
        except Exception:
            return False

    def get_url(self, url: str, config: CloudConfig) -> str:
        """Central URL resolution and signing method"""
        raise NotImplementedError


class AWSProvider(CloudProvider):
    def __init__(self, profile: Optional[str] = None, region: str = "us-west-2"):
        if not self.check_aws_credentials():
            raise ValueError("AWS credentials not configured")

        try:
            session = (
                boto3.Session(profile_name=profile) if profile else boto3.Session()
            )
            self.client = session.client("s3", region_name=region)
        except Exception as e:
            logger.error(f"Failed to initialize AWS client: {str(e)}")
            raise ValueError("AWS provider initialization failed")

    def get_url(self, url: str, config: CloudConfig) -> Optional[str]:
        """Resolve and sign URL based on configuration"""
        # First check for alternate S3 URL in STAC metadata
        if isinstance(url, dict) and "alternate" in url and "s3" in url["alternate"]:
            s3_url = url["alternate"]["s3"]["href"]
            logger.debug(f"Using alternate S3 URL: {s3_url}")
            url = s3_url
        # Then check URL patterns if defined
        elif config.url_patterns:
            for http_pattern, s3_pattern in config.url_patterns.items():
                if url.startswith(http_pattern):
                    url = url.replace(http_pattern, s3_pattern)
                    logger.debug(f"Converted to S3 URL: {url}")
                    break

        # Sign URL if it's an S3 URL
        if url.startswith("s3://"):
            try:
                bucket = url.split("/")[2]
                key = "/".join(url.split("/")[3:])

                params = {
                    "Bucket": bucket,
                    "Key": key,
                }
                if config.requester_pays:
                    params["RequestPayer"] = "requester"

                return self.client.generate_presigned_url(
                    "get_object", Params=params, ExpiresIn=3600
                )
            except Exception as e:
                logger.error(f"Failed to sign URL {url}: {str(e)}")
                return None

        return url


__all__ = ["CloudConfig", "AWSProvider"]
