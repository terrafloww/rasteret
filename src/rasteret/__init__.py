'''
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
'''

"""Rasteret package."""

from importlib.metadata import version as get_version

from rasteret.core.processor import Rasteret
from rasteret.core.collection import Collection
from rasteret.cloud import CloudConfig, AWSProvider
from rasteret.constants import DataSources
from rasteret.logging import setup_logger

# Set up logging
setup_logger("INFO")


def version():
    """Return the version of the rasteret package."""
    return get_version("rasteret")


__version__ = version()


__all__ = [
    "Collection",
    "Rasteret",
    "CloudConfig",
    "AWSProvider",
    "DataSources",
]
