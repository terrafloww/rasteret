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


__all__ = ["Collection", "Rasteret", "CloudConfig", "AWSProvider", "DataSources"]
