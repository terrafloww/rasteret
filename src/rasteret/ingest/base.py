# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

"""Base class for collection builders.

Every builder takes an external data source and produces a
:class:`~rasteret.core.collection.Collection` backed by Rasteret's
Parquet schema.  The four required columns are ``id``, ``datetime``,
``geometry``, ``assets``; everything else is source-specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from rasteret.core.collection import Collection


class CollectionBuilder(ABC):
    """Abstract base class for all collection builders.

    Subclasses implement :meth:`build` to acquire data from their
    specific source, normalise it, and return a ``Collection``.

    Parameters
    ----------
    name : str
        Human-readable collection name.
    data_source : str
        Data source identifier for band mapping and URL policy.
    workspace_dir : Path, optional
        If set, persist the collection as partitioned Parquet.
    """

    def __init__(
        self,
        *,
        name: str = "",
        data_source: str = "",
        workspace_dir: str | Path | None = None,
    ) -> None:
        self.name = name
        self.data_source = data_source
        if workspace_dir is None:
            self.workspace_dir: str | Path | None = None
        elif isinstance(workspace_dir, Path):
            self.workspace_dir = workspace_dir
        else:
            ws = str(workspace_dir)
            if "://" in ws and not ws.startswith("file://"):
                self.workspace_dir = ws
            else:
                self.workspace_dir = Path(ws)

    @abstractmethod
    def build(self, **kwargs: Any) -> "Collection":
        """Build and return a Collection.

        Each subclass decides how to acquire data and normalise it
        into the Collection contract.
        """
        ...
