"""Abstract base class shared by all agents."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    """Base class for all agents in the fake-news detection system.

    Each agent encapsulates a distinct analytical responsibility and exposes
    a single ``process`` entry-point.  Agents are intentionally stateless so
    that they can be reused across batches without side-effects.
    """

    def __init__(self, name: str, verbose: bool = False) -> None:
        self.name = name
        self.verbose = verbose
        self._log = logging.getLogger(f"agent.{name}")
        if verbose:
            logging.basicConfig(level=logging.DEBUG)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Run the agent on *input_data* and return the result."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _info(self, msg: str) -> None:
        self._log.info("[%s] %s", self.name, msg)

    def _debug(self, msg: str) -> None:
        self._log.debug("[%s] %s", self.name, msg)

    def _warn(self, msg: str) -> None:
        self._log.warning("[%s] %s", self.name, msg)

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(name={self.name!r})"
