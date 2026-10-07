from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pyfedappwrap.engine.federated.models import FLNetMessageMetaDTO


class AppAggregator(ABC):
    def secure_round(self, round_nr: int) -> bool:
        """
        Whether the clients send round ``round_nr`` (counted from 1) with SMPC.

        On a real controller a secure round delivers one package, the sum of all relay clients,
        instead of one package per client. The in-memory test controller ignores SMPC and still
        delivers one package per client. The clients pass ``smpc=`` in exactly these rounds.
        """
        return False

    @abstractmethod
    def aggregate(self, data: list[Any], n_clients: int,
                  meta: Optional[FLNetMessageMetaDTO] = None) -> Any:
        """
        Aggregate client payloads into a single result.

        ``n_clients`` is the number of sites whose data is in ``data``. It equals ``len(data)``
        except in a secure round on a real controller, where ``data`` holds the relay clients' sum.
        When the coordinator also trains, its own payload is the last element of ``data``, never
        part of a sum.
        """
        raise NotImplementedError
