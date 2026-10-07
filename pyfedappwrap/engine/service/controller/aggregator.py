from __future__ import annotations

from typing import Any, Optional, List, Dict
import time

from pyfedappwrap.engine.config.system_config import FLNetDPSettings
from pyfedappwrap.engine.federated.aggregator import AppAggregator
from pyfedappwrap.engine.federated.models import FLNetDataPackageDTO
from pyfedappwrap.engine.service.controller.common import FLNetCommunicator


class FLNetCommunicatorAggregator(FLNetCommunicator):
    """Communicator for the aggregator (coordinator) side of a federated run.

    Aggregator developers interact with three methods:
    - ``await_data_from_clients`` — wait for client submissions for a round
    - ``broadcast``               — send the aggregated result back to all clients
    - ``receive_setup``           — query run metadata (client order, max clients)

    Aggregation logic itself is intentionally **not** provided here — implement
    your own ``AppAggregator`` and call it between ``await_data_from_clients``
    and ``broadcast``.
    """

    def __init__(self, *args, aggregators: Optional[dict[str, AppAggregator]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._aggregators: dict[str, AppAggregator] = dict(aggregators or {})

    def register_aggregator(self, aggregator: AppAggregator,
                            key: str = "default") -> None:
        self._aggregators[key] = aggregator

    def aggregate(self, packages: List[FLNetDataPackageDTO],
                  aggregator_name: str, n_clients: Optional[int] = None) -> Any:
        aggregator = self._aggregators.get(aggregator_name)
        if aggregator is None:
            raise KeyError(
                f"No runtime aggregator registered for {aggregator_name!r}."
            )
        if not packages:
            raise ValueError("Cannot aggregate an empty package list.")
        return aggregator.aggregate(
            [package.data for package in packages],
            n_clients=len(packages) if n_clients is None else n_clients,
            meta=packages[-1].meta,
        )

    def await_data_from_clients(
        self,
        *,
        to_aggregator: Optional[str] = None,
        num_data_packages_per_communication_round: int = 1,
        communication_id: Optional[str] = None,
        data_type: Any = None,
    ) -> Dict[str, List[FLNetDataPackageDTO]]:
        """Wait until at least one complete communication round is available.

        Polls ``/receive-data-from-clients`` until at least one group of
        ``num_data_packages_per_communication_round`` packages arrives under the
        same communication_id.

        Args:
            to_aggregator: The aggregator name this communicator is collecting for.
            num_data_packages_per_communication_round: Minimum number of client
                packages required before a round is considered complete.
            communication_id: If given, wait only for that specific round.
                If None, accept any round.
            data_type: Optional type hint for deserialising the payload data.

        Returns:
            A dict mapping communication_id → list of packages. Only rounds with
            at least ``num_data_packages_per_communication_round`` packages are
            included.

        Raises:
            TimeoutError: If the deadline passes without a complete round.
        """
        poll_count = 0
        deadline = time.monotonic() + self.timeout
        while poll_count < self.max_polls:
            grouped_packages = self._receive_from_clients_grouped(
                min_packages_per_round=num_data_packages_per_communication_round,
                from_clients=None,  # accept from any client
                to_aggregator=to_aggregator,
                communication_id=communication_id,
                data_type=data_type,
            )
            if grouped_packages:
                return grouped_packages
            else:
                poll_count += 1
                if not self._wait_before_next_receive(
                    poll_count=poll_count,
                    communication_id=communication_id,
                    from_clients=None,
                    deadline=deadline,
                ):
                    break
        raise TimeoutError(
            f"Did not receive data from clients for aggregator {to_aggregator!r} "
            f"after {self.max_polls} polls with interval {self.poll_interval}s."
        )

    def broadcast(
        self,
        communication_id: str,
        data: Any,
        from_aggregator: str,
        *,
        dp: Optional[FLNetDPSettings] = None,
    ) -> str:
        """Broadcast the aggregated result to all clients for this round.

        Args:
            communication_id: The communication_id of the round being responded to.
                Clients use this to match the broadcast to the round they submitted.
                Only ONE package per communication_id is allowed to be broadcast by the aggregator.
            data: The aggregated payload to send.
            from_aggregator: The logical aggregator name to echo back in the broadcast.
                Clients use this to filter the reply via ``await_data_from_aggregator``.
                Defaults to ``self.aggregator_id`` if not set.
            dp: Optional differential-privacy settings to apply on send.

        Returns:
            The communication_id that was used.
        """
        return self._send_to_clients_broadcast(
            data,
            communication_id=communication_id,
            from_aggregator=from_aggregator,
            dp=dp,
        )
