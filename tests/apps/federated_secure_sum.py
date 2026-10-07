"""A federated app that sums lists: round 1 in plaintext, every later round with secure aggregation (SMPC).

Round 1 carries strings (the site names), which SMPC cannot sum. Every later round sends about 240,000
floats per site as a list of sub-lists, because the controller decodes at most 131,072 elements per CBOR
array. The values are quarters, so FL-Net's fixed-point conversion at exponent 8 keeps them exact.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass

from pyfedappwrap.engine.config.system_config import FLNetSMPCSettings
from pyfedappwrap.engine.federated import AppAggregator, FLNetMessageMetaDTO
from pyfedappwrap.learning.federated import BaseFederatedApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig

AGGREGATOR = "secure-sum"
MAX_LIST_LENGTH = 131_072


@dataclass
class SecureSumConfig(AppConfig):
    federated_rounds: int = 4
    n_values: int = 240_000


@dataclass
class SecureSumInput(AppInputConfig):
    input: Any = None


@dataclass
class SecureSumOutput(AppOutputConfig):
    output: Any = None


def site_values(site_id: str, round_nr: int, n_values: int) -> np.ndarray:
    rng = np.random.default_rng([round_nr, *site_id.encode()])
    return rng.integers(-4000, 4000, n_values) / 4


def chunked(values: np.ndarray) -> list[list[float]]:
    return [values[i:i + MAX_LIST_LENGTH].tolist() for i in range(0, len(values), MAX_LIST_LENGTH)]


class SecureSumClientApp(BaseFederatedApp[SecureSumConfig, SecureSumInput, SecureSumOutput]):
    def run_train(self, data: SecureSumInput) -> SecureSumOutput:
        config = self.config or SecureSumConfig()
        replies = [self.communicator.aggregate(
            {"site": self.federated_client_id}, AGGREGATOR, communication_id="secure-sum-round-1").data]
        for round_nr in range(2, config.federated_rounds + 1):
            payload = chunked(site_values(self.federated_client_id, round_nr, config.n_values))
            replies.append(self.communicator.aggregate(
                payload, AGGREGATOR, communication_id=f"secure-sum-round-{round_nr}",
                smpc=FLNetSMPCSettings(exponent=8)).data)
        return SecureSumOutput(output=pd.DataFrame(replies))

    def run_prediction(self, data: SecureSumInput) -> SecureSumOutput:
        return self.run_train(data)

    def _save(self) -> str:
        return "client"

    def _load(self, path: str):
        return path


class SecureSumAggregator(AppAggregator):
    """Round 1 collects the site names; every later round sums the packages element-wise.

    The packages of a secure round are the controller's sum of the relay clients plus the coordinator's
    own payload, or one payload per site on the in-memory controller. Summing all of them serves both.
    """

    def __init__(self):
        self.round_nr = 0
        self.sites: list[str] = []
        self.sums: dict[int, np.ndarray] = {}
        self.packages: dict[int, int] = {}
        self.n_clients: dict[int, int] = {}

    def secure_round(self, round_nr: int) -> bool:
        return round_nr > 1

    def aggregate(self, data: list[Any], n_clients: int,
                  meta: Optional[FLNetMessageMetaDTO] = None) -> Any:
        self.round_nr += 1
        self.packages[self.round_nr] = len(data)
        self.n_clients[self.round_nr] = n_clients
        if self.round_nr == 1:
            self.sites = sorted(payload["site"] for payload in data)
            return {"round": 1, "sites": self.sites}
        self.sums[self.round_nr] = np.sum([np.concatenate(payload) for payload in data], axis=0)
        return {"round": self.round_nr, "n_sites": n_clients}
