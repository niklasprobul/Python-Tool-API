"""Secure aggregation (SMPC) in some rounds of a run: plaintext round 1, secure rounds after it.

On a real controller a secure round delivers one summed package for all relay clients; the in-memory
controller ignores SMPC and delivers one package per site. The same app must complete on both.

Round 1 carries strings (the site names), which SMPC cannot sum. Every later round sends 240,000 floats
per site as a list of sub-lists, because the controller decodes at most 131,072 elements per CBOR array.
The values are quarters, so FL-Net's fixed-point conversion at exponent 8 keeps them exact.
"""
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
from pydantic.dataclasses import dataclass

from pyfedappwrap.engine.config.system_config import FLNetSMPCSettings, system_settings
from pyfedappwrap.engine.federated import (
    AppAggregator,
    FLNetLocalParticipantConfigDTO,
    FLNetLocalTestConfigDTO,
    FLNetMessageMetaDTO,
)
from pyfedappwrap.engine.service.socket.messages.federated import FederatedTestRunCreateDTO
from pyfedappwrap.engine.tests.federated.controller_client import FederatedDockerizedControllerClient
from pyfedappwrap.engine.tests.federated.dockerized_services import COMPOSE_ENV_VAR
from pyfedappwrap.engine.tests.federated.runner import LocalFederatedRunner
from pyfedappwrap.learning.federated import BaseFederatedApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig

AGGREGATOR = "secure-sum"
MAX_LIST_LENGTH = 131_072
SITES = ["coordinator", "site-1", "site-2"]
ROUNDS = 4
N_VALUES = 240_000
GHCR_COMPOSE_FILE = Path(__file__).parent / "configs" / "fed-learn-sim-docker-compose.ghcr.yml"


def is_secure_round(round_nr: int) -> bool:
    return round_nr > 1


def site_values(site_id: str, round_nr: int) -> np.ndarray:
    rng = np.random.default_rng([round_nr, *site_id.encode()])
    return rng.integers(-4000, 4000, N_VALUES) / 4


def chunked(values: np.ndarray) -> list[list[float]]:
    return [values[i:i + MAX_LIST_LENGTH].tolist() for i in range(0, len(values), MAX_LIST_LENGTH)]


@dataclass
class SecureSumConfig(AppConfig):
    federated_rounds: int = ROUNDS


@dataclass
class SecureSumInput(AppInputConfig):
    input: Any = None


@dataclass
class SecureSumOutput(AppOutputConfig):
    output: Any = None


class SecureSumClientApp(BaseFederatedApp[SecureSumConfig, SecureSumInput, SecureSumOutput]):
    def run_train(self, data: SecureSumInput) -> SecureSumOutput:
        replies = []
        for round_nr in range(1, self.config.federated_rounds + 1):
            if is_secure_round(round_nr):
                payload = chunked(site_values(self.federated_client_id, round_nr))
            else:
                payload = {"site": self.federated_client_id}
            replies.append(self.communicator.aggregate(
                payload, AGGREGATOR, communication_id=f"secure-sum-round-{round_nr}",
                smpc=FLNetSMPCSettings(exponent=8) if is_secure_round(round_nr) else None).data)
        return SecureSumOutput(output=pd.DataFrame(replies))

    def run_prediction(self, data: SecureSumInput) -> SecureSumOutput:
        return self.run_train(data)

    def _save(self) -> str:
        return "client"

    def _load(self, path: str):
        return path


class SecureSumAggregator(AppAggregator):
    """Round 1 collects the site names; every later round sums the packages element-wise, which serves
    both the relay clients' sum plus the coordinator's payload and one payload per site."""

    def __init__(self):
        self.round_nr = 0
        self.sites: list[str] = []
        self.sums: dict[int, np.ndarray] = {}
        self.package_counts: dict[int, int] = {}
        self.n_clients: dict[int, int] = {}

    def secure_round(self, round_nr: int) -> bool:
        return is_secure_round(round_nr)

    def aggregate(self, data: list[Any], n_clients: int,
                  meta: Optional[FLNetMessageMetaDTO] = None) -> Any:
        self.round_nr += 1
        self.package_counts[self.round_nr] = len(data)
        self.n_clients[self.round_nr] = n_clients
        if self.round_nr == 1:
            self.sites = sorted(payload["site"] for payload in data)
            return {"round": 1, "sites": self.sites}
        self.sums[self.round_nr] = np.sum([np.concatenate(payload) for payload in data], axis=0)
        return {"round": self.round_nr, "n_sites": n_clients}


def _participants(tmp_path: Path) -> list[FLNetLocalParticipantConfigDTO]:
    participants = []
    for site in SITES:
        base_dir = tmp_path / site
        (base_dir / "data").mkdir(parents=True)
        (base_dir / "output").mkdir(parents=True)
        participants.append(FLNetLocalParticipantConfigDTO(
            participant_id=site,
            role="aggregator" if site == "coordinator" else "client",
            base_dir=base_dir,
            hyper_params={"federated_rounds": ROUNDS},
        ))
    return participants


def _run(config: FLNetLocalTestConfigDTO) -> tuple[dict, SecureSumAggregator]:
    system_settings.config_settings_path = "app_federated.yml"
    aggregator = SecureSumAggregator()
    runner = LocalFederatedRunner(config, aggregators={AGGREGATOR: aggregator})
    results = runner.run({site: SecureSumClientApp() for site in SITES})
    return results, aggregator


def _assert_every_round_summed(results: dict, aggregator: SecureSumAggregator) -> None:
    assert all(result.success for result in results.values()), results
    assert aggregator.sites == SITES
    for round_nr in range(2, ROUNDS + 1):
        expected = np.sum([site_values(site, round_nr) for site in SITES], axis=0)
        np.testing.assert_array_equal(aggregator.sums[round_nr], expected)
        assert aggregator.n_clients[round_nr] == len(SITES)
    for site in SITES[1:]:
        assert results[site].result.output["round"].tolist() == list(range(1, ROUNDS + 1))


def test_secure_rounds_complete_on_the_in_memory_controller_with_one_package_per_site(tmp_path: Path):
    results, aggregator = _run(FLNetLocalTestConfigDTO(participants=_participants(tmp_path), timeout=60.0))

    _assert_every_round_summed(results, aggregator)
    assert aggregator.package_counts == {round_nr: len(SITES) for round_nr in range(1, ROUNDS + 1)}


@pytest.mark.skipif(
    not system_settings.fl_test.use_dockerized_controller,
    reason="needs Docker; set FL_TEST__USE_DOCKERIZED_CONTROLLER=true",
)
def test_secure_rounds_complete_on_the_dockerized_controller_with_one_summed_package(tmp_path: Path,
                                                                                       monkeypatch):
    if not os.getenv(COMPOSE_ENV_VAR):
        monkeypatch.setenv(COMPOSE_ENV_VAR, str(GHCR_COMPOSE_FILE))
    participants = _participants(tmp_path)
    config = FLNetLocalTestConfigDTO(participants=participants, timeout=120.0, poll_interval=0.05,
                                     max_polls=2400)
    FederatedDockerizedControllerClient().setup_dockerized_fl_run(
        FederatedTestRunCreateDTO(id=1), config, participants)

    results, aggregator = _run(config)

    _assert_every_round_summed(results, aggregator)
    # The relay clients' sum, then the coordinator's own payload.
    assert aggregator.package_counts == {1: len(SITES), **{round_nr: 2 for round_nr in range(2, ROUNDS + 1)}}
