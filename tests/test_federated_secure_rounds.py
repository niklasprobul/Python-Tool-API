"""Secure aggregation (SMPC) in some rounds of a run: plaintext round 1, secure rounds after it.

On a real controller a secure round delivers one summed package for all relay clients; the in-memory
controller ignores SMPC and delivers one package per site. The same app must complete on both.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from pyfedappwrap.engine.config.system_config import system_settings
from pyfedappwrap.engine.federated import FLNetLocalParticipantConfigDTO, FLNetLocalTestConfigDTO
from pyfedappwrap.engine.service.socket.messages.federated import FederatedTestRunCreateDTO
from pyfedappwrap.engine.tests.federated.controller_client import FederatedDockerizedControllerClient
from pyfedappwrap.engine.tests.federated.dockerized_services import COMPOSE_ENV_VAR
from pyfedappwrap.engine.tests.federated.runner import LocalFederatedRunner
from tests.apps.federated_secure_sum import (
    AGGREGATOR,
    SecureSumAggregator,
    SecureSumClientApp,
    site_values,
)

SITES = ["coordinator", "site-1", "site-2"]
ROUNDS = 4
N_VALUES = 240_000
GHCR_COMPOSE_FILE = Path(__file__).parent / "configs" / "fed-learn-sim-docker-compose.ghcr.yml"


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
            hyper_params={"federated_rounds": ROUNDS, "n_values": N_VALUES},
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
        expected = np.sum([site_values(site, round_nr, N_VALUES) for site in SITES], axis=0)
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
