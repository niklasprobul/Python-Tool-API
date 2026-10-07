from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import TypeAdapter

from pyfedappwrap.engine.config.config import ModeType
from pyfedappwrap.engine.config.system_config import local_runtime_paths, system_settings
from pyfedappwrap.engine.federated.aggregator import AppAggregator
from pyfedappwrap.engine.federated.reporter import FederatedStatusReporter
from pyfedappwrap.engine.federated.models import (
    FederatedParticipantType,
    FLNetLocalParticipantConfigDTO,
    FLNetLocalTestConfigDTO,
    FLNetLocalThreadResultDTO,
)
from pyfedappwrap.engine.runtime_lifecycle import EngineLifecycle
from pyfedappwrap.engine.service.socket.messages.app import SendFinishRunDTO, SendUpdateRunDTO
from pyfedappwrap.engine.service.socket.messages.message import (
    RunMessageTypes,
    SendLogDTO,
    TestRunMessageLogDTO,
)
from pyfedappwrap.engine.service.controller.aggregator import FLNetCommunicatorAggregator
from pyfedappwrap.engine.enums.test_embed_states import RunType
from pyfedappwrap.engine.tests.federated.coordinator_bridge import CoordinatorTrainingBridge
from pyfedappwrap.engine.tests.federated.controller import FLNetInMemoryController, \
    FLNetInMemoryControllerSession
from pyfedappwrap.engine.tests.test_upload_client import TestUploadClient
from pyfedappwrap.engine.worker.data_manager import get_data


logger = logging.getLogger(__name__)


class _FederatedAppSocketProxy:
    def __init__(self, delegate):
        self.delegate = delegate
        self._observers: list[Any] = []

    def register(self, observer):
        if observer not in self._observers:
            self._observers.append(observer)

    def unregister(self, observer):
        try:
            self._observers.remove(observer)
        except ValueError:
            pass

    def send(self, message):
        if isinstance(message, (SendFinishRunDTO, SendUpdateRunDTO)):
            return
        if self.delegate is None:
            return
        self.delegate.send(message)

    def send_silent(self, message):
        if self.delegate is None:
            return
        send_silent = getattr(self.delegate, "send_silent", None)
        if callable(send_silent):
            send_silent(message)


class LocalFederatedRunner:
    def __init__(self, config: FLNetLocalTestConfigDTO,
                 aggregators: dict[str, AppAggregator] | None = None,
                 ws_client=None,
                 upload_client=None,
                 lifecycle: EngineLifecycle | None = None,
                 reporter: FederatedStatusReporter | None = None,
                 run_id: int | None = None,
                 run_type: RunType = RunType.FEDERATED_RUN):
        self.config = config
        self.controller = None if self._uses_external_controller() else FLNetInMemoryController(config)
        if self.controller is None:
            logger.info(
                "LocalFederatedRunner using external federated controller url=%s channel=%s simulate_participants_locally=%s",
                self.config.controller_url,
                self.config.channel,
                self.config.simulate_participants_locally,
            )
        else:
            logger.info(
                "LocalFederatedRunner using in-memory federated controller channel=%s simulate_participants_locally=%s",
                self.config.channel,
                self.config.simulate_participants_locally,
            )
        self._aggregators: dict[str, AppAggregator] = dict(aggregators or {})
        self.ws_client = _FederatedAppSocketProxy(ws_client)
        self.upload_client = upload_client or TestUploadClient(config.app_key)
        self.lifecycle = lifecycle or EngineLifecycle()
        self.reporter = reporter
        self.run_id = run_id or 0
        self.run_type = run_type

    def register_aggregator(self, aggregator: AppAggregator, key: str = "default"):
        self._aggregators[key] = aggregator

    def build_session(self, participant_id: str) -> FLNetInMemoryControllerSession:
        if self.controller is None:
            raise RuntimeError("External federated controller mode does not use in-memory sessions.")
        return FLNetInMemoryControllerSession(self.controller, participant_id)

    def _uses_external_controller(self) -> bool:
        return self.config.use_external_controller

    def _controller_url(self) -> str:
        if self._uses_external_controller():
            controller_url = self.config.controller_url
            if not controller_url:
                raise ValueError(
                    "External federated runs require a controller URL. "
                    "Set FL_RUN__CONTROLLER_COMM_URL for real runs, "
                    "or FL_TEST__DOCKERIZED_CONTROLLER_COMM_URL for dockerized tests."
                )
            return controller_url.rstrip("/")
        return "http://local-flnet"

    def _make_traffic_logger(self, participant_id: str):
        """Build the callable the communicator invokes after each controller HTTP call. Persists the
        traffic as a run log message with group CONTROLLER (bytes sent/received per endpoint)."""
        def _log_traffic(path: str, bytes_sent: int, bytes_received: int, status_code: int):
            # Skip empty polls (HTTP 204 = no data yet) to avoid flooding the log while a round waits;
            # only record traffic that actually carried a payload.
            if status_code == 204:
                return
            log = TestRunMessageLogDTO(
                process="controller-traffic",
                type=RunMessageTypes.LOG,
                message=f"POST {path} sent={bytes_sent}B received={bytes_received}B status={status_code}",
                run_id=self.run_id,
                worker_id=participant_id,
                severity="INFO",
                caller="FLNetCommunicator",
                stack_trace=None,
                group="CONTROLLER",
            )
            self.ws_client.send(SendLogDTO(message=log, run_type=self.run_type))
        return _log_traffic

    def _controller_app_key(self, participant: FLNetLocalParticipantConfigDTO) -> str:
        # The controller stores the run under RunKey{channel, appKey}, registered by the learning
        # API with appKey = relay client id. Every data call must present the same key, so use the
        # participant id (= relay client id) - system_settings.app_id is the tool id, identical in
        # every clinic, and the controller answers 404 "no such run key" for it.
        return participant.participant_id

    @staticmethod
    def _participant_data_dir(participant: FLNetLocalParticipantConfigDTO) -> str:
        return str(participant.data_dir or (participant.base_dir / "data"))

    @staticmethod
    def _participant_output_dir(participant: FLNetLocalParticipantConfigDTO) -> str:
        return str(participant.output_dir or (participant.base_dir / "output"))

    @staticmethod
    def _noop_stop_worker(_error=None):
        return None

    def configure_app(self, app: Any, participant: FLNetLocalParticipantConfigDTO):
        if hasattr(app, "configure_federation"):
            app.configure_federation(
                controller_url=self._controller_url(),
                app_key=self.config.app_key,
                controller_app_key=self._controller_app_key(participant),
                client_id=participant.participant_id,
                client_ids=self.config.resolve_client_ids(),
                aggregator_id=self.config.aggregator_id,
                channel=self.config.channel,
            )

        config_type = getattr(app, "c_type", None)
        if config_type is not None:
            config_adapter = TypeAdapter(config_type)
            config_instance = config_adapter.validate_python(participant.hyper_params)
            if hasattr(app, "on_config_loaded"):
                app.on_config_loaded(config_instance)
            else:
                app.config = config_instance

        if hasattr(app, "set_startup"):
            app.set_startup(
                self.ws_client,
                self.upload_client,
                participant.participant_id,
                self.run_id,
                self.run_type,
                self.lifecycle,
                self._noop_stop_worker,
            )

        communicator = getattr(app, "communicator", None)
        if communicator is not None:
            if not self._uses_external_controller():
                communicator.session = self.build_session(participant.participant_id)
            communicator.max_polls = self.config.max_polls
            communicator.timeout = self.config.timeout
            communicator.poll_interval = self.config.poll_interval
            if (
                not self._uses_external_controller()
                and self.controller is not None
                and hasattr(communicator, "bind_notice_subject")
            ):
                communicator.bind_notice_subject(self.controller.notice_subject)
            if hasattr(communicator, "attach_status_reporter") and self.reporter is not None:
                communicator.attach_status_reporter(self.reporter, participant.participant_id)
            if hasattr(communicator, "attach_traffic_logger"):
                communicator.attach_traffic_logger(self._make_traffic_logger(participant.participant_id))

    def _build_aggregator_communicator(self) -> FLNetCommunicatorAggregator:
        communicator = FLNetCommunicatorAggregator(
            self._controller_url(),
            self.config.app_key,
            controller_app_key=self._controller_app_key(self.config.aggregator_participant),
            client_id=self.config.aggregator_id,
            client_ids=self.config.resolve_client_ids(),
            aggregator_id=self.config.aggregator_id,
            channel=self.config.channel,
            session=None if self._uses_external_controller() else self.build_session(self.config.aggregator_id),
            timeout=self.config.timeout,
            poll_interval=self.config.poll_interval,
            max_polls=self.config.max_polls,
        )
        if not self._uses_external_controller() and self.controller is not None:
            communicator.bind_notice_subject(self.controller.notice_subject)
        if hasattr(communicator, "attach_status_reporter") and self.reporter is not None:
            communicator.attach_status_reporter(self.reporter, self.config.aggregator_id)
        if hasattr(communicator, "attach_traffic_logger"):
            communicator.attach_traffic_logger(self._make_traffic_logger(self.config.aggregator_id))
        for key, aggregator in self._aggregators.items():
            communicator.register_aggregator(aggregator, key)
        return communicator

    def _resolve_rounds_from_app(self, app: Any,
                                 participant: FLNetLocalParticipantConfigDTO) -> "int | None":
        """Resolve the federated round count the way the *training app* does: validate the
        participant's hyper_params through the app's config dataclass so dataclass defaults (e.g.
        ``federated_rounds=10``) are applied. This keeps the aggregator's loop length in lockstep with
        the training apps even when hyper_params omits the key (otherwise the aggregator falls back to
        1 round while the apps loop 10, and every round after the first deadlocks)."""
        config_type = getattr(app, "c_type", None)
        if config_type is None:
            return None
        try:
            config_instance = TypeAdapter(config_type).validate_python(participant.hyper_params)
        except Exception:  # noqa: BLE001 - resolution is best-effort; callers fall back
            return None
        for attr in ("federated_rounds", "total_rounds"):
            value = getattr(config_instance, attr, None)
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
        return None

    def _aggregator_round_count(self, participant: FLNetLocalParticipantConfigDTO) -> int:
        candidate_values = [
            participant.hyper_params.get("federated_rounds"),
            participant.hyper_params.get("total_rounds"),
        ]
        candidate_values.extend(
            item.hyper_params.get("federated_rounds")
            for item in self.config.participants
            if item.role == FederatedParticipantType.CLIENT
        )
        candidate_values.extend(
            item.hyper_params.get("total_rounds")
            for item in self.config.participants
            if item.role == FederatedParticipantType.CLIENT
        )
        for value in candidate_values:
            if value is None:
                continue
            try:
                return max(1, int(value))
            except (TypeError, ValueError):
                continue
        return 1

    def _secure_round(self, round_nr: int) -> bool:
        """Ask the registered aggregators whether the clients send this round with SMPC. The round's
        aggregator is known only from its packages, so all of them must agree."""
        answers = {aggregator.secure_round(round_nr) for aggregator in self._aggregators.values()}
        if len(answers) > 1:
            raise ValueError(f"The registered aggregators disagree on whether round {round_nr} is secure.")
        return answers == {True}

    @staticmethod
    def _aggregator_output_filename(participant: FLNetLocalParticipantConfigDTO) -> str:
        return participant.hyper_params.get("output_filename", "aggregated.csv")

    @staticmethod
    def _aggregator_enabled(config: FLNetLocalTestConfigDTO) -> bool:
        return config.start_aggregator

    def _participants_to_execute(self) -> list[FLNetLocalParticipantConfigDTO]:
        if self.config.simulate_participants_locally:
            return list(self.config.participants)

        app_id_matches = [
            participant for participant in self.config.participants
            if participant.participant_id == system_settings.app_id
        ]
        if len(app_id_matches) == 1:
            return app_id_matches

        if len(self.config.participants) == 1:
            return list(self.config.participants)

        hyperparam_matches = [
            participant for participant in self.config.participants
            if str(participant.hyper_params.get("app_id") or "") == system_settings.app_id
        ]
        if len(hyperparam_matches) == 1:
            return hyperparam_matches

        local_candidates = [
            participant for participant in self.config.participants
            if participant.hyper_params.get("local") is True
        ]
        if len(local_candidates) == 1:
            return local_candidates

        if len(app_id_matches) > 1:
            raise ValueError(
                f"Multiple participants match app_id {system_settings.app_id!r}; "
                "real federated runs require one local participant."
            )

        raise ValueError(
            "Real federated runs require the participants payload to identify "
            "this app instance, either by participantId matching system_settings.app_id "
            "or by providing only one participant."
        )

    def _run_aggregator(self, participant: FLNetLocalParticipantConfigDTO,
                        bridge: "CoordinatorTrainingBridge | None" = None,
                        n_rounds: "int | None" = None):
        communicator = self._build_aggregator_communicator()
        if n_rounds is None:
            n_rounds = self._aggregator_round_count(participant)
        output_dir = self._participant_output_dir(participant)
        aggregated = None

        n_relay_clients = len(communicator.client_ids)
        try:
            for round_nr in range(1, n_rounds + 1):
                # In a secure round a real controller delivers the relay clients' SMPC sum as one
                # package; the in-memory controller ignores SMPC and delivers one per client.
                summed = self._secure_round(round_nr) and self._uses_external_controller()
                n_packages = 1 if summed else n_relay_clients
                logger.info("[AGGREGATOR] round %d/%d: awaiting %d relay client package(s)%s%s",
                            round_nr, n_rounds, n_packages,
                            " (SMPC sum)" if summed else "",
                            " + 1 coordinator (bridge)" if bridge is not None else "")
                # communication_id=None: accept whatever round comm-id the clients send (they use a
                # per-round id like '<base>-round-N'); passing the base would never match it (the
                # controller does exact matching on a manual comm-id) and the round would never complete.
                grouped_packages = communicator.await_data_from_clients(
                    to_aggregator=None,
                    num_data_packages_per_communication_round=n_packages,
                    communication_id=None,
                )
                communication_id, packages = next(iter(grouped_packages.items()))
                communication_id = packages[0].communication_id
                latest_meta = packages[-1].meta
                aggregator_key = latest_meta.to_aggregator or "default"
                logger.info("[AGGREGATOR] round %d: received %d relay package(s) for comm_id=%s",
                            round_nr, len(packages), communication_id)
                # When the coordinator also trains, fold its own local update (handed over the
                # in-process bridge, since it cannot reach its own aggregator via the relay) into
                # this round's packages so the global model includes the coordinator's data.
                if bridge is not None:
                    own = bridge.take_contribution(communication_id)
                    if own is not None:
                        packages = [*packages, own]
                    else:
                        logger.warning(
                            "Coordinator contribution missing for %s; aggregating relay clients only",
                            communication_id,
                        )
                n_sites = len(packages) + (n_relay_clients - 1 if summed else 0)
                aggregated = communicator.aggregate(packages, aggregator_key, n_clients=n_sites)
                logger.info("[AGGREGATOR] round %d: aggregated %d package(s); broadcasting comm_id=%r from_aggregator=%r "
                            "(clients await from_aggregator they sent to)",
                            round_nr, len(packages), communication_id, aggregator_key)
                communicator.broadcast(
                    communication_id,
                    aggregated,
                    from_aggregator=aggregator_key,
                )
                # Release the coordinator's training thread with the same aggregated result the
                # relay clients just received via broadcast.
                if bridge is not None:
                    bridge.publish_result(communication_id, aggregated)
                logger.info("[AGGREGATOR] round %d complete (comm_id=%s)", round_nr, communication_id)
            with local_runtime_paths(output_dir=output_dir):
                self._save_aggregated_result(
                    aggregated,
                    self._aggregator_output_filename(participant),
                )
            return aggregated
        finally:
            communicator.unbind_notice_subject()

    @staticmethod
    def _save_aggregated_result(result: Any, filename: str) -> None:
        from pyfedappwrap.engine.config.system_config import get_output_dir
        output_path = Path(get_output_dir())
        output_path.mkdir(parents=True, exist_ok=True)
        target = output_path / filename

        if isinstance(result, pd.DataFrame):
            dataframe = result
        elif hasattr(result, "tolist"):
            dataframe = pd.DataFrame([result.tolist()])
        elif isinstance(result, (list, tuple)):
            dataframe = pd.DataFrame([list(result)])
        else:
            dataframe = pd.DataFrame([{"result": result}])
        dataframe.to_csv(target, index=False)

    def build_input(self, app: Any, participant: FLNetLocalParticipantConfigDTO):
        input_type = getattr(app, "i_type", None)
        if input_type is None:
            return None

        with local_runtime_paths(
            data_dir=self._participant_data_dir(participant),
            output_dir=self._participant_output_dir(participant),
        ):
            raw_input = get_data(
                input_data_dic={},
                input_file_paths=participant.input_file_paths,
                input_model_type=input_type,
                token=None,
            )

        return TypeAdapter(input_type).validate_python(raw_input)

    def _build_app_thread(self, participant, app, results, results_lock, mode,
                          result_key=None, communicator_override=None) -> threading.Thread:
        """Build (but don't start) the thread that runs a participant's training app. configure_app is
        done synchronously here so app.config/communicator are ready before the thread runs; an
        optional communicator_override replaces the relay client with the coordinator's in-process
        bridge."""
        result_key = result_key or participant.participant_id
        self.configure_app(app, participant)
        if communicator_override is not None:
            app.communicator = communicator_override
        input_config = self.build_input(app, participant)
        data_dir = self._participant_data_dir(participant)
        output_dir = self._participant_output_dir(participant)

        def _target(rk=result_key, bound_app=app, bound_input=input_config,
                    bound_output_dir=output_dir, bound_data_dir=data_dir, bound_config=app.config):
            try:
                with local_runtime_paths(data_dir=bound_data_dir, output_dir=bound_output_dir):
                    captured_output: dict[str, Any] = {}
                    original_send_output = bound_app.send_output

                    def _capturing_send_output(output, output_mode=ModeType.TRAINING):
                        captured_output["value"] = output
                        return original_send_output(output, output_mode)

                    bound_app.send_output = _capturing_send_output
                    try:
                        bound_app.start(bound_config, bound_input, mode)
                    finally:
                        bound_app.send_output = original_send_output
                    result = captured_output.get("value")
                outcome = FLNetLocalThreadResultDTO(participant_id=rk, success=True, result=result)
            except Exception as exc:
                outcome = FLNetLocalThreadResultDTO(
                    participant_id=rk, success=False, error=f"{type(exc).__name__}: {exc}")
            with results_lock:
                results[rk] = outcome

        return threading.Thread(target=_target, name=f"fl-local-{result_key}", daemon=True)

    def _build_aggregator_thread(self, participant, results, results_lock, bridge,
                                 n_rounds=None) -> threading.Thread:
        def _target(bound_participant=participant, bound_bridge=bridge, bound_n_rounds=n_rounds):
            try:
                result = self._run_aggregator(bound_participant, bound_bridge, bound_n_rounds)
                outcome = FLNetLocalThreadResultDTO(
                    participant_id=bound_participant.participant_id, success=True, result=result)
            except Exception as exc:
                outcome = FLNetLocalThreadResultDTO(
                    participant_id=bound_participant.participant_id, success=False,
                    error=f"{type(exc).__name__}: {exc}")
            with results_lock:
                results[bound_participant.participant_id] = outcome

        return threading.Thread(target=_target,
                                name=f"fl-local-agg-{participant.participant_id}", daemon=True)

    def run(self, apps_by_participant: dict[str, Any],
            mode: ModeType = ModeType.TRAINING) -> dict[
        str, FLNetLocalThreadResultDTO]:
        results: dict[str, FLNetLocalThreadResultDTO] = {}
        results_lock = threading.Lock()
        threads: list[threading.Thread] = []
        participants_to_execute = self._participants_to_execute()

        missing_clients = [
            p.participant_id for p in participants_to_execute
            if p.role == FederatedParticipantType.CLIENT and p.participant_id not in apps_by_participant
        ]
        if missing_clients:
            raise KeyError(
                f"No app provided for participant(s): {missing_clients}. "
                f"Provide an entry in apps_by_participant for every configured client participant."
            )
        if (
            self._aggregator_enabled(self.config)
            and any(p.role == FederatedParticipantType.AGGREGATOR for p in participants_to_execute)
            and not self._aggregators
        ):
            raise KeyError("No runtime aggregator registered for the configured aggregator participant.")

        # (participant_id, training-thread result key) for coordinators that also train.
        coordinator_train_keys: list[tuple[str, str]] = []

        for participant in participants_to_execute:
            if participant.role == FederatedParticipantType.AGGREGATOR:
                if not self._aggregator_enabled(self.config):
                    with results_lock:
                        results[participant.participant_id] = FLNetLocalThreadResultDTO(
                            participant_id=participant.participant_id, success=True, result=None)
                    continue

                # The coordinator runs a training app too when one was provided for it: spawn the
                # aggregator thread AND the training-app thread, bridged in-process.
                coordinator_app = apps_by_participant.get(participant.participant_id)
                bridge = (CoordinatorTrainingBridge(participant.participant_id, self.config.aggregator_id)
                          if coordinator_app is not None else None)
                # Lock the aggregator's round count to the coordinator training app's resolved config
                # so both loop the same number of rounds (the app gets federated_rounds from the
                # dataclass default when hyper_params omits it; the raw-dict fallback would give 1).
                agg_rounds = (self._resolve_rounds_from_app(coordinator_app, participant)
                              if coordinator_app is not None else None)
                threads.append(self._build_aggregator_thread(
                    participant, results, results_lock, bridge, agg_rounds))
                if coordinator_app is not None:
                    train_key = f"{participant.participant_id}::train"
                    threads.append(self._build_app_thread(
                        participant, coordinator_app, results, results_lock, mode,
                        result_key=train_key, communicator_override=bridge))
                    coordinator_train_keys.append((participant.participant_id, train_key))
                continue

            app = apps_by_participant[participant.participant_id]
            threads.append(self._build_app_thread(participant, app, results, results_lock, mode))

        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Fold each coordinator's training outcome into its participant outcome so a training failure
        # surfaces, then drop the synthetic training key (callers key results by participant id).
        with results_lock:
            for pid, train_key in coordinator_train_keys:
                train_outcome = results.pop(train_key, None)
                if train_outcome is not None and not train_outcome.success:
                    agg_outcome = results.get(pid)
                    results[pid] = FLNetLocalThreadResultDTO(
                        participant_id=pid, success=False, error=train_outcome.error,
                        result=getattr(agg_outcome, "result", None))

        # Release notice subscriptions so reusing apps/controllers across runs does not leak
        # observers on the shared Subject. The coordinator's bridge has no such subscription.
        for participant in participants_to_execute:
            app = apps_by_participant.get(participant.participant_id)
            communicator = getattr(app, "communicator", None)
            unbind = getattr(communicator, "unbind_notice_subject", None)
            if callable(unbind):
                unbind()

        return results
