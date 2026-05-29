"""Bridge between PLC hardware and LSTM+PPO inference engine."""

from __future__ import annotations

import time
import logging
from collections import deque

from src.utils import load_config, PROJ_ROOT
from src.plc.snap7_client import PLCClient, ElevatorPLCInterface, DB_INPUT, DB_OUTPUT
from src.inference import ElevatorScheduler

logger = logging.getLogger(__name__)


class PLCInferenceBridge:
    """Main loop: reads PLC state → runs inference → writes decisions to PLC.

    The PLC handles hard-real-time elevator control (motor, doors, safety).
    This bridge provides soft-real-time scheduling recommendations via the
    trained LSTM+PPO model.
    """

    def __init__(self, config_path: str | None = None,
                 checkpoint_path: str | None = None):
        self.cfg = load_config(config_path)

        plc_cfg = self.cfg.get("plc", {})
        self.poll_interval = plc_cfg.get("poll_interval_ms", 100) / 1000.0
        self.db_input = plc_cfg.get("db_input", DB_INPUT)
        self.db_output = plc_cfg.get("db_output", DB_OUTPUT)

        ckpt = checkpoint_path or str(PROJ_ROOT / "checkpoints" / "best_model.pt")
        self.scheduler = ElevatorScheduler(checkpoint_path=ckpt, config_path=config_path)

        self.client = PLCClient(
            ip=plc_cfg.get("ip", "192.168.0.1"),
            rack=plc_cfg.get("rack", 0),
            slot=plc_cfg.get("slot", 1),
            timeout_ms=plc_cfg.get("timeout_ms", 1000),
        )
        self.interface: ElevatorPLCInterface | None = None

        self.running = False
        self._decision_log: deque[dict] = deque(maxlen=1000)
        self._stats: dict = {"decisions": 0, "errors": 0, "last_decision_time": 0.0}

    def connect(self) -> bool:
        ok = self.client.connect()
        if ok:
            self.interface = ElevatorPLCInterface(
                self.client, self.db_input, self.db_output
            )
            self.scheduler.reset()
            logger.info("Bridge connected to PLC")
        return ok

    def disconnect(self):
        self.running = False
        self.client.disconnect()
        logger.info("Bridge disconnected")

    def run(self):
        """Main control loop."""
        if not self.client.connected:
            logger.error("Not connected to PLC")
            return

        self.running = True
        logger.info(f"Bridge running, poll interval={self.poll_interval*1000:.0f}ms")

        while self.running:
            try:
                self._step()
            except ConnectionError:
                logger.warning("PLC connection lost, attempting reconnect...")
                time.sleep(1.0)
                if not self._reconnect():
                    logger.error("Reconnect failed, stopping bridge")
                    break
            except Exception:
                logger.exception("Unexpected error in bridge loop")
                self._stats["errors"] += 1
                time.sleep(self.poll_interval)

            time.sleep(self.poll_interval)

    def _step(self):
        """Single decision loop iteration — single PLC read for all data."""
        # Single bulk read of all DB10 data (22 bytes)
        data = self.interface.read_all_inputs()

        # Parse all elevator states and floor calls from the same buffer
        elevators = []
        for el_id in range(ElevatorPLCInterface.NUM_ELEVATORS):
            try:
                state = self.interface._parse_elevator_state(el_id, data)
                elevators.append(state)
            except Exception:
                elevators.append({
                    "floor": 1, "direction": 0, "load_ratio": 0.0,
                    "is_moving": False, "door_open": False,
                    "has_car_calls": False,
                })

        up_calls, down_calls = self.interface.parse_floor_calls(data)

        if not (any(up_calls) or any(down_calls)):
            return

        state_dict = {
            "elevators": elevators,
            "floor_up_calls": up_calls,
            "floor_down_calls": down_calls,
        }

        t0 = time.perf_counter()
        action = self.scheduler.query(state_dict)
        decision_time = (time.perf_counter() - t0) * 1000

        self._stats["decisions"] += 1
        self._stats["last_decision_time"] = decision_time

        # Assign all active calls
        for calls, direction in ((up_calls, 1), (down_calls, -1)):
            for f in range(10):
                if calls[f]:
                    self.interface.write_call_indicator(f + 1, direction, True)
                    self._decision_log.append({
                        "timestamp": time.time(),
                        "action": action,
                        "floor": f + 1,
                        "direction": direction,
                        "decision_time_ms": decision_time,
                    })

        if self._stats["decisions"] % 100 == 0:
            n_calls = sum(up_calls) + sum(down_calls)
            logger.info(
                f"Decision #{self._stats['decisions']}: "
                f"elevator={action}, calls={n_calls}, "
                f"latency={decision_time:.2f}ms"
            )

    def _reconnect(self) -> bool:
        for attempt in range(3):
            try:
                self.client.disconnect()
                time.sleep(1.0)
                if self.client.connect():
                    self.interface = ElevatorPLCInterface(
                        self.client, self.db_input, self.db_output
                    )
                    self.scheduler.reset()
                    return True
            except Exception:
                pass
        return False

    def stop(self):
        self.running = False

    def get_stats(self) -> dict:
        return dict(self._stats)

    def get_recent_decisions(self, n: int = 20) -> list[dict]:
        return list(self._decision_log)[-n:]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    bridge = PLCInferenceBridge()
    print("Bridge initialized (no PLC connection in test mode)")
    print(f"Config: poll_interval={bridge.poll_interval*1000:.0f}ms, db_input={bridge.db_input}, db_output={bridge.db_output}")
    print(f"Scheduler model loaded: {bridge.scheduler.trainer.policy.training if hasattr(bridge.scheduler.trainer.policy, 'training') else False}")
    print("PLC Bridge module: OK")
