"""python-snap7 client wrapper for S7-1200 PLC communication."""

from __future__ import annotations

import time
import struct
import logging
from typing import Optional

import snap7
from snap7.util import get_bool, set_bool, get_word, set_word

logger = logging.getLogger(__name__)

# S7 Area codes
AREA_INPUTS = 0x81
AREA_OUTPUTS = 0x82
AREA_MERKERS = 0x83
AREA_DB = 0x84


class PLCClient:
    """Low-level S7-1200 PLC client via snap7."""

    def __init__(self, ip: str = "192.168.0.1", rack: int = 0, slot: int = 1,
                 timeout_ms: int = 1000):
        self.ip = ip
        self.rack = rack
        self.slot = slot
        self.timeout_ms = timeout_ms
        self._client: snap7.client.Client | None = None

    def connect(self) -> bool:
        try:
            self._client = snap7.client.Client()
            self._client.set_connection_params(self.ip, 0x0101, 0x0101)
            self._client.set_timeout(self.timeout_ms)
            self._client.connect(self.ip, self.rack, self.slot)
            logger.info(f"Connected to PLC at {self.ip} (rack={self.rack}, slot={self.slot})")
            return True
        except Exception as e:
            logger.error(f"PLC connection failed: {e}")
            self._client = None
            return False

    def disconnect(self):
        if self._client:
            try:
                self._client.disconnect()
            except Exception:
                pass
            self._client = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._client.get_connected()

    def _check_connected(self):
        if not self.connected:
            raise ConnectionError("Not connected to PLC")

    def read_area(self, area: int, db_number: int, start: int, size: int) -> bytearray:
        self._check_connected()
        return self._client.read_area(area, db_number, start, size)

    def write_area(self, area: int, db_number: int, start: int, data: bytearray):
        self._check_connected()
        self._client.write_area(area, db_number, start, data)

    def db_read(self, db_number: int, start: int, size: int) -> bytearray:
        return self.read_area(AREA_DB, db_number, start, size)

    def db_write(self, db_number: int, start: int, data: bytearray):
        self.write_area(AREA_DB, db_number, start, data)


class ElevatorPLCInterface:
    """Maps EET elevator I/O addresses to structured Python data.

    Based on the competition device description PDF I/O tables.
    Single elevator DB layout (relative addressing within DB block):
      DBX0.0-DBX0.4: floor 1-5 up call buttons
      DBX0.5-DBX1.1: floor 2-6 down call buttons
      DBX1.2-DBX1.7: car call buttons floor 1-6
      DBX2.0-DBX2.1: door open/close buttons
      DBX2.2: light curtain
      DBX2.3: maintenance signal
      DBX2.4-DBX3.2: door lock signals
      DBX3.3-DBX3.4: door open/close limit
      DBX3.5-DBX3.6: upper/lower leveling
      DBX3.7-DBX4.2: terminal limit switches
      DBX4.3: auto run signal
      DBW6: current load weight (0-2000 kg)
    """

    # Number of elevators controlled
    NUM_ELEVATORS = 3

    # DB block base offsets per elevator
    # Each elevator's I/O is mapped to a separate DB block offset region
    # Adjust these based on actual PLC program DB layout
    DB_OFFSET_PER_ELEVATOR = 100  # bytes per elevator region

    def __init__(self, client: PLCClient, db_number: int = 1):
        self.client = client
        self.db_number = db_number

    def read_elevator_state(self, elevator_id: int) -> dict:
        """Read complete state for one elevator from PLC."""
        base = elevator_id * self.DB_OFFSET_PER_ELEVATOR
        data = self.client.db_read(self.db_number, base, 20)

        return {
            "floor": self._read_current_floor(data),
            "direction": self._read_direction(data),
            "load_ratio": self._read_load_ratio(data),
            "is_moving": get_bool(data, 3, 5),       # DBX3.5 motor start
            "door_open": get_bool(data, 3, 3),       # DBX3.3 door open limit
            "has_car_calls": self._has_any_car_call(data),
            "fault": get_bool(data, 3, 1),           # DBX3.1 fault indicator
            "overload": get_bool(data, 3, 4),        # DBX3.4 full load indicator
            "auto_run": get_bool(data, 4, 3),        # DBX4.3 auto run signal
            "car_calls": self._read_car_calls(data),
        }

    def read_floor_calls(self) -> tuple[list[bool], list[bool]]:
        """Read all floor up/down calls across all elevators.

        Returns (up_calls[10], down_calls[10])
        """
        up_calls = [False] * 10
        down_calls = [False] * 10

        for el_id in range(self.NUM_ELEVATORS):
            base = el_id * self.DB_OFFSET_PER_ELEVATOR
            data = self.client.db_read(self.db_number, base, 12)

            for f in range(1, 6):  # floors 1-5 up: DBX0.0-DBX0.4
                if get_bool(data, 0, f - 1):
                    up_calls[f - 1] = True
            for f in range(2, 7):  # floors 2-6 down: DBX0.5-DBX1.1
                bit_idx = 5 + f - 2
                byte_idx = 0 if bit_idx < 8 else 1
                bit_in_byte = bit_idx % 8
                if get_bool(data, byte_idx, bit_in_byte):
                    down_calls[f - 1] = True

        return up_calls, down_calls

    def write_call_assignment(self, elevator_id: int, floor: int,
                              direction: int, active: bool = True):
        """Write button indicator light to acknowledge a call assignment.

        This tells the EET model that this elevator will handle this call.
        The PLC program reads these outputs and routes accordingly.
        """
        base = elevator_id * self.DB_OFFSET_PER_ELEVATOR
        # Read current output state
        data = self.client.db_read(self.db_number, base, 6)

        if direction > 0:  # up call
            idx_byte = (floor - 1) // 8 if (floor - 1) < 8 else (floor - 1 - 8) // 8
            idx_bit = (floor - 1) % 8
            if floor <= 5:
                set_bool(data, 0, idx_bit, active)  # up call indicator
        elif direction < 0:  # down call
            idx_bit = (floor - 2) % 8
            if 2 <= floor <= 6:
                set_bool(data, 0, 5 + idx_bit, active)  # down call indicator

        self.client.db_write(self.db_number, base, data)

    def write_elevator_command(self, elevator_id: int, command: str,
                               value: bool = True):
        """Write a command to a specific elevator.

        Commands: motor_start, up_contactor, down_contactor, high_speed,
                 low_speed, door_open_cmd, door_close_cmd, brake_1, brake_2, brake_3
        """
        base = elevator_id * self.DB_OFFSET_PER_ELEVATOR
        cmd_map = {
            "motor_start": (3, 5),
            "up_contactor": (3, 6),
            "down_contactor": (3, 7),
            "high_speed": (4, 0),
            "low_speed": (4, 1),
            "door_open_cmd": (4, 2),
            "door_close_cmd": (4, 3),
            "brake_1": (4, 4),
            "brake_2": (4, 5),
            "brake_3": (4, 6),
            "ready_signal": (4, 7),
        }
        if command not in cmd_map:
            raise ValueError(f"Unknown command: {command}")

        byte_idx, bit_idx = cmd_map[command]
        data = self.client.db_read(self.db_number, base + byte_idx, 1)
        set_bool(data, 0, bit_idx, value)
        self.client.db_write(self.db_number, base + byte_idx, data)

    def send_ready_signal(self, elevator_id: int):
        self.write_elevator_command(elevator_id, "ready_signal", True)

    def clear_ready_signal(self, elevator_id: int):
        self.write_elevator_command(elevator_id, "ready_signal", False)

    def check_auto_run(self) -> bool:
        """Check if auto-run signal is active on elevator 1."""
        try:
            state = self.read_elevator_state(0)
            return state.get("auto_run", False)
        except Exception:
            return False

    # 7-segment display mapping: (a, b, c, d, e, f, g) → digit
    _SEGMENT_MAP = {
        (1, 1, 1, 1, 1, 1, 0): 0,
        (0, 1, 1, 0, 0, 0, 0): 1,
        (1, 1, 0, 1, 1, 0, 1): 2,
        (1, 1, 1, 1, 0, 0, 1): 3,
        (0, 1, 1, 0, 0, 1, 1): 4,
        (1, 0, 1, 1, 0, 1, 1): 5,
        (1, 0, 1, 1, 1, 1, 1): 6,
        (1, 1, 1, 0, 0, 0, 0): 7,
        (1, 1, 1, 1, 1, 1, 1): 8,
        (1, 1, 1, 1, 0, 1, 1): 9,
    }

    @staticmethod
    def _read_current_floor(data: bytearray) -> int:
        bits = (
            get_bool(data, 2, 0),  # a
            get_bool(data, 2, 1),  # b
            get_bool(data, 2, 2),  # c
            get_bool(data, 2, 3),  # d
            get_bool(data, 2, 4),  # e
            get_bool(data, 2, 5),  # f
            get_bool(data, 2, 6),  # g
        )
        return ElevatorPLCInterface._SEGMENT_MAP.get(bits, 1)  # default to 1 if unknown

    @staticmethod
    def _read_direction(data: bytearray) -> int:
        up = get_bool(data, 2, 7)
        down = get_bool(data, 3, 0)
        if up and not down:
            return 1
        elif down and not up:
            return -1
        return 0

    @staticmethod
    def _read_load_ratio(data: bytearray) -> float:
        raw = get_word(data, 6)
        return min(raw / 2000.0, 1.0)

    @staticmethod
    def _has_any_car_call(data: bytearray) -> bool:
        for i in range(6):
            if get_bool(data, 1, 2 + i):
                return True
        return False

    @staticmethod
    def _read_car_calls(data: bytearray) -> list[bool]:
        return [get_bool(data, 1, 2 + i) for i in range(6)] + [False] * 4


if __name__ == "__main__":
    # Dry-run test (no PLC required)
    print("Testing PLC interface structure (no PLC required)...")
    print(f"ElevatorPLCInterface.NUM_ELEVATORS = {ElevatorPLCInterface.NUM_ELEVATORS}")
    print(f"DB_OFFSET_PER_ELEVATOR = {ElevatorPLCInterface.DB_OFFSET_PER_ELEVATOR}")

    # Mock bytearray to test bit operations
    data = bytearray(20)
    set_bool(data, 0, 0, True)
    set_bool(data, 0, 5, True)
    set_bool(data, 3, 5, True)
    set_bool(data, 4, 3, True)

    print(f"Floor 1 up call: {get_bool(data, 0, 0)}")
    print(f"Floor 2 down call: {get_bool(data, 0, 5)}")
    print(f"Motor start: {get_bool(data, 3, 5)}")
    print(f"Auto run: {get_bool(data, 4, 3)}")
    print("PLC client module: OK")
