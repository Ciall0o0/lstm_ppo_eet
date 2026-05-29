"""python-snap7 client wrapper for S7-1200 PLC communication.

Address mapping based on:
  - EET_IO_LIST_PLC输入变量.xls (DB10)
  - EET_IO_LIST_PLC输出变量.xls (DB11)
"""

from __future__ import annotations

import time
import struct
import logging
from typing import Literal

import snap7
from snap7.util import get_bool, set_bool, get_word, set_word

logger = logging.getLogger(__name__)

# S7 Area codes
AREA_INPUTS = 0x81
AREA_OUTPUTS = 0x82
AREA_MERKERS = 0x83
AREA_DB = 0x84

# DB block numbers (from XLS)
DB_INPUT = 10
DB_OUTPUT = 11

# Elevator base byte offsets in DB10/DB11 (non-uniform spacing from XLS)
ELEVATOR_INPUT_BASES = (2, 6, 10)   # DB10 byte offsets for elevators 1-3
ELEVATOR_OUTPUT_BASES = (2, 7, 12)  # DB11 byte offsets for elevators 1-3

# DB10 read size: covers floor calls (3 bytes) + all 3 elevator inputs + load weights + auto-run
_DB10_READ_SIZE = 22  # bytes 0-21

CommandName = Literal[
    "motor_start", "up_contactor", "down_contactor",
    "high_speed", "low_speed", "door_open_cmd", "door_close_cmd",
    "brake_1", "brake_2", "brake_3",
]

# Command byte/bit offsets relative to elevator output base
_CMD_MAP: dict[CommandName, tuple[int, int]] = {
    "motor_start":    (6, 0),
    "up_contactor":   (6, 1),
    "down_contactor": (6, 2),
    "high_speed":     (6, 3),
    "low_speed":      (6, 4),
    "door_open_cmd":  (6, 5),
    "door_close_cmd": (6, 6),
    "brake_1":        (6, 7),
    "brake_2":        (7, 0),
    "brake_3":        (7, 1),
}


def _floor_call_bit(floor: int, direction: int) -> tuple[int, int]:
    """Convert floor+direction to (byte, bit) in the call indicator region.

    Up calls:   floors 1-8 at DBX0.0-0.7, floor 9 at DBX1.0
    Down calls: floors 2-9 at DBX1.1-2.0, floor 10 at DBX2.1
    """
    if direction > 0:
        return (0 if floor <= 8 else 1, (floor - 1) % 8)
    else:
        bit_idx = floor - 2 + 1  # offset by 1 because starts at DBX1.1
        return (1 if bit_idx < 8 else 2, bit_idx % 8)


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
            tsap = (self.rack << 8) | self.slot
            self._client.set_connection_params(self.ip, tsap, tsap)
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

    Based on competition XLS files:
      - EET_IO_LIST_PLC输入变量.xls → DB10 (inputs)
      - EET_IO_LIST_PLC输出变量.xls → DB11 (outputs)

    Building: 10 floors, 3 elevators
    """

    NUM_ELEVATORS = 3
    NUM_FLOORS = 10

    def __init__(self, client: PLCClient, db_input: int = DB_INPUT,
                 db_output: int = DB_OUTPUT):
        self.client = client
        self.db_input = db_input
        self.db_output = db_output
        self._current_floor = [1] * self.NUM_ELEVATORS
        self._was_upper_leveling = [False] * self.NUM_ELEVATORS
        self._was_lower_leveling = [False] * self.NUM_ELEVATORS

    # ── Input read methods (DB10) ──────────────────────────────────────────

    def read_all_inputs(self) -> bytearray:
        """Read all DB10 input data in a single PLC round-trip (22 bytes)."""
        return self.client.db_read(self.db_input, 0, _DB10_READ_SIZE)

    def parse_floor_calls(self, data: bytearray) -> tuple[list[bool], list[bool]]:
        """Parse shared floor up/down call buttons from DB10 data.

        Layout (DB10.DBX0.0 ~ DB10.DBX2.1):
          DBX0.0~0.7: 1~8层上行呼梯
          DBX1.0:      9层上行呼梯
          DBX1.1~2.0: 2~9层下行呼梯
          DBX2.1:      10层下行呼梯
        """
        up_calls = [False] * self.NUM_FLOORS
        down_calls = [False] * self.NUM_FLOORS

        for f in range(1, 9):
            up_calls[f - 1] = get_bool(data, 0, f - 1)
        up_calls[8] = get_bool(data, 1, 0)

        for f in range(2, 10):
            byte_idx, bit_idx = _floor_call_bit(f, -1)
            down_calls[f - 1] = get_bool(data, byte_idx, bit_idx)
        down_calls[9] = get_bool(data, 2, 1)

        return up_calls, down_calls

    def read_floor_calls(self) -> tuple[list[bool], list[bool]]:
        """Read shared floor up/down call buttons (convenience wrapper)."""
        return self.parse_floor_calls(self.read_all_inputs())

    def parse_elevator_inputs(self, elevator_id: int, data: bytearray) -> dict:
        """Parse input signals for one elevator from DB10 data.

        Elevator input layout (relative to elevator base):
          +0.0~+1.1: car call buttons 1~10 (10 bits)
          +1.2: door open button, +1.3: door close button
          +1.4: light curtain, +1.5: maintenance, +1.6: car door lock
          +1.7~+2.6: floor door locks 1~10 (10 bits)
          +2.7: door open limit, +3.0: door close limit
          +3.1: upper leveling, +3.2: lower leveling
          +3.3~+3.6: terminal limit switches (4 bits)
        """
        base = ELEVATOR_INPUT_BASES[elevator_id]
        # Offset into the bulk-read buffer
        off = base

        car_calls = [get_bool(data, off + i // 8, i % 8) for i in range(10)]

        door_open_btn = get_bool(data, off + 1, 2)
        door_close_btn = get_bool(data, off + 1, 3)
        light_curtain = get_bool(data, off + 1, 4)
        maintenance = get_bool(data, off + 1, 5)
        car_door_lock = get_bool(data, off + 1, 6)

        floor_locks = []
        for i in range(10):
            bit_idx = 7 + i
            byte_idx = off + 1 + bit_idx // 8
            bit_in_byte = bit_idx % 8
            floor_locks.append(get_bool(data, byte_idx, bit_in_byte))

        door_open_limit = get_bool(data, off + 2, 7)
        door_close_limit = get_bool(data, off + 3, 0)
        upper_leveling = get_bool(data, off + 3, 1)
        lower_leveling = get_bool(data, off + 3, 2)
        terminal_limits = [get_bool(data, off + 3, 3 + i) for i in range(4)]

        self._update_floor_from_leveling(elevator_id, upper_leveling, lower_leveling)

        return {
            "car_calls": car_calls,
            "has_car_calls": any(car_calls),
            "door_open_btn": door_open_btn,
            "door_close_btn": door_close_btn,
            "light_curtain": light_curtain,
            "maintenance": maintenance,
            "car_door_lock": car_door_lock,
            "floor_locks": floor_locks,
            "door_open_limit": door_open_limit,
            "door_close_limit": door_close_limit,
            "upper_leveling": upper_leveling,
            "lower_leveling": lower_leveling,
            "terminal_limits": terminal_limits,
        }

    def parse_load_weight(self, elevator_id: int, data: bytearray) -> float:
        """Parse elevator load weight from DB10 data.

        Load weights at: DB10.DBW16, DB10.DBW18, DB10.DBW20
        """
        addr = 16 + elevator_id * 2
        raw = get_word(data, addr)
        return min(raw / 2000.0, 1.0)

    def parse_auto_run(self, data: bytearray) -> bool:
        """Parse auto-run signal from DB10.DBX14.5."""
        return get_bool(data, 14, 5)

    def read_elevator_state(self, elevator_id: int) -> dict:
        """Read complete state for one elevator (triggers PLC read)."""
        data = self.read_all_inputs()
        return self._parse_elevator_state(elevator_id, data)

    def read_all_elevator_states(self) -> list[dict]:
        """Read all elevator states in a single PLC round-trip."""
        data = self.read_all_inputs()
        return [self._parse_elevator_state(i, data) for i in range(self.NUM_ELEVATORS)]

    def _parse_elevator_state(self, elevator_id: int, data: bytearray) -> dict:
        """Parse elevator state from pre-read DB10 data."""
        inputs = self.parse_elevator_inputs(elevator_id, data)
        load_ratio = self.parse_load_weight(elevator_id, data)
        return {
            "floor": self._current_floor[elevator_id],
            "direction": 0,
            "load_ratio": load_ratio,
            "is_moving": False,
            "door_open": inputs["door_open_limit"],
            "has_car_calls": inputs["has_car_calls"],
            "fault": False,
            "overload": load_ratio >= 1.0,
            "auto_run": self.parse_auto_run(data),
            "car_calls": inputs["car_calls"],
        }

    def _update_floor_from_leveling(self, elevator_id: int,
                                    upper: bool, lower: bool):
        """Track floor position using leveling sensor transitions."""
        was_upper = self._was_upper_leveling[elevator_id]
        was_lower = self._was_lower_leveling[elevator_id]

        if upper and not was_upper:
            self._current_floor[elevator_id] = min(
                self._current_floor[elevator_id] + 1, self.NUM_FLOORS
            )
        if lower and not was_lower:
            self._current_floor[elevator_id] = max(
                self._current_floor[elevator_id] - 1, 1
            )

        self._was_upper_leveling[elevator_id] = upper
        self._was_lower_leveling[elevator_id] = lower

    # ── Output write methods (DB11) ────────────────────────────────────────

    def write_call_indicator(self, floor: int, direction: int, active: bool = True):
        """Write floor call indicator light to DB11.

        Layout (DB11.DBX0.0 ~ DB11.DBX2.1):
          DBX0.0~1.0: 1~9层上行指示灯
          DBX1.1~2.1: 2~10层下行指示灯
        """
        data = self.client.db_read(self.db_output, 0, 3)
        byte_idx, bit_idx = _floor_call_bit(floor, direction)
        set_bool(data, byte_idx, bit_idx, active)
        self.client.db_write(self.db_output, 0, data)

    def write_car_call_indicator(self, elevator_id: int, floor: int,
                                 active: bool = True):
        """Write car call button indicator for one elevator."""
        base = ELEVATOR_OUTPUT_BASES[elevator_id]
        data = self.client.db_read(self.db_output, base, 2)

        if 1 <= floor <= 10:
            bit_idx = floor - 1
            set_bool(data, bit_idx // 8, bit_idx % 8, active)

        self.client.db_write(self.db_output, base, data)

    def write_direction_indicator(self, elevator_id: int, direction: int):
        """Write up/down direction indicator for one elevator."""
        base = ELEVATOR_OUTPUT_BASES[elevator_id]
        data = self.client.db_read(self.db_output, 5, 1)
        set_bool(data, 0, 2, direction > 0)
        set_bool(data, 0, 3, direction < 0)
        self.client.db_write(self.db_output, 5, data)

    def write_elevator_command(self, elevator_id: int, command: CommandName,
                               value: bool = True):
        """Write a command to a specific elevator.

        cmd_map values are byte/bit offsets relative to the elevator's output base.
        E.g., for elevator 1 (base=2): (6, 0) → absolute DB11.DBX8.0
        """
        if command not in _CMD_MAP:
            raise ValueError(f"Unknown command: {command}")

        byte_idx, bit_idx = _CMD_MAP[command]
        abs_byte = ELEVATOR_OUTPUT_BASES[elevator_id] + byte_idx
        data = self.client.db_read(self.db_output, abs_byte, 1)
        set_bool(data, 0, bit_idx, value)
        self.client.db_write(self.db_output, abs_byte, data)

    def write_ready_signal(self, elevator_id: int, active: bool = True):
        """Write/clear ready signal to DB11.DBX17.2 (global)."""
        data = self.client.db_read(self.db_output, 17, 1)
        set_bool(data, 0, 2, active)
        self.client.db_write(self.db_output, 17, data)


if __name__ == "__main__":
    print("Testing PLC interface structure (no PLC required)...")
    print(f"NUM_ELEVATORS = {ElevatorPLCInterface.NUM_ELEVATORS}")
    print(f"NUM_FLOORS = {ElevatorPLCInterface.NUM_FLOORS}")
    print(f"DB_INPUT = {DB_INPUT}, DB_OUTPUT = {DB_OUTPUT}")

    # Test floor call bit math
    for f in range(1, 10):
        b, bit = _floor_call_bit(f, 1)
        assert 0 <= b <= 2 and 0 <= bit <= 7, f"up call floor {f}: byte={b}, bit={bit}"
    for f in range(2, 11):
        b, bit = _floor_call_bit(f, -1)
        assert 0 <= b <= 2 and 0 <= bit <= 7, f"down call floor {f}: byte={b}, bit={bit}"
    print("Floor call bit math: OK")

    # Test bit operations on mock data
    data = bytearray(3)
    set_bool(data, *_floor_call_bit(1, 1), True)
    set_bool(data, *_floor_call_bit(9, 1), True)
    set_bool(data, *_floor_call_bit(10, -1), True)
    print(f"Floor 1 up call: {get_bool(data, *_floor_call_bit(1, 1))}")
    print(f"Floor 9 up call: {get_bool(data, *_floor_call_bit(9, 1))}")
    print(f"Floor 10 down call: {get_bool(data, *_floor_call_bit(10, -1))}")

    # Test elevator input parsing
    el_data = bytearray(_DB10_READ_SIZE)
    set_bool(el_data, ELEVATOR_INPUT_BASES[0] + 0, 0, True)   # car call floor 1 (byte 2, bit 0)
    set_bool(el_data, ELEVATOR_INPUT_BASES[0] + 1, 1, True)   # car call floor 10 (byte 3, bit 1)
    set_bool(el_data, ELEVATOR_INPUT_BASES[0] + 3, 1, True)   # upper leveling

    iface = ElevatorPLCInterface.__new__(ElevatorPLCInterface)
    iface.NUM_ELEVATORS = 3
    iface.NUM_FLOORS = 10
    iface._current_floor = [1] * 3
    iface._was_upper_leveling = [False] * 3
    iface._was_lower_leveling = [False] * 3
    inputs = iface.parse_elevator_inputs(0, el_data)
    print(f"Car call floor 1: {inputs['car_calls'][0]}")
    print(f"Car call floor 10: {inputs['car_calls'][9]}")
    print(f"Upper leveling: {inputs['upper_leveling']}")

    print("PLC client module: OK")
