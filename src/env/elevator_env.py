"""Elevator group-control simulation environment (Gymnasium)."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .metrics import compute_episode_metrics

MAX_FLOOR = 10


@dataclass
class Elevator:
    id: int
    current_floor: float = 1.0
    target_floor: float = 1.0
    direction: int = 0          # -1=down, 0=idle, 1=up
    state: str = "idle"         # idle | moving | doors_open | doors_close
    load_kg: float = 0.0
    max_load: float = 900.0
    door_timer: float = 0.0
    car_calls: set = field(default_factory=set)
    assigned_passengers: list = field(default_factory=list)
    floor_travel_time: float = 2.0
    door_open_time: float = 2.0
    door_close_time: float = 2.5
    accel_decel_time: float = 1.0

    @property
    def load_ratio(self) -> float:
        return min(self.load_kg / self.max_load, 1.0) if self.max_load > 0 else 0.0

    @property
    def is_moving(self) -> bool:
        return self.state == "moving"

    @property
    def is_door_open(self) -> bool:
        return self.state == "doors_open"

    def reset(self):
        self.current_floor = 1.0
        self.target_floor = 1.0
        self.direction = 0
        self.state = "idle"
        self.load_kg = 0.0
        self.door_timer = 0.0
        self.car_calls.clear()
        self.assigned_passengers.clear()

    def step(self, dt: float) -> list:
        """Advance elevator state by dt seconds. Returns list of delivered passengers."""
        delivered = []
        if self.state == "moving":
            self._step_moving(dt, delivered)
        elif self.state == "doors_open":
            self._step_doors_open(dt)
        elif self.state == "doors_close":
            self._step_doors_close(dt)
        return delivered

    def _step_moving(self, dt: float, delivered: list):
        travel_dist = dt / self.floor_travel_time
        if self.direction == 1:
            self.current_floor += travel_dist
        elif self.direction == -1:
            self.current_floor -= travel_dist
        self.current_floor = max(0.5, min(MAX_FLOOR + 0.5, self.current_floor))

        if (self.direction == 1 and self.current_floor >= self.target_floor) or \
           (self.direction == -1 and self.current_floor <= self.target_floor):
            self.current_floor = round(self.target_floor)
            self.direction = 0
            self.state = "doors_open"
            self.door_timer = self.door_open_time
            delivered.extend(self._disembark_passengers())

    def _step_doors_open(self, dt: float):
        self.door_timer -= dt
        self._board_passengers()
        if self.door_timer <= 0:
            if self.car_calls:
                self._select_next_target()
                self.state = "doors_close"
                self.door_timer = self.door_close_time
            else:
                self.state = "idle"

    def _step_doors_close(self, dt: float):
        self.door_timer -= dt
        if self.door_timer <= 0:
            self.state = "moving"
            self.direction = 1 if self.target_floor > self.current_floor else -1

    def assign_call(self, pickup_floor: int, dest_floor: int, passenger_id: int):
        """Assign a passenger call to this elevator."""
        self.car_calls.add(dest_floor)
        p = {"id": passenger_id, "pickup": pickup_floor, "dest": dest_floor,
             "arrive_time": None, "boarded": False}
        self.assigned_passengers.append(p)
        if self.state == "idle":
            self._select_next_target()

    def _select_next_target(self):
        if not self.car_calls:
            return
        current = self.current_floor
        nearest = min(self.car_calls, key=lambda f: abs(f - current))
        self.target_floor = float(nearest)
        if self.state == "idle":
            if abs(self.target_floor - current) < 0.1:
                self.state = "doors_open"
                self.door_timer = self.door_open_time
            else:
                self.direction = 1 if self.target_floor > current else -1
                self.state = "moving"

    def _board_passengers(self):
        current = round(self.current_floor)
        for p in self.assigned_passengers:
            if not p["boarded"] and p["pickup"] == current:
                p["boarded"] = True
                p["arrive_time"] = None  # will be set on delivery
                self.load_kg += 75.0  # average passenger weight

    def _disembark_passengers(self) -> list:
        current = round(self.current_floor)
        delivered = []
        self.car_calls.discard(current)
        remaining = []
        for p in self.assigned_passengers:
            if p["boarded"] and p["dest"] == current:
                p["arrive_time"] = current
                self.load_kg = max(0, self.load_kg - 75.0)
                delivered.append(p)
            else:
                remaining.append(p)
        self.assigned_passengers = remaining
        return delivered

    def to_vector(self) -> np.ndarray:
        """Encode elevator state as fixed-length vector."""
        out = np.empty(MAX_FLOOR + 3 + 4, dtype=np.float32)
        self._write_to_buffer(out, 0)
        return out

    def _write_to_buffer(self, out: np.ndarray, offset: int) -> int:
        """Write elevator state into out[offset:] and return new offset."""
        f = int(round(self.current_floor))
        out[offset:offset + MAX_FLOOR] = 0.0
        if 1 <= f <= MAX_FLOOR:
            out[offset + f - 1] = 1.0
        offset += MAX_FLOOR

        out[offset:offset + 3] = 0.0
        out[offset + self.direction + 1] = 1.0
        offset += 3

        out[offset] = self.load_ratio; offset += 1
        out[offset] = 1.0 if self.is_moving else 0.0; offset += 1
        out[offset] = 1.0 if self.state == "doors_open" else 0.0; offset += 1
        out[offset] = 1.0 if self.car_calls else 0.0; offset += 1

        return offset


class ElevatorEnv(gym.Env):
    """Multi-elevator group-control environment driven by .eet event sequences."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = config or {}
        self.num_elevators = cfg.get("num_elevators", 3)
        self.state_dim = self._compute_state_dim(self.num_elevators)
        self.floor_travel_time = cfg.get("floor_travel_time", 2.0)
        self.door_open_time = cfg.get("door_open_time", 2.0)
        self.door_close_time = cfg.get("door_close_time", 2.5)
        self.dwell_time = cfg.get("dwell_time", 3.0)
        self.long_wait_threshold = cfg.get("long_wait_threshold", 60.0)
        self.max_total_time = cfg.get("max_total_time", 3600.0)  # 1 hour cap

        self.observation_space = spaces.Box(
            low=-1.0, high=10.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.num_elevators)

        # Pre-allocated buffer for observation construction
        self._obs_buffer = np.empty(self.state_dim, dtype=np.float32)

        self.elevators: list[Elevator] = []
        self._init_elevators()

        # Episode state
        self.events: np.ndarray | None = None
        self.event_idx: int = 0
        self.elapsed: float = 0.0
        self.passenger_id_counter: int = 0
        self.pending_calls: deque = deque()
        self.floors_up_calls: set[int] = set()
        self.floors_down_calls: set[int] = set()
        self.completed_passengers: list[dict] = []

        # Stats
        self.total_empty_floors: float = 0.0
        self.total_loaded_floors: float = 0.0
        self.start_stop_count: int = 0
        self.elevator_active_time: float = 0.0

        # Reward config
        self.r_passenger = cfg.get("passenger_delivered", 2.0)
        self.r_wait_sec = cfg.get("wait_time_per_sec", -0.05)
        self.r_empty_floor = cfg.get("empty_distance_per_floor", -0.1)
        self.r_start_stop = cfg.get("energy_per_start_stop", -0.05)
        self.r_idle_sec = cfg.get("idle_penalty_per_sec", 0.0)

        self._event_wait_buffers: dict[int, float] = {}  # passenger_id → arrival_time
        self._passenger_arrival_times: dict[int, float] = {}  # passenger_id → time entered pending_calls

    def _init_elevators(self):
        self.elevators = [
            Elevator(
                id=i, floor_travel_time=self.floor_travel_time,
                door_open_time=self.door_open_time,
                door_close_time=self.door_close_time,
            )
            for i in range(self.num_elevators)
        ]

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        for el in self.elevators:
            el.reset()

        options = options or {}
        self.events = options.get("events", None)
        self.event_idx = 0
        self.passenger_id_counter = 0
        self.pending_calls.clear()
        self.floors_up_calls.clear()
        self.floors_down_calls.clear()
        self.completed_passengers.clear()
        self._event_wait_buffers.clear()

        self.total_empty_floors = 0.0
        self.total_loaded_floors = 0.0
        self.start_stop_count = 0
        self.elevator_active_time = 0.0

        # Normalize event times: shift to start at 0, scale to max 3600s
        if self.events is not None and len(self.events) > 0:
            # Copy to avoid mutating original data
            events = np.array(self.events, copy=True)
            raw_times = events[:, 2]
            t_min = float(raw_times.min())
            t_max = float(raw_times.max())
            if t_max - t_min > 1e-6:
                scale = 3600.0 / (t_max - t_min)
                events[:, 2] = (raw_times - t_min) * scale
            else:
                events[:, 2] = 0.0
            self.events = events
            self._time_offset = t_min
            self.elapsed = 0.0
            self._advance_to_next_events()
        else:
            self._time_offset = 0.0
            self.elapsed = 0.0

        return self._get_obs(), self._get_info()

    def step(self, action: int):
        """Process one scheduling decision.

        action: elevator index (0..num_elevators-1) to assign to the oldest pending call.
        If no pending calls, action is ignored and time advances to next event.
        """
        reward = 0.0

        # --- Assign pending call ---
        if self.pending_calls and 0 <= action < self.num_elevators:
            call = self.pending_calls.popleft()
            elevator = self.elevators[action]
            elevator.assign_call(call["floor"], call["dest"], call["passenger_id"])
            # Wait clock starts from passenger's true arrival, not assignment time
            self._event_wait_buffers[call["passenger_id"]] = call["arrival_time"]
            if call["direction"] == 1:
                self.floors_up_calls.discard(call["floor"])
            else:
                self.floors_down_calls.discard(call["floor"])

        # --- Determine time delta ---
        if self.pending_calls:
            dt = 0.0  # still have calls to assign, don't advance time
        else:
            dt = self._time_to_next_event()

        # --- Advance simulation by dt ---
        if dt > 0:
            prev_positions = [el.current_floor for el in self.elevators]
            prev_states = [el.state for el in self.elevators]

            for el in self.elevators:
                delivered = el.step(dt)
                for p in delivered:
                    wait_time = self.elapsed + dt - self._event_wait_buffers.pop(p["id"], self.elapsed + dt)
                    self.completed_passengers.append({
                        **p,
                        "wait_time": wait_time,
                        "ride_time": abs(p["dest"] - p["pickup"]) * self.floor_travel_time,
                    })
                    reward += self.r_passenger

            self.elapsed += dt

            for i, el in enumerate(self.elevators):
                if el.is_moving:
                    dist = abs(el.current_floor - prev_positions[i])
                    if el.load_ratio > 0.05:
                        self.total_loaded_floors += dist
                    else:
                        self.total_empty_floors += dist
                        reward += self.r_empty_floor * dist

                if prev_states[i] != "moving" and el.state == "moving":
                    self.start_stop_count += 1
                    reward += self.r_start_stop

                if el.state != "idle":
                    self.elevator_active_time += dt

            # Penalize waiting passengers (from true arrival time)
            for arrival in self._event_wait_buffers.values():
                reward += self.r_wait_sec * dt

            # Penalize idle elevators
            for el in self.elevators:
                if el.state == "idle":
                    reward += self.r_idle_sec * dt

        # --- Inject new events ---
        if self.events is not None:
            self._advance_to_next_events()

        done = self._is_done()
        info = self._get_info()

        return self._get_obs(), reward, done, False, info

    def _time_to_next_event(self) -> float:
        """Compute dt to advance: next event time or next elevator arrival."""
        candidates = []

        # Next event in sequence
        if self.events is not None and self.event_idx < len(self.events):
            next_et = float(self.events[self.event_idx, 2])
            gap = next_et - self.elapsed
            if gap > 0:
                candidates.append(gap)

        # Next elevator state change
        for el in self.elevators:
            if el.is_moving:
                remaining = abs(el.target_floor - el.current_floor) * el.floor_travel_time
                candidates.append(max(0.01, remaining))
            elif el.state == "doors_open":
                candidates.append(el.door_timer + 0.01)
            elif el.state == "doors_close":
                candidates.append(el.door_timer + 0.01)

        return min(candidates) if candidates else 0.5  # fallback

    def _advance_to_next_events(self):
        """Inject pending calls from event sequence based on event_time."""
        if self.events is None:
            return
        while self.event_idx < len(self.events):
            ev = self.events[self.event_idx]
            et = float(ev[2])  # event_time
            if et > self.elapsed + 1e-6:
                break
            src = max(1, min(MAX_FLOOR, int(ev[0])))
            dst = max(1, min(MAX_FLOOR, int(ev[1])))
            if src != dst:
                direction = 1 if dst > src else -1
                pid = self.passenger_id_counter
                self.passenger_id_counter += 1
                self.pending_calls.append({
                    "floor": src, "dest": dst, "direction": direction,
                    "passenger_id": pid,
                    "arrival_time": self.elapsed,  # record true arrival time
                })
                if direction == 1:
                    self.floors_up_calls.add(src)
                else:
                    self.floors_down_calls.add(src)
            self.event_idx += 1

    def _get_obs(self) -> np.ndarray:
        offset = 0
        for el in self.elevators:
            offset = el._write_to_buffer(self._obs_buffer, offset)

        # up calls one-hot
        self._obs_buffer[offset:offset + MAX_FLOOR] = 0.0
        for f in self.floors_up_calls:
            if 1 <= f <= MAX_FLOOR:
                self._obs_buffer[offset + f - 1] = 1.0
        offset += MAX_FLOOR

        # down calls one-hot
        self._obs_buffer[offset:offset + MAX_FLOOR] = 0.0
        for f in self.floors_down_calls:
            if 1 <= f <= MAX_FLOOR:
                self._obs_buffer[offset + f - 1] = 1.0
        offset += MAX_FLOOR

        # up-call destinations one-hot (which floors are up-call passengers going to)
        self._obs_buffer[offset:offset + MAX_FLOOR] = 0.0
        for c in self.pending_calls:
            if c["direction"] == 1 and 1 <= c["dest"] <= MAX_FLOOR:
                self._obs_buffer[offset + int(c["dest"]) - 1] = 1.0
        offset += MAX_FLOOR

        # down-call destinations one-hot
        self._obs_buffer[offset:offset + MAX_FLOOR] = 0.0
        for c in self.pending_calls:
            if c["direction"] == -1 and 1 <= c["dest"] <= MAX_FLOOR:
                self._obs_buffer[offset + int(c["dest"]) - 1] = 1.0
        offset += MAX_FLOOR

        # global features
        self._obs_buffer[offset] = min(self.elapsed / 3600.0, 1.0); offset += 1
        self._obs_buffer[offset] = min(len(self.pending_calls) / 30.0, 1.0)

        return self._obs_buffer

    def _get_info(self) -> dict:
        return {
            "elapsed": self.elapsed,
            "pending_calls": len(self.pending_calls),
            "completed": len(self.completed_passengers),
            "total_empty_floors": self.total_empty_floors,
            "total_loaded_floors": self.total_loaded_floors,
            "start_stop_count": self.start_stop_count,
            "elevator_active_time": self.elevator_active_time,
        }

    def _is_done(self) -> bool:
        if self.elapsed >= self.max_total_time:
            return True
        if self.events is None or self.event_idx < len(self.events):
            return False
        if self.pending_calls:
            return False
        for el in self.elevators:
            if el.is_moving or el.assigned_passengers:
                return False
        return bool(self.completed_passengers)

    @staticmethod
    def _compute_state_dim(num_elevators: int) -> int:
        return (MAX_FLOOR + 3 + 1 + 1 + 1 + 1) * num_elevators + MAX_FLOOR * 4 + 2

    @property
    def STATE_DIM(self) -> int:
        return self.state_dim

    def get_episode_metrics(self) -> dict:
        return compute_episode_metrics({
            "completed": self.completed_passengers,
            "total_time": self.elapsed,
            "empty_movement_floors": self.total_empty_floors,
            "loaded_movement_floors": self.total_loaded_floors,
            "start_stop_count": self.start_stop_count,
            "elevator_uptime": self.elevator_active_time,
            "elevator_idle_time": max(0, self.elapsed * self.num_elevators - self.elevator_active_time),
            "num_elevators": self.num_elevators,
        })


if __name__ == "__main__":
    # Quick smoke test with random data
    env = ElevatorEnv()
    fake_events = np.array([
        [1, 5, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [3, 7, 2.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [8, 2, 4.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [5, 1, 6.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
        [2, 9, 8.0, 0.0, 0.0, 0, 0.0, 0.0, 0, 10],
    ], dtype=np.float32)

    obs, info = env.reset(options={"events": fake_events})
    print(f"State dim: {len(obs)}, Action space: {env.action_space.n}")
    print(f"Initial obs[:20]: {obs[:20]}")

    total_reward = 0
    for step in range(200):
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        total_reward += reward
        if done:
            print(f"Done at step {step+1}, reward={total_reward:.2f}")
            break

    metrics = env.get_episode_metrics()
    print(f"\nEpisode metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
