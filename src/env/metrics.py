"""Elevator scheduling performance metrics."""

import numpy as np

MAX_FLOOR = 10
FLOOR_HEIGHT = 3.0
GRAVITY = 9.81
MASS_KG = 900.0
SPEED_MPS = FLOOR_HEIGHT / 2.0


def avg_wait_time(wait_times: list) -> float:
    if not wait_times:
        return 0.0
    return float(np.mean(wait_times))


def long_wait_rate(wait_times: list, threshold: float = 60.0) -> float:
    if not wait_times:
        return 0.0
    return float(np.mean(np.array(wait_times) > threshold))


def avg_ride_time(ride_times: list) -> float:
    if not ride_times:
        return 0.0
    return float(np.mean(ride_times))


def energy_estimate(
    distance_floors: float,
    start_stop_count: int,
    avg_load_ratio: float = 0.0,
) -> dict:
    """Estimate energy consumption in watt-hours.

    Uses simplified physics: potential energy change for loaded travel,
    kinetic energy for starts/stops.

    Returns:
        dict with keys: potential_wh, kinetic_wh, total_wh
    """
    height = distance_floors * FLOOR_HEIGHT
    potential_energy = MASS_KG * GRAVITY * height * avg_load_ratio
    potential_wh = potential_energy / 3600.0

    v = SPEED_MPS
    kinetic_per_start = 0.5 * MASS_KG * v * v
    kinetic_energy = kinetic_per_start * start_stop_count * (1.0 + avg_load_ratio)
    kinetic_wh = kinetic_energy / 3600.0

    return {
        "potential_wh": potential_wh,
        "kinetic_wh": kinetic_wh,
        "total_wh": potential_wh + kinetic_wh,
    }


def compute_episode_metrics(episode_stats: dict) -> dict:
    """Compute all efficiency metrics from accumulated episode stats.

    episode_stats keys:
        completed: list of passenger dicts with wait_time, ride_time
        total_time: float, total simulation time
        empty_movement_floors: float
        loaded_movement_floors: float
        start_stop_count: int
        elevator_uptime: float (sum of time elevators were active)
        elevator_idle_time: float
    """
    wait_times = [p["wait_time"] for p in episode_stats.get("completed", [])]
    ride_times = [p["ride_time"] for p in episode_stats.get("completed", [])]
    total_movement = (
        episode_stats.get("empty_movement_floors", 0)
        + episode_stats.get("loaded_movement_floors", 0)
    )

    total_time = max(episode_stats.get("total_time", 1.0), 1.0)
    uptime = episode_stats.get("elevator_uptime", 0.0)
    idle = episode_stats.get("elevator_idle_time", 0.0)
    n_elevators = max(episode_stats.get("num_elevators", 3), 1)

    op_eff = uptime / (total_time * n_elevators) if total_time > 0 else 0.0
    empty_rate = (
        episode_stats.get("empty_movement_floors", 0) / total_movement
        if total_movement > 0
        else 0.0
    )
    energy = energy_estimate(
        total_movement,
        episode_stats.get("start_stop_count", 0),
        avg_load_ratio=(1.0 - empty_rate) if total_movement > 0 else 0.0,
    )

    return {
        "operational_efficiency": op_eff,
        "empty_load_rate": empty_rate,
        "avg_wait_time": avg_wait_time(wait_times),
        "avg_ride_time": avg_ride_time(ride_times),
        "long_wait_rate": long_wait_rate(wait_times),
        "total_passengers": len(wait_times),
        "energy_wh": energy["total_wh"],
        "energy_potential_wh": energy["potential_wh"],
        "energy_kinetic_wh": energy["kinetic_wh"],
        "total_movement_floors": total_movement,
        "start_stop_count": episode_stats.get("start_stop_count", 0),
    }
