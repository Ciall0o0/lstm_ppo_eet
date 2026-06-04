"""PLC real-time monitor with live visualization.

Connects to PLC via bridge, runs scheduling inference, and displays
real-time charts of elevator status, dispatch decisions, and performance.

Usage:
    PYTHONPATH=. .venv/bin/python src/plc/run_monitor.py --ip 192.168.0.1
    PYTHONPATH=. .venv/bin/python src/plc/run_monitor.py --dry-run
"""

from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
# Auto-detect backend: prefer interactive, fall back to Agg
_GUI_BACKEND = None
for backend in ("TkAgg", "Qt5Agg", "GTK3Agg"):
    try:
        matplotlib.use(backend)
        # Verify backend actually works by creating a test figure
        import matplotlib.pyplot as _plt
        _fig = _plt.figure()
        _plt.close(_fig)
        _GUI_BACKEND = backend
        break
    except (ImportError, Exception):
        continue
if _GUI_BACKEND is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.utils import load_config
from src.plc.bridge import PLCInferenceBridge

logger = logging.getLogger(__name__)

ELEVATOR_COLORS = ["#2196F3", "#4CAF50", "#FF9800"]
LATENCY_COLOR = "#E91E63"
ELEVATOR_LABELS = ["Elevator 1", "Elevator 2", "Elevator 3"]


class PLCMonitor:
    """Real-time visualization dashboard for PLC bridge."""

    def __init__(self, bridge: PLCInferenceBridge):
        self.bridge = bridge
        self.start_time = time.time()

        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle("Elevator PLC Monitor", fontsize=14, fontweight="bold")
        self.fig.tight_layout(pad=3.0)

    def update(self, frame):
        """Called by FuncAnimation to refresh all plots."""
        stats = self.bridge.get_stats()
        decisions = self.bridge.get_recent_decisions(200)

        self._update_floor_status(decisions)
        self._update_action_distribution(decisions)
        self._update_latency_timeline(decisions)
        self._update_stats_display(stats)

    def _update_floor_status(self, decisions: list[dict]):
        """Bar chart showing elevator positions."""
        ax = self.axes[0, 0]
        ax.clear()

        # Read cached elevator states from bridge (thread-safe snapshot)
        floors = self.bridge.get_elevator_floors()

        bars = ax.bar(ELEVATOR_LABELS, floors, color=ELEVATOR_COLORS,
                      edgecolor="black", linewidth=0.5)

        for bar, floor in zip(bars, floors):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"Floor {floor}", ha="center", va="bottom", fontweight="bold")

        ax.set_ylim(0, 11)
        ax.set_ylabel("Floor")
        ax.set_title("Elevator Positions")
        ax.set_yticks(range(1, 11))
        ax.grid(axis="y", alpha=0.3)

    def _update_action_distribution(self, decisions: list[dict]):
        """Pie chart showing which elevators were dispatched."""
        ax = self.axes[0, 1]
        ax.clear()

        if not decisions:
            ax.text(0.5, 0.5, "No decisions yet", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Dispatch Distribution")
            return

        counts = Counter(d["action"] for d in decisions)
        sizes = [counts.get(i, 0) for i in range(3)]

        non_zero = [(s, l, c) for s, l, c in zip(sizes, ELEVATOR_LABELS, ELEVATOR_COLORS) if s > 0]
        if non_zero:
            sizes, labels, colors = zip(*non_zero)
            ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                   startangle=90, pctdistance=0.85)
            centre = plt.Circle((0, 0), 0.70, fc="white")
            ax.add_artist(centre)

        ax.set_title(f"Dispatch Distribution (n={sum(sizes)})")

    def _update_latency_timeline(self, decisions: list[dict]):
        """Line chart showing inference latency over time."""
        ax = self.axes[1, 0]
        ax.clear()

        if not decisions:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="gray")
            ax.set_title("Inference Latency")
            return

        latencies = [d["decision_time_ms"] for d in decisions]
        ax.plot(latencies, color=LATENCY_COLOR, linewidth=1.0, alpha=0.8)
        ax.fill_between(range(len(latencies)), latencies, alpha=0.2, color=LATENCY_COLOR)

        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        ax.axhline(y=avg, color="gray", linestyle="--", alpha=0.5, label=f"Avg: {avg:.1f}ms")
        ax.axhline(y=p95, color="orange", linestyle="--", alpha=0.5, label=f"P95: {p95:.1f}ms")

        ax.set_xlabel("Decision #")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Inference Latency")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)

    def _update_stats_display(self, stats: dict):
        """Text panel showing cumulative statistics."""
        ax = self.axes[1, 1]
        ax.clear()
        ax.axis("off")

        elapsed = time.time() - self.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)

        lines = [
            ("Runtime", f"{hours:02d}:{minutes:02d}:{seconds:02d}"),
            ("Total Decisions", f"{stats['decisions']}"),
            ("Errors", f"{stats['errors']}"),
            ("Last Latency", f"{stats['last_decision_time']:.2f} ms"),
            ("", ""),
            ("PLC Status", "Connected" if self.bridge.client.connected else "Disconnected"),
        ]

        y_pos = 0.9
        for label, value in lines:
            if label:
                ax.text(0.1, y_pos, f"{label}:", fontsize=11, fontweight="bold",
                        transform=ax.transAxes, va="top")
                ax.text(0.6, y_pos, value, fontsize=11,
                        transform=ax.transAxes, va="top")
            y_pos -= 0.15

        ax.set_title("Statistics")

    def start(self, interval_ms: int = 1000):
        """Start the live visualization."""
        self.ani = FuncAnimation(self.fig, self.update, interval=interval_ms,
                                 cache_frame_data=False)
        plt.show()

    def save_snapshot(self, path: Path):
        """Save current figure to file."""
        self.fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved snapshot to {path}")


def export_decisions_csv(decisions: list[dict], path: Path):
    """Export decision log to CSV."""
    if not decisions:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=decisions[0].keys())
        writer.writeheader()
        writer.writerows(decisions)
    logger.info(f"Exported {len(decisions)} decisions to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="PLC Elevator Monitor")
    parser.add_argument("--ip", default="192.168.0.1", help="PLC IP address")
    parser.add_argument("--rack", type=int, default=0, help="PLC rack")
    parser.add_argument("--slot", type=int, default=1, help="PLC slot")
    parser.add_argument("--poll-interval", type=int, default=100,
                        help="Poll interval in ms")
    parser.add_argument("--log-file", type=str, help="Log to file")
    parser.add_argument("--save-dir", type=str, default="checkpoints/monitor",
                        help="Directory to save outputs")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without GUI (headless mode)")
    parser.add_argument("--duration", type=int, default=0,
                        help="Run duration in seconds (0 = indefinite)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without PLC connection")
    return parser.parse_args()


def main():
    args = parse_args()

    log_format = "%(asctime)s [%(levelname)s] %(message)s"
    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file))
    logging.basicConfig(level=logging.INFO, format=log_format, handlers=handlers)

    cfg = load_config()
    cfg.setdefault("plc", {})
    cfg["plc"]["ip"] = args.ip
    cfg["plc"]["rack"] = args.rack
    cfg["plc"]["slot"] = args.slot
    cfg["plc"]["poll_interval_ms"] = args.poll_interval

    bridge = PLCInferenceBridge()
    bridge.cfg = cfg
    bridge.poll_interval = args.poll_interval / 1000.0
    bridge.client.ip = args.ip
    bridge.client.rack = args.rack
    bridge.client.slot = args.slot

    if args.dry_run:
        logger.info("Dry run mode — skipping PLC connection")
        logger.info(f"Config: ip={args.ip}, rack={args.rack}, slot={args.slot}, "
                    f"poll={args.poll_interval}ms")
        logger.info("Bridge module OK")
        return

    logger.info(f"Connecting to PLC at {args.ip} (rack={args.rack}, slot={args.slot})...")
    if not bridge.connect():
        logger.error("Failed to connect to PLC")
        sys.exit(1)

    logger.info("Connected! Starting monitor...")

    def shutdown(sig, frame):
        logger.info("Shutting down...")
        bridge.stop()
        bridge.disconnect()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    import threading
    bridge_thread = threading.Thread(target=bridge.run, daemon=True)
    bridge_thread.start()

    monitor = None
    use_gui = not args.no_gui and _GUI_BACKEND is not None

    if not use_gui:
        if not args.no_gui and _GUI_BACKEND is None:
            logger.warning("No GUI backend available, falling back to headless mode")
            logger.info("To enable GUI, install PyQt5: uv add PyQt5")
        logger.info("Headless mode — press Ctrl+C to stop")
        try:
            if args.duration > 0:
                time.sleep(args.duration)
            else:
                while bridge.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            pass
    else:
        monitor = PLCMonitor(bridge)

        if args.duration > 0:
            def stop_after_duration():
                time.sleep(args.duration)
                bridge.stop()
            threading.Thread(target=stop_after_duration, daemon=True).start()

        try:
            monitor.start(interval_ms=1000)
        except Exception as e:
            logger.error(f"GUI error: {e}")

    bridge.stop()
    bridge.disconnect()

    decisions = bridge.get_recent_decisions(1000)
    if decisions:
        export_decisions_csv(decisions, save_dir / "decisions.csv")

    if monitor is not None:
        try:
            monitor.save_snapshot(save_dir / "final_snapshot.png")
        except Exception:
            pass

    stats = bridge.get_stats()
    logger.info(f"Session complete: {stats['decisions']} decisions, "
                f"{stats['errors']} errors")


if __name__ == "__main__":
    main()
