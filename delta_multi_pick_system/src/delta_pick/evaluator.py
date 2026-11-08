from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


class Evaluator:
    def __init__(self, output_dir: Path, pool, controller):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pool, self.controller = pool, controller
        self.events_path = self.output_dir / "object_events.csv"
        self.telemetry_path = self.output_dir / "telemetry.csv"
        self._seen_status: dict[int, str] = {}
        self._event_file = self.events_path.open("w", newline="")
        self._telemetry_file = self.telemetry_path.open("w", newline="")
        self._events = csv.writer(self._event_file)
        self._telemetry = csv.writer(self._telemetry_file)
        self._events.writerow(["sim_time_s", "spawn_id", "slot", "event", "x_m", "y_m", "spawn_time_s", "pick_time_s", "place_time_s"])
        self._telemetry.writerow(["sim_time_s", "phase", "spawned", "placed", "missed", "active", "failed_grasps", "failed_placements", "mean_cycle_time_s"])

    def update(self, sim_time: float, log_telemetry: bool = False) -> None:
        for slot, obj in self.pool.objects.items():
            if self._seen_status.get(slot) != obj.status:
                self._seen_status[slot] = obj.status
                self._events.writerow([
                    f"{sim_time:.4f}", obj.spawn_id, slot, obj.status,
                    f"{obj.x:.5f}", f"{obj.y:.5f}", f"{obj.spawn_time:.4f}",
                    "" if obj.pick_time is None else f"{obj.pick_time:.4f}",
                    "" if obj.place_time is None else f"{obj.place_time:.4f}",
                ])
        if log_telemetry:
            state = self.controller.status()
            self._telemetry.writerow([
                f"{sim_time:.4f}", state["phase"], self.pool.spawned,
                self.pool.placed, self.pool.missed,
                len(self.pool.active_objects()), state["failed_grasps"], state["failed_placements"],
                "" if state["mean_cycle_time_s"] is None else f"{state['mean_cycle_time_s']:.5f}",
            ])
        self._event_file.flush()
        self._telemetry_file.flush()

    def summary(self, elapsed_s: float) -> dict[str, Any]:
        completed = len(self.controller.completed_cycles)
        attempts = self.pool.placed + self.pool.missed
        summary = {
            "elapsed_simulation_s": round(elapsed_s, 4),
            "spawned": self.pool.spawned,
            "placed": self.pool.placed,
            "missed": self.pool.missed,
            "active_on_conveyor": len(self.pool.active_objects()),
            "failed_grasps": self.controller.failed_grasps,
            "failed_placements": self.controller.failed_placements,
            "completed_pick_cycles": completed,
            "mean_cycle_time_s": (sum(self.controller.completed_cycles) / completed) if completed else None,
            "throughput_objects_per_min": (self.pool.placed / elapsed_s * 60.0) if elapsed_s > 0 else 0.0,
            "placement_success_rate": (self.pool.placed / attempts) if attempts else None,
            "event_log": str(self.events_path),
            "telemetry_log": str(self.telemetry_path),
        }
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        return summary

    def close(self) -> None:
        self._event_file.close()
        self._telemetry_file.close()
