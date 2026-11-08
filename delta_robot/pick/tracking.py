from __future__ import annotations

from dataclasses import asdict


class ObjectTracker:
    """Expose a consistent tracking snapshot independent of the scheduler."""

    def __init__(self, pool):
        self.pool = pool

    def snapshot(self) -> list[dict]:
        rows = []
        for obj in self.pool.objects.values():
            row = asdict(obj)
            row["velocity"] = [self.pool.speed, 0.0, 0.0] if obj.status == "on_belt" else [0.0, 0.0, 0.0]
            rows.append(row)
        return rows
