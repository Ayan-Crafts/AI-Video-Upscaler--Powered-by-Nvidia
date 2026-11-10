from __future__ import annotations

import psutil


def system_metrics() -> dict[str, float | int]:
    memory = psutil.virtual_memory()
    return {
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_used_gb": round((memory.total - memory.available) / (1024 ** 3), 2),
        "ram_total_gb": round(memory.total / (1024 ** 3), 2),
        "ram_percent": memory.percent,
    }
