# utils.py
# Ruta: utils.py
# ============================================================
# D.A.P.S Ω — Utilidades
# ============================================================

import logging
import math
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def fmt_ts(dt: datetime) -> str:
    try:
        return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
    except Exception:
        return "N/A"


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except Exception:
        return default


def fmt_pct(x: float, decimals: int = 2) -> str:
    return f"{safe_float(x):.{decimals}f}%"


def fmt_price(x: float) -> str:
    v = safe_float(x)
    if v >= 1000:
        return f"{v:,.2f}"
    if v >= 1:
        return f"{v:.4f}"
    if v >= 0.01:
        return f"{v:.5f}"
    return f"{v:.8f}"


def humanize_minutes(minutes: float) -> str:
    m = int(max(0, minutes))
    if m < 60:
        return f"{m} min"
    h, mm = divmod(m, 60)
    return f"{h}h {mm}min"


def build_status_scan(progress: int, total: int, current: str = "") -> str:
    if total <= 0:
        return f"{current}"
    pct = int(progress * 100 / total)
    return f"[{pct}%] {current} ({progress}/{total})"


def aggregate_summary(results: List[Dict]) -> Dict:
    if not results:
        return {"count": 0}
    return {
        "count": len(results),
        "avg": sum(r.get("score", 0) for r in results) / len(results),
    }
