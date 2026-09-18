# ranking.py
# Ruta: ranking.py
# ============================================================
# D.A.P.S Ω — Rankings LONG y SHORT independientes
# ============================================================

from typing import List, Dict
from models import ScoreResult


_TIER_ORDER = {"OMEGA": 0, "A": 1, "B": 2, "S": 3, "NO_TRADE": 4}


def _sort_key(r: ScoreResult):
    return (_TIER_ORDER.get(r.tier, 5), -r.total_score)


def rank_all(results: List[ScoreResult]) -> List[ScoreResult]:
    """Ranking completo (todas las señales)."""
    return sorted(results, key=_sort_key)


def rank_long(results: List[ScoreResult]) -> List[ScoreResult]:
    """Ranking LONG ordenado por tier y score."""
    longs = [r for r in results if r.direction == "LONG" and r.is_valid]
    return sorted(longs, key=_sort_key)


def rank_short(results: List[ScoreResult]) -> List[ScoreResult]:
    """Ranking SHORT ordenado por tier y score."""
    shorts = [r for r in results if r.direction == "SHORT" and r.is_valid]
    return sorted(shorts, key=_sort_key)


def best_long(results: List[ScoreResult]):
    r = rank_long(results)
    return r[0] if r else None


def best_short(results: List[ScoreResult]):
    r = rank_short(results)
    return r[0] if r else None


def summary(results: List[ScoreResult]) -> Dict:
    return {
        "total": len(results),
        "valid": sum(1 for r in results if r.is_valid),
        "long_valid": sum(1 for r in results if r.is_valid and r.direction == "LONG"),
        "short_valid": sum(1 for r in results if r.is_valid and r.direction == "SHORT"),
        "neutral": sum(1 for r in results if r.direction == "NEUTRAL"),
        "avg_score": (sum(r.total_score for r in results) / len(results)) if results else 0.0,
        "max_score": max((r.total_score for r in results), default=0.0),
    }
