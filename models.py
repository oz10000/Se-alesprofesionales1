# models.py
# Ruta: models.py
# ============================================================
# D.A.P.S Ω — Modelos de datos
# ============================================================

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class IndicatorSnapshot:
    """Snapshot de indicadores para un activo en un momento dado."""
    symbol: str
    price: float
    ema_9: float = 0.0
    ema_21: float = 0.0
    ema_50: float = 0.0
    ema_200: float = 0.0
    adx: float = 0.0
    atr: float = 0.0
    atr_pct: float = 0.0
    rsi: float = 50.0
    macd_hist: float = 0.0
    roc: float = 0.0
    bb_width: float = 0.0
    ker: float = 0.0
    rvol: float = 1.0
    tfi: float = 0.0
    ofi: float = 0.0
    cvd: float = 0.0
    regime: str = "Chop"
    mtf_aligned: bool = False
    avg_volume_usd: float = 0.0


@dataclass
class ScoreBreakdown:
    """Componentes del score normalizados [0, 1]."""
    trend: float = 0.0
    momentum: float = 0.0
    volume: float = 0.0
    volatility: float = 0.0
    liquidity: float = 0.0
    regime: float = 0.0


@dataclass
class ScoreResult:
    """Resultado del scoring para un activo."""
    symbol: str
    total_score: float = 0.0
    direction: str = "NEUTRAL"
    prob_long: float = 0.5
    prob_short: float = 0.5
    confidence: float = 0.0

    tier: str = "NO_TRADE"
    breakdown: ScoreBreakdown = field(default_factory=ScoreBreakdown)

    adx: float = 0.0
    atr_pct: float = 0.0
    atr_abs: float = 0.0
    rsi: float = 50.0
    rvol: float = 1.0
    tfi: float = 0.0
    ofi: float = 0.0
    cvd: float = 0.0
    regime: str = "Chop"
    spread_est: float = 0.0
    cost_est: float = 0.0

    is_valid: bool = False
    reason: str = "Sin evaluar"
    conditions: List[str] = field(default_factory=list)

    # Predicción operativa (calculada en risk_manager)
    entry_price: float = 0.0
    entry_zone_low: float = 0.0
    entry_zone_high: float = 0.0
    entry_tolerance_pct: float = 0.0
    entry_type: str = "market"  # market | pullback | breakout

    sl_price: float = 0.0
    sl_pct: float = 0.0
    sl_method: str = "ATR"

    tp_price: float = 0.0
    tp_pct: float = 0.0
    tp_method: str = "ATR"
    rr_ratio: float = 0.0

    trailing_distance_pct: float = 0.0
    trailing_activation_pct: float = 0.0
    breakeven_trigger_pct: float = 0.0

    expected_duration_min: float = 0.0
    expected_duration_conf: float = 0.0
    time_to_tp_min: float = 0.0
    time_to_sl_min: float = 0.0
    next_entry_eta_min: float = 0.0

    risk_level: str = "MEDIUM"   # LOW | MEDIUM | HIGH
    reward_level: str = "MEDIUM" # LOW | MEDIUM | HIGH

    timestamp: str = ""
