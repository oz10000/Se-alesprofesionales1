# scoring.py
# Ruta: scoring.py
# ============================================================
# D.A.P.S Ω — Motor de scoring conservador
# ============================================================

import logging
import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from models import IndicatorSnapshot, ScoreBreakdown, ScoreResult
from indicators import (
    compute_adx, compute_atr, compute_rsi, compute_macd,
    compute_roc, compute_bollinger, compute_ema,
    compute_volume_ratio, compute_delta_volume, compute_cvd,
    compute_tfi, compute_ofi, compute_ker, compute_regime,
)

logger = logging.getLogger(__name__)


def _nan_to_zero(x: float) -> float:
    if x is None:
        return 0.0
    try:
        if math.isnan(x) or math.isinf(x):
            return 0.0
    except Exception:
        return 0.0
    return float(x)


class CompositeScorer:
    """Score compuesto normalizado 0–100."""

    def __init__(self, config):
        self.config = config
        self.weights = config.weights
        self.volume_weights = config.volume_sub_weights
        self.tiers = config.tiers
        self.regime_thresholds = config.regime_thresholds
        self.min_prob = config.min_prob_directional
        self.min_move = config.min_expected_move_pct
        self.cost_total = config.total_cost_round_trip
        self.min_bars = config.min_bars_required

    # --------------------------------------------------------
    # CÁLCULO PÚBLICO
    # --------------------------------------------------------

    def compute(self, symbol: str, data: Dict[str, pd.DataFrame]) -> ScoreResult:
        try:
            df = data.get("entry")
            if df is None or df.empty or len(df) < self.min_bars:
                return self._empty(symbol, f"Datos insuficientes ({len(df) if df is not None else 0})")

            # Eliminar última vela (sin look-ahead)
            df_use = df.iloc[:-1]
            if len(df_use) < self.min_bars // 2:
                return self._empty(symbol, "Datos insuficientes tras limpieza")

            snap = self._snapshot(symbol, df_use, data.get("confirm"), data.get("trend"))
            br = self._breakdown(snap)
            total = self._weighted_score(br)

            # Tier
            tier = self._classify_tier(total)

            # Probabilidades
            raw_long = self._prob_long(snap)
            prob_long = max(0.01, min(0.99, raw_long))
            prob_short = 1.0 - prob_long

            # Dirección
            if prob_long > self.min_prob:
                direction = "LONG"
            elif prob_short > self.min_prob:
                direction = "SHORT"
            else:
                direction = "NEUTRAL"

            # Confianza (0–100)
            confidence = round(min(total, 100.0), 2)

            # Validación
            is_valid, reason = self._validate(snap, total, direction)

            # Condiciones
            conditions = self._conditions(snap, direction)

            # Costos estimados
            spread_est = self._spread_estimate(snap)
            cost_est = self.cost_total

            res = ScoreResult(
                symbol=symbol,
                total_score=round(total, 2),
                direction=direction,
                prob_long=round(prob_long, 4),
                prob_short=round(prob_short, 4),
                confidence=confidence,
                tier=tier,
                breakdown=br,
                adx=round(_nan_to_zero(snap.adx), 2),
                atr_pct=round(_nan_to_zero(snap.atr_pct), 5),
                atr_abs=round(_nan_to_zero(snap.atr), 6),
                rsi=round(_nan_to_zero(snap.rsi), 2),
                rvol=round(_nan_to_zero(snap.rvol), 3),
                tfi=round(_nan_to_zero(snap.tfi), 4),
                ofi=round(_nan_to_zero(snap.ofi), 4),
                cvd=round(_nan_to_zero(snap.cvd), 4),
                regime=snap.regime,
                spread_est=round(spread_est, 5),
                cost_est=round(cost_est, 5),
                is_valid=is_valid,
                reason=reason,
                conditions=conditions,
            )
            return res
        except Exception as e:
            logger.error(f"Error scoring {symbol}: {e}", exc_info=True)
            return self._empty(symbol, f"Error: {str(e)[:80]}")

    # --------------------------------------------------------
    # SNAPSHOT
    # --------------------------------------------------------

    def _snapshot(self, symbol: str, df_use: pd.DataFrame,
                  df_confirm: Optional[pd.DataFrame],
                  df_trend: Optional[pd.DataFrame]) -> IndicatorSnapshot:
        close = float(df_use["close"].iloc[-1])
        snap = IndicatorSnapshot(symbol=symbol, price=close)

        if len(df_use) >= 9:
            snap.ema_9 = float(compute_ema(df_use, 9).iloc[-1])
        if len(df_use) >= 21:
            snap.ema_21 = float(compute_ema(df_use, 21).iloc[-1])
        if len(df_use) >= 50:
            snap.ema_50 = float(compute_ema(df_use, 50).iloc[-1])
        if len(df_use) >= 200:
            snap.ema_200 = float(compute_ema(df_use, 200).iloc[-1])
        else:
            snap.ema_200 = snap.ema_50

        snap.adx = float(compute_adx(df_use, 14).iloc[-1])
        atr_s = compute_atr(df_use, 14)
        snap.atr = float(atr_s.iloc[-1])
        snap.atr_pct = snap.atr / close if close > 0 else 0.0

        snap.rsi = float(compute_rsi(df_use, 14).iloc[-1])
        macd_df = compute_macd(df_use)
        snap.macd_hist = float(macd_df["histogram"].iloc[-1])
        snap.roc = float(compute_roc(df_use, 10).iloc[-1])
        bb = compute_bollinger(df_use)
        snap.bb_width = float(bb["bb_width"].iloc[-1])
        snap.ker = float(compute_ker(df_use, 10).iloc[-1])

        snap.rvol = float(compute_volume_ratio(df_use, 20).iloc[-1])
        snap.tfi = float(compute_tfi(df_use, 5).iloc[-1])
        snap.ofi = float(compute_ofi(df_use, 5).iloc[-1])
        snap.cvd = float(compute_cvd(df_use, 20).iloc[-1])

        snap.regime = compute_regime(snap.adx, snap.atr_pct)

        # MTF alignment
        if df_confirm is not None and not df_confirm.empty and len(df_confirm) >= 50:
            try:
                ema_50c = float(compute_ema(df_confirm, 50).iloc[-1])
                ema_200c = (float(compute_ema(df_confirm, 200).iloc[-1])
                            if len(df_confirm) >= 200 else ema_50c)
                snap.mtf_aligned = bool(ema_50c > ema_200c)
            except Exception:
                snap.mtf_aligned = False

        # Liquidez estimada
        try:
            avg_v = float(df_use["volume"].tail(100).mean())
            avg_p = float(df_use["close"].tail(100).mean())
            snap.avg_volume_usd = avg_v * avg_p
        except Exception:
            snap.avg_volume_usd = 0.0

        # Sanitizar
        for k in ("ema_9", "ema_21", "ema_50", "ema_200", "adx", "atr",
                  "atr_pct", "rsi", "macd_hist", "roc", "bb_width", "ker",
                  "rvol", "tfi", "ofi", "cvd", "avg_volume_usd"):
            setattr(snap, k, _nan_to_zero(getattr(snap, k)))

        return snap

    # --------------------------------------------------------
    # BREAKDOWN [0,1]
    # --------------------------------------------------------

    def _breakdown(self, s: IndicatorSnapshot) -> ScoreBreakdown:
        b = ScoreBreakdown()

        # Tendencia
        ema_score = 0.0
        if s.ema_9 > s.ema_21: ema_score += 0.33
        if s.ema_21 > s.ema_50: ema_score += 0.33
        if s.ema_50 > s.ema_200: ema_score += 0.34
        adx_n = min(s.adx / 40.0, 1.0)
        ker_n = min(max(s.ker, 0.0), 1.0)
        b.trend = 0.40 * ema_score + 0.35 * adx_n + 0.25 * ker_n

        # Momentum
        rsi_n = 1.0 - abs(s.rsi - 50.0) / 50.0
        macd_n = min(max(s.macd_hist * 100 + 0.5, 0), 1)
        roc_n = min(max(s.roc / 5.0 + 0.5, 0), 1)
        b.momentum = 0.40 * rsi_n + 0.30 * macd_n + 0.30 * roc_n

        # Volumen inteligente
        rvol_n = min(s.rvol / 3.0, 1.0)
        tfi_n = min(abs(s.tfi) * 2.5, 1.0)
        ofi_n = min(abs(s.ofi) * 2.5, 1.0)
        cvd_n = min(abs(s.cvd) * 0.5, 1.0)
        w = self.volume_weights
        b.volume = (
            w.get("tfi", 0.35) * tfi_n
            + w.get("ofi", 0.25) * ofi_n
            + w.get("rvol", 0.25) * rvol_n
            + w.get("cvd", 0.15) * cvd_n
        )

        # Volatilidad (rango operable)
        ap = s.atr_pct
        if 0.005 <= ap <= 0.015:
            v = 1.0
        elif 0.003 <= ap <= 0.025:
            v = 0.7
        else:
            v = 0.3
        bb_w = s.bb_width if s.bb_width else 0.02
        bb_n = 1.0 - abs(bb_w - 0.02) / 0.03
        bb_n = max(0.0, min(bb_n, 1.0))
        b.volatility = 0.60 * v + 0.40 * bb_n

        # Liquidez
        v = s.avg_volume_usd
        if v > 1e9: liq = 1.0
        elif v > 1e8: liq = 0.85
        elif v > 1e7: liq = 0.65
        elif v > 1e6: liq = 0.40
        else: liq = 0.20
        b.liquidity = liq

        # Régimen
        regime_map = {
            "Expansion": 1.0,
            "Trend_Strong": 0.9,
            "Trend_Weak": 0.6,
            "Chop": 0.2,
        }
        b.regime = regime_map.get(s.regime, 0.3)

        # Sanitizar
        for k in ("trend", "momentum", "volume", "volatility", "liquidity", "regime"):
            setattr(b, k, max(0.0, min(1.0, _nan_to_zero(getattr(b, k)))))

        return b

    # --------------------------------------------------------
    # SCORE PONDERADO
    # --------------------------------------------------------

    def _weighted_score(self, b: ScoreBreakdown) -> float:
        w = self.weights
        total = (
            w.get("trend", 0.17) * b.trend
            + w.get("momentum", 0.15) * b.momentum
            + w.get("volume_intelligence", 0.40) * b.volume
            + w.get("volatility", 0.11) * b.volatility
            + w.get("liquidity", 0.10) * b.liquidity
            + w.get("regime", 0.07) * b.regime
        ) * 100.0
        return max(0.0, min(100.0, _nan_to_zero(total)))

    # --------------------------------------------------------
    # PROBABILIDAD LONG
    # --------------------------------------------------------

    def _prob_long(self, s: IndicatorSnapshot) -> float:
        ema_bull = 1.0 if s.ema_9 > s.ema_21 else 0.0
        rsi_bull = 1.0 if s.rsi > 50 else 0.0
        mtf = 1.0 if s.mtf_aligned else 0.0

        score = (
            0.35 * (s.tfi * 2.5 + 0.5)
            + 0.25 * (s.ofi * 2.5 + 0.5)
            + 0.20 * ema_bull
            + 0.10 * rsi_bull
            + 0.10 * mtf
        )
        return float(np.clip(score, 0.0, 1.0))

    # --------------------------------------------------------
    # TIER
    # --------------------------------------------------------

    def _classify_tier(self, score: float) -> str:
        ordered = sorted(self.tiers.items(), key=lambda kv: -kv[1])
        for tier, threshold in ordered:
            if score >= threshold:
                return tier
        return "NO_TRADE"

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    def _validate(self, s: IndicatorSnapshot, score: float,
                  direction: str) -> tuple:
        regime = s.regime
        th = self.regime_thresholds.get(
            regime,
            self.regime_thresholds.get("Chop",
                {"min_score": 65, "min_adx": 15, "min_rvol": 2.0})
        )
        min_score = th["min_score"]
        min_adx = th["min_adx"]
        min_rvol = th["min_rvol"]

        if score < min_score:
            return False, f"Score {score:.1f} < {min_score}"
        if direction == "NEUTRAL":
            return False, "Dirección neutral"
        if s.rvol < min_rvol:
            return False, f"RVOL {s.rvol:.2f} < {min_rvol}"
        if s.adx < min_adx:
            return False, f"ADX {s.adx:.1f} < {min_adx}"

        # Cost-aware filter
        expected_move = s.atr_pct * 1.5
        if expected_move < self.min_move:
            return False, f"Mov. esperado {expected_move:.4f} < {self.min_move}"

        return True, "OK"

    # --------------------------------------------------------
    # CONDICIONES
    # --------------------------------------------------------

    def _conditions(self, s: IndicatorSnapshot, direction: str) -> List[str]:
        c: List[str] = []
        if s.adx >= 30:
            c.append(f"ADX≥30 ({s.adx:.0f})")
        if s.ker >= 0.55:
            c.append(f"KER≥0.55 ({s.ker:.2f})")
        if s.rvol >= 2.0:
            c.append(f"RVOL≥2.0 ({s.rvol:.2f})")
        if abs(s.tfi) >= 0.2:
            c.append(f"TFI {'+' if s.tfi > 0 else ''}{s.tfi:.2f}")
        if abs(s.ofi) >= 0.2:
            c.append(f"OFI {'+' if s.ofi > 0 else ''}{s.ofi:.2f}")
        if s.mtf_aligned:
            c.append("MTF✓")
        if s.regime in ("Expansion", "Trend_Strong"):
            c.append(f"Régimen:{s.regime}")
        return c

    # --------------------------------------------------------
    # SPREAD ESTIMADO
    # --------------------------------------------------------

    def _spread_estimate(self, s: IndicatorSnapshot) -> float:
        # Aproximación conservadora por liquidez
        v = s.avg_volume_usd
        if v > 1e9: return 0.0001
        if v > 1e8: return 0.0002
        if v > 1e7: return 0.0004
        if v > 1e6: return 0.0008
        return 0.0015

    # --------------------------------------------------------
    # VACÍO
    # --------------------------------------------------------

    def _empty(self, symbol: str, reason: str) -> ScoreResult:
        return ScoreResult(
            symbol=symbol,
            total_score=0.0,
            direction="NEUTRAL",
            prob_long=0.5,
            prob_short=0.5,
            tier="NO_TRADE",
            is_valid=False,
            reason=reason,
        )
