# risk_manager.py
# Ruta: risk_manager.py
# ============================================================
# D.A.P.S Ω — Gestión de riesgo y predicción operativa
# ============================================================

import math
import logging
from typing import Optional

from models import ScoreResult

logger = logging.getLogger(__name__)


class RiskManager:
    """Calcula entrada, SL, TP, trailing, tiempos esperados."""

    def __init__(self, config):
        self.config = config
        self.cost = config.total_cost_round_trip

    # --------------------------------------------------------
    # COMPLETAR SCORERESULT
    # --------------------------------------------------------

    def enrich(self, res: ScoreResult) -> ScoreResult:
        try:
            if res.entry_price <= 0:
                # Si no tenemos precio, no podemos calcular
                return res

            entry = res.entry_price
            atr = res.atr_abs if res.atr_abs > 0 else entry * 0.005
            atr_pct = res.atr_pct if res.atr_pct > 0 else atr / entry

            # -------- ENTRADA --------
            res.entry_price = entry
            zone = self.config.entry_zone_atr_mult() * atr
            res.entry_zone_low = max(entry - zone, 0.0)
            res.entry_zone_high = entry + zone
            res.entry_tolerance_pct = self.config.entry_max_chase_pct()

            # Tipo de entrada según régimen
            if res.regime in ("Expansion", "Trend_Strong"):
                res.entry_type = "breakout"
            elif res.regime == "Trend_Weak":
                res.entry_type = "market"
            else:
                res.entry_type = "pullback"

            # -------- STOP LOSS --------
            sl_mult = self.config.atr_mult_sl(res.symbol)
            sl_dist = sl_mult * atr_pct
            sl_dist = max(min(sl_dist, 0.02), 0.0015)

            if res.direction == "LONG":
                res.sl_price = entry * (1 - sl_dist)
            else:
                res.sl_price = entry * (1 + sl_dist)
            res.sl_pct = sl_dist * 100
            res.sl_method = f"ATR×{sl_mult:.1f}"

            # -------- TAKE PROFIT --------
            tp_mult = self.config.atr_mult_tp()
            tp_dist = max(tp_mult * atr_pct, 0.003)

            if res.direction == "LONG":
                res.tp_price = entry * (1 + tp_dist)
            else:
                res.tp_price = entry * (1 - tp_dist)
            res.tp_pct = tp_dist * 100
            res.tp_method = f"ATR×{tp_mult:.1f}"

            # -------- R:R --------
            if res.sl_pct > 0:
                res.rr_ratio = res.tp_pct / res.sl_pct
            else:
                res.rr_ratio = 0.0

            # -------- TRAILING / BE --------
            trailing_dist = self.config.trailing_atr_mult() * atr_pct
            trailing_dist = max(self.config.trailing_floor(),
                                min(self.config.trailing_cap(), trailing_dist))
            res.trailing_distance_pct = trailing_dist * 100
            res.trailing_activation_pct = (
                self.config.trailing_activation_atr() * atr_pct * 100
            )
            res.breakeven_trigger_pct = (
                self.config.breakeven_trigger_atr() * atr_pct * 100
            )

            # -------- TIEMPOS ESPERADOS --------
            self._estimate_times(res)

            # -------- RIESGO / RECOMPENSA --------
            res.risk_level = self._classify_risk(res.sl_pct, res.adx, res.rvol)
            res.reward_level = self._classify_reward(res.tp_pct, res.rr_ratio)

            return res
        except Exception as e:
            logger.error(f"Error enriquecimiento {res.symbol}: {e}", exc_info=True)
            return res

    # --------------------------------------------------------
    # ESTIMACIÓN DE TIEMPOS
    # --------------------------------------------------------

    def _estimate_times(self, res: ScoreResult) -> None:
        # Base: cuanto mayor movimiento esperado y menor ADX, más tiempo
        atr_pct = res.atr_pct if res.atr_pct > 0 else 0.005
        adx = max(res.adx, 1.0)
        rvol = max(res.rvol, 0.5)

        # Factor de velocidad: más ADX y RVOL => más rápido
        speed_factor = (adx / 30.0) * (rvol / 1.5)
        speed_factor = max(0.3, min(speed_factor, 3.0))

        # Duración base en minutos
        base = 60.0 / speed_factor

        res.expected_duration_min = round(base, 1)
        res.expected_duration_conf = round(
            min(100.0, 40.0 + 20.0 * (res.total_score / 100.0) + 15.0 * min(adx / 40.0, 1.0)),
            1,
        )

        # Tiempo a TP: proporcional a distancia / velocidad
        tp_factor = res.tp_pct / (atr_pct * 100.0) if atr_pct > 0 else 2.0
        res.time_to_tp_min = round(base * max(tp_factor, 1.0), 1)

        # Tiempo a SL: típicamente más rápido
        sl_factor = res.sl_pct / (atr_pct * 100.0) if atr_pct > 0 else 1.5
        res.time_to_sl_min = round(base * max(sl_factor, 0.5) * 0.7, 1)

        # ETA hasta próxima entrada (si no es válida)
        if res.is_valid:
            res.next_entry_eta_min = 0.0
        else:
            res.next_entry_eta_min = round(
                max(15.0, (100.0 - res.total_score) * 1.2),
                1,
            )

    # --------------------------------------------------------
    # CLASIFICACIÓN RIESGO / RECOMPENSA
    # --------------------------------------------------------

    def _classify_risk(self, sl_pct: float, adx: float, rvol: float) -> str:
        if sl_pct <= 0:
            return "HIGH"
        score = 0.0
        if sl_pct < 1.0:
            score += 1
        elif sl_pct < 1.8:
            score += 2
        else:
            score += 3

        if adx >= 30: score += 1
        if rvol >= 2.0: score += 1

        if score >= 4: return "LOW"
        if score >= 2: return "MEDIUM"
        return "HIGH"

    def _classify_reward(self, tp_pct: float, rr: float) -> str:
        if tp_pct <= 0:
            return "LOW"
        score = 0.0
        if tp_pct >= 2.5: score += 2
        elif tp_pct >= 1.2: score += 1

        if rr >= 2.0: score += 2
        elif rr >= 1.4: score += 1

        if score >= 3: return "HIGH"
        if score >= 1: return "MEDIUM"
        return "LOW"