# config.py
# Ruta: config.py
# ============================================================
# D.A.P.S Ω — Cargador de configuración
# ============================================================

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import yaml


logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Error de configuración."""
    pass


def _load_yaml(path: str) -> dict:
    """Carga YAML con manejo robusto de errores."""
    if not os.path.exists(path):
        raise ConfigError(f"Archivo no encontrado: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"YAML inválido en {path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error leyendo {path}: {e}")
    if not isinstance(data, dict):
        raise ConfigError(f"YAML debe ser diccionario en {path}")
    return data


def _normalize(d: Dict[str, float], name: str) -> Dict[str, float]:
    """Normaliza un dict de pesos para que sumen 1.0."""
    if not d:
        return {}
    total = sum(d.values())
    if total <= 0:
        n = len(d)
        return {k: 1.0 / n for k in d}
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"⚠️ {name} suma {total:.4f}; normalizando")
        return {k: v / total for k, v in d.items()}
    return d


@dataclass
class ScannerConfig:
    """Configuración del scanner."""

    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str = "config.yaml") -> "ScannerConfig":
        raw = _load_yaml(path)
        inst = cls(raw=raw)
        inst._validate()
        inst._ensure_dirs()
        return inst

    def _validate(self) -> None:
        # Claves mínimas
        for key in ("project", "exchanges", "symbols", "scoring", "risk", "tiers"):
            if key not in self.raw:
                raise ConfigError(f"Falta clave '{key}' en config")

        # Exchanges soportados
        supported = self.raw["exchanges"].get("supported", [])
        if not supported:
            raise ConfigError("No hay exchanges en 'exchanges.supported'")

        # Símbolos
        total_symbols = sum(
            len(v) for v in self.raw["symbols"].values() if isinstance(v, list)
        )
        if total_symbols == 0:
            raise ConfigError("No hay símbolos definidos")

        # Pesos del scoring
        w = self.raw["scoring"].get("weights", {})
        self.raw["scoring"]["weights"] = _normalize(w, "scoring.weights")

        # Sub-pesos de volumen
        sw = self.raw["scoring"].get("sub_weights", {}).get("volume", {})
        self.raw["scoring"]["sub_weights"]["volume"] = _normalize(
            sw, "scoring.sub_weights.volume"
        )

    def _ensure_dirs(self) -> None:
        for key, path in self.raw.get("directories", {}).items():
            try:
                os.makedirs(path, exist_ok=True)
            except Exception as e:
                logger.warning(f"No se pudo crear {path}: {e}")

    # ---------------- PROPIEDADES ----------------

    @property
    def project(self) -> Dict:
        return self.raw.get("project", {})

    @property
    def exchanges(self) -> Dict:
        return self.raw.get("exchanges", {})

    @property
    def supported_exchanges(self) -> List[str]:
        return list(self.exchanges.get("supported", []))

    @property
    def exchange_priority(self) -> List[str]:
        return list(self.exchanges.get("priority", []))

    @property
    def symbols(self) -> Dict[str, List[str]]:
        return self.raw.get("symbols", {})

    @property
    def symbols_for_exchange(self) -> Dict[str, List[str]]:
        """Devuelve símbolos por exchange."""
        out: Dict[str, List[str]] = {}
        for ex in self.supported_exchanges:
            out[ex] = list(self.symbols.get(ex, []))
        return out

    @property
    def timeframes(self) -> Dict:
        return self.raw.get("timeframes", {})

    @property
    def supported_timeframes(self) -> List[str]:
        return list(self.timeframes.get("supported", []))

    @property
    def default_timeframe(self) -> str:
        return self.timeframes.get("default", "15m")

    @property
    def history_limit(self) -> int:
        return int(self.timeframes.get("history_limit", 500))

    @property
    def min_bars_required(self) -> int:
        return int(self.timeframes.get("min_bars_required", 200))

    @property
    def scoring(self) -> Dict:
        return self.raw.get("scoring", {})

    @property
    def weights(self) -> Dict[str, float]:
        return _normalize(self.scoring.get("weights", {}), "scoring.weights")

    @property
    def volume_sub_weights(self) -> Dict[str, float]:
        sw = self.scoring.get("sub_weights", {}).get("volume", {})
        return _normalize(sw, "sub_weights.volume")

    @property
    def thresholds(self) -> Dict:
        return self.raw.get("thresholds", {})

    @property
    def regime_thresholds(self) -> Dict:
        return self.thresholds.get("regime_thresholds", {})

    @property
    def min_prob_directional(self) -> float:
        return float(self.thresholds.get("min_prob_directional", 0.60))

    @property
    def min_expected_move_pct(self) -> float:
        return float(self.thresholds.get("min_expected_move_pct", 0.0032))

    @property
    def tiers(self) -> Dict[str, float]:
        return self.raw.get("tiers", {})

    @property
    def risk(self) -> Dict:
        return self.raw.get("risk", {})

    @property
    def costs(self) -> Dict:
        return self.raw.get("costs", {})

    @property
    def total_cost_round_trip(self) -> float:
        c = self.costs
        return (
            c.get("fee_per_side", 0.0005) * 2
            + c.get("slippage", 0.0003)
            + c.get("spread", 0.0002)
        )

    @property
    def verification(self) -> Dict:
        return self.raw.get("verification", {})

    # ---------------- HELPERS ----------------

    def atr_mult_sl(self, symbol: str) -> float:
        sl = self.risk.get("stop_loss", {})
        m = sl.get("atr_mult_by_symbol", {})
        return float(m.get(symbol, sl.get("atr_mult_default", 1.8)))

    def atr_mult_tp(self) -> float:
        return float(self.risk.get("take_profit", {}).get("atr_mult_default", 2.5))

    def trailing_atr_mult(self) -> float:
        return float(self.risk.get("trailing", {}).get("atr_mult_default", 2.0))

    def trailing_floor(self) -> float:
        return float(self.risk.get("trailing", {}).get("floor", 0.03))

    def trailing_cap(self) -> float:
        return float(self.risk.get("trailing", {}).get("cap", 0.08))

    def trailing_activation_atr(self) -> float:
        return float(self.risk.get("trailing", {}).get("activation_atr", 0.50))

    def breakeven_trigger_atr(self) -> float:
        return float(self.risk.get("trailing", {}).get("breakeven_trigger_atr", 0.25))

    def entry_zone_atr_mult(self) -> float:
        return float(self.risk.get("entry", {}).get("zone_atr_mult", 0.30))

    def entry_max_chase_pct(self) -> float:
        return float(self.risk.get("entry", {}).get("max_chase_pct", 0.005))

    def min_bars(self) -> int:
        return self.min_bars_required