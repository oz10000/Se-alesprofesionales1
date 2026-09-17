# data_engine.py
# Ruta: data_engine.py
# ============================================================
# D.A.P.S Ω — Motor de datos (Binance / Bybit) con caché
# ============================================================

import os
import time
import logging
from typing import Dict, List, Optional

import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

CACHE_TTL = 3600


class DataEngine:
    """Motor de datos conservador con Binance y Bybit como prioridad."""

    def __init__(self, config):
        self.config = config
        self.cache_dir = config.raw.get("directories", {}).get("cache", "./cache")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self._connect()

    # --------------------------------------------------------
    # CONEXIÓN
    # --------------------------------------------------------

    def _connect(self) -> None:
        priority = self.config.exchange_priority or ["binance", "bybit"]
        for ex_id in priority:
            try:
                cls = getattr(ccxt, ex_id, None)
                if cls is None:
                    logger.warning(f"⚠️ Exchange {ex_id} no existe en CCXT")
                    continue
                ex = cls({
                    "enableRateLimit": True,
                    "options": {"defaultType": "spot"},
                    "timeout": self.config.exchanges.get("timeout_ms", 30000),
                })
                ex.load_markets()
                self.exchanges[ex_id] = ex
                logger.info(f"✅ Conectado a {ex_id}")
            except Exception as e:
                logger.warning(f"⚠️ {ex_id}: {e}")

        if not self.exchanges:
            logger.error("❌ Ningún exchange disponible")

    # --------------------------------------------------------
    # FETCH OHLCV
    # --------------------------------------------------------

    def fetch(self, symbol: str, exchange_id: str, timeframe: str,
              limit: int = 500) -> Optional[pd.DataFrame]:
        cache_file = os.path.join(
            self.cache_dir,
            f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{limit}.parquet"
        )

        # Caché
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and self._fresh(df):
                    return df
            except Exception:
                pass

        ex = self.exchanges.get(exchange_id)
        if ex is None:
            # Fallback a otro exchange disponible
            for alt_id, alt in self.exchanges.items():
                df = self._try_fetch(alt, alt_id, symbol, timeframe, limit)
                if df is not None:
                    return df
            return None

        df = self._try_fetch(ex, exchange_id, symbol, timeframe, limit)
        if df is None:
            # Fallback
            for alt_id, alt in self.exchanges.items():
                if alt_id == exchange_id:
                    continue
                df = self._try_fetch(alt, alt_id, symbol, timeframe, limit)
                if df is not None:
                    return df

        if df is not None:
            try:
                df.to_parquet(cache_file)
            except Exception:
                pass
            return df

        # Último recurso: caché obsoleta
        if os.path.exists(cache_file):
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        return None

    def _try_fetch(self, ex: ccxt.Exchange, ex_id: str, symbol: str,
                   timeframe: str, limit: int) -> Optional[pd.DataFrame]:
        retries = self.config.exchanges.get("max_retries", 3)
        for attempt in range(retries):
            try:
                raw = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not raw:
                    continue
                df = pd.DataFrame(
                    raw,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep="last")]
                df = df.dropna(subset=["open", "high", "low", "close"])
                if df.empty:
                    continue
                return df
            except Exception as e:
                logger.debug(f"Intento {attempt+1}/{retries} {symbol}@{ex_id}: {e}")
                if attempt < retries - 1:
                    time.sleep(1 + attempt)
        return None

    # --------------------------------------------------------
    # MULTI-TIMEFRAME
    # --------------------------------------------------------

    def fetch_multi_timeframe(self, symbol: str, exchange_id: str) -> Dict[str, Optional[pd.DataFrame]]:
        tf = self.config.timeframes
        limit = self.config.history_limit
        return {
            "entry": self.fetch(symbol, exchange_id, tf.get("default", "15m"), limit),
            "confirm": self.fetch(symbol, exchange_id, "15m", limit),
            "trend": self.fetch(symbol, exchange_id, "1h", limit),
        }

    # --------------------------------------------------------
    # HELPERS
    # --------------------------------------------------------

    def _fresh(self, df: pd.DataFrame) -> bool:
        try:
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            age = (pd.Timestamp.now(tz="UTC") - last).total_seconds()
            return age < CACHE_TTL
        except Exception:
            return False

    def get_symbols(self, exchange_id: str) -> List[str]:
        return list(self.config.symbols.get(exchange_id, []))

    @property
    def available_exchanges(self) -> List[str]:
        return list(self.exchanges.keys())
