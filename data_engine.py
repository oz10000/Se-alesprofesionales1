# data_engine.py
# Ruta: data_engine.py
# ============================================================
# D.A.P.S Ω — Motor de datos multi-exchange con detección de bloqueo
#
# Características:
#   - Detecta exchanges bloqueados en cloud automáticamente
#   - Prioriza exchanges funcionales (OKX, Kraken, MEXC, Bitget)
#   - Filtra stablecoins y tokens sintéticos
#   - Top 40 activos reales por volumen
#   - Multi-timeframe: 5m, 15m, 1h, 4h
# ============================================================

import os
import time
import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------
CACHE_TTL_SECONDS = 3600
MAX_RETRIES = 2
RETRY_DELAY_SECONDS = 1.0
TOP_N_ASSETS = 40
SUPPORTED_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
OHLCV_LIMIT = 500

# ------------------------------------------------------------
# BLACKLIST — Excluir stablecoins y tokens sintéticos
# ------------------------------------------------------------
STABLECOINS = {
    "USDC", "USDT", "BUSD", "DAI", "TUSD", "USDD", "FRAX",
    "USDP", "UST", "USTC", "GUSD", "USDE", "PYUSD", "FDUSD",
    "USDG", "USDE", "SUSD", "MIM", "LUSD", "ALUSD",
}

SYNTHETICS = {
    "XAUT", "PAXG",   # tokenized gold
    "WBTC", "WETH", "WSTETH", "STETH", "CBETH", "RETH",  # wrapped
}

# Tokens sintéticos tipo Binance X-stocks (X-prefix + mayúsculas)
# XAUT, XSOXL, XSNDK, XMSTR, XCRCL, etc.
def _is_synthetic_prefix(base: str) -> bool:
    if len(base) < 3:
        return False
    # X seguido de 2+ letras mayúsculas = token sintético
    return base.startswith("X") and base[1:].isalpha() and base[1:].isupper()


def _is_valid_base(base: str) -> bool:
    """Determina si una base es un activo cripto real."""
    if not base or not base.isalpha():
        return False
    if base in STABLECOINS:
        return False
    if base in SYNTHETICS:
        return False
    if _is_synthetic_prefix(base):
        return False
    return True


# ============================================================
# DATA ENGINE
# ============================================================

class DataEngine:
    """
    Motor de datos multi-exchange con detección de bloqueo.

    - Prueba cada exchange al conectar
    - Marca como "blocked" los que fallan
    - Reordena prioridad: exchanges funcionales primero
    - Filtra stablecoins y sintéticos
    """

    MAIN_EXCHANGES = ["binance", "bybit"]
    FALLBACK_EXCHANGES = ["okx", "kraken", "mexc", "bitget"]
    ALL_EXCHANGES = MAIN_EXCHANGES + FALLBACK_EXCHANGES

    # --------------------------------------------------------
    # INICIALIZACIÓN
    # --------------------------------------------------------

    def __init__(self, config):
        self.config = config
        self.cache_dir = config.raw.get("directories", {}).get("cache", "./cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.available: List[str] = []         # Conectados OK
        self.blocked: List[str] = []           # Bloqueados/failed
        self.symbols_cache: Dict[str, List[str]] = {}

        self._connect_all()

    # --------------------------------------------------------
    # CONEXIÓN + TEST
    # --------------------------------------------------------

    def _connect_all(self) -> None:
        """Conecta y prueba cada exchange."""
        for ex_id in self.ALL_EXCHANGES:
            try:
                ex_class = getattr(ccxt, ex_id, None)
                if ex_class is None:
                    self.blocked.append(ex_id)
                    continue

                ex = ex_class({
                    "enableRateLimit": True,
                    "timeout": 15000,
                    "options": {"defaultType": "spot"},
                })
                ex.load_markets()

                # Test funcional: fetch de 2 velas de BTC
                if self._test_exchange(ex):
                    self.exchanges[ex_id] = ex
                    self.available.append(ex_id)
                    n_sym = len(ex.symbols) if hasattr(ex, "symbols") else 0
                    logger.info(f"✅ {ex_id}: OK · {n_sym} símbolos")
                else:
                    self.blocked.append(ex_id)
                    logger.warning(f"🚫 {ex_id}: bloqueado o inaccesible desde esta IP")

            except Exception as e:
                self.blocked.append(ex_id)
                logger.warning(f"🚫 {ex_id}: {str(e)[:80]}")

        if not self.available:
            logger.error("❌ CRÍTICO: Ningún exchange disponible")
        else:
            logger.info(
                f"✅ {len(self.available)} exchanges operativos · "
                f"{len(self.blocked)} bloqueados"
            )

    def _test_exchange(self, ex: ccxt.Exchange) -> bool:
        """Prueba si el exchange responde desde esta IP."""
        try:
            ohlcv = ex.fetch_ohlcv("BTC/USDT", "1h", limit=2)
            return bool(ohlcv) and len(ohlcv) >= 1
        except ccxt.ExchangeError as e:
            err = str(e).lower()
            # Detectar bloqueos típicos (451, geo-restricted, etc.)
            if any(k in err for k in ["451", "403", "restricted", "blocked", "forbidden"]):
                return False
            return False
        except Exception:
            return False

    # --------------------------------------------------------
    # TOP 40 SÍMBOLOS POR VOLUMEN (FILTRADOS)
    # --------------------------------------------------------

    def fetch_top_symbols(self, exchange_id: str, top_n: int = TOP_N_ASSETS) -> List[str]:
        """Obtiene top N símbolos reales por volumen."""
        cache_key = f"{exchange_id}_top{top_n}"
        if cache_key in self.symbols_cache:
            return self.symbols_cache[cache_key]

        ex = self.exchanges.get(exchange_id)
        if ex is None:
            logger.warning(f"⚠️ {exchange_id} no disponible")
            return []

        candidates: List[str] = []

        # ---- Intento 1: fetch_tickers por quoteVolume ----
        try:
            if ex.has.get("fetchTickers", False):
                tickers = ex.fetch_tickers()
                filtered = {}
                for sym, t in tickers.items():
                    if not sym.endswith("/USDT"):
                        continue
                    base = sym.split("/")[0]
                    if not _is_valid_base(base):
                        continue
                    qv = t.get("quoteVolume")
                    if qv is None or qv <= 0:
                        continue
                    filtered[sym] = float(qv)

                candidates = sorted(filtered, key=filtered.get, reverse=True)[:top_n]
                logger.info(f"✅ {exchange_id}: {len(candidates)} símbolos filtrados por volumen")
        except Exception as e:
            logger.warning(f"⚠️ {exchange_id} fetch_tickers: {e}")

        # ---- Intento 2: markets ordenados ----
        if not candidates:
            try:
                markets = ex.markets
                filtered = []
                for sym, m in markets.items():
                    if not sym.endswith("/USDT"):
                        continue
                    base = sym.split("/")[0]
                    if not _is_valid_base(base):
                        continue
                    if not m.get("active", True):
                        continue
                    filtered.append(sym)
                candidates = filtered[:top_n]
                logger.info(f"✅ {exchange_id}: {len(candidates)} símbolos filtrados (fallback)")
            except Exception as e:
                logger.error(f"❌ {exchange_id}: {e}")

        # ---- Último fallback: lista estática filtrada ----
        if not candidates:
            try:
                for sym in ex.symbols:
                    if not sym.endswith("/USDT"):
                        continue
                    base = sym.split("/")[0]
                    if _is_valid_base(base):
                        candidates.append(sym)
                        if len(candidates) >= top_n:
                            break
                logger.info(f"✅ {exchange_id}: {len(candidates)} símbolos (orden alfabético)")
            except Exception:
                pass

        self.symbols_cache[cache_key] = candidates
        return candidates

    # --------------------------------------------------------
    # FETCH OHLCV CON FALLBACK INTELIGENTE
    # --------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        exchange_id: str,
        timeframe: str = "15m",
        limit: int = OHLCV_LIMIT,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """Obtiene OHLCV con fallback entre exchanges operativos."""
        if timeframe not in SUPPORTED_TIMEFRAMES:
            return None

        # ---- Caché fresca ----
        cache_file = os.path.join(
            self.cache_dir,
            f"{exchange_id}_{symbol.replace('/', '_')}_{timeframe}_{limit}.parquet",
        )
        if use_cache and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty and self._cache_fresh(df):
                    return df
            except Exception:
                pass

        # ---- Orden de intentos: preferido primero, luego el resto ----
        try_order = [exchange_id] + [e for e in self.available if e != exchange_id]

        for ex_id in try_order:
            ex = self.exchanges.get(ex_id)
            if ex is None:
                continue

            df = self._try_fetch_ohlcv(ex, ex_id, symbol, timeframe, limit)
            if df is not None:
                if use_cache:
                    try:
                        df.to_parquet(cache_file)
                    except Exception:
                        pass
                return df

        # ---- Caché obsoleta ----
        if os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.warning(f"⚠️ Caché obsoleta para {symbol} {timeframe}")
                    return df
            except Exception:
                pass

        return None

    def _try_fetch_ohlcv(
        self,
        ex: ccxt.Exchange,
        ex_id: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Optional[pd.DataFrame]:
        """Intenta fetch en un exchange con reintentos."""
        # Verificar que el símbolo exista
        if hasattr(ex, "symbols") and symbol not in ex.symbols:
            return None

        for attempt in range(MAX_RETRIES):
            try:
                ohlcv = ex.fetch_ohlcv(symbol, timeframe, limit=limit)
                if not ohlcv or len(ohlcv) < 10:
                    return None

                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep="last")]

                if not self._validate(df):
                    return None

                return df

            except ccxt.RateLimitExceeded:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
            except ccxt.NetworkError:
                time.sleep(RETRY_DELAY_SECONDS)
            except ccxt.ExchangeError:
                return None
            except Exception:
                time.sleep(RETRY_DELAY_SECONDS)

        return None

    # --------------------------------------------------------
    # MULTI-TIMEFRAME
    # --------------------------------------------------------

    def fetch_multi_timeframe(
        self,
        symbol: str,
        exchange_id: str,
    ) -> Dict[str, Optional[pd.DataFrame]]:
        """Obtiene 5m, 15m, 1h, 4h para un símbolo."""
        return {
            tf: self.fetch(symbol, exchange_id, tf, OHLCV_LIMIT)
            for tf in SUPPORTED_TIMEFRAMES
        }

    # --------------------------------------------------------
    # VALIDACIÓN
    # --------------------------------------------------------

    def _validate(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty or len(df) < 20:
            return False
        required = ["open", "high", "low", "close", "volume"]
        if not all(c in df.columns for c in required):
            return False
        if df[required].isna().any().any():
            return False
        if (df["high"] < df["low"]).any():
            return False
        if (df[["open", "high", "low", "close"]] <= 0).any().any():
            return False
        return True

    def _cache_fresh(self, df: pd.DataFrame) -> bool:
        try:
            last = df.index[-1]
            if last.tzinfo is None:
                last = last.tz_localize("UTC")
            age = (pd.Timestamp.now(tz="UTC") - last).total_seconds()
            return age < CACHE_TTL_SECONDS
        except Exception:
            return False

    # --------------------------------------------------------
    # UTILIDADES PÚBLICAS
    # --------------------------------------------------------

    def get_available_exchanges(self) -> List[str]:
        return list(self.available)

    def get_blocked_exchanges(self) -> List[str]:
        return list(self.blocked)

    def get_symbols_for_exchange(self, exchange_id: str) -> List[str]:
        return self.fetch_top_symbols(exchange_id)

    def is_symbol_available(self, symbol: str, exchange_id: str) -> bool:
        ex = self.exchanges.get(exchange_id)
        if ex is None:
            return False
        return symbol in ex.symbols if hasattr(ex, "symbols") else False

    @property
    def status(self) -> Dict:
        return {
            "available": self.available,
            "blocked": self.blocked,
            "n_available": len(self.available),
            "n_blocked": len(self.blocked),
            "timeframes": SUPPORTED_TIMEFRAMES,
            "top_n": TOP_N_ASSETS,
        }
