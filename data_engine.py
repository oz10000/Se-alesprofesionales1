# data_engine.py
# Ruta: data_engine.py
# ============================================================
# D.A.P.S Ω — Motor de datos multi-exchange con fallbacks
#
# Características:
#   - Binance y Bybit como exchanges principales
#   - MEXC, Bitget, OKX, Kraken como fallback
#   - Top 40 activos por volumen en cada exchange principal
#   - Multi-timeframe: 5m, 15m, 1h, 4h
#   - Caché Parquet con TTL
#   - Validación de continuidad de velas
#   - Manejo de rate limits según documentación CCXT
# ============================================================

import os
import time
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import ccxt

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# CONSTANTES
# ------------------------------------------------------------
CACHE_TTL_SECONDS = 3600          # 1 hora
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 1.5
TOP_N_ASSETS = 40                 # Top 40 por volumen
SUPPORTED_TIMEFRAMES = ["5m", "15m", "1h", "4h"]
OHLCV_LIMIT = 500                 # Velas por timeframe


# ============================================================
# DATA ENGINE
# ============================================================

class DataEngine:
    """
    Motor de datos multi-exchange con fallbacks en cascada.

    Principales:
        - binance
        - bybit

    Fallback:
        - mexc
        - bitget
        - okx
        - kraken
    """

    MAIN_EXCHANGES = ["binance", "bybit"]
    FALLBACK_EXCHANGES = ["mexc", "bitget", "okx", "kraken"]
    ALL_EXCHANGES = MAIN_EXCHANGES + FALLBACK_EXCHANGES

    # --------------------------------------------------------
    # INICIALIZACIÓN
    # --------------------------------------------------------

    def __init__(self, config):
        self.config = config
        self.cache_dir = config.raw.get("directories", {}).get("cache", "./cache")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.exchanges: Dict[str, ccxt.Exchange] = {}
        self.available: List[str] = []
        self.symbols_cache: Dict[str, List[str]] = {}

        self._connect_all()

    # --------------------------------------------------------
    # CONEXIÓN A EXCHANGES
    # --------------------------------------------------------

    def _connect_all(self) -> None:
        """Conecta a todos los exchanges disponibles."""
        for ex_id in self.ALL_EXCHANGES:
            try:
                ex_class = getattr(ccxt, ex_id, None)
                if ex_class is None:
                    logger.warning(f"⚠️ {ex_id}: no existe en CCXT")
                    continue

                ex = ex_class({
                    "enableRateLimit": True,
                    "timeout": 30000,
                    "options": {"defaultType": "spot"},
                })
                ex.load_markets()
                self.exchanges[ex_id] = ex
                self.available.append(ex_id)
                n_symbols = len(ex.symbols) if hasattr(ex, "symbols") else 0
                logger.info(f"✅ {ex_id}: {n_symbols} símbolos cargados")

            except Exception as e:
                logger.warning(f"⚠️ {ex_id}: {e}")

        if not self.available:
            logger.error("❌ Ningún exchange disponible")

    # --------------------------------------------------------
    # TOP 40 ACTIVOS POR VOLUMEN
    # --------------------------------------------------------

    def fetch_top_symbols(self, exchange_id: str, top_n: int = TOP_N_ASSETS) -> List[str]:
        """
        Obtiene los top N símbolos por volumen (quoteVolume) en un exchange.

        Usa fetch_tickers si está disponible; si no, usa load_markets
        ordenado por volumen base.

        Args:
            exchange_id: ID del exchange (ej: 'binance').
            top_n: Número de activos a retornar.

        Returns:
            Lista de símbolos unificados (ej: ['BTC/USDT', ...]).
        """
        cache_key = f"{exchange_id}_top{top_n}"
        if cache_key in self.symbols_cache:
            return self.symbols_cache[cache_key]

        ex = self.exchanges.get(exchange_id)
        if ex is None:
            logger.warning(f"⚠️ {exchange_id} no disponible para top symbols")
            return []

        symbols: List[str] = []

        # ---- Intento 1: fetch_tickers ordenado por quoteVolume ----
        try:
            if ex.has.get("fetchTickers", False):
                tickers = ex.fetch_tickers()
                if tickers:
                    # Filtrar solo pares /USDT spot
                    filtered = {
                        k: v for k, v in tickers.items()
                        if k.endswith("/USDT")
                        and v.get("quoteVolume") is not None
                        and v.get("quoteVolume") > 0
                    }
                    # Ordenar por quoteVolume descendente
                    sorted_tickers = sorted(
                        filtered.items(),
                        key=lambda kv: float(kv[1].get("quoteVolume", 0)),
                        reverse=True,
                    )
                    symbols = [k for k, _ in sorted_tickers[:top_n]]
                    logger.info(
                        f"✅ {exchange_id}: {len(symbols)} top symbols por quoteVolume"
                    )
        except Exception as e:
            logger.warning(f"⚠️ {exchange_id} fetch_tickers falló: {e}")

        # ---- Intento 2: load_markets ordenado por volumen base ----
        if not symbols:
            try:
                markets = ex.markets
                usdt_markets = {
                    k: v for k, v in markets.items()
                    if k.endswith("/USDT")
                    and v.get("active", True)
                    and v.get("spot", True)
                }
                # Ordenar por volumen base si está disponible
                sorted_markets = sorted(
                    usdt_markets.items(),
                    key=lambda kv: float(kv[1].get("baseVolume", 0) or 0),
                    reverse=True,
                )
                symbols = [k for k, _ in sorted_markets[:top_n]]
                logger.info(
                    f"✅ {exchange_id}: {len(symbols)} top symbols por baseVolume"
                )
            except Exception as e:
                logger.warning(f"⚠️ {exchange_id} load_markets falló: {e}")

        # ---- Fallback: primeros N símbolos /USDT ----
        if not symbols:
            try:
                symbols = [
                    s for s in ex.symbols
                    if s.endswith("/USDT")
                ][:top_n]
                logger.info(
                    f"✅ {exchange_id}: {len(symbols)} top symbols por orden alfabético"
                )
            except Exception as e:
                logger.error(f"❌ {exchange_id} no se pudieron obtener símbolos: {e}")

        self.symbols_cache[cache_key] = symbols
        return symbols

    # --------------------------------------------------------
    # FETCH OHLCV CON FALLBACK
    # --------------------------------------------------------

    def fetch_ohlcv(
        self,
        symbol: str,
        exchange_id: str,
        timeframe: str = "15m",
        limit: int = OHLCV_LIMIT,
        use_cache: bool = True,
    ) -> Optional[pd.DataFrame]:
        """
        Obtiene velas OHLCV con fallback entre exchanges.

        Args:
            symbol: Símbolo unificado (ej: 'BTC/USDT').
            exchange_id: Exchange preferido.
            timeframe: Temporalidad ('5m', '15m', '1h', '4h').
            limit: Número de velas a obtener.
            use_cache: Usar caché si está disponible.

        Returns:
            DataFrame con columnas OHLCV, o None si falla.
        """
        # Validar timeframe
        if timeframe not in SUPPORTED_TIMEFRAMES:
            logger.warning(f"⚠️ Timeframe {timeframe} no soportado")
            return None

        # ---- Caché ----
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

        # ---- Intentar exchange preferido ----
        df = self._try_fetch_ohlcv(exchange_id, symbol, timeframe, limit)

        # ---- Fallback a otros exchanges ----
        if df is None:
            for fallback_id in self.available:
                if fallback_id == exchange_id:
                    continue
                df = self._try_fetch_ohlcv(fallback_id, symbol, timeframe, limit)
                if df is not None:
                    logger.info(
                        f"✅ {symbol} {timeframe} obtenido de {fallback_id} "
                        f"(fallback desde {exchange_id})"
                    )
                    break

        # ---- Guardar en caché ----
        if df is not None and use_cache:
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.debug(f"No se pudo guardar caché: {e}")

        # ---- Último recurso: caché obsoleta ----
        if df is None and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if not df.empty:
                    logger.warning(f"⚠️ Usando caché obsoleta para {symbol} {timeframe}")
            except Exception:
                pass

        return df

    # --------------------------------------------------------
    # FETCH INTERNO
    # --------------------------------------------------------

    def _try_fetch_ohlcv(
        self,
        exchange_id: str,
        symbol: str,
        timeframe: str,
        limit: int,
    ) -> Optional[pd.DataFrame]:
        """
        Intenta obtener OHLCV de un exchange específico con reintentos.

        Sigue la documentación de CCXT:
        - Usa `limit` para acotar las velas (max 1000-5000 según exchange).
        - Usa `enableRateLimit` para no exceder límites.
        - Valida continuidad de velas.
        """
        ex = self.exchanges.get(exchange_id)
        if ex is None:
            return None

        # Verificar que el símbolo exista en el exchange
        if hasattr(ex, "symbols") and symbol not in ex.symbols:
            logger.debug(f"⚠️ {symbol} no existe en {exchange_id}")
            return None

        # Verificar que soporte el timeframe
        if hasattr(ex, "timeframes") and timeframe not in ex.timeframes:
            logger.debug(f"⚠️ {timeframe} no soportado en {exchange_id}")
            return None

        for attempt in range(MAX_RETRIES):
            try:
                # ---- fetch_ohlcv según documentación CCXT ----
                # Parámetros: symbol, timeframe, since=None, limit=limit
                # limit: máximo de velas a obtener
                ohlcv = ex.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=None,          # desde la vela más reciente hacia atrás
                    limit=limit,         # máximo de velas
                )

                if not ohlcv or len(ohlcv) < 10:
                    logger.debug(
                        f"⚠️ {exchange_id} {symbol} {timeframe}: "
                        f"solo {len(ohlcv) if ohlcv else 0} velas"
                    )
                    continue

                # ---- Construir DataFrame ----
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                df = df[~df.index.duplicated(keep="last")]

                # ---- Validar ----
                if not self._validate_ohlcv(df):
                    continue

                # ---- Validar continuidad ----
                if not self._validate_continuity(df, timeframe):
                    logger.warning(
                        f"⚠️ {exchange_id} {symbol} {timeframe}: "
                        f"velas no continuas, saltando"
                    )
                    continue

                logger.debug(
                    f"✅ {exchange_id} {symbol} {timeframe}: {len(df)} velas"
                )
                return df

            except ccxt.RateLimitExceeded as e:
                wait = RETRY_DELAY_SECONDS * (attempt + 1) * 2
                logger.warning(
                    f"⚠️ {exchange_id} rate limit, esperando {wait}s: {e}"
                )
                time.sleep(wait)
            except ccxt.NetworkError as e:
                logger.warning(f"⚠️ {exchange_id} red: {e}")
                time.sleep(RETRY_DELAY_SECONDS)
            except ccxt.ExchangeError as e:
                logger.warning(f"⚠️ {exchange_id} exchange: {e}")
                break
            except Exception as e:
                logger.warning(f"⚠️ {exchange_id} {symbol}: {e}")
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
        """
        Obtiene múltiples timeframes para un símbolo.

        Timeframes: 5m, 15m, 1h, 4h.

        Returns:
            Dict con keys '5m', '15m', '1h', '4h'.
        """
        result: Dict[str, Optional[pd.DataFrame]] = {}
        for tf in SUPPORTED_TIMEFRAMES:
            try:
                result[tf] = self.fetch_ohlcv(
                    symbol=symbol,
                    exchange_id=exchange_id,
                    timeframe=tf,
                    limit=OHLCV_LIMIT,
                )
            except Exception as e:
                logger.warning(f"⚠️ {symbol} {tf}: {e}")
                result[tf] = None
        return result

    # --------------------------------------------------------
    # VALIDACIONES
    # --------------------------------------------------------

    def _validate_ohlcv(self, df: pd.DataFrame) -> bool:
        """Valida integridad básica de las velas."""
        if df is None or df.empty:
            return False
        if len(df) < 50:
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
        if (df["volume"] < 0).any():
            return False
        return True

    def _validate_continuity(self, df: pd.DataFrame, timeframe: str) -> bool:
        """
        Valida que las velas sean continuas (sin gaps).

        Sigue la documentación de CCXT: los gaps en OHLCV son comunes
        pero no deben superar un umbral.
        """
        if df is None or len(df) < 10:
            return False

        # Duración esperada en milisegundos
        tf_ms = self._timeframe_to_ms(timeframe)
        if tf_ms <= 0:
            return True  # No validar si no podemos calcular

        # Calcular diferencias entre timestamps
        diffs = df.index.to_series().diff().dropna()
        diffs_ms = diffs.dt.total_seconds() * 1000

        # Permitir gaps de hasta 2× la duración esperada
        max_allowed = tf_ms * 2

        # Contar gaps
        n_gaps = (diffs_ms > max_allowed).sum()
        gap_ratio = n_gaps / len(diffs_ms) if len(diffs_ms) > 0 else 0

        # Permitir hasta 10% de gaps (datos de exchange no siempre perfectos)
        return gap_ratio <= 0.10

    @staticmethod
    def _timeframe_to_ms(timeframe: str) -> float:
        """Convierte timeframe a milisegundos."""
        units = {"m": 60 * 1000, "h": 60 * 60 * 1000, "d": 24 * 60 * 60 * 1000}
        try:
            num = int(timeframe[:-1])
            unit = timeframe[-1]
            return num * units.get(unit, 0)
        except Exception:
            return 0.0

    # --------------------------------------------------------
    # CACHÉ
    # --------------------------------------------------------

    def _cache_fresh(self, df: pd.DataFrame) -> bool:
        """Verifica si la caché está fresca."""
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
        """Retorna lista de exchanges conectados."""
        return list(self.available)

    def get_symbols_for_exchange(self, exchange_id: str) -> List[str]:
        """Retorna los top 40 símbolos para un exchange."""
        return self.fetch_top_symbols(exchange_id)

    def get_all_symbols(self) -> Dict[str, List[str]]:
        """Retorna top 40 símbolos para cada exchange principal."""
        result: Dict[str, List[str]] = {}
        for ex_id in self.MAIN_EXCHANGES:
            if ex_id in self.exchanges:
                result[ex_id] = self.fetch_top_symbols(ex_id)
        return result

    def is_symbol_available(self, symbol: str, exchange_id: str) -> bool:
        """Verifica si un símbolo está disponible en un exchange."""
        ex = self.exchanges.get(exchange_id)
        if ex is None:
            return False
        return symbol in ex.symbols if hasattr(ex, "symbols") else False

    @property
    def status(self) -> Dict:
        """Retorna estado del motor de datos."""
        return {
            "available": self.available,
            "main_exchanges": [e for e in self.MAIN_EXCHANGES if e in self.available],
            "fallback_exchanges": [e for e in self.FALLBACK_EXCHANGES if e in self.available],
            "n_exchanges": len(self.available),
            "supported_timeframes": SUPPORTED_TIMEFRAMES,
            "top_n_assets": TOP_N_ASSETS,
        }
