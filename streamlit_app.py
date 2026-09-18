# streamlit_app.py
# Ruta: streamlit_app.py
# ============================================================
# D.A.P.S Ω — Scanner Conservador Multi-Exchange
#
# Características:
#   - Detección automática de exchanges bloqueados
#   - Solo muestra exchanges operativos (OKX/Kraken/MEXC/Bitget en cloud)
#   - Filtrado de stablecoins y tokens sintéticos
#   - Análisis multi-timeframe (5m / 15m / 1h / 4h)
#   - Rankings LONG y SHORT independientes
#   - Predicción operativa completa
#   - Temporizador hasta próximo trade aprobado
# ============================================================

import logging
import traceback
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import streamlit as st

from config import ScannerConfig
from data_engine import DataEngine
from scoring import CompositeScorer
from risk_manager import RiskManager
from ranking import (
    rank_all, rank_long, rank_short,
    best_long, best_short, summary,
)
from utils import fmt_pct, fmt_price, humanize_minutes, now_utc


# ------------------------------------------------------------
# LOGGING
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="D.A.P.S Ω — Scanner",
    page_icon="Ω",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------
# CSS
# ------------------------------------------------------------
st.markdown(
    """
    <style>
        .stApp { background-color: white; color: black; }
        h1, h2, h3, h4, h5, h6 { color: black; }
        .stMetric {
            background-color: #f7f7f7;
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #eee;
        }
        .stDataFrame { background-color: white; color: black; }
        .stExpander { background-color: #fafafa; border: 1px solid #e0e0e0; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] {
            background-color: #f0f0f0;
            border-radius: 6px;
            padding: 8px 16px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# CARGA DE CONFIGURACIÓN
# ------------------------------------------------------------
try:
    config = ScannerConfig.from_yaml("config.yaml")
except Exception as e:
    st.error(f"❌ Error cargando config.yaml: {e}")
    st.code(traceback.format_exc())
    st.stop()


# ------------------------------------------------------------
# INICIALIZACIÓN DE RECURSOS
# ------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def init_engine():
    return DataEngine(config)


@st.cache_resource(show_spinner=False)
def init_scorer():
    return CompositeScorer(config)


@st.cache_resource(show_spinner=False)
def init_risk():
    return RiskManager(config)


try:
    engine = init_engine()
    scorer = init_scorer()
    risk = init_risk()
except Exception as e:
    st.error(f"❌ Error inicializando componentes: {e}")
    st.code(traceback.format_exc())
    st.stop()


# ------------------------------------------------------------
# VERIFICAR EXCHANGES DISPONIBLES
# ------------------------------------------------------------
available_exchanges = engine.get_available_exchanges()
blocked_exchanges = engine.get_blocked_exchanges()

if not available_exchanges:
    st.error(
        "❌ **Ningún exchange disponible.** "
        "Todos están bloqueados desde esta IP.\n\n"
        "**Solución:** Ejecutar el proyecto localmente con `streamlit run streamlit_app.py`."
    )
    if blocked_exchanges:
        st.warning(f"Exchanges bloqueados: {', '.join(blocked_exchanges)}")
    st.stop()


# ------------------------------------------------------------
# ESTADO DE SESIÓN
# ------------------------------------------------------------
_defaults = {
    "scan_results": [],
    "last_scan": None,
    "next_scan": None,
    "current_exchange": available_exchanges[0],
    "current_timeframe": config.default_timeframe,
    "next_trade_eta": None,
    "next_trade_symbol": None,
    "next_trade_confidence": 0.0,
    "scan_duration": 0.0,
    "n_scanned": 0,
    "n_errors": 0,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------
st.title("Ω D.A.P.S — Scanner Conservador")
st.caption(
    f"Versión {config.project.get('version', '1.1.0')} · "
    f"{len(available_exchanges)} exchanges activos · "
    f"Multi-Timeframe (5m / 15m / 1h / 4h)"
)

if blocked_exchanges:
    st.info(
        f"ℹ️ **{len(blocked_exchanges)} exchanges bloqueados** desde esta IP: "
        f"{', '.join(blocked_exchanges)}. "
        f"El sistema usa automáticamente los exchanges operativos."
    )


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    # ---- Exchange (solo disponibles) ----
    exchange = st.selectbox(
        "Exchange",
        options=available_exchanges,
        index=0,
        help="Solo se muestran exchanges accesibles desde esta IP.",
    )
    st.session_state.current_exchange = exchange

    # ---- Info de exchanges bloqueados ----
    if blocked_exchanges:
        with st.expander(f"⚠️ {len(blocked_exchanges)} bloqueados"):
            for ex_id in blocked_exchanges:
                st.caption(f"🚫 {ex_id} (no accesible)")

    # ---- Timeframe ----
    timeframe = st.selectbox(
        "Timeframe de entrada",
        options=config.supported_timeframes,
        index=(
            config.supported_timeframes.index(config.default_timeframe)
            if config.default_timeframe in config.supported_timeframes
            else 0
        ),
        help="El TF seleccionado se usa como entrada. Los otros 3 confirman.",
    )
    st.session_state.current_timeframe = timeframe

    st.markdown("---")

    # ---- Botón escanear ----
    scan_btn = st.button(
        "🔄 ESCANEAR",
        type="primary",
        use_container_width=True,
    )

    st.markdown("---")
    st.header("📊 Estado")

    # ---- Símbolos ----
    try:
        symbols = engine.get_symbols_for_exchange(exchange)
        n_symbols = len(symbols)
    except Exception:
        n_symbols = 0

    st.caption(f"Exchange: **{exchange}**")
    st.caption(f"Timeframe: **{timeframe}**")
    st.caption(f"Activos disponibles: **{n_symbols}**")
    st.caption(f"Exchanges OK: **{len(available_exchanges)}**")

    if st.session_state.last_scan:
        st.caption(
            f"Último escaneo: "
            f"**{st.session_state.last_scan.strftime('%H:%M:%S UTC')}**"
        )
    else:
        st.caption("Último escaneo: **nunca**")

    if st.session_state.scan_duration > 0:
        st.caption(f"Duración: **{st.session_state.scan_duration:.1f}s**")

    if st.session_state.next_scan:
        remaining = (st.session_state.next_scan - now_utc()).total_seconds()
        if remaining > 0:
            m, s = divmod(int(remaining), 60)
            st.caption(f"Próximo escaneo: **{m:02d}:{s:02d}**")
        else:
            st.caption("Próximo escaneo: **disponible**")

    st.markdown("---")
    with st.expander("🌐 Exchanges conectados"):
        for ex_id in available_exchanges:
            is_main = ex_id in engine.MAIN_EXCHANGES
            prefix = "🟢" if is_main else "🔵"
            st.caption(f"{prefix} {ex_id}")


# ------------------------------------------------------------
# FUNCIÓN DE ESCANEO
# ------------------------------------------------------------
def do_scan(exchange_id: str, entry_tf: str) -> tuple:
    """
    Ejecuta el escaneo completo.

    Returns:
        (results, n_scanned, n_errors, duration_seconds)
    """
    t_start = now_utc()

    symbols = engine.get_symbols_for_exchange(exchange_id)
    total = len(symbols)

    if total == 0:
        st.warning(f"⚠️ No hay símbolos disponibles en {exchange_id}")
        return [], 0, 0, 0.0

    results = []
    n_errors = 0

    progress = st.progress(0.0)
    status = st.empty()
    metrics_row = st.empty()

    for i, sym in enumerate(symbols):
        status.text(f"Escaneando {sym} ({i + 1}/{total})...")

        try:
            # ---- Multi-timeframe: 5m, 15m, 1h, 4h ----
            data = engine.fetch_multi_timeframe(sym, exchange_id)

            entry_df = data.get(entry_tf)
            if entry_df is None or entry_df.empty:
                n_errors += 1
                progress.progress((i + 1) / total)
                continue

            # ---- Scoring ----
            res = scorer.compute(sym, data)

            # ---- Precio actual ----
            res.entry_price = float(entry_df["close"].iloc[-1])

            # ---- Enriquecer con risk manager ----
            res = risk.enrich(res)
            res.timestamp = now_utc().isoformat()

            results.append(res)

        except Exception as e:
            logger.warning(f"Error escaneando {sym}: {e}")
            n_errors += 1

        progress.progress((i + 1) / total)

        # Métricas en vivo
        if i % 5 == 0 or i == total - 1:
            valid_now = sum(1 for r in results if r.is_valid)
            metrics_row.caption(
                f"✅ Escaneados: {len(results)} · "
                f"Válidas: {valid_now} · "
                f"Errores: {n_errors}"
            )

    progress.empty()
    status.empty()
    metrics_row.empty()

    duration = (now_utc() - t_start).total_seconds()
    return results, len(results), n_errors, duration


# ------------------------------------------------------------
# TRIGGER DE ESCANEO
# ------------------------------------------------------------
if scan_btn:
    exchange_id = st.session_state.current_exchange
    entry_tf = st.session_state.current_timeframe

    with st.spinner(f"Escaneando {exchange_id} (timeframe {entry_tf})..."):
        try:
            results, n_scanned, n_errors, duration = do_scan(exchange_id, entry_tf)

            st.session_state.scan_results = results
            st.session_state.last_scan = now_utc()
            st.session_state.next_scan = now_utc() + timedelta(minutes=15)
            st.session_state.scan_duration = duration
            st.session_state.n_scanned = n_scanned
            st.session_state.n_errors = n_errors

            # ---- Calcular ETA próximo trade ----
            valid = [r for r in results if r.is_valid]
            if valid:
                best = max(valid, key=lambda x: x.total_score)
                st.session_state.next_trade_eta = best.next_entry_eta_min
                st.session_state.next_trade_symbol = best.symbol
                st.session_state.next_trade_confidence = best.confidence
            else:
                all_etas = [
                    r.next_entry_eta_min
                    for r in results
                    if r.next_entry_eta_min > 0
                ]
                st.session_state.next_trade_eta = min(all_etas) if all_etas else None
                st.session_state.next_trade_symbol = None
                st.session_state.next_trade_confidence = 0.0

        except Exception as e:
            st.error(f"❌ Error durante el escaneo: {e}")
            st.code(traceback.format_exc())

    st.rerun()


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_dashboard, tab_long, tab_short, tab_all = st.tabs(
    [
        "📡 Dashboard",
        "🟢 Ranking LONG",
        "🔴 Ranking SHORT",
        "📊 Todas las señales",
    ]
)


# ------------------------------------------------------------
# HELPERS DE TABLAS
# ------------------------------------------------------------
def _ranking_rows(rlist):
    rows = []
    for i, r in enumerate(rlist, 1):
        rows.append({
            "#": i,
            "Símbolo": r.symbol,
            "Tier": r.tier,
            "Score": round(r.total_score, 1),
            "Confianza": f"{r.confidence:.0f}%",
            "Dir": r.direction,
            "P(L)": f"{r.prob_long * 100:.0f}%",
            "P(S)": f"{r.prob_short * 100:.0f}%",
            "Régimen": r.regime,
            "Tend": round(r.breakdown.trend * 100, 0),
            "Mom": round(r.breakdown.momentum * 100, 0),
            "Vol": round(r.breakdown.volume * 100, 0),
            "ADX": round(r.adx, 1),
            "ATR%": f"{r.atr_pct * 100:.2f}%",
            "RSI": round(r.rsi, 1),
            "RVOL": round(r.rvol, 2),
            "TFI": round(r.tfi, 3),
            "OFI": round(r.ofi, 3),
            "Spread": f"{r.spread_est * 100:.3f}%",
            "Coste": f"{r.cost_est * 100:.3f}%",
            "Riesgo": r.risk_level,
            "Reward": r.reward_level,
            "Válida": "✅" if r.is_valid else "❌",
            "Razón": r.reason,
        })
    return rows


def _operativo_rows(rlist):
    rows = []
    for i, r in enumerate(rlist, 1):
        rows.append({
            "#": i,
            "Símbolo": r.symbol,
            "Tier": r.tier,
            "Score": round(r.total_score, 1),
            "Confianza": f"{r.confidence:.0f}%",
            "Dir": r.direction,
            "Entrada": fmt_price(r.entry_price),
            "Zona baja": fmt_price(r.entry_zone_low),
            "Zona alta": fmt_price(r.entry_zone_high),
            "Tipo": r.entry_type,
            "SL $": fmt_price(r.sl_price),
            "SL %": f"{r.sl_pct:.2f}%",
            "TP $": fmt_price(r.tp_price),
            "TP %": f"{r.tp_pct:.2f}%",
            "R:R": f"{r.rr_ratio:.2f}",
            "Trailing %": f"{r.trailing_distance_pct:.2f}%",
            "BE %": f"{r.breakeven_trigger_pct:.2f}%",
            "Dur. (min)": round(r.expected_duration_min, 1),
            "T a TP (min)": round(r.time_to_tp_min, 1),
            "T a SL (min)": round(r.time_to_sl_min, 1),
            "ETA (min)": round(r.next_entry_eta_min, 1),
        })
    return rows


# ------------------------------------------------------------
# TAB 1 — DASHBOARD
# ------------------------------------------------------------
with tab_dashboard:
    st.subheader("📡 Estado general")

    results = st.session_state.scan_results

    if not results:
        st.info("Presioná **ESCANEAR** en la barra lateral para comenzar.")
    else:
        # ---- Resumen ----
        s = summary(results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Activos escaneados", s["total"])
        c2.metric("Señales válidas", s["valid"])
        c3.metric("LONG válidas", s["long_valid"])
        c4.metric("SHORT válidas", s["short_valid"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Score promedio", f"{s['avg_score']:.1f}")
        c6.metric("Score máximo", f"{s['max_score']:.1f}")
        c7.metric("Neutrales", s["neutral"])
        c8.metric("Exchange", st.session_state.current_exchange or "N/A")

        st.markdown("---")

        # ---- Temporizador ----
        eta = st.session_state.next_trade_eta
        symbol_eta = st.session_state.next_trade_symbol
        conf_eta = st.session_state.next_trade_confidence

        if eta is not None and eta > 0:
            st.info(
                f"⏱️ **Próximo trade aprobado:** "
                f"{symbol_eta or 'N/A'} · "
                f"ETA: **{humanize_minutes(eta)}** · "
                f"Confianza: {conf_eta:.0f}%"
            )
        else:
            st.warning(
                "⏳ **Sin señales aprobadas en este momento.** "
                "Esperá el próximo escaneo o cambiá de exchange/timeframe."
            )

        st.markdown("---")

        # ---- Mejor LONG ----
        st.markdown("### 🌟 Mejor LONG")
        bl = best_long(results)
        if bl:
            st.success(
                f"**{bl.symbol}** · Tier {bl.tier} · "
                f"Score {bl.total_score:.1f} · "
                f"Confianza {bl.confidence:.0f}% · "
                f"Régimen {bl.regime}"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada", fmt_price(bl.entry_price))
            col2.metric("SL", f"{fmt_price(bl.sl_price)} ({bl.sl_pct:.2f}%)")
            col3.metric("TP", f"{fmt_price(bl.tp_price)} ({bl.tp_pct:.2f}%)")
            col1.metric("R:R", f"{bl.rr_ratio:.2f}")
            col2.metric("Duración esperada", humanize_minutes(bl.expected_duration_min))
            col3.metric("Tiempo a TP", humanize_minutes(bl.time_to_tp_min))
        else:
            st.info("Sin señales LONG aprobadas.")

        st.markdown("---")

        # ---- Mejor SHORT ----
        st.markdown("### 🌟 Mejor SHORT")
        bs = best_short(results)
        if bs:
            st.error(
                f"**{bs.symbol}** · Tier {bs.tier} · "
                f"Score {bs.total_score:.1f} · "
                f"Confianza {bs.confidence:.0f}% · "
                f"Régimen {bs.regime}"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada", fmt_price(bs.entry_price))
            col2.metric("SL", f"{fmt_price(bs.sl_price)} ({bs.sl_pct:.2f}%)")
            col3.metric("TP", f"{fmt_price(bs.tp_price)} ({bs.tp_pct:.2f}%)")
            col1.metric("R:R", f"{bs.rr_ratio:.2f}")
            col2.metric("Duración esperada", humanize_minutes(bs.expected_duration_min))
            col3.metric("Tiempo a TP", humanize_minutes(bs.time_to_tp_min))
        else:
            st.info("Sin señales SHORT aprobadas.")

        st.markdown("---")

        # ---- Estado del escaneo ----
        col1, col2, col3 = st.columns(3)
        col1.metric("Símbolos escaneados", st.session_state.n_scanned)
        col2.metric("Errores", st.session_state.n_errors)
        col3.metric("Duración", f"{st.session_state.scan_duration:.1f}s")


# ------------------------------------------------------------
# TAB 2 — RANKING LONG
# ------------------------------------------------------------
with tab_long:
    st.subheader("🟢 Ranking LONG")

    results = st.session_state.scan_results
    longs = rank_long(results)

    if not longs:
        st.info("Sin señales LONG válidas en este momento.")
    else:
        st.markdown("#### 📋 Tabla operativa")
        st.dataframe(
            pd.DataFrame(_operativo_rows(longs)),
            use_container_width=True,
            height=420,
        )

        st.markdown("#### 🎯 Detalle de scores")
        st.dataframe(
            pd.DataFrame(_ranking_rows(longs)),
            use_container_width=True,
            height=320,
        )

        # ---- Detalle por activo ----
        st.markdown("---")
        st.markdown("### 🔍 Detalle por activo")

        for r in longs[:10]:
            with st.expander(
                f"{r.symbol} — Tier {r.tier} — Score {r.total_score:.1f} — "
                f"Confianza {r.confidence:.0f}%"
            ):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Score total", f"{r.total_score:.1f}")
                    st.metric("Tendencia", f"{r.breakdown.trend * 100:.0f}")
                    st.metric("Momentum", f"{r.breakdown.momentum * 100:.0f}")
                    st.metric("Volumen", f"{r.breakdown.volume * 100:.0f}")

                with c2:
                    st.metric("ADX", f"{r.adx:.1f}")
                    st.metric("ATR%", f"{r.atr_pct * 100:.2f}%")
                    st.metric("RSI", f"{r.rsi:.1f}")
                    st.metric("RVOL", f"{r.rvol:.2f}")

                with c3:
                    st.metric("TFI", f"{r.tfi:+.3f}")
                    st.metric("OFI", f"{r.ofi:+.3f}")
                    st.metric("Spread", f"{r.spread_est * 100:.3f}%")
                    st.metric("Coste", f"{r.cost_est * 100:.3f}%")

                st.markdown("---")

                c4, c5, c6 = st.columns(3)
                with c4:
                    st.metric("Entrada", fmt_price(r.entry_price))
                    st.metric("Zona baja", fmt_price(r.entry_zone_low))
                    st.metric("Zona alta", fmt_price(r.entry_zone_high))
                    st.metric("Tipo entrada", r.entry_type)

                with c5:
                    st.metric("SL $", fmt_price(r.sl_price))
                    st.metric("SL %", f"{r.sl_pct:.2f}%")
                    st.metric("SL método", r.sl_method)
                    st.metric("R:R", f"{r.rr_ratio:.2f}")

                with c6:
                    st.metric("TP $", fmt_price(r.tp_price))
                    st.metric("TP %", f"{r.tp_pct:.2f}%")
                    st.metric("TP método", r.tp_method)
                    st.metric("Trailing %", f"{r.trailing_distance_pct:.2f}%")

                st.markdown("---")

                c7, c8, c9 = st.columns(3)
                with c7:
                    st.metric("Duración esperada", humanize_minutes(r.expected_duration_min))
                    st.metric("Confianza duración", f"{r.expected_duration_conf:.0f}%")

                with c8:
                    st.metric("Tiempo a TP", humanize_minutes(r.time_to_tp_min))
                    st.metric("Tiempo a SL", humanize_minutes(r.time_to_sl_min))

                with c9:
                    st.metric("ETA entrada", humanize_minutes(r.next_entry_eta_min))
                    st.metric("Riesgo", r.risk_level)
                    st.metric("Reward", r.reward_level)

                if r.conditions:
                    st.markdown("**Condiciones favorables:**")
                    for c in r.conditions:
                        st.caption(f"✓ {c}")


# ------------------------------------------------------------
# TAB 3 — RANKING SHORT
# ------------------------------------------------------------
with tab_short:
    st.subheader("🔴 Ranking SHORT")

    results = st.session_state.scan_results
    shorts = rank_short(results)

    if not shorts:
        st.info("Sin señales SHORT válidas en este momento.")
    else:
        st.markdown("#### 📋 Tabla operativa")
        st.dataframe(
            pd.DataFrame(_operativo_rows(shorts)),
            use_container_width=True,
            height=420,
        )

        st.markdown("#### 🎯 Detalle de scores")
        st.dataframe(
            pd.DataFrame(_ranking_rows(shorts)),
            use_container_width=True,
            height=320,
        )

        # ---- Detalle por activo ----
        st.markdown("---")
        st.markdown("### 🔍 Detalle por activo")

        for r in shorts[:10]:
            with st.expander(
                f"{r.symbol} — Tier {r.tier} — Score {r.total_score:.1f} — "
                f"Confianza {r.confidence:.0f}%"
            ):
                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric("Score total", f"{r.total_score:.1f}")
                    st.metric("Tendencia", f"{r.breakdown.trend * 100:.0f}")
                    st.metric("Momentum", f"{r.breakdown.momentum * 100:.0f}")
                    st.metric("Volumen", f"{r.breakdown.volume * 100:.0f}")

                with c2:
                    st.metric("ADX", f"{r.adx:.1f}")
                    st.metric("ATR%", f"{r.atr_pct * 100:.2f}%")
                    st.metric("RSI", f"{r.rsi:.1f}")
                    st.metric("RVOL", f"{r.rvol:.2f}")

                with c3:
                    st.metric("TFI", f"{r.tfi:+.3f}")
                    st.metric("OFI", f"{r.ofi:+.3f}")
                    st.metric("Spread", f"{r.spread_est * 100:.3f}%")
                    st.metric("Coste", f"{r.cost_est * 100:.3f}%")

                st.markdown("---")

                c4, c5, c6 = st.columns(3)
                with c4:
                    st.metric("Entrada", fmt_price(r.entry_price))
                    st.metric("Zona baja", fmt_price(r.entry_zone_low))
                    st.metric("Zona alta", fmt_price(r.entry_zone_high))
                    st.metric("Tipo entrada", r.entry_type)

                with c5:
                    st.metric("SL $", fmt_price(r.sl_price))
                    st.metric("SL %", f"{r.sl_pct:.2f}%")
                    st.metric("SL método", r.sl_method)
                    st.metric("R:R", f"{r.rr_ratio:.2f}")

                with c6:
                    st.metric("TP $", fmt_price(r.tp_price))
                    st.metric("TP %", f"{r.tp_pct:.2f}%")
                    st.metric("TP método", r.tp_method)
                    st.metric("Trailing %", f"{r.trailing_distance_pct:.2f}%")

                st.markdown("---")

                c7, c8, c9 = st.columns(3)
                with c7:
                    st.metric("Duración esperada", humanize_minutes(r.expected_duration_min))
                    st.metric("Confianza duración", f"{r.expected_duration_conf:.0f}%")

                with c8:
                    st.metric("Tiempo a TP", humanize_minutes(r.time_to_tp_min))
                    st.metric("Tiempo a SL", humanize_minutes(r.time_to_sl_min))

                with c9:
                    st.metric("ETA entrada", humanize_minutes(r.next_entry_eta_min))
                    st.metric("Riesgo", r.risk_level)
                    st.metric("Reward", r.reward_level)

                if r.conditions:
                    st.markdown("**Condiciones favorables:**")
                    for c in r.conditions:
                        st.caption(f"✓ {c}")


# ------------------------------------------------------------
# TAB 4 — TODAS LAS SEÑALES
# ------------------------------------------------------------
with tab_all:
    st.subheader("📊 Todas las señales")

    results = st.session_state.scan_results
    if not results:
        st.info("Sin resultados. Presioná ESCANEAR.")
    else:
        all_ranked = rank_all(results)

        st.markdown("#### 📋 Tabla completa")
        df = pd.DataFrame(_ranking_rows(all_ranked))
        st.dataframe(df, use_container_width=True, height=600)

        # ---- Descarga CSV ----
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar CSV",
            data=csv,
            file_name=(
                f"daps_ranking_"
                f"{st.session_state.current_exchange}_"
                f"{st.session_state.current_timeframe}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
            ),
            mime="text/csv",
        )

        # ---- Filtros ----
        st.markdown("---")
        st.markdown("#### 🔍 Filtros")

        c1, c2, c3 = st.columns(3)
        with c1:
            filter_tier = st.multiselect(
                "Tier",
                options=["OMEGA", "A", "B", "S", "NO_TRADE"],
                default=["OMEGA", "A", "B", "S"],
            )
        with c2:
            filter_dir = st.multiselect(
                "Dirección",
                options=["LONG", "SHORT", "NEUTRAL"],
                default=["LONG", "SHORT"],
            )
        with c3:
            filter_valid = st.selectbox(
                "Validez",
                options=["Todas", "Solo válidas", "Solo no válidas"],
                index=0,
            )

        filtered = all_ranked
        if filter_tier:
            filtered = [r for r in filtered if r.tier in filter_tier]
        if filter_dir:
            filtered = [r for r in filtered if r.direction in filter_dir]
        if filter_valid == "Solo válidas":
            filtered = [r for r in filtered if r.is_valid]
        elif filter_valid == "Solo no válidas":
            filtered = [r for r in filtered if not r.is_valid]

        st.caption(f"Mostrando {len(filtered)} de {len(all_ranked)} activos")

        if filtered:
            st.dataframe(
                pd.DataFrame(_operativo_rows(filtered)),
                use_container_width=True,
                height=420,
            )


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
footer_parts = [
    "D.A.P.S Ω Scanner",
    f"v{config.project.get('version', '1.1.0')}",
]
if st.session_state.last_scan:
    footer_parts.append(
        f"Último escaneo: {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S UTC')}"
    )
else:
    footer_parts.append("Último escaneo: nunca")

if st.session_state.n_scanned > 0:
    footer_parts.append(f"{st.session_state.n_scanned} activos analizados")

st.caption(" · ".join(footer_parts))
