# streamlit_app.py
# Ruta: streamlit_app.py
# ============================================================
# D.A.P.S Ω — Scanner Conservador (Streamlit)
# ============================================================

import logging
import traceback
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st

from config import ScannerConfig
from data_engine import DataEngine
from scoring import CompositeScorer
from risk_manager import RiskManager
from ranking import rank_long, rank_short, rank_all, summary, best_long, best_short
from utils import fmt_pct, fmt_price, humanize_minutes, now_utc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# CONFIGURACIÓN DE PÁGINA
# ------------------------------------------------------------
st.set_page_config(
    page_title="D.A.P.S Ω — Scanner",
    page_icon="Ω",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: white; color: black; }
    h1, h2, h3, h4, h5, h6 { color: black; }
    .stMetric { background-color: #f7f7f7; border-radius: 8px; padding: 8px; }
    .stDataFrame { background-color: white; color: black; }
</style>
""", unsafe_allow_html=True)


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
# INICIALIZACIÓN
# ------------------------------------------------------------
@st.cache_resource
def init_engine():
    return DataEngine(config)


@st.cache_resource
def init_scorer():
    return CompositeScorer(config)


@st.cache_resource
def init_risk():
    return RiskManager(config)


engine = init_engine()
scorer = init_scorer()
risk = init_risk()

for k, v in [
    ("scan_results", []),
    ("last_scan", None),
    ("next_scan", None),
    ("current_exchange", None),
    ("current_timeframe", config.default_timeframe),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ------------------------------------------------------------
# TÍTULO
# ------------------------------------------------------------
st.title("Ω D.A.P.S — Scanner Conservador")
st.caption(f"Versión {config.project.get('version', '1.0.0')} · Solo Binance y Bybit")


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuración")

    exchange = st.selectbox(
        "Exchange",
        options=config.supported_exchanges,
        index=0,
    )
    st.session_state.current_exchange = exchange

    timeframe = st.selectbox(
        "Timeframe",
        options=config.supported_timeframes,
        index=config.supported_timeframes.index(config.default_timeframe)
        if config.default_timeframe in config.supported_timeframes else 0,
    )
    st.session_state.current_timeframe = timeframe

    st.markdown("---")
    scan_btn = st.button("🔄 ESCANEAR", type="primary", use_container_width=True)

    st.markdown("---")
    st.header("📊 Estado")
    st.caption(f"Exchange activo: **{exchange}**")
    st.caption(f"Timeframe: **{timeframe}**")
    st.caption(f"Activos: {len(engine.get_symbols(exchange))}")
    st.caption(f"Exchanges conectados: {len(engine.available_exchanges)}")

    if st.session_state.last_scan:
        st.caption(f"Último escaneo: {st.session_state.last_scan.strftime('%H:%M:%S UTC')}")
    else:
        st.caption("Último escaneo: nunca")

    if st.session_state.next_scan:
        remaining = (st.session_state.next_scan - now_utc()).total_seconds()
        if remaining > 0:
            m, s = divmod(int(remaining), 60)
            st.caption(f"Próximo escaneo en: {m:02d}:{s:02d}")
        else:
            st.caption("Próximo escaneo: ya disponible")


# ------------------------------------------------------------
# FUNCIÓN DE ESCANEO
# ------------------------------------------------------------
def do_scan(exchange_id: str, timeframe: str):
    symbols = engine.get_symbols(exchange_id)
    total = len(symbols)
    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, sym in enumerate(symbols):
        status.text(f"Escaneando {sym} ({i+1}/{total})...")
        try:
            tf_map = {
                "entry": timeframe,
                "confirm": "15m",
                "trend": "1h",
            }
            data = {
                "entry": engine.fetch(sym, exchange_id, tf_map["entry"], config.history_limit),
                "confirm": engine.fetch(sym, exchange_id, tf_map["confirm"], config.history_limit),
                "trend": engine.fetch(sym, exchange_id, tf_map["trend"], config.history_limit),
            }

            entry_df = data.get("entry")
            if entry_df is None or entry_df.empty:
                continue

            # Precio actual
            price = float(entry_df["close"].iloc[-1])

            res = scorer.compute(sym, data)
            res.entry_price = price
            res = risk.enrich(res)
            res.timestamp = now_utc().isoformat()
            results.append(res)
        except Exception as e:
            logger.warning(f"Error {sym}: {e}")
        progress.progress((i + 1) / total)

    progress.empty()
    status.empty()
    return results


# ------------------------------------------------------------
# TRIGGER DE ESCANEO
# ------------------------------------------------------------
if scan_btn:
    exchange_id = st.session_state.current_exchange
    timeframe = st.session_state.current_timeframe
    with st.spinner(f"Escaneando {exchange_id} ({timeframe})..."):
        results = do_scan(exchange_id, timeframe)
    st.session_state.scan_results = results
    st.session_state.last_scan = now_utc()
    st.session_state.next_scan = now_utc() + timedelta(minutes=15)
    st.rerun()


# ------------------------------------------------------------
# TABS
# ------------------------------------------------------------
tab_dashboard, tab_long, tab_short, tab_all = st.tabs(
    ["📡 Dashboard", "🟢 Ranking LONG", "🔴 Ranking SHORT", "📊 Todas las señales"]
)


# ------------------------------------------------------------
# TAB 1: DASHBOARD
# ------------------------------------------------------------
with tab_dashboard:
    st.subheader("📡 Estado general")

    results = st.session_state.scan_results
    if not results:
        st.info("Presioná **ESCANEAR** para comenzar.")
    else:
        s = summary(results)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Activos", s["total"])
        c2.metric("Señales válidas", s["valid"])
        c3.metric("LONG válidas", s["long_valid"])
        c4.metric("SHORT válidas", s["short_valid"])

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Score promedio", f"{s['avg_score']:.1f}")
        c6.metric("Score máximo", f"{s['max_score']:.1f}")
        c7.metric("Neutrales", s["neutral"])
        c8.metric("Exchange", st.session_state.current_exchange or "N/A")

        st.markdown("---")

        st.markdown("### 🌟 Mejor LONG")
        bl = best_long(results)
        if bl:
            st.success(
                f"**{bl.symbol}** · {bl.tier} · Score {bl.total_score:.1f} · "
                f"Confianza {bl.confidence:.0f}%"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada", fmt_price(bl.entry_price))
            col2.metric("SL", fmt_price(bl.sl_price) + f" ({bl.sl_pct:.2f}%)")
            col3.metric("TP", fmt_price(bl.tp_price) + f" ({bl.tp_pct:.2f}%)")
            col1.metric("R:R", f"{bl.rr_ratio:.2f}")
            col2.metric("Duración esperada", humanize_minutes(bl.expected_duration_min))
            col3.metric("Próxima señal ETA", humanize_minutes(bl.next_entry_eta_min))
        else:
            st.info("Sin LONG válidas en este momento.")

        st.markdown("### 🌟 Mejor SHORT")
        bs = best_short(results)
        if bs:
            st.error(
                f"**{bs.symbol}** · {bs.tier} · Score {bs.total_score:.1f} · "
                f"Confianza {bs.confidence:.0f}%"
            )
            col1, col2, col3 = st.columns(3)
            col1.metric("Entrada", fmt_price(bs.entry_price))
            col2.metric("SL", fmt_price(bs.sl_price) + f" ({bs.sl_pct:.2f}%)")
            col3.metric("TP", fmt_price(bs.tp_price) + f" ({bs.tp_pct:.2f}%)")
            col1.metric("R:R", f"{bs.rr_ratio:.2f}")
            col2.metric("Duración esperada", humanize_minutes(bs.expected_duration_min))
            col3.metric("Próxima señal ETA", humanize_minutes(bs.next_entry_eta_min))
        else:
            st.info("Sin SHORT válidas en este momento.")


# ------------------------------------------------------------
# HELPERS DE TABLA
# ------------------------------------------------------------
def _rows_ranking(rlist):
    rows = []
    for i, r in enumerate(rlist, 1):
        rows.append({
            "#": i,
            "Símbolo": r.symbol,
            "Tier": r.tier,
            "Score": round(r.total_score, 1),
            "Confianza": f"{r.confidence:.0f}%",
            "P(L)": f"{r.prob_long*100:.0f}%",
            "P(S)": f"{r.prob_short*100:.0f}%",
            "Régimen": r.regime,
            "Tend": round(r.breakdown.trend * 100, 0),
            "Mom": round(r.breakdown.momentum * 100, 0),
            "Vol": round(r.breakdown.volume * 100, 0),
            "ADX": round(r.adx, 1),
            "ATR%": f"{r.atr_pct*100:.2f}%",
            "RSI": round(r.rsi, 1),
            "RVOL": round(r.rvol, 2),
            "TFI": round(r.tfi, 3),
            "OFI": round(r.ofi, 3),
            "Spread": f"{r.spread_est*100:.3f}%",
            "Coste": f"{r.cost_est*100:.3f}%",
            "Riesgo": r.risk_level,
            "Reward": r.reward_level,
        })
    return rows


def _rows_operativo(rlist):
    rows = []
    for i, r in enumerate(rlist, 1):
        rows.append({
            "#": i,
            "Símbolo": r.symbol,
            "Tier": r.tier,
            "Score": round(r.total_score, 1),
            "Dir": r.direction,
            "Entrada": fmt_price(r.entry_price),
            "Zona Baja": fmt_price(r.entry_zone_low),
            "Zona Alta": fmt_price(r.entry_zone_high),
            "Tipo": r.entry_type,
            "SL $": fmt_price(r.sl_price),
            "SL %": f"{r.sl_pct:.2f}%",
            "SL Método": r.sl_method,
            "TP $": fmt_price(r.tp_price),
            "TP %": f"{r.tp_pct:.2f}%",
            "TP Método": r.tp_method,
            "R:R": f"{r.rr_ratio:.2f}",
            "Trailing %": f"{r.trailing_distance_pct:.2f}%",
            "Activación %": f"{r.trailing_activation_pct:.2f}%",
            "BE %": f"{r.breakeven_trigger_pct:.2f}%",
            "Dur. (min)": round(r.expected_duration_min, 1),
            "Conf. dur.": f"{r.expected_duration_conf:.0f}%",
            "T a TP (min)": round(r.time_to_tp_min, 1),
            "T a SL (min)": round(r.time_to_sl_min, 1),
            "ETA entrada (min)": round(r.next_entry_eta_min, 1),
        })
    return rows


# ------------------------------------------------------------
# TAB 2: LONG
# ------------------------------------------------------------
with tab_long:
    st.subheader("🟢 Ranking LONG")
    results = st.session_state.scan_results
    longs = rank_long(results)

    if not longs:
        st.info("Sin señales LONG válidas.")
    else:
        st.markdown("#### Tabla operativa")
        st.dataframe(pd.DataFrame(_rows_operativo(longs)),
                     use_container_width=True, height=400)

        st.markdown("#### Detalle de scores")
        st.dataframe(pd.DataFrame(_rows_ranking(longs)),
                     use_container_width=True, height=300)


# ------------------------------------------------------------
# TAB 3: SHORT
# ------------------------------------------------------------
with tab_short:
    st.subheader("🔴 Ranking SHORT")
    results = st.session_state.scan_results
    shorts = rank_short(results)

    if not shorts:
        st.info("Sin señales SHORT válidas.")
    else:
        st.markdown("#### Tabla operativa")
        st.dataframe(pd.DataFrame(_rows_operativo(shorts)),
                     use_container_width=True, height=400)

        st.markdown("#### Detalle de scores")
        st.dataframe(pd.DataFrame(_rows_ranking(shorts)),
                     use_container_width=True, height=300)


# ------------------------------------------------------------
# TAB 4: TODAS
# ------------------------------------------------------------
with tab_all:
    st.subheader("📊 Todas las señales (incluye NO_TRADE)")
    results = st.session_state.scan_results
    if not results:
        st.info("Sin resultados. Presioná ESCANEAR.")
    else:
        all_ranked = rank_all(results)
        df = pd.DataFrame(_rows_ranking(all_ranked))
        df["Válida"] = ["✅" if r.is_valid else "❌" for r in all_ranked]
        st.dataframe(df, use_container_width=True, height=700)


# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------
st.markdown("---")
st.caption(
    f"D.A.P.S Ω Scanner · "
    f"Último escaneo: {st.session_state.last_scan.strftime('%Y-%m-%d %H:%M:%S UTC') if st.session_state.last_scan else 'Nunca'}"
)
