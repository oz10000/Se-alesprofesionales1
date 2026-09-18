# D.A.P.S Ω — Scanner Conservador

![CI/CD](https://github.com/Se-alesprofesionales1/Se-alesprofesionales1/actions/workflows/deploy.yml/badge.svg)
![Health Check](https://github.com/Se-alesprofesionales1/Se-alesprofesionales1/actions/workflows/health-check.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)

**Versión:** 1.0.0
**Autor:** Walter Armando Ponce
**Status:** Production Ready

---

## Descripción

**D.A.P.S Ω — Scanner Conservador** es un scanner profesional de oportunidades para **trading manual** sobre **Binance** y **Bybit**.

Clasifica oportunidades mediante un **score compuesto normalizado** y muestra dos rankings independientes:

- **Ranking LONG** — ordenado por score descendente
- **Ranking SHORT** — ordenado por score descendente

Para cada activo aprobado, el sistema calcula automáticamente:

- Entrada estimada (precio + zona + tolerancia)
- Stop Loss (ATR-based, con piso y techo)
- Take Profit (ATR-based, con piso)
- Trailing Stop (distancia y activación)
- Break-Even (trigger)
- Duración esperada del trade
- Tiempo estimado hasta TP
- Tiempo estimado hasta SL
- ETA hasta la próxima entrada aprobada
- Riesgo y recompensa clasificados

---

## Características

- **Multi-exchange:** Binance y Bybit (con fallback en cascada)
- **Multi-timeframe:** 5m / 15m / 1h / 4h
- **Score compuesto:** 6 componentes ponderados (tendencia, momentum, volumen inteligente, volatilidad, liquidez, régimen)
- **Tiers:** Ω > A > B > S > NO_TRADE
- **Umbrales dinámicos:** Se ajustan según régimen de mercado (Expansion / Trend_Strong / Trend_Weak / Chop)
- **Datos públicos:** API CCXT pública, sin API keys
- **Caché local:** Parquet con TTL de 1 hora
- **Sin backtesting:** El scanner es puramente operativo
- **Sin optimización automática:** Los parámetros son fijos y conservadores
- **Interfaz limpia:** Un solo botón, dos rankings

---

## Instalación

### Requisitos

- Python 3.11 o superior
- pip

### Pasos

```bash
# 1. Clonar repositorio
git clone https://github.com/Se-alesprofesionales1/Se-alesprofesionales1.git
cd Se-alesprofesionales1

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar el scanner
streamlit run streamlit_app.py
