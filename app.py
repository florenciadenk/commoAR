import os
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core import (
    ORDER, EXPORTS_USD, WEIGHTS, download_prices, build_analysis,
    monte_carlo, individual_shock, systemic_shock
)
from ai import generate_economic_analysis

st.set_page_config(
    page_title="CommoAR AI | Riesgo exportador",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Estilo ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0b1220; color: #edf3fb; }
[data-testid="stSidebar"] { background: #101c2d; border-right: 1px solid #26374d; }
[data-testid="stMetric"] { background: #142238; border: 1px solid #263b55; padding: 16px; border-radius: 14px; }
.hero { padding: 28px 30px; border-radius: 18px; background: linear-gradient(120deg,#142c45,#173b51 60%,#1b4b55); border: 1px solid #2d5268; margin-bottom: 22px; }
.hero h1 { font-family: 'Manrope',sans-serif; font-size: 2.2rem; color: #f4f8fc; margin: 0; }
.hero p { color: #bfd0df; font-size: 1rem; margin: 8px 0 0 0; }
.section-title { font-family: 'Manrope',sans-serif; color: #e8f1fb; font-weight: 700; margin-top: 12px; }
.small-note { color: #9fb1c5; font-size: .85rem; }
div[data-testid="stButton"] button[kind="primary"] { background: #1f8a9b; border: 0; color: white; border-radius: 9px; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ---------- Datos y cálculos ----------
@st.cache_data(ttl=3600, show_spinner="Descargando precios internacionales desde Yahoo Finance...")
def load_model():
    prices = download_prices("5y")
    analysis = build_analysis(prices)
    mc = monte_carlo(analysis["index_returns"], simulations=5000, days=252, seed=42)
    return analysis, mc

try:
    analysis, mc = load_model()
except Exception as exc:
    st.error(f"No se pudieron cargar los datos del mercado: {exc}")
    st.info("Verificá tu conexión e intentá actualizar los datos nuevamente.")
    if st.button("Reintentar descarga"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# Persistir escenarios por sesión
if "scenario_individual" not in st.session_state:
    st.session_state.scenario_individual = None
if "scenario_systemic" not in st.session_state:
    st.session_state.scenario_systemic = None

# ---------- Navegación ----------
with st.sidebar:
    st.markdown("## 📊 CommoAR AI")
    st.caption("MONITOR DE COMMODITIES Y RIESGO EXPORTADOR")
    page = st.radio(
        "NAVEGACIÓN",
        ["Dashboard", "Simuladores", "Analista IA", "Fuentes y metodología"],
        label_visibility="visible",
    )
    st.divider()
    st.caption("DATOS DE MERCADO")
    st.write(f"Desde: **{analysis['start_date']:%d/%m/%Y}**")
    st.write(f"Hasta: **{analysis['end_date']:%d/%m/%Y}**")
    st.caption("Precios de referencia internacionales · Yahoo Finance")
    if st.button("Actualizar datos", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ---------- Encabezado ----------
st.markdown("""
<div class="hero">
  <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
    <div style="font-size:1.65rem;">◈</div>
    <div style="font-size:.78rem;letter-spacing:.16em;font-weight:700;color:#91c9d7;">
      COMMODITY MONITOR & EXPORT RISK
    </div>
  </div>
  <h1>CommoAR AI</h1>
  <p>Monitor de commodities y análisis de riesgo exportador de Argentina</p>
</div>
""", unsafe_allow_html=True)

if page == "Dashboard":
    st.markdown('<h3 class="section-title">Panel de exposición exportadora</h3>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Complejos analizados", "5")
    c2.metric("Exportaciones de referencia", f"USD {EXPORTS_USD.sum():,.0f} M")
    c3.metric("Participación nacional", "55,5%")
    c4.metric("Observaciones comunes", f"{analysis['observations']:,}")

    st.caption(
        "Base exportadora: cinco complejos seleccionados, primer semestre de 2026. "
        "Las ponderaciones se mantienen constantes durante el período histórico."
    )

    left, right = st.columns([1.55, 1])
    with left:
        st.markdown("#### Evolución de precios e índice CommoAR")
        chart_df = analysis["base100"].reset_index()
        date_col = chart_df.columns[0]
        long_df = chart_df.melt(
            id_vars=date_col,
            value_vars=ORDER + ["Indice_CommoAR"],
            var_name="Serie",
            value_name="Índice (base 100)",
        )
        fig = px.line(
            long_df, x=date_col, y="Índice (base 100)", color="Serie",
            color_discrete_map={"Indice_CommoAR": "#f5c56b"},
        )
        fig.update_layout(
            template="plotly_dark", height=430, margin=dict(l=10,r=10,t=20,b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        fig.add_hline(y=100, line_dash="dash", line_color="#75869a")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("#### Ponderaciones exportadoras")
        weight_df = pd.DataFrame({
            "Complejo": ORDER,
            "Participación": [WEIGHTS[x] * 100 for x in ORDER],
        })
        fig_w = px.pie(
            weight_df, names="Complejo", values="Participación", hole=.62,
            color_discrete_sequence=["#4f9da6","#e2b96f","#6c8ebf","#9b83bd","#76a88d"],
        )
        fig_w.update_layout(
            template="plotly_dark", height=430, margin=dict(l=5,r=5,t=20,b=5),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend_title_text="",
        )
        st.plotly_chart(fig_w, use_container_width=True)

    st.markdown("#### Rendimiento y riesgo histórico")
    st.dataframe(
        analysis["summary"].rename(index={"Indice_CommoAR": "Índice CommoAR"}),
        use_container_width=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Volatilidad anualizada")
        vol = analysis["summary"].loc[ORDER, "Volatilidad_anualizada_%"].sort_values()
        vol_df = vol.rename_axis("Complejo").reset_index(name="Volatilidad anual (%)")
        fig_v = px.bar(
            vol_df, x="Volatilidad anual (%)", y="Complejo", orientation="h",
            color="Volatilidad anual (%)", color_continuous_scale=["#4f9da6","#e2b96f","#d76b64"],
        )
        fig_v.update_layout(template="plotly_dark", height=330, coloraxis_showscale=False,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_v, use_container_width=True)
    with col_b:
        st.markdown("#### Correlaciones históricas")
        corr = analysis["correlations"]
        fig_c = px.imshow(
            corr, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu",
            aspect="auto",
        )
        fig_c.update_layout(template="plotly_dark", height=330,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_c, use_container_width=True)

    st.markdown("#### Riesgo simulado · Monte Carlo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mediana a un año", f"USD {mc['median']:,.0f} M")
    m2.metric("Percentil 5", f"USD {mc['p5']:,.0f} M")
    m3.metric("VaR 95% estimado", f"USD {mc['var_usd']:,.0f} M")
    m4.metric("Percentil 95", f"USD {mc['p95']:,.0f} M")

    fig_mc = px.histogram(
        x=mc["finals"], nbins=40, labels={"x": "Valor de referencia al final del horizonte (M USD)", "y": "Escenarios"},
    )
    fig_mc.add_vline(x=mc["base"], line_dash="dash", line_color="#f5c56b",
                     annotation_text="Base")
    fig_mc.add_vline(x=mc["p5"], line_dash="dash", line_color="#df7777",
                     annotation_text="Percentil 5")
    fig_mc.update_layout(template="plotly_dark", height=350,
                         paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_mc, use_container_width=True)

    with st.expander("Metodología y limitaciones"):
        st.markdown("""
        - Los precios son referencias internacionales de futuros y no precios FOB efectivos argentinos.
        - Las ponderaciones corresponden al primer semestre de 2026 y se mantienen fijas en el período histórico.
        - El oro representa una referencia para el complejo de oro y plata; WTI es una referencia para petróleo.
        - Monte Carlo utiliza retornos logarítmicos diarios del índice CommoAR, con media y volatilidad históricas constantes, distribución normal e independencia temporal.
        - El valor inicial de referencia corresponde a exportaciones del primer semestre de 2026; no es una proyección de exportaciones anuales.
        """)

elif page == "Simuladores":
    st.markdown('<h3 class="section-title">Simulador de escenarios</h3>', unsafe_allow_html=True)
    mode = st.radio(
        "Seleccioná el tipo de simulación",
        ["Impacto individual", "Shock sistémico y contagio"],
        horizontal=True,
    )

    if mode == "Impacto individual":
        st.write("Modificá individualmente los precios de referencia y observá el efecto conjunto sobre la canasta.")
        with st.form("individual_form"):
            cols = st.columns(2)
            shock_values = {}
            for i, name in enumerate(ORDER):
                with cols[i % 2]:
                    shock_values[name] = st.slider(
                        f"{name}: variación del precio (%)",
                        min_value=-50, max_value=50, value=0, step=5,
                        key=f"ind_{name}",
                    )
            submitted = st.form_submit_button("Calcular impacto", type="primary", use_container_width=True)

        if submitted:
            result = individual_shock(shock_values)
            st.session_state.scenario_individual = {
                "type": result["type"],
                "shocks_pct": result["shocks_pct"],
                "impacts": result["impacts"].to_dict(),
                "total_impact": result["total_impact"],
                "variation_pct": result["variation_pct"],
                "base": result["base"],
                "new_total": result["new_total"],
                "assumption": result["assumption"],
            }

        result = st.session_state.scenario_individual
        if result:
            a, b, c = st.columns(3)
            a.metric("Impacto total", f"USD {result['total_impact']:+,.1f} M")
            b.metric("Variación de la canasta", f"{result['variation_pct']:+.2f}%")
            c.metric("Nuevo valor de referencia", f"USD {result['new_total']:,.1f} M")

            impacts_df = pd.DataFrame({
                "Complejo": ORDER,
                "Impacto (M USD)": [result["impacts"][x] for x in ORDER],
            })
            fig = px.bar(
                impacts_df, x="Complejo", y="Impacto (M USD)",
                color="Impacto (M USD)",
                color_continuous_scale=["#d65f65","#536b83","#45a58a"],
            )
            fig.add_hline(y=0, line_color="white")
            fig.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Escenario hipotético con cantidades exportadas constantes.")
        else:
            st.info("Ajustá los sliders y calculá el impacto para generar un escenario que luego podrá interpretar Gemini.")

    else:
        st.write("Elegí un complejo disparador y evaluá cómo se propaga el shock según las betas históricas.")
        with st.form("systemic_form"):
            trigger = st.selectbox("Complejo disparador", ORDER)
            shock = st.slider("Magnitud del shock inicial (%)", -40, 40, -20, 5)
            contagion = st.checkbox("Incluir contagio histórico mediante betas", value=True)
            submitted = st.form_submit_button("Simular shock sistémico", type="primary", use_container_width=True)

        if submitted:
            result = systemic_shock(trigger, shock, contagion, analysis["returns"], mc)
            st.session_state.scenario_systemic = {
                "type": result["type"],
                "trigger": result["trigger"],
                "initial_shock_pct": result["initial_shock_pct"],
                "include_contagion": result["include_contagion"],
                "betas": result["betas"].to_dict(),
                "shocks_pct": result["shocks_pct"].to_dict(),
                "impacts": result["impacts"].to_dict(),
                "direct_impact": result["direct_impact"],
                "contagion_impact": result["contagion_impact"],
                "total_impact": result["total_impact"],
                "variation_pct": result["variation_pct"],
                "agro_impact": result["agro_impact"],
                "energy_mining_impact": result["energy_mining_impact"],
                "base": result["base"],
                "new_total": result["new_total"],
                "mc_text": result["mc_text"],
                "assumption": result["assumption"],
            }

        result = st.session_state.scenario_systemic
        if result:
            a, b, c = st.columns(3)
            a.metric("Impacto directo", f"USD {result['direct_impact']:+,.1f} M")
            b.metric("Impacto por contagio", f"USD {result['contagion_impact']:+,.1f} M")
            c.metric("Impacto total", f"USD {result['total_impact']:+,.1f} M",
                     delta=f"{result['variation_pct']:+.2f}%")

            impacts_df = pd.DataFrame({
                "Complejo": ORDER,
                "Impacto (M USD)": [result["impacts"][x] for x in ORDER],
                "Shock aplicado (%)": [result["shocks_pct"][x] for x in ORDER],
            })
            fig = px.bar(
                impacts_df, x="Complejo", y="Impacto (M USD)",
                color="Impacto (M USD)",
                color_continuous_scale=["#d65f65","#536b83","#45a58a"],
                hover_data=["Shock aplicado (%)"],
            )
            fig.add_hline(y=0, line_color="white")
            fig.update_layout(template="plotly_dark", height=420, coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True)

            st.info(result["mc_text"])
            st.caption("Las betas representan asociaciones históricas, no relaciones causales. Cantidades constantes.")

elif page == "Analista IA":
    st.markdown('<h3 class="section-title">Analista económico con inteligencia artificial</h3>', unsafe_allow_html=True)
    st.write("Elegí qué escenario querés que Gemini interprete. La IA recibe los resultados del simulador seleccionado y los indicadores históricos del modelo.")

    analysis_choice = st.selectbox(
        "Escenario para analizar",
        ["Impacto individual de commodities", "Shock sistémico y efecto contagio"],
    )
    analysis_type = "individual" if analysis_choice.startswith("Impacto") else "systemic"
    scenario_key = "scenario_individual" if analysis_type == "individual" else "scenario_systemic"
    scenario = st.session_state.get(scenario_key)

    if scenario is None:
        st.warning("Todavía no generaste un escenario de este tipo. Primero ejecutá el simulador correspondiente.")
    else:
        st.success("Hay un escenario disponible para analizar.")
        with st.expander("Ver datos del escenario enviado a Gemini"):
            st.json(scenario)

    question = st.text_area(
        "Pregunta adicional (opcional)",
        placeholder="Ej.: ¿Qué complejo compensa mejor la caída de la soja?",
        height=100,
    )

    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        api_key = os.getenv("GEMINI_API_KEY")

    if st.button("Generar análisis con Gemini", type="primary", use_container_width=True):
        if scenario is None:
            st.error("Primero generá un escenario en el simulador seleccionado.")
        elif not api_key:
            st.error("Falta configurar GEMINI_API_KEY en los secretos de Streamlit.")
            st.code('GEMINI_API_KEY = "tu_clave_de_Gemini"', language="toml")
        else:
            try:
                with st.spinner("Gemini está interpretando los resultados..."):
                    report, used_model = generate_economic_analysis(
                        api_key=api_key,
                        analysis_type=analysis_type,
                        scenario=scenario,
                        summary=analysis["summary"],
                        correlations=analysis["correlations"],
                        extra_question=question,
                    )
                st.caption(f"Informe generado con {used_model}")
                st.markdown(report)
            except Exception as exc:
                st.error("No se pudo generar el informe.")
                st.exception(exc)


elif page == "Fuentes y metodología":
    st.markdown('<h3 class="section-title">Fuentes, datos y metodología</h3>', unsafe_allow_html=True)
    st.write(
        "Esta sección documenta el origen de los datos, los criterios de construcción "
        "del índice y los principales supuestos de los escenarios."
    )

    st.markdown("#### 1. Exportaciones argentinas")
    st.markdown(
        """
        **Fuente primaria:** Instituto Nacional de Estadística y Censos (INDEC), "
        [Complejos exportadores — Primer semestre de 2026](https://www.indec.gob.ar/indec/web/Nivel3-Tema-3-2).

        Los valores incorporados en el modelo son:
        """
    )
    export_table = pd.DataFrame({
        "Complejo": ORDER,
        "Exportaciones (M USD)": [EXPORTS_USD[x] for x in ORDER],
        "Peso dentro de la canasta (%)": [WEIGHTS[x] * 100 for x in ORDER],
    })
    st.dataframe(
        export_table.style.format({
            "Exportaciones (M USD)": "{:,.0f}",
            "Peso dentro de la canasta (%)": "{:.1f}%"
        }),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "La suma de los cinco complejos es USD 27.483 millones. "
        "El modelo utiliza estos valores como ponderaciones fijas de referencia."
    )

    st.markdown("#### 2. Precios internacionales")
    st.write(
        "Las series diarias se descargan desde Yahoo Finance mediante la biblioteca "
        "`yfinance`, utilizando contratos futuros internacionales como proxies."
    )
    ticker_rows = [
        ("Soja", "ZS=F", "CBOT", "https://finance.yahoo.com/quote/ZS=F/"),
        ("Petróleo WTI", "CL=F", "NYMEX", "https://finance.yahoo.com/quote/CL=F/"),
        ("Maíz", "ZC=F", "CBOT", "https://finance.yahoo.com/quote/ZC=F/"),
        ("Oro", "GC=F", "COMEX", "https://finance.yahoo.com/quote/GC=F/"),
        ("Trigo", "ZW=F", "CBOT", "https://finance.yahoo.com/quote/ZW=F/"),
    ]
    ticker_df = pd.DataFrame(ticker_rows, columns=["Serie", "Ticker", "Referencia", "Yahoo Finance"])
    st.dataframe(ticker_df, use_container_width=True, hide_index=True)
    st.markdown(
        """
        - [Yahoo Finance — Soja (ZS=F)](https://finance.yahoo.com/quote/ZS=F/)
        - [Yahoo Finance — Petróleo WTI (CL=F)](https://finance.yahoo.com/quote/CL=F/)
        - [Yahoo Finance — Maíz (ZC=F)](https://finance.yahoo.com/quote/ZC=F/)
        - [Yahoo Finance — Oro (GC=F)](https://finance.yahoo.com/quote/GC=F/)
        - [Yahoo Finance — Trigo (ZW=F)](https://finance.yahoo.com/quote/ZW=F/)
        """
    )

    st.markdown("#### 3. Construcción del índice CommoAR")
    st.latex(r"CommoAR_t = \\sum_{i=1}^{5} w_i \\cdot P_{i,t}^{(100)}")
    st.write(
        "Cada precio se normaliza tomando la primera observación común como base 100. "
        "Luego se calcula un promedio ponderado por la participación de cada complejo "
        "en las exportaciones de referencia del primer semestre de 2026."
    )

    st.markdown("#### 4. Indicadores de riesgo")
    st.markdown(
        """
        - **Retornos logarítmicos:** se calculan sobre las series de precios.
        - **Volatilidad anualizada:** desviación estándar diaria multiplicada por √252.
        - **Correlaciones:** correlaciones históricas entre retornos diarios.
        - **Monte Carlo:** 5.000 trayectorias a 252 días, utilizando la media y la volatilidad históricas de los retornos logarítmicos del índice CommoAR.
        - **Shocks individuales:** variaciones elegidas por el usuario, manteniendo constantes las cantidades exportadas.
        - **Shock sistémico:** propagación mediante betas históricas respecto del complejo disparador.
        """
    )

    st.markdown("#### 5. Limitaciones que deben tenerse presentes")
    st.warning(
        "El índice es una medida sintética de exposición a precios internacionales. "
        "No mide ingresos efectivos de exportación ni constituye un pronóstico."
    )
    st.markdown(
        """
        - Los futuros internacionales son referencias y no precios FOB efectivos argentinos.
        - WTI es un proxy del complejo petrolero-petroquímico; oro es un proxy del complejo oro y plata.
        - Las ponderaciones del primer semestre de 2026 se mantienen constantes en los cinco años.
        - Las simulaciones mantienen constantes las cantidades exportadas.
        - Monte Carlo supone media y volatilidad constantes, retornos normales e independencia temporal.
        - Las correlaciones y betas no implican causalidad.
        - Los resultados de Gemini son interpretaciones de los resultados cuantitativos y no reemplazan el análisis económico.
        """
    )
    st.caption(
        f"Período de precios utilizado en esta sesión: "
        f"{analysis['start_date']:%d/%m/%Y} a {analysis['end_date']:%d/%m/%Y}. "
        f"Observaciones comunes: {analysis['observations']:,}."
    )

st.divider()
st.caption(
    "CommoAR AI · Herramienta académica de análisis de exposición a precios internacionales. "
    "Los resultados son escenarios hipotéticos y no constituyen pronósticos de exportaciones efectivas."
)
