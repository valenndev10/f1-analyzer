import io
import os
import streamlit as st
import matplotlib.pyplot as plt

from data import cargar_sesion
from claude_api import build_quali_context, build_race_context, query_claude
from visualizaciones import (
    grafica_clasificacion,
    grafica_telemetria,
    grafica_mapa_velocidad,
    grafica_sectores,
    grafica_posicion,
    grafica_estrategia,
    grafica_degradacion,
)

# ── Configuración de página ───────────────────────────────────────────────────

st.set_page_config(
    page_title="F1 Data Analyzer",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Data Analyzer")
st.caption("Análisis de datos de Fórmula 1 con Python y FastF1")

# ── Lista de GPs disponibles ──────────────────────────────────────────────────

GPS_2024 = [
    "Bahrain", "Saudi Arabia", "Australia", "Japan", "China",
    "Miami", "Emilia Romagna", "Monaco", "Canada", "Spain",
    "Austria", "Great Britain", "Hungary", "Belgium", "Netherlands",
    "Italy", "Azerbaijan", "Singapore", "United States", "Mexico City",
    "Brazil", "Las Vegas", "Qatar", "Abu Dhabi",
]

# ── Cache de sesiones ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando datos de FastF1...")
def get_session(anio, gp, tipo):
    return cargar_sesion(anio, gp, tipo)

# ── Helpers: figura → bytes PNG (cacheable) ───────────────────────────────────

def _fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

# ── Cache de figuras ──────────────────────────────────────────────────────────
# Cada función recibe parámetros primitivos (str/int/tuple) → hashables.
# La figura se genera una sola vez por combinación y se sirve desde caché.

@st.cache_data(show_spinner="Generando clasificación...")
def cached_clasificacion(anio, gp):
    return _fig_to_bytes(grafica_clasificacion(get_session(anio, gp, "Q")))

@st.cache_data(show_spinner="Generando mapa de velocidad...")
def cached_mapa_velocidad(anio, gp, piloto):
    return _fig_to_bytes(grafica_mapa_velocidad(get_session(anio, gp, "Q"), piloto))

@st.cache_data(show_spinner="Generando comparativa de sectores...")
def cached_sectores(anio, gp, pilotos):
    return _fig_to_bytes(grafica_sectores(get_session(anio, gp, "Q"), list(pilotos)))

@st.cache_data(show_spinner="Generando telemetría...")
def cached_telemetria(anio, gp, tipo, p1, p2):
    return _fig_to_bytes(grafica_telemetria(get_session(anio, gp, tipo), p1, p2))

@st.cache_data(show_spinner="Generando posición vuelta a vuelta...")
def cached_posicion(anio, gp, pilotos):
    return _fig_to_bytes(grafica_posicion(get_session(anio, gp, "R"), list(pilotos)))

@st.cache_data(show_spinner="Generando estrategia...")
def cached_estrategia(anio, gp):
    return _fig_to_bytes(grafica_estrategia(get_session(anio, gp, "R")))

@st.cache_data(show_spinner="Generando degradación...")
def cached_degradacion(anio, gp, pilotos):
    return _fig_to_bytes(grafica_degradacion(get_session(anio, gp, "R"), list(pilotos)))

# ── Selector global en sidebar ────────────────────────────────────────────────

with st.sidebar:
    st.header("Configuración")
    anio = st.selectbox("Temporada", [2024, 2023, 2022], index=0)
    gp   = st.selectbox("Gran Premio", GPS_2024, index=GPS_2024.index("Spain"))


# ── Pestañas ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏁 Clasificación",
    "📡 Telemetría",
    "🏆 Carrera",
    "🔧 Estrategia",
    "💬 Chat IA",
])

# ── TAB 1: Clasificación ──────────────────────────────────────────────────────

with tab1:
    st.subheader(f"Clasificación — {gp} {anio}")

    try:
        quali = get_session(anio, gp, "Q")
        drivers = quali.laps["Driver"].unique().tolist()

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Tiempos por piloto")
            st.image(cached_clasificacion(anio, gp), use_container_width=True)

        with col2:
            st.markdown("#### Mapa de velocidad")
            piloto_mapa = st.selectbox("Piloto", drivers, key="mapa_piloto")
            st.image(cached_mapa_velocidad(anio, gp, piloto_mapa), use_container_width=True)

        st.markdown("#### Comparativa de sectores")
        pilotos_sec = st.multiselect(
            "Pilotos a comparar", drivers, default=drivers[:5], key="sectores_pilotos"
        )
        if len(pilotos_sec) >= 2:
            st.image(cached_sectores(anio, gp, tuple(pilotos_sec)), use_container_width=True)
        else:
            st.info("Selecciona al menos 2 pilotos.")

    except Exception as e:
        st.error(f"Error cargando clasificación: {e}")

# ── TAB 2: Telemetría ─────────────────────────────────────────────────────────

with tab2:
    st.subheader(f"Comparativa de telemetría — {gp} {anio}")

    sesion_tipo = st.radio(
        "Sesión", ["Clasificación (Q)", "Carrera (R)"], horizontal=True, key="tel_sesion"
    )
    tipo_tel = "Q" if sesion_tipo == "Clasificación (Q)" else "R"

    try:
        session_tel = get_session(anio, gp, tipo_tel)
        drivers_tel = session_tel.laps["Driver"].unique().tolist()

        col1, col2 = st.columns(2)
        with col1:
            p1 = st.selectbox("Piloto 1", drivers_tel, index=0, key="tel_p1")
        with col2:
            p2 = st.selectbox("Piloto 2", drivers_tel, index=1, key="tel_p2")

        if p1 != p2:
            st.image(cached_telemetria(anio, gp, tipo_tel, p1, p2), use_container_width=True)
        else:
            st.warning("Selecciona dos pilotos diferentes.")

    except Exception as e:
        st.error(f"Error cargando telemetría: {e}")

# ── TAB 3: Carrera ────────────────────────────────────────────────────────────

with tab3:
    st.subheader(f"Posición vuelta a vuelta — {gp} {anio}")

    try:
        race = get_session(anio, gp, "R")
        drivers_race = race.laps["Driver"].unique().tolist()

        pilotos_dest = st.multiselect(
            "Pilotos a destacar", drivers_race, default=drivers_race[:5], key="carrera_pilotos"
        )

        if pilotos_dest:
            st.image(cached_posicion(anio, gp, tuple(pilotos_dest)), use_container_width=True)
        else:
            st.info("Selecciona al menos un piloto.")

    except Exception as e:
        st.error(f"Error cargando carrera: {e}")

# ── TAB 4: Estrategia ─────────────────────────────────────────────────────────

with tab4:
    st.subheader(f"Estrategia y degradación — {gp} {anio}")

    try:
        race_strat = get_session(anio, gp, "R")
        drivers_strat = race_strat.laps["Driver"].unique().tolist()

        st.markdown("#### Estrategia de neumáticos")
        st.image(cached_estrategia(anio, gp), use_container_width=True)

        st.markdown("#### Degradación por stint")
        pilotos_deg = st.multiselect(
            "Pilotos a comparar", drivers_strat, default=drivers_strat[:3], key="deg_pilotos"
        )
        if pilotos_deg:
            st.image(cached_degradacion(anio, gp, tuple(pilotos_deg)), use_container_width=True)
        else:
            st.info("Selecciona al menos un piloto.")

    except Exception as e:
        st.error(f"Error cargando estrategia: {e}")

# ── TAB 5: Chat IA ────────────────────────────────────────────────────────────

with tab5:
    st.subheader(f"💬 Analista IA — {gp} {anio}")
    st.caption("Pregunta en lenguaje natural sobre los datos de esta sesión.")

    # Contexto F1 para Claude (cacheado por GP/año)
    @st.cache_data(show_spinner=False)
    def get_context(anio, gp):
        try:
            quali = get_session(anio, gp, "Q")
            race  = get_session(anio, gp, "R")
            return build_quali_context(quali, gp, anio), build_race_context(race, gp, anio)
        except Exception:
            return "", ""

    quali_ctx, race_ctx = get_context(anio, gp)

    # Historial de chat en session_state (se reinicia al cambiar GP/año)
    chat_key = f"chat_{anio}_{gp}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []  # lista de {"role": ..., "content": ...}

    history = st.session_state[chat_key]

    # Mostrar historial
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Input del usuario
    pregunta = st.chat_input("Ej: ¿Por qué fue Verstappen más rápido que Norris en el S2?")

    if pregunta:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.warning("El chat IA no está disponible en este momento.")
        else:
            # Mostrar pregunta
            with st.chat_message("user"):
                st.markdown(pregunta)

            # Obtener respuesta
            with st.chat_message("assistant"):
                with st.spinner("Analizando datos..."):
                    respuesta = query_claude(pregunta, quali_ctx, race_ctx, history)
                st.markdown(respuesta)

            # Guardar en historial
            history.append({"role": "user", "content": pregunta})
            history.append({"role": "assistant", "content": respuesta})
            st.session_state[chat_key] = history

    # Botón para limpiar historial
    if history:
        if st.button("🗑️ Limpiar conversación", key="clear_chat"):
            st.session_state[chat_key] = []
            st.rerun()
