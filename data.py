import fastf1
import os
from config import CACHE_DIR, TEAM_COLORS

os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def cargar_sesion(anio, gp, tipo):
    """Carga y devuelve una sesión de FastF1.
    tipo: 'Q' (clasificación) | 'R' (carrera) | 'FP1' | 'FP2' | 'FP3'
    """
    session = fastf1.get_session(anio, gp, tipo)
    session.load()
    return session


def get_team_color(session, driver):
    team = session.laps[session.laps['Driver'] == driver]['Team'].iloc[0]
    return TEAM_COLORS.get(team, '#aaaaaa')


def get_team_name(session, driver):
    return session.laps[session.laps['Driver'] == driver]['Team'].iloc[0]


def get_drivers(session):
    """Devuelve lista de pilotos con tiempo válido, ordenados por tiempo."""
    return (session.laps.groupby('Driver')['LapTime']
            .min().dropna().sort_values().index.tolist())
