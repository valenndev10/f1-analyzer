import fastf1
import pandas as pd
import os

CACHE_DIR = os.path.join(os.path.dirname(__file__), "f1_cache")
fastf1.Cache.enable_cache(CACHE_DIR)


def get_pilotos_temporada(anio):
    """Lista de pilotos de la temporada, cargando solo el primer GP."""
    schedule = fastf1.get_event_schedule(anio, include_testing=False)
    first_gp = schedule.iloc[0]["EventName"]
    session = fastf1.get_session(anio, first_gp, "R")
    session.load(laps=False, telemetry=False, weather=False, messages=False)
    return sorted(session.results["Abbreviation"].dropna().tolist())


def cargar_temporada(anio, piloto):
    """
    Carga los resultados de carrera de todos los GPs de la temporada
    para un piloto dado. Solo descarga resultados (sin telemetría).
    """
    schedule = fastf1.get_event_schedule(anio, include_testing=False)
    resultados = []

    for _, event in schedule.iterrows():
        gp_name = event["EventName"]
        try:
            session = fastf1.get_session(anio, gp_name, "R")
            session.load(laps=False, telemetry=False, weather=False, messages=False)

            results = session.results
            if piloto not in results["Abbreviation"].values:
                continue

            row = results[results["Abbreviation"] == piloto].iloc[0]

            pos   = row.get("Position", None)
            grid  = row.get("GridPosition", None)
            pts   = row.get("Points", 0)

            resultados.append({
                "GP":       gp_name,
                "grid":     int(grid) if pd.notna(grid) and grid != 0 else None,
                "posicion": int(pos)  if pd.notna(pos)  else None,
                "puntos":   float(pts) if pd.notna(pts) else 0.0,
                "status":   str(row.get("Status", "")),
                "team":     str(row.get("TeamName", "")),
            })
        except Exception:
            continue

    df = pd.DataFrame(resultados)
    if not df.empty:
        df["puntos_acum"] = df["puntos"].cumsum()
        df["ganadas"]     = df.apply(
            lambda r: (r["posicion"] or 20) - (r["grid"] or 20), axis=1
        )  # negativo = ganó posiciones
    return df
