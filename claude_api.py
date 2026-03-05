import os
import pandas as pd
import anthropic


# ── Construcción de contexto ──────────────────────────────────────────────────

def _format_time(td):
    """Convierte timedelta a string legible (mm:ss.mmm)."""
    if pd.isna(td):
        return "N/A"
    total = td.total_seconds()
    mins = int(total // 60)
    secs = total % 60
    return f"{mins}:{secs:06.3f}" if mins > 0 else f"{secs:.3f}s"


def build_quali_context(session, gp, anio):
    """Extrae datos de clasificación en texto estructurado para Claude."""
    laps = session.laps.copy()
    best = (
        laps.groupby("Driver")[["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]]
        .min()
        .dropna(subset=["LapTime"])
        .sort_values("LapTime")
    )

    pole_s = best["LapTime"].iloc[0].total_seconds()

    lines = [f"=== CLASIFICACIÓN {gp.upper()} {anio} ==="]
    lines.append(f"{'Pos':<4} {'Piloto':<6} {'Tiempo':<12} {'Gap':<10} {'S1':>8} {'S2':>8} {'S3':>8}")
    lines.append("-" * 60)

    for i, (driver, row) in enumerate(best.iterrows()):
        lap_s = row["LapTime"].total_seconds()
        gap = lap_s - pole_s
        s1 = row["Sector1Time"].total_seconds() if pd.notna(row["Sector1Time"]) else 0
        s2 = row["Sector2Time"].total_seconds() if pd.notna(row["Sector2Time"]) else 0
        s3 = row["Sector3Time"].total_seconds() if pd.notna(row["Sector3Time"]) else 0
        gap_str = "POLE" if i == 0 else f"+{gap:.3f}s"
        lines.append(
            f"P{i+1:<3} {driver:<6} {_format_time(row['LapTime']):<12} {gap_str:<10}"
            f" {s1:>7.3f}s {s2:>7.3f}s {s3:>7.3f}s"
        )

    return "\n".join(lines)


def build_race_context(session, gp, anio):
    """Extrae datos de carrera en texto estructurado para Claude."""
    laps = session.laps.copy()
    lines = [f"\n=== CARRERA {gp.upper()} {anio} ==="]

    # Clasificación final
    final_pos = (
        laps.dropna(subset=["Position"])
        .groupby("Driver")["Position"]
        .last()
        .sort_values()
        .astype(int)
    )

    lines.append("\nCLASIFICACIÓN FINAL:")
    for driver, pos in final_pos.items():
        lines.append(f"  P{pos} {driver}")

    # Estrategia de neumáticos por piloto
    lines.append("\nESTRATEGIA (compuesto · vueltas por stint):")
    for driver in final_pos.index:
        driver_laps = laps[laps["Driver"] == driver]
        if driver_laps.empty:
            continue
        stints = (
            driver_laps.groupby("Stint")
            .agg(compound=("Compound", "first"), n_laps=("LapNumber", "count"))
        )
        strategy = " → ".join(
            f"{row['compound']}({row['n_laps']}v)" for _, row in stints.iterrows()
        )
        lines.append(f"  {driver}: {strategy}")

    # Ritmo medio por piloto (median LapTime)
    lines.append("\nRITMO MEDIO EN CARRERA (mediana, sin inlaps/outlaps):")
    pace = (
        laps[laps["IsPersonalBest"] == False]
        .groupby("Driver")["LapTime"]
        .median()
        .dropna()
        .sort_values()
    )
    for driver, t in pace.items():
        lines.append(f"  {driver}: {_format_time(t)}")

    return "\n".join(lines)


# ── Query a Claude ────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Eres un experto analista de Fórmula 1 que responde preguntas en español con datos reales de timing y telemetría.

Datos de la sesión actual:
{context}

Instrucciones:
- Responde siempre en español, de forma clara y técnica pero accesible.
- Apóyate en los números concretos del contexto para justificar tus respuestas.
- Si la pregunta requiere datos que no están en el contexto, indícalo claramente.
- Sé conciso: respuestas de 3-6 frases salvo que se pida más detalle.
- Si es una pregunta de comparativa, usa los datos de sectores/tiempos directamente."""


def query_claude(question: str, quali_ctx: str, race_ctx: str, history: list) -> str:
    """Envía la pregunta a Claude con el contexto F1 y devuelve la respuesta."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ Falta la API key de Anthropic. Añádela en el sidebar."

    client = anthropic.Anthropic(api_key=api_key)

    context = quali_ctx + "\n" + race_ctx
    system = SYSTEM_PROMPT.format(context=context)

    messages = history + [{"role": "user", "content": question}]

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        system=system,
        messages=messages,
    )

    return response.content[0].text
