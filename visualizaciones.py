import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from config import COMPOUND_COLORS
from data import get_team_color, get_team_name


# ── CLASIFICACIÓN ─────────────────────────────────────────────────────────────

def grafica_clasificacion(session):
    laps = session.laps.groupby('Driver')['LapTime'].min().dropna().sort_values(ascending=False)
    tiempos = laps.dt.total_seconds()
    pole = tiempos.min()
    gaps = tiempos - pole

    colores = [get_team_color(session, d) for d in laps.index]
    etiquetas = [f"{d}  {get_team_name(session, d)}" for d in laps.index]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(etiquetas, gaps.values, color=colores)

    for i, (gap, t) in enumerate(zip(gaps.values, tiempos.values)):
        label = f"POLE  {pole:.3f}s" if gap == 0 else f"+{gap:.3f}s"
        ax.text(gap + 0.02, i, label, va='center', fontsize=8)

    ax.set_xlabel('Diferencia respecto a la pole (segundos)')
    ax.set_title('Clasificación', fontsize=14, fontweight='bold')
    ax.set_xlim(0, gaps.max() + 1.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    return fig


# ── COMPARATIVA TELEMETRÍA ────────────────────────────────────────────────────

def grafica_telemetria(session, piloto1, piloto2):
    v1 = session.laps.pick_driver(piloto1).pick_fastest()
    v2 = session.laps.pick_driver(piloto2).pick_fastest()

    tel1 = v1.get_telemetry().add_distance()
    tel2 = v2.get_telemetry().add_distance()

    color1 = get_team_color(session, piloto1)
    color2 = get_team_color(session, piloto2)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f'Comparativa {piloto1} vs {piloto2}', fontsize=14, fontweight='bold')

    axes[0].plot(tel1['Distance'], tel1['Speed'], color=color1, label=piloto1)
    axes[0].plot(tel2['Distance'], tel2['Speed'], color=color2, label=piloto2)
    axes[0].set_ylabel('Velocidad (km/h)')
    axes[0].legend(loc='lower right')

    axes[1].plot(tel1['Distance'], tel1['Throttle'], color=color1)
    axes[1].plot(tel2['Distance'], tel2['Throttle'], color=color2)
    axes[1].set_ylabel('Acelerador (%)')

    axes[2].plot(tel1['Distance'], tel1['Brake'].astype(int) * 100, color=color1)
    axes[2].plot(tel2['Distance'], tel2['Brake'].astype(int) * 100, color=color2)
    axes[2].set_ylabel('Freno')
    axes[2].set_xlabel('Distancia (m)')

    for ax in axes:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


# ── MAPA DE VELOCIDAD ─────────────────────────────────────────────────────────

def grafica_mapa_velocidad(session, piloto):
    vuelta = session.laps.pick_driver(piloto).pick_fastest()
    tel = vuelta.get_telemetry().add_distance()

    x = tel['X'].values
    y = tel['Y'].values
    speed = tel['Speed'].values

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mcolors.Normalize(vmin=speed.min(), vmax=speed.max())
    cmap = plt.cm.RdYlGn

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidth=3)
    lc.set_array(speed[:-1])

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.add_collection(lc)
    ax.autoscale()
    ax.set_aspect('equal')
    ax.axis('off')

    cbar = fig.colorbar(lc, ax=ax, orientation='horizontal', pad=0.02, fraction=0.03)
    cbar.set_label('Velocidad (km/h)', fontsize=10)
    ax.set_title(f'Mapa de velocidad — {piloto}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig


# ── SECTORES ──────────────────────────────────────────────────────────────────

def grafica_sectores(session, pilotos):
    datos = {}
    for d in pilotos:
        v = session.laps.pick_driver(d).pick_fastest()
        datos[d] = {
            'S1': v['Sector1Time'].total_seconds(),
            'S2': v['Sector2Time'].total_seconds(),
            'S3': v['Sector3Time'].total_seconds(),
        }

    sectores = ['S1', 'S2', 'S3']
    best = {s: min(datos[d][s] for d in pilotos) for s in sectores}
    gaps = {d: {s: datos[d][s] - best[s] for s in sectores} for d in pilotos}

    x = np.arange(len(sectores))
    width = 0.8 / len(pilotos)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    for i, d in enumerate(pilotos):
        color = get_team_color(session, d)
        valores = [gaps[d][s] for s in sectores]
        bars = ax.bar(x + i * width, valores, width, label=d, color=color)

        for bar, val in zip(bars, valores):
            if val == 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        '★', ha='center', va='bottom', fontsize=10, color='gold')
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                        f'+{val:.3f}', ha='center', va='bottom', fontsize=7, color='white')

    ax.set_xticks(x + width * (len(pilotos) - 1) / 2)
    ax.set_xticklabels(sectores, fontsize=13, fontweight='bold', color='white')
    ax.set_ylabel('Gap respecto al mejor (s)')
    ax.set_title('Comparativa de sectores', fontsize=14, fontweight='bold')
    ax.tick_params(colors='white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')

    plt.tight_layout()
    return fig


# ── POSICIÓN VUELTA A VUELTA ──────────────────────────────────────────────────

def grafica_posicion(session, pilotos_destacados):
    all_drivers = session.laps['Driver'].unique()

    fig, ax = plt.subplots(figsize=(14, 8))

    for driver in all_drivers:
        laps_driver = session.laps.pick_driver(driver)[['LapNumber', 'Position']].dropna()
        if driver in pilotos_destacados:
            color = get_team_color(session, driver)
            ax.plot(laps_driver['LapNumber'], laps_driver['Position'],
                    color=color, linewidth=2, label=driver, zorder=3)
            last = laps_driver.iloc[-1]
            ax.text(last['LapNumber'] + 0.5, last['Position'], driver,
                    color=color, fontsize=8, va='center')
        else:
            ax.plot(laps_driver['LapNumber'], laps_driver['Position'],
                    color='#444444', linewidth=0.8, alpha=0.5, zorder=1)

    ax.invert_yaxis()
    ax.set_yticks(range(1, 21))
    ax.set_ylabel('Posición')
    ax.set_xlabel('Vuelta')
    ax.set_title('Posición vuelta a vuelta', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


# ── ESTRATEGIA DE NEUMÁTICOS ──────────────────────────────────────────────────

def grafica_estrategia(session):
    orden_final = (session.laps.groupby('Driver')['Position']
                   .last().dropna().sort_values().index.tolist())

    fig, ax = plt.subplots(figsize=(14, 9))

    for i, driver in enumerate(orden_final):
        stints = session.laps.pick_driver(driver)[['LapNumber', 'Stint', 'Compound']].dropna()
        for _, stint_laps in stints.groupby('Stint'):
            compound = stint_laps['Compound'].iloc[0]
            color = COMPOUND_COLORS.get(compound, '#aaaaaa')
            start = stint_laps['LapNumber'].min()
            end = stint_laps['LapNumber'].max()
            ax.barh(i, end - start + 1, left=start, height=0.6,
                    color=color, edgecolor='#222', linewidth=0.4)

    ax.set_yticks(range(len(orden_final)))
    ax.set_yticklabels(orden_final, fontsize=8)
    ax.set_xlabel('Vuelta')
    ax.set_title('Estrategia de neumáticos', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_elements = [Patch(facecolor=c, label=k) for k, c in COMPOUND_COLORS.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    return fig


# ── DEGRADACIÓN POR STINT ─────────────────────────────────────────────────────

def grafica_degradacion(session, pilotos):
    fig, ax = plt.subplots(figsize=(12, 6))

    for driver in pilotos:
        laps_driver = session.laps.pick_driver(driver).pick_quicklaps()[
            ['LapNumber', 'LapTime', 'Stint', 'Compound']].dropna()
        laps_driver = laps_driver.copy()
        laps_driver['LapTimeSec'] = laps_driver['LapTime'].dt.total_seconds()
        color = get_team_color(session, driver)

        for _, stint_laps in laps_driver.groupby('Stint'):
            compound = stint_laps['Compound'].iloc[0]
            marker = 'o' if compound == 'SOFT' else 's' if compound == 'MEDIUM' else '^'
            ax.scatter(stint_laps['LapNumber'], stint_laps['LapTimeSec'],
                       color=color, marker=marker, s=25, zorder=3)
            ax.plot(stint_laps['LapNumber'], stint_laps['LapTimeSec'],
                    color=color, linewidth=1.2, alpha=0.7)

        last = laps_driver.iloc[-1]
        ax.text(last['LapNumber'] + 0.3, last['LapTimeSec'], driver,
                color=color, fontsize=9, va='center')

    ax.set_xlabel('Vuelta')
    ax.set_ylabel('Tiempo de vuelta (s)')
    ax.set_title('Degradación por stint', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    legend_markers = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', label='SOFT'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', label='MEDIUM'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', label='HARD'),
    ]
    ax.legend(handles=legend_markers, loc='upper right')

    plt.tight_layout()
    return fig


# ── ANÁLISIS DE FRENADA Y TRAZADA ────────────────────────────────────────────

def _extraer_datos_curvas(tel, corners):
    """Para cada curva: velocidad mínima y distancia de frenada."""
    resultados = []
    for _, corner in corners.iterrows():
        dist_corner = corner['Distance']
        label = f"T{int(corner['Number'])}"

        # Velocidad mínima en ventana ±75 m alrededor de la curva
        mask_vel = (
            (tel['Distance'] >= dist_corner - 75) &
            (tel['Distance'] <= dist_corner + 75)
        )
        seg_vel = tel[mask_vel]
        min_speed = float(seg_vel['Speed'].min()) if not seg_vel.empty else None

        # Primer frenazo en los 300 m previos a la curva
        mask_brake = (
            (tel['Distance'] >= dist_corner - 300) &
            (tel['Distance'] < dist_corner)
        )
        seg_brake = tel[mask_brake]
        brake_dist = 0.0
        if not seg_brake.empty:
            braking = seg_brake[seg_brake['Brake'].astype(bool)]
            if not braking.empty:
                brake_dist = float(dist_corner - braking.iloc[0]['Distance'])

        resultados.append({'curva': label, 'min_speed': min_speed, 'brake_dist': brake_dist})
    return resultados


def grafica_mapa_frenada(session, piloto1, piloto2):
    """Circuito coloreado por zonas de freno para dos pilotos."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#1a1a2e')

    for ax, piloto in zip(axes, [piloto1, piloto2]):
        vuelta = session.laps.pick_driver(piloto).pick_fastest()
        tel = vuelta.get_telemetry().add_distance()

        x = tel['X'].values
        y = tel['Y'].values
        brake = tel['Brake'].values.astype(bool)

        # Fondo del circuito
        ax.plot(x, y, color='#333333', linewidth=5, zorder=1)

        # Segmentos coloreados: rojo=frenando, verde=acelerando
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        colors = ['#ff3333' if b else '#00cc66' for b in brake[:-1]]
        lc = LineCollection(segments, colors=colors, linewidth=3, zorder=2)
        ax.add_collection(lc)

        ax.autoscale()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')
        color_piloto = get_team_color(session, piloto)
        ax.set_title(piloto, color=color_piloto, fontsize=13, fontweight='bold')

    legend_elems = [
        Patch(facecolor='#ff3333', label='Frenando'),
        Patch(facecolor='#00cc66', label='Sin frenar'),
    ]
    fig.legend(handles=legend_elems, loc='lower center', ncol=2,
               facecolor='#1a1a2e', labelcolor='white', fontsize=10)
    fig.suptitle('Zonas de frenada', fontsize=14, fontweight='bold', color='white')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


def grafica_velocidad_curvas(session, piloto1, piloto2):
    """Velocidad mínima por curva comparando dos pilotos."""
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    v1 = session.laps.pick_driver(piloto1).pick_fastest().get_telemetry().add_distance()
    v2 = session.laps.pick_driver(piloto2).pick_fastest().get_telemetry().add_distance()

    datos1 = _extraer_datos_curvas(v1, corners)
    datos2 = _extraer_datos_curvas(v2, corners)

    curvas = [d['curva'] for d in datos1]
    speeds1 = [d['min_speed'] or 0 for d in datos1]
    speeds2 = [d['min_speed'] or 0 for d in datos2]

    color1 = get_team_color(session, piloto1)
    color2 = get_team_color(session, piloto2)

    x = np.arange(len(curvas))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(curvas) * 0.7), 6))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    bars1 = ax.bar(x - width / 2, speeds1, width, color=color1, label=piloto1)
    bars2 = ax.bar(x + width / 2, speeds2, width, color=color2, label=piloto2)

    # Marcar al más rápido en cada curva
    for i, (s1, s2) in enumerate(zip(speeds1, speeds2)):
        winner = bars1[i] if s1 > s2 else bars2[i]
        ax.text(winner.get_x() + winner.get_width() / 2,
                winner.get_height() + 1, '★',
                ha='center', va='bottom', fontsize=8, color='gold')

    ax.set_xticks(x)
    ax.set_xticklabels(curvas, fontsize=8, color='white', rotation=45, ha='right')
    ax.set_ylabel('Velocidad mínima (km/h)', color='white')
    ax.set_title('Velocidad mínima por curva', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    return fig


def grafica_punto_frenada(session, piloto1, piloto2):
    """Distancia de frenada antes de cada curva para dos pilotos."""
    circuit_info = session.get_circuit_info()
    corners = circuit_info.corners

    v1 = session.laps.pick_driver(piloto1).pick_fastest().get_telemetry().add_distance()
    v2 = session.laps.pick_driver(piloto2).pick_fastest().get_telemetry().add_distance()

    datos1 = _extraer_datos_curvas(v1, corners)
    datos2 = _extraer_datos_curvas(v2, corners)

    # Filtrar solo curvas con datos de frenada en ambos
    curvas, bd1, bd2 = [], [], []
    for d1, d2 in zip(datos1, datos2):
        if d1['brake_dist'] > 0 or d2['brake_dist'] > 0:
            curvas.append(d1['curva'])
            bd1.append(d1['brake_dist'])
            bd2.append(d2['brake_dist'])

    color1 = get_team_color(session, piloto1)
    color2 = get_team_color(session, piloto2)

    y = np.arange(len(curvas))
    height = 0.38

    fig, ax = plt.subplots(figsize=(10, max(6, len(curvas) * 0.45)))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    ax.barh(y + height / 2, bd1, height, color=color1, label=piloto1)
    ax.barh(y - height / 2, bd2, height, color=color2, label=piloto2)

    ax.set_yticks(y)
    ax.set_yticklabels(curvas, fontsize=9, color='white')
    ax.set_xlabel('Distancia de frenada antes de la curva (m)', color='white')
    ax.set_title('Punto de frenada por curva', fontsize=14, fontweight='bold', color='white')
    ax.tick_params(colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#444')
    ax.spines['bottom'].set_color('#444')
    ax.legend(facecolor='#1a1a2e', labelcolor='white')
    plt.tight_layout()
    return fig


# ── DASHBOARD DE TEMPORADA ────────────────────────────────────────────────────

def grafica_posiciones_temporada(df, piloto, anio):
    """Posición final en cada GP de la temporada."""
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    gps = df["GP"].tolist()
    posiciones = df["posicion"].tolist()

    colores = []
    for pos, status in zip(df["posicion"], df["status"]):
        if pos is None or "DNF" in str(status) or "Retired" in str(status):
            colores.append("#e74c3c")
        elif pos <= 3:
            colores.append("#f1c40f")
        elif pos <= 10:
            colores.append("#3498db")
        else:
            colores.append("#7f8c8d")

    bars = ax.bar(range(len(gps)), posiciones, color=colores, edgecolor="#222", linewidth=0.5)

    for i, (bar, pos) in enumerate(zip(bars, posiciones)):
        if pos is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                    f"P{pos}", ha="center", va="bottom", fontsize=8, color="white")

    for i, grid in enumerate(df["grid"]):
        if grid is not None:
            ax.plot(i, grid, "D", color="white", markersize=5, zorder=5, alpha=0.7)

    ax.set_xticks(range(len(gps)))
    ax.set_xticklabels(
        [g.replace(" Grand Prix", "").replace(" ", "\n") for g in gps],
        fontsize=7, color="white", rotation=0
    )
    ax.invert_yaxis()
    ax.set_ylim(21, 0)
    ax.set_yticks(range(1, 21))
    ax.set_yticklabels([f"P{i}" for i in range(1, 21)], fontsize=7, color="white")
    ax.set_title(f"{piloto} — Posiciones {anio}  (◆ = salida)", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    legend = [
        Patch(facecolor="#f1c40f", label="Podio"),
        Patch(facecolor="#3498db", label="Puntos"),
        Patch(facecolor="#7f8c8d", label="Fuera de puntos"),
        Patch(facecolor="#e74c3c", label="DNF"),
        Line2D([0], [0], marker="D", color="white", linestyle="None", label="Salida"),
    ]
    ax.legend(handles=legend, loc="lower right", facecolor="#1a1a2e", labelcolor="white", fontsize=8)

    plt.tight_layout()
    return fig


def grafica_puntos_temporada(df, piloto, anio):
    """Puntos acumulados a lo largo de la temporada."""
    fig, ax = plt.subplots(figsize=(14, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    x = range(len(df))
    ax.fill_between(x, df["puntos_acum"], alpha=0.3, color="#e74c3c")
    ax.plot(x, df["puntos_acum"], color="#e74c3c", linewidth=2)
    ax.scatter(x, df["puntos_acum"], color="white", s=30, zorder=5)

    total = df["puntos_acum"].iloc[-1] if not df.empty else 0
    ax.set_title(
        f"{piloto} — Puntos acumulados {anio}  (Total: {total:.0f} pts)",
        fontsize=13, fontweight="bold", color="white"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(
        [g.replace(" Grand Prix", "").replace(" ", "\n") for g in df["GP"]],
        fontsize=7, color="white"
    )
    ax.tick_params(colors="white")
    ax.set_ylabel("Puntos", color="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    plt.tight_layout()
    return fig


def grafica_grid_vs_carrera(df, piloto, anio):
    """Scatter: posición de salida vs posición final."""
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    valid = df.dropna(subset=["grid", "posicion"])
    ax.plot([1, 20], [1, 20], "--", color="#555", linewidth=1)

    sc = ax.scatter(
        valid["grid"], valid["posicion"],
        c=valid["puntos"], cmap="RdYlGn", s=80,
        edgecolors="white", linewidths=0.5, zorder=5,
        vmin=0, vmax=25,
    )

    for _, row in valid.iterrows():
        label = row["GP"].replace(" Grand Prix", "")
        ax.annotate(label, (row["grid"], row["posicion"]),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6, color="white", alpha=0.8)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Puntos", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_xlim(0, 21)
    ax.set_ylim(21, 0)
    ax.set_xlabel("Posición de salida (grid)", color="white")
    ax.set_ylabel("Posición final", color="white")
    ax.set_title(f"{piloto} — Grid vs Carrera {anio}", fontsize=13, fontweight="bold", color="white")
    ax.tick_params(colors="white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#444")
    ax.spines["bottom"].set_color("#444")

    plt.tight_layout()
    return fig
