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
