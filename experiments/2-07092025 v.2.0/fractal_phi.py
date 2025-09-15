# -*- coding: utf-8 -*-
"""
ZFSC: Фрактальная матрица на золотом сечении
Визуализация как "космический подсолнух":
- узлы по спирали Фибоначчи (XY)
- Z = слой (поколение частиц)
- ребра: цвет = тип связи (внутрислой / межслой), толщина = сила связи (|вес|)
- дополнительный вывод: спектр собственных значений в PNG
"""

import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# ---------- Параметры ----------
LEVEL = 4                 # уровень рекурсии (4 -> матрица 16x16)
Z_STEP = 1.5              # шаг слоев по оси Z
SUNFLOWER_SCALE = 0.5     # масштаб спирали Фибоначчи в XY
EDGE_MIN_ABS = 1e-6       # порог для отсечения очень слабых связей
INTRA_COLOR = "rgba(50,50,50,0.55)"   # цвет внутрислойных ребер
INTER_COLOR = "rgba(150,150,150,0.22)"# цвет межслойных ребер

# Поколения (6 слоёв) — физический порядок
LAYER_NAMES = [
    "Гравитон (нулевая мода)",
    "Нейтрино (νe, νμ, ντ)",
    "Лептоны (e, μ, τ)",
    "Кварки (u, d, c, s, t, b)",
    "Бозоны (γ, W, Z, gluons, Higgs)",
    "Тёмная материя (DM states)"
]
LAYER_COLORS = ["#636EFA", "#19D3F3", "#00CC96", "#AB63FA", "#EF553B", "#FFA15A"]


# ---------- Математика: фрактальная матрица на золотом сечении ----------
phi = (1 + np.sqrt(5)) / 2  # золотое сечение

def fractal_matrix(level: int) -> np.ndarray:
    """
    Рекурсивная фрактальная матрица:
        M_{n+1} = [[M_n,      φ M_n],
                   [φ M_n,    M_n ]]
    """
    if level == 0:
        return np.array([[1.0]])
    M = fractal_matrix(level - 1)
    top = np.hstack([M, phi * M])
    bottom = np.hstack([phi * M, M])
    return np.vstack([top, bottom])


# ---------- Геометрия узлов ----------
def sunflower_points(n: int, scale: float = 1.0) -> np.ndarray:
    """
    Спираль Фибоначчи (подсолнух) в плоскости XY.
    Возвращает массив shape (n, 2).
    """
    pts = np.zeros((n, 2), dtype=float)
    golden_angle = np.pi * (3 - np.sqrt(5))
    for k in range(n):
        r = scale * np.sqrt(k)
        theta = k * golden_angle
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts[k] = (x, y)
    return pts


# ---------- Построение графа ----------
def build_graph(level: int):
    """
    Из фрактальной матрицы строим:
    - coords: координаты узлов (x,y,z)
    - node_layers: слой (поколение) для каждого узла
    - edges: список ребер (i, j, weight, intra_layer_flag)
    - M: сама матрица
    """
    M = fractal_matrix(level)
    N = M.shape[0]

    # Узлы: XY — подсолнух, Z — номер слоя (поколения)
    xy = sunflower_points(N, scale=SUNFLOWER_SCALE)
    node_layers = {i: i % 6 for i in range(N)}  # распределяем по 6 поколениям циклично
    coords = np.zeros((N, 3), dtype=float)
    for i in range(N):
        x, y = xy[i]
        z = node_layers[i] * Z_STEP
        coords[i] = (x, y, z)

    # Ребра: по ненулевым элементам матрицы
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            w = M[i, j]
            if abs(w) > EDGE_MIN_ABS:
                intra = (node_layers[i] == node_layers[j])
                edges.append((i, j, float(w), intra))
    return coords, node_layers, edges, M


# ---------- Визуализация ----------
def visualize(coords, node_layers, edges, level: int):
    """
    Делает интерактивный 3D HTML: zfsc_sunflower.html
    """
    N = coords.shape[0]

    # Узлы — отдельные трейсы по слоям (чтобы была легенда)
    node_traces = []
    for l in range(6):
        nodes = [i for i in range(N) if node_layers[i] == l]
        if not nodes:
            continue
        X = [coords[i, 0] for i in nodes]
        Y = [coords[i, 1] for i in nodes]
        Z = [coords[i, 2] for i in nodes]
        text = [f"{LAYER_NAMES[l]}<br>ID={i}" for i in nodes]
        node_traces.append(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode="markers",
            marker=dict(size=6, color=LAYER_COLORS[l], line=dict(color="black", width=0.5)),
            text=text, hoverinfo="text",
            name=LAYER_NAMES[l],
            showlegend=True
        ))

    # Рёбра — по одному трейсу на ребро (чтобы варьировать толщину)
    edge_traces = []
    for (i, j, w, intra) in edges:
        x0, y0, z0 = coords[i]
        x1, y1, z1 = coords[j]
        color = INTRA_COLOR if intra else INTER_COLOR
        width = 1.0 + 3.0 * abs(w)  # сила связи -> толщина
        edge_traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="none",
            showlegend=False
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=f"ZFSC: Космический подсолнух (уровень={level}, размер={2**level}×{2**level})",
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False, title=""),
            yaxis=dict(showbackground=False, title=""),
            zaxis=dict(showbackground=False, title="")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.write_html("zfsc_sunflower.html", include_plotlyjs="cdn")
    print("✅ Сохранено: zfsc_sunflower.html")


def save_spectrum_png(M: np.ndarray, level: int):
    """
    Сохраняет спектр собственных значений в PNG.
    """
    eigvals = np.linalg.eigvals(M)
    eigvals = np.sort(np.real(eigvals))

    plt.figure(figsize=(8, 4))
    plt.plot(eigvals, "o-", label="Eigenvalues")
    plt.title(f"Спектр собственных значений (уровень={level}, размер={M.shape[0]}×{M.shape[1]})")
    plt.xlabel("Индекс")
    plt.ylabel("Значение")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("zfsc_sunflower_spectrum.png", dpi=150)
    plt.close()
    print("✅ Сохранено: zfsc_sunflower_spectrum.png")


def main():
    coords, node_layers, edges, M = build_graph(LEVEL)
    visualize(coords, node_layers, edges, LEVEL)
    save_spectrum_png(M, LEVEL)


if __name__ == "__main__":
    main()
