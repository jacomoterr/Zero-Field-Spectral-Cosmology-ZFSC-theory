# -*- coding: utf-8 -*-
"""
ZFSC onion model visualization (6 слоёв, физический порядок поколений)
"""

import math
import random
import numpy as np
import networkx as nx
import plotly.graph_objects as go

# --------------------------
# ПАРАМЕТРЫ
# --------------------------
N_NODES = 180
N_LAYERS = 6
BASE_RADIUS = 1.0
RADIUS_STEP = 0.7
RNG_SEED = 42

# Физически корректные слои
LAYER_NAMES = [
    "Гравитон (нулевая мода)",
    "Нейтрино (νe, νμ, ντ)",
    "Лептоны (e, μ, τ)",
    "Кварки (u, d, c, s, t, b)",
    "Бозоны (γ, W, Z, gluons, Higgs)",
    "Тёмная материя (DM states)"
]

# Цвета
LAYER_COLORS = ["#636EFA", "#19D3F3", "#00CC96", "#AB63FA", "#EF553B", "#FFA15A"]

# Вероятности связей
P_INTRA = 0.3
P_INTER_ADJ = 0.1
P_INTER_FAR = 0.02


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def fibonacci_sphere(n, radius):
    pts = np.zeros((n, 3))
    phi = math.pi * (3. - math.sqrt(5.))
    for i in range(n):
        y = 1 - (i / float(n - 1)) * 2
        r = math.sqrt(max(0.0, 1 - y * y))
        theta = phi * i
        x = math.cos(theta) * r
        z = math.sin(theta) * r
        pts[i] = radius * np.array([x, y, z])
    return pts


def build_graph():
    G = nx.Graph()
    attrs = {}

    per_layer = [N_NODES // N_LAYERS] * N_LAYERS
    for i in range(N_NODES - sum(per_layer)):
        per_layer[i] += 1

    node_id = 0
    for l in range(N_LAYERS):
        for _ in range(per_layer[l]):
            eigval = abs(np.random.randn()) * (l + 1)
            attrs[node_id] = dict(layer=l, eigval=eigval)
            G.add_node(node_id)
            node_id += 1

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            li, lj = attrs[i]["layer"], attrs[j]["layer"]
            if li == lj:
                p = P_INTRA
            elif abs(li - lj) == 1:
                p = P_INTER_ADJ
            else:
                p = P_INTER_FAR
            if random.random() < p:
                diff = abs(attrs[i]["eigval"] - attrs[j]["eigval"])
                weight = math.exp(-0.3 * diff)
                G.add_edge(i, j, weight=weight, intra=(li == lj))
    return G, attrs


def build_layout(attrs):
    coords = {}
    by_layer = {l: [] for l in range(N_LAYERS)}
    for n, a in attrs.items():
        by_layer[a["layer"]].append(n)

    for l in range(N_LAYERS):
        r = BASE_RADIUS + l * RADIUS_STEP
        pts = fibonacci_sphere(len(by_layer[l]), r)
        for idx, n in enumerate(by_layer[l]):
            coords[n] = tuple(pts[idx])
    return coords


def edges_to_traces(G, coords):
    traces = []
    for u, v, data in G.edges(data=True):
        x0, y0, z0 = coords[u]
        x1, y1, z1 = coords[v]
        w = data["weight"]

        color = "rgba(50,50,50,0.4)" if data["intra"] else "rgba(50,50,50,0.15)"
        width = 2 + 4 * w if data["intra"] else 1 + 2 * w

        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="none",
            showlegend=False
        ))
    return traces


def build_figure(G, attrs):
    coords = build_layout(attrs)
    edge_traces = edges_to_traces(G, coords)

    node_traces = []
    for l in range(N_LAYERS):
        nodes = [n for n in G.nodes() if attrs[n]["layer"] == l]
        X = [coords[n][0] for n in nodes]
        Y = [coords[n][1] for n in nodes]
        Z = [coords[n][2] for n in nodes]
        text = [
            f"{LAYER_NAMES[l]}<br>ID={n} eig≈{attrs[n]['eigval']:.3f}"
            for n in nodes
        ]
        node_traces.append(go.Scatter3d(
            x=X, y=Y, z=Z,
            mode="markers",
            marker=dict(size=6, color=LAYER_COLORS[l], line=dict(color="black", width=0.5)),
            text=text,
            hoverinfo="text",
            name=LAYER_NAMES[l]
        ))

    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title="ZFSC: Луковичная матрица (6 физических слоёв)",
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False, title=""),
            yaxis=dict(showbackground=False, title=""),
            zaxis=dict(showbackground=False, title="")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig


def main():
    set_seed(RNG_SEED)
    G, attrs = build_graph()
    fig = build_figure(G, attrs)
    fig.write_html("onion_zfsc.html", include_plotlyjs="cdn")
    print("✅ Сохранено: onion_zfsc.html")


if __name__ == "__main__":
    main()
