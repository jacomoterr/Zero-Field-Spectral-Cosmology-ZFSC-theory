# -*- coding: utf-8 -*-
"""
ZFSC: 6-мерная матрица, проекция PCA в 3D
"""

import math
import random
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# --------------------------
# ПАРАМЕТРЫ
# --------------------------
N_NODES = 180
N_LAYERS = 6
RNG_SEED = 42

LAYER_NAMES = [
    "Гравитон (нулевая мода)",
    "Нейтрино (νe, νμ, ντ)",
    "Лептоны (e, μ, τ)",
    "Кварки (u, d, c, s, t, b)",
    "Бозоны (γ, W, Z, gluons, Higgs)",
    "Тёмная материя (DM states)"
]

LAYER_COLORS = ["#636EFA", "#19D3F3", "#00CC96", "#AB63FA", "#EF553B", "#FFA15A"]

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

def build_graph():
    G = nx.Graph()
    attrs = {}

    per_layer = [N_NODES // N_LAYERS] * N_LAYERS
    for i in range(N_NODES - sum(per_layer)):
        per_layer[i] += 1

    node_id = 0
    for l in range(N_LAYERS):
        for _ in range(per_layer[l]):
            # 6D-вектор (каждое измерение можно привязать к Δ, r, gL, gR, h1,h2,h3)
            features = np.random.normal(loc=l, scale=0.5, size=6)
            eigval = np.linalg.norm(features)
            attrs[node_id] = dict(layer=l, eigval=eigval, features=features)
            G.add_node(node_id)
            node_id += 1

    nodes = list(G.nodes())
    for i in range(len(nodes)):
        for j in range(i+1, len(nodes)):
            fi, fj = attrs[i]["features"], attrs[j]["features"]
            dist = np.linalg.norm(fi - fj)
            weight = math.exp(-0.3 * dist)
            if random.random() < 0.1:  # разрежаем связи
                G.add_edge(i, j, weight=weight)
    return G, attrs

def project_pca(attrs):
    X = np.vstack([attrs[n]["features"] for n in sorted(attrs.keys())])
    pca = PCA(n_components=3, random_state=RNG_SEED)
    coords = pca.fit_transform(X)
    return {n: tuple(coords[i]) for i, n in enumerate(sorted(attrs.keys()))}, pca.explained_variance_ratio_

def edges_to_traces(G, coords):
    traces = []
    for u, v, data in G.edges(data=True):
        x0, y0, z0 = coords[u]
        x1, y1, z1 = coords[v]
        w = data["weight"]
        width = 1 + 3 * w
        color = "rgba(50,50,50,0.2)"
        traces.append(go.Scatter3d(
            x=[x0, x1], y=[y0, y1], z=[z0, z1],
            mode="lines",
            line=dict(color=color, width=width),
            hoverinfo="none",
            showlegend=False
        ))
    return traces

def build_figure(G, attrs, coords, var_ratio):
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

    title = f"ZFSC: 6D матрица → PCA(3D) (доля дисперсии: {var_ratio[0]:.2f}, {var_ratio[1]:.2f}, {var_ratio[2]:.2f})"
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=title,
        showlegend=True,
        scene=dict(
            xaxis=dict(showbackground=False, title="PC1"),
            yaxis=dict(showbackground=False, title="PC2"),
            zaxis=dict(showbackground=False, title="PC3")
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def main():
    set_seed(RNG_SEED)
    G, attrs = build_graph()
    coords, var_ratio = project_pca(attrs)
    fig = build_figure(G, attrs, coords, var_ratio)
    fig.write_html("matrix6d_zfsc.html", include_plotlyjs="cdn")
    print("✅ Сохранено: matrix6d_zfsc.html")

if __name__ == "__main__":
    main()
