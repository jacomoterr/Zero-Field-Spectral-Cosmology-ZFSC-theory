import argparse
import math
import random
import numpy as np
import networkx as nx
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from geometry import compose_matrix

# ===============================
# Цвета для секторов
# ===============================
SECTOR_COLORS = {
    "nu": "blue",
    "lep": "green",
    "u": "red",
    "d": "orange",
    "bos": "purple",
    "dark": "gray",
    "grav": "black",
}

# Цвета для связей
EDGE_COLORS = {
    "default": "lightgray",
    "spin": "cyan",
    "color": "red",
    "isospin": "blue",
    "charge": "yellow",
    "baryon": "brown",
    "nambu": "magenta",
}

# ===============================
# Основная функция
# ===============================
def visualize(matrix_size, delta, r, output_html="geometry.html"):
    # строим матрицу
    M, offsets = compose_matrix(matrix_size, delta, r)

    # граф
    G = nx.Graph()

    # узлы
    for sector, (i0, i1) in offsets.items():
        for i in range(i0, i1):
            label = f"{sector}_{i-i0+1}"
            G.add_node(i, sector=sector, label=label)

    # рёбра
    N = M.shape[0]
    for i in range(N):
        for j in range(i+1, N):
            w = M[i, j]
            if abs(w) > 1e-12:
                # тип связи (по месту можно сделать умнее, пока default)
                edge_type = "default"
                G.add_edge(i, j, weight=abs(w), type=edge_type)

    # позиции через PCA для "луковицы"
    coords = PCA(n_components=2).fit_transform(np.identity(N))
    pos = {i: coords[i] for i in range(N)}

    # plotly trace: рёбра
    edge_x = []
    edge_y = []
    edge_widths = []
    edge_colors = []
    for u,v,data in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        edge_widths.append(2 + 4*data["weight"])
        edge_colors.append(EDGE_COLORS.get(data["type"], EDGE_COLORS["default"]))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="lightgray"),
        hoverinfo="none",
        mode="lines"
    )

    # plotly trace: узлы
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    for i, data in G.nodes(data=True):
        x, y = pos[i]
        node_x.append(x)
        node_y.append(y)
        node_text.append(data["label"])
        node_color.append(SECTOR_COLORS.get(data["sector"], "gray"))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            color=node_color,
            size=10,
            line_width=2
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title="Onion Matrix Visualization",
                        titlefont_size=16,
                        showlegend=False,
                        hovermode="closest",
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
                    ))
    fig.write_html(output_html)
    print(f"[INFO] Visualization saved to {output_html}")

# ===============================
# CLI
# ===============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix_size", type=int, default=11)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--r", type=float, default=0.1)
    parser.add_argument("--output", type=str, default="geometry.html")
    args = parser.parse_args()

    visualize(args.matrix_size, args.delta, args.r, args.output)
