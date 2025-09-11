import networkx as nx
import numpy as np
import plotly.graph_objects as go

# ------------------------------
# Параметры
N = 60          # общее число узлов
layers = 4      # число слоёв "луковицы"
radius = 0.35   # радиус для генерации связей
# ------------------------------

# Создаём граф
G = nx.random_geometric_graph(N, radius=radius, dim=3)

# Определяем слои: разбиваем узлы на группы
nodes_per_layer = N // layers
node_layers = {}
for i, node in enumerate(G.nodes()):
    layer = i // nodes_per_layer
    node_layers[node] = layer

# Извлекаем координаты узлов
pos = nx.get_node_attributes(G, 'pos')
Xn = [pos[k][0] for k in G.nodes()]
Yn = [pos[k][1] for k in G.nodes()]
Zn = [pos[k][2] for k in G.nodes()]

# Линии (рёбра графа)
Xe, Ye, Ze = [], [], []
for e in G.edges():
    Xe += [pos[e[0]][0], pos[e[1]][0], None]
    Ye += [pos[e[0]][1], pos[e[1]][1], None]
    Ze += [pos[e[0]][2], pos[e[1]][2], None]

edge_trace = go.Scatter3d(
    x=Xe, y=Ye, z=Ze,
    mode='lines',
    line=dict(color='rgba(150,150,150,0.4)', width=2),
    hoverinfo='none'
)

# Палитра для слоёв
colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan']

# Узлы по слоям
node_traces = []
for layer in range(layers):
    layer_nodes = [n for n in G.nodes() if node_layers[n] == layer]
    Xl = [pos[n][0] for n in layer_nodes]
    Yl = [pos[n][1] for n in layer_nodes]
    Zl = [pos[n][2] for n in layer_nodes]
    texts = [f"Layer {layer+1} – Node {n}" for n in layer_nodes]

    trace = go.Scatter3d(
        x=Xl, y=Yl, z=Zl,
        mode='markers+text',
        marker=dict(
            size=7,
            color=colors[layer % len(colors)],
            line=dict(color='black', width=0.5)
        ),
        text=texts,
        textposition="top center",
        hoverinfo='text',
        name=f"Layer {layer+1}"
    )
    node_traces.append(trace)

# Собираем финальную фигуру
fig = go.Figure(data=[edge_trace] + node_traces)
fig.update_layout(
    title="3D Луковичная матрица с пометками",
    showlegend=True,
    scene=dict(
        xaxis=dict(showbackground=False),
        yaxis=dict(showbackground=False),
        zaxis=dict(showbackground=False)
    ),
    margin=dict(l=0, r=0, b=0, t=40)
)

fig.show()
