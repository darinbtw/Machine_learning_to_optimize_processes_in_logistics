"""
Route Optimization Web Application
Реализация алгоритмов TSP: Nearest Neighbor, Global NN, Greedy Edge
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import time
import math
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

app = Flask(__name__)


# ─────────────────────────────────────────────
#  Утилиты расстояния
# ─────────────────────────────────────────────

def haversine(p1, p2):
    """Геодезическое расстояние между двумя точками (км)."""
    R = 6371
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def euclidean(p1, p2):
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)


def build_distance_matrix(points, use_geo=True):
    n = len(points)
    dist_fn = haversine if use_geo else euclidean
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i][j] = dist_fn(points[i], points[j])
    return D


def total_distance(path, D):
    n = len(path)
    return sum(D[path[i]][path[(i+1) % n]] for i in range(n))


# ─────────────────────────────────────────────
#  Алгоритм 1: Nearest Neighbor (ближайший сосед)
# ─────────────────────────────────────────────

def nearest_neighbor(D, start=0):
    """
    Жадный алгоритм ближайшего соседа.
    Сложность: O(n²)
    """
    n = len(D)
    visited = [False] * n
    path = [start]
    visited[start] = True

    for _ in range(n - 1):
        cur = path[-1]
        best, best_d = -1, float('inf')
        for j in range(n):
            if not visited[j] and D[cur][j] < best_d:
                best_d = D[cur][j]
                best = j
        path.append(best)
        visited[best] = True

    return path, total_distance(path, D)


# ─────────────────────────────────────────────
#  Алгоритм 2: Global Nearest Neighbor
# ─────────────────────────────────────────────

def global_nearest_neighbor(D):
    """
    Глобальный ближайший сосед: запускает NN из каждой вершины,
    возвращает лучший результат.
    Сложность: O(n³)
    """
    n = len(D)
    best_path, best_dist = None, float('inf')
    for start in range(n):
        path, dist = nearest_neighbor(D, start)
        if dist < best_dist:
            best_path, best_dist = path, dist
    return best_path, best_dist


# ─────────────────────────────────────────────
#  Алгоритм 3: Greedy Edge Insertion
# ─────────────────────────────────────────────

def greedy_edge(D):
    """
    Жадное добавление рёбер по возрастанию стоимости.
    Сложность: O(n² log n)
    """
    n = len(D)

    # Формируем список всех рёбер, сортируем по весу
    edges = sorted(
        [(D[i][j], i, j) for i in range(n) for j in range(i+1, n)],
        key=lambda e: e[0]
    )

    degree = [0] * n
    adj = [[] for _ in range(n)]
    parent = list(range(n))
    edge_count = 0

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        parent[px] = py

    for d, i, j in edges:
        if edge_count == n:
            break
        if degree[i] >= 2 or degree[j] >= 2:
            continue
        # Предотвращаем преждевременное замыкание цикла
        if edge_count < n - 1 and find(i) == find(j):
            continue
        adj[i].append(j)
        adj[j].append(i)
        degree[i] += 1
        degree[j] += 1
        union(i, j)
        edge_count += 1

    # Восстанавливаем путь из структуры смежности
    path = [0]
    vis = [False] * n
    vis[0] = True
    cur = 0
    for _ in range(n - 1):
        for nb in adj[cur]:
            if not vis[nb]:
                path.append(nb)
                vis[nb] = True
                cur = nb
                break

    # Добавляем пропущенные вершины (если граф не связный)
    for i in range(n):
        if not vis[i]:
            path.append(i)

    return path, total_distance(path, D)


# ─────────────────────────────────────────────
#  Визуализация маршрута
# ─────────────────────────────────────────────

ALGO_COLORS = {
    'nn': '#BA7517',
    'gnn': '#1D9E75',
    'greedy': '#378ADD'
}

def visualize_routes(points, results, labels=None):
    """
    Строит сравнительный график маршрутов всех алгоритмов.
    Возвращает base64-строку PNG.
    """
    coords = np.array(points)
    xs, ys = coords[:, 1], coords[:, 0]  # lon, lat

    n_algo = len(results)
    fig, axes = plt.subplots(1, n_algo, figsize=(6 * n_algo, 5), facecolor='#1a1a1a')

    for ax, (algo_key, path, dist_val) in zip(axes if n_algo > 1 else [axes], results):
        col = ALGO_COLORS.get(algo_key, '#aaaaaa')
        ax.set_facecolor('#111111')
        ax.tick_params(colors='#666666', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')

        # Рисуем рёбра маршрута
        full_path = path + [path[0]]
        for i in range(len(full_path) - 1):
            p1 = points[full_path[i]]
            p2 = points[full_path[i+1]]
            ax.plot([p1[1], p2[1]], [p1[0], p2[0]],
                    color=col, linewidth=1.2, alpha=0.7, zorder=1)
            # Стрелка направления
            mx, my = (p1[1]+p2[1])/2, (p1[0]+p2[0])/2
            dx, dy = (p2[1]-p1[1])*0.001, (p2[0]-p1[0])*0.001
            ax.annotate('', xy=(mx+dx, my+dy), xytext=(mx-dx, my-dy),
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.8))

        # Узлы
        ax.scatter(xs, ys, s=50, color='white', zorder=3, linewidths=0)
        ax.scatter([xs[path[0]]], [ys[path[0]]], s=100,
                   color=col, zorder=4, marker='*')  # старт

        if labels:
            for i, (x, y) in enumerate(zip(xs, ys)):
                ax.annotate(labels[i], (x, y),
                            textcoords='offset points', xytext=(4, 4),
                            fontsize=6.5, color='#aaaaaa')

        title = {'nn': 'Nearest Neighbor',
                 'gnn': 'Global NN',
                 'greedy': 'Greedy Edge'}.get(algo_key, algo_key)
        ax.set_title(f'{title}\n{dist_val:.1f} ед.',
                     color=col, fontsize=10, pad=8)
        ax.grid(True, color='#222222', linewidth=0.5)

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight',
                facecolor='#1a1a1a')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ─────────────────────────────────────────────
#  Flask Routes
# ─────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    """
    API: принимает координаты, запускает все алгоритмы, возвращает JSON.

    Request body:
    {
        "coordinates": [[lat, lon], ...],
        "labels": ["Москва", ...],   // опционально
        "use_geo": true              // использовать геодезическое расстояние
    }
    """
    data = request.get_json(force=True)
    coordinates = data.get('coordinates', [])
    labels = data.get('labels', [f'P{i+1}' for i in range(len(coordinates))])
    use_geo = data.get('use_geo', False)

    if len(coordinates) < 3:
        return jsonify({'error': 'Минимум 3 точки'}), 400
    if len(coordinates) > 200:
        return jsonify({'error': 'Максимум 200 точек'}), 400

    points = [tuple(c) for c in coordinates]
    D = build_distance_matrix(points, use_geo=use_geo)

    results = {}

    t0 = time.perf_counter()
    nn_path, nn_dist = nearest_neighbor(D)
    results['nn'] = {
        'path': nn_path,
        'path_names': [labels[i] for i in nn_path],
        'distance': round(nn_dist, 2),
        'time_ms': round((time.perf_counter() - t0) * 1000, 3)
    }

    t0 = time.perf_counter()
    gnn_path, gnn_dist = global_nearest_neighbor(D)
    results['gnn'] = {
        'path': gnn_path,
        'path_names': [labels[i] for i in gnn_path],
        'distance': round(gnn_dist, 2),
        'time_ms': round((time.perf_counter() - t0) * 1000, 3)
    }

    t0 = time.perf_counter()
    ge_path, ge_dist = greedy_edge(D)
    results['greedy'] = {
        'path': ge_path,
        'path_names': [labels[i] for i in ge_path],
        'distance': round(ge_dist, 2),
        'time_ms': round((time.perf_counter() - t0) * 1000, 3)
    }

    # Определяем лучший алгоритм
    best_key = min(results, key=lambda k: results[k]['distance'])
    results['best'] = best_key

    nn_d = results['nn']['distance']
    for k in ['gnn', 'greedy']:
        d = results[k]['distance']
        results[k]['improvement_pct'] = round((nn_d - d) / nn_d * 100, 2)

    # Визуализация
    vis_data = [
        ('nn', nn_path, nn_dist),
        ('gnn', gnn_path, gnn_dist),
        ('greedy', ge_path, ge_dist)
    ]
    try:
        img_b64 = visualize_routes(points, vis_data, labels if len(points) <= 20 else None)
        results['chart_base64'] = img_b64
    except Exception as e:
        results['chart_error'] = str(e)

    return jsonify(results)


@app.route('/upload', methods=['POST'])
def upload_csv():
    """Загрузка CSV-файла с координатами."""
    if 'file' not in request.files:
        return jsonify({'error': 'Файл не найден'}), 400

    f = request.files['file']
    try:
        df = pd.read_csv(f)
        # Ожидаемые колонки: lat/latitude/y, lon/longitude/x, name (опционально)
        lat_col = next((c for c in df.columns if c.lower() in ('lat','latitude','y')), None)
        lon_col = next((c for c in df.columns if c.lower() in ('lon','longitude','lng','x')), None)
        name_col = next((c for c in df.columns if c.lower() in ('name','city','label','title')), None)

        if not lat_col or not lon_col:
            return jsonify({'error': f'Не найдены колонки широты/долготы. Доступны: {list(df.columns)}'}), 400

        df = df.dropna(subset=[lat_col, lon_col])
        coords = df[[lat_col, lon_col]].values.tolist()
        labels = df[name_col].astype(str).tolist() if name_col else [f'P{i+1}' for i in range(len(coords))]

        return jsonify({'coordinates': coords, 'labels': labels, 'count': len(coords)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/benchmark', methods=['POST'])
def benchmark():
    """Бенчмарк алгоритмов на наборах разного размера."""
    data = request.get_json(force=True)
    sizes = data.get('sizes', [10, 20, 50, 100])
    runs = data.get('runs', 3)

    results = []
    for n in sizes:
        pts = [(np.random.uniform(0, 100), np.random.uniform(0, 100)) for _ in range(n)]
        D = build_distance_matrix(pts, use_geo=False)
        row = {'n': n}
        for algo_name, fn in [('nn', lambda D: nearest_neighbor(D)),
                               ('gnn', lambda D: global_nearest_neighbor(D)),
                               ('greedy', lambda D: greedy_edge(D))]:
            times, dists = [], []
            for _ in range(runs):
                t = time.perf_counter()
                _, d = fn(D)
                times.append((time.perf_counter() - t) * 1000)
                dists.append(d)
            row[f'{algo_name}_dist'] = round(np.mean(dists), 2)
            row[f'{algo_name}_time'] = round(np.mean(times), 3)
        results.append(row)

    return jsonify({'benchmark': results})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
