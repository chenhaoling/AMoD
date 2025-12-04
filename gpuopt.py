# optimized_fleet_optimizer.py
"""
高性能车队优化器 - 支持GPU加速和多核并行
适用于大规模订单数据（10万+订单）
"""

import pandas as pd
import numpy as np
from datetime import datetime
from collections import defaultdict
import os
import json
import pickle
from multiprocessing import Pool, cpu_count
from functools import partial
import time
import networkx as nx
from networkx.algorithms import bipartite

try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("✓ 检测到GPU，将使用CUDA加速")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠ 未检测到CuPy，使用CPU模式")

try:
    from scipy.spatial import cKDTree

    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class OptimizedFleetOptimizer:
    """高性能车队优化器"""

    def __init__(self, max_connection_time=900, use_gpu=True, n_workers=None):
        self.max_connection_time = max_connection_time
        self.avg_speed_kmh = 25
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.n_workers = n_workers or max(1, cpu_count() - 1)

        # 数据存储
        self.trips = []
        self.trip_data = None  # numpy数组

    def add_trips_batch(self, orders):
        """批量添加订单（性能更好）"""
        n = len(orders)

        # 预分配numpy数组
        self.trip_data = np.zeros((n, 6), dtype=np.float32)
        self.trip_ids = []

        for i, order in enumerate(orders):
            self.trip_data[i] = [
                order['departure_time'],
                order['arrival_time'],
                order['origin_lat'],
                order['origin_lon'],
                order['dest_lat'],
                order['dest_lon']
            ]
            self.trip_ids.append(order['order_id'])

        print(f"✓ 批量加载 {n} 条订单")

    def haversine_vectorized(self, lat1, lon1, lat2, lon2):
        """向量化的Haversine距离计算"""
        if self.use_gpu:
            lat1, lon1 = cp.asarray(lat1), cp.asarray(lon1)
            lat2, lon2 = cp.asarray(lat2), cp.asarray(lon2)

            R = 6371
            lat1_rad = cp.radians(lat1)
            lon1_rad = cp.radians(lon1)
            lat2_rad = cp.radians(lat2)
            lon2_rad = cp.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = cp.sin(dlat / 2) ** 2 + cp.cos(lat1_rad) * cp.cos(lat2_rad) * cp.sin(dlon / 2) ** 2
            c = 2 * cp.arcsin(cp.sqrt(a))

            return cp.asnumpy(R * c)
        else:
            # NumPy版本
            R = 6371
            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
            c = 2 * np.arcsin(np.sqrt(a))

            return R * c

    def build_spatial_index(self):
        """构建空间索引加速邻近查询"""
        if not SCIPY_AVAILABLE:
            return None

        # 提取下车点坐标
        dropoff_coords = self.trip_data[:, 4:6]  # (dest_lat, dest_lon)

        # 构建KD树
        print("构建空间索引...")
        tree = cKDTree(dropoff_coords)
        print(f"✓ 空间索引构建完成")

        return tree

    def find_connectable_pairs_optimized(self, chunk_size=5000):
        """优化的可连接订单对查找 - 使用时间窗口+空间索引"""
        print("\n开始构建共享网络（优化算法）...")

        n = len(self.trip_data)

        # 1. 按接客时间排序
        pickup_order = np.argsort(self.trip_data[:, 0])
        sorted_data = self.trip_data[pickup_order]

        # 2. 构建空间索引
        spatial_tree = self.build_spatial_index()

        # 3. 预计算所有可能的连接
        edges = []

        print(f"使用时间窗口过滤...")
        start_time = time.time()

        for i in range(0, n, chunk_size):
            chunk_end = min(i + chunk_size, n)

            if i % (chunk_size * 10) == 0:
                elapsed = time.time() - start_time
                progress = i / n * 100
                eta = elapsed / max(progress, 0.01) * (100 - progress)
                print(f"  进度: {i}/{n} ({progress:.1f}%) - 已用时{elapsed:.0f}s, 预计剩余{eta:.0f}s")

            # 处理当前chunk
            chunk_edges = self._process_chunk(i, chunk_end, sorted_data, pickup_order, spatial_tree)
            edges.extend(chunk_edges)

        print(f"✓ 找到 {len(edges)} 个可连接的订单对")
        return edges, pickup_order

    def _process_chunk(self, start_idx, end_idx, sorted_data, pickup_order, spatial_tree):
        """处理一个数据块"""
        edges = []

        for i in range(start_idx, end_idx):
            trip1_data = sorted_data[i]
            trip1_idx = pickup_order[i]

            dropoff_time1 = trip1_data[1]
            dropoff_loc1 = trip1_data[4:6]

            # 时间窗口：只考虑在合理时间范围内的订单
            earliest_pickup = dropoff_time1
            latest_pickup = dropoff_time1 + self.max_connection_time

            # 二分查找时间窗口
            time_window_start = np.searchsorted(sorted_data[:, 0], earliest_pickup)
            time_window_end = np.searchsorted(sorted_data[:, 0], latest_pickup)

            if time_window_end <= time_window_start:
                continue

            # 空间过滤（如果有空间索引）
            if spatial_tree:
                # 查找距离下车点一定范围内的上车点
                max_dist_km = (self.max_connection_time / 3600) * self.avg_speed_kmh / 1.3
                candidates_spatial = spatial_tree.query_ball_point(dropoff_loc1, max_dist_km / 111)  # 粗略转换为度
                candidates = list(set(candidates_spatial) & set(range(time_window_start, time_window_end)))
            else:
                candidates = range(time_window_start, time_window_end)

            if not candidates:
                continue

            # 批量计算候选订单的行驶时间
            candidate_indices = list(candidates)
            trip2_pickup_locs = sorted_data[candidate_indices, 2:4]

            # 向量化距离计算
            distances = self.haversine_vectorized(
                dropoff_loc1[0], dropoff_loc1[1],
                trip2_pickup_locs[:, 0], trip2_pickup_locs[:, 1]
            )

            # 计算行驶时间
            travel_times = (distances * 1.3 / self.avg_speed_kmh) * 3600

            # 检查时间约束
            for j, cand_idx in enumerate(candidate_indices):
                if cand_idx == i:
                    continue

                trip2_idx = pickup_order[cand_idx]
                trip2_pickup_time = sorted_data[cand_idx, 0]

                # 检查是否可连接
                arrival_at_pickup2 = dropoff_time1 + travel_times[j]

                if arrival_at_pickup2 <= trip2_pickup_time:
                    connection_time = trip2_pickup_time - dropoff_time1
                    if connection_time <= self.max_connection_time:
                        edges.append((trip1_idx, trip2_idx))

        return edges

    def calculate_minimum_fleet_optimal(self, edges):
        """使用最小路径覆盖算法计算最优车队规模"""
        print("\n计算最小车队规模（最大匹配算法）...")

        n = len(self.trip_data)

        if len(edges) == 0:
            print(f"✓ 没有可连接的边，需要 {n} 辆车")
            return n

        # 构建二分图：左侧 0~n-1，右侧 n~2n-1
        G = nx.Graph()  # 用Graph而不是DiGraph

        for u, v in edges:
            G.add_edge(u, v + n)

        # 显式标记两个集合
        left_nodes = set(range(n))
        right_nodes = set(range(n, 2 * n))

        try:
            # 计算最大匹配
            matching = nx.bipartite.maximum_matching(G, top_nodes=left_nodes)
            max_matching = len(matching) // 2

            # 最小路径覆盖 = 节点数 - 最大匹配数
            fleet_size = n - max_matching

            print(f"  最大匹配数: {max_matching}")
            print(f"✓ 最优车队规模: {fleet_size} 辆")

            return fleet_size
        except Exception as e:
            print(f"⚠ 匹配计算失败: {e}，回退到启发式算法")
            return self.calculate_minimum_fleet_fast(edges)

    def optimize(self):
        """执行优化"""
        print(f"\n{'=' * 80}")
        print(f"  开始车队优化（高性能版本）")
        print(f"{'=' * 80}")
        print(f"订单数: {len(self.trip_data)}")
        print(f"使用GPU: {self.use_gpu}")
        print(f"CPU核心数: {self.n_workers}")

        total_start = time.time()

        # 构建共享网络
        edges, _ = self.find_connectable_pairs_optimized()

        # 计算车队规模
        fleet_size = self.calculate_minimum_fleet_optimal(edges)

        total_time = time.time() - total_start

        efficiency = (1 - fleet_size / len(self.trip_data)) * 100

        result = {
            'total_orders': len(self.trip_data),
            'optimal_fleet_size': fleet_size,
            'efficiency_improvement': efficiency,
            'network_edges': len(edges),
            'computation_time_seconds': total_time,
            'used_gpu': self.use_gpu
        }

        return result


def main():
    """主函数"""
    print("=" * 80)
    print("  高性能车队规模优化工具")
    print("=" * 80)

    # 检查GPU - 修复后的代码
    if GPU_AVAILABLE:
        try:
            # 获取当前设备
            device = cp.cuda.Device()
            # 使用 compute_capability 或其他属性来获取GPU信息
            print(f"✓ GPU可用: 设备ID {device.id}, 计算能力 {device.compute_capability}")
        except Exception as e:
            print(f"✓ GPU可用 (无法获取详细信息: {e})")

    data_file = input("\n请输入OD数据文件路径: ").strip()
    if not os.path.exists(data_file):
        print(f"文件不存在: {data_file}")
        return

    # 加载数据
    print(f"\n加载数据...")
    df = pd.read_csv(data_file, encoding='utf-8')
    print(f"读取 {len(df)} 条记录")

    # 解析订单
    orders = []
    for idx, row in df.iterrows():
        if idx % 50000 == 0:
            print(f"  解析进度: {idx}/{len(df)}")

        try:
            order = {
                'order_id': str(row.get('order_id', f"order_{idx}")),
                'departure_time': (pd.to_datetime(row['begin_time']) -
                                   pd.to_datetime(row['begin_time']).replace(hour=0, minute=0,
                                                                             second=0)).total_seconds(),
                'arrival_time': (pd.to_datetime(row['finish_time']) -
                                 pd.to_datetime(row['finish_time']).replace(hour=0, minute=0,
                                                                            second=0)).total_seconds(),
                'origin_lat': float(row['starting_lat']),
                'origin_lon': float(row['starting_lng']),
                'dest_lat': float(row['dest_lat']),
                'dest_lon': float(row['dest_lng'])
            }
            orders.append(order)
        except:
            continue

    print(f"✓ 成功解析 {len(orders)} 条订单")

    # 采样选项（对于超大数据集）
    if len(orders) > 100000:
        sample = input(f"\n数据量较大({len(orders)}条)，是否采样？(y/n，回车跳过): ").strip().lower()
        if sample == 'y':
            sample_size = int(input("采样数量: "))
            import random
            orders = random.sample(orders, min(sample_size, len(orders)))
            print(f"采样后订单数: {len(orders)}")

    # 创建优化器
    max_conn_time = input("\n最大连接时间（秒，默认900）: ").strip()
    max_conn_time = int(max_conn_time) if max_conn_time else 900

    use_gpu = input("使用GPU加速？(y/n，默认y): ").strip().lower()
    use_gpu = use_gpu != 'n'

    optimizer = OptimizedFleetOptimizer(
        max_connection_time=max_conn_time,
        use_gpu=use_gpu
    )

    # 批量添加订单
    optimizer.add_trips_batch(orders)

    # 执行优化
    result = optimizer.optimize()

    # 输出结果
    print(f"\n{'=' * 80}")
    print("  优化结果")
    print(f"{'=' * 80}")
    print(f"  原始订单数: {result['total_orders']}")
    print(f"  最优车队规模: {result['optimal_fleet_size']} 辆")
    print(f"  效率提升: {result['efficiency_improvement']:.1f}%")
    print(f"  可连接订单对: {result['network_edges']}")
    print(f"  计算耗时: {result['computation_time_seconds']:.1f} 秒")
    print(f"  使用GPU: {'是' if result['used_gpu'] else '否'}")

    # 保存结果
    output_file = 'optimization_result1.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()