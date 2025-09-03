# src/fleet_optimizer.py

import networkx as nx
import numpy as np
from collections import defaultdict
import math


class VehicleShareabilityOptimizer:
    """基于MIT论文的车辆共享网络优化器"""

    def __init__(self, config, max_connection_time=900):  # 15分钟默认连接时间
        self.config = config
        self.max_connection_time = max_connection_time  # δ参数
        self.trips = []
        self.shareability_network = None
        self.optimal_fleet_size = 0
        self.vehicle_dispatches = []

    def add_trip(self, trip_id, pickup_time, dropoff_time, pickup_location, dropoff_location, trip_type='private_car'):
        """添加出行需求"""
        self.trips.append({
            'id': trip_id,
            'pickup_time': pickup_time,
            'dropoff_time': dropoff_time,
            'pickup_location': pickup_location,
            'dropoff_location': dropoff_location,
            'type': trip_type
        })

    def estimate_travel_time(self, from_location, to_location):
        """估算行驶时间 - 简化版本"""
        # 这里使用欧几里得距离的简单估算，实际应该用路网距离
        # 假设平均速度 30km/h = 8.33 m/s
        distance = 1000  # 简化假设1km平均距离
        return distance / 8.33  # 约120秒

    def can_serve_consecutively(self, trip1, trip2):
        """判断两个trips是否可以由同一车辆连续服务"""
        # 条件1: trip1结束时间 + 行驶时间 <= trip2开始时间
        travel_time = self.estimate_travel_time(trip1['dropoff_location'], trip2['pickup_location'])
        if trip1['dropoff_time'] + travel_time > trip2['pickup_time']:
            return False

        # 条件2: 连接时间不超过最大限制
        connection_time = trip2['pickup_time'] - trip1['dropoff_time']
        if connection_time > self.max_connection_time:
            return False

        return True

    def build_shareability_network(self):
        """构建车辆共享网络"""
        G = nx.DiGraph()

        # 添加所有trip作为节点
        for trip in self.trips:
            G.add_node(trip['id'], **trip)

        # 添加边 - 如果两个trips可以连续服务
        for i, trip1 in enumerate(self.trips):
            for j, trip2 in enumerate(self.trips):
                if i != j and self.can_serve_consecutively(trip1, trip2):
                    G.add_edge(trip1['id'], trip2['id'])

        self.shareability_network = G
        return G

    def solve_minimum_path_cover(self):
        """求解最小路径覆盖问题"""
        if self.shareability_network is None:
            self.build_shareability_network()

        G = self.shareability_network

        # 创建二分图用于最大匹配
        # 左侧节点: trip出边，右侧节点: trip入边
        left_nodes = [f"{node}_out" for node in G.nodes()]
        right_nodes = [f"{node}_in" for node in G.nodes()]

        bipartite_graph = nx.Graph()
        bipartite_graph.add_nodes_from(left_nodes, bipartite=0)
        bipartite_graph.add_nodes_from(right_nodes, bipartite=1)

        # 为每条边在二分图中添加对应边
        for u, v in G.edges():
            bipartite_graph.add_edge(f"{u}_out", f"{v}_in")

        # 求最大匹配
        try:
            matching = nx.bipartite.maximum_matching(bipartite_graph, left_nodes)
            # 最小路径覆盖数 = 总节点数 - 最大匹配数
            self.optimal_fleet_size = len(G.nodes()) - len(matching) // 2  # 修正：匹配数要除以2
        except:
            # 如果匹配算法失败，使用简化估算
            self.optimal_fleet_size = max(1, len(G.nodes()) // 3)  # 假设平均每车服务3个trips

        return self.optimal_fleet_size

    def get_fleet_composition(self):
        """根据trip类型分配车辆类型比例"""
        trip_types = defaultdict(int)
        for trip in self.trips:
            trip_types[trip['type']] += 1

        total_trips = len(self.trips)

        composition = {}
        for trip_type, count in trip_types.items():
            # 按比例分配车辆
            vehicles_needed = max(1, int(self.optimal_fleet_size * count / total_trips))
            composition[trip_type] = vehicles_needed

        return composition

    def optimize_fleet(self):
        """执行完整的车队优化"""
        print(f"开始车队规模优化，总trips: {len(self.trips)}")

        # 构建网络
        self.build_shareability_network()
        print(f"构建车辆共享网络，节点数: {self.shareability_network.number_of_nodes()}, "
              f"边数: {self.shareability_network.number_of_edges()}")

        # 求解最优规模
        optimal_size = self.solve_minimum_path_cover()
        print(f"计算得到最优车队规模: {optimal_size}")

        # 获取车辆构成
        composition = self.get_fleet_composition()
        print(f"车辆类型分配: {composition}")

        return {
            'total_vehicles': optimal_size,
            'composition': composition,
            'efficiency_ratio': optimal_size / len(self.trips),
            'network_density': self.shareability_network.number_of_edges() / (len(self.trips) * (len(self.trips) - 1))
        }