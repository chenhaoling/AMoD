# src/interfaces.py - 模块化接口实现

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class TripDemand:
    """出行需求数据结构"""
    demand_id: str
    origin: str
    destination: str
    departure_time: float
    demand_type: str  # 'private_car' or 'ride_hailing'
    pickup_lat: float = 0.0
    pickup_lon: float = 0.0
    dest_lat: float = 0.0
    dest_lon: float = 0.0


@dataclass
class Vehicle:
    """车辆数据结构"""
    vehicle_id: str
    vehicle_type: str  # 'private_car' or 'ridehail'
    current_location: str
    status: str  # 'idle', 'busy', 'parked'
    current_order_id: Optional[str] = None


@dataclass
class Order:
    """订单数据结构"""
    order_id: str
    pickup_location: str
    destination: str
    passenger_count: int = 1
    order_time: float = 0.0
    assigned_vehicle: Optional[str] = None


# =============================================================================
# 1. 需求生成接口
# =============================================================================
class IDemandGenerator(ABC):
    """需求生成器接口"""

    @abstractmethod
    def generate_large_demand_set(self, size: int) -> List[TripDemand]:
        """生成大量需求用于优化分析"""
        pass

    @abstractmethod
    def generate_simulation_demands(self, size: int, vehicle_composition: Dict) -> List[TripDemand]:
        """基于车辆构成生成仿真用需求"""
        pass


class RealDemandGenerator(IDemandGenerator):
    """实际需求生成器实现"""

    def __init__(self, config, road_edges: Dict):
        self.config = config
        self.road_edges = road_edges
        self.edge_list = list(road_edges.keys())

    def generate_large_demand_set(self, size: int) -> List[TripDemand]:
        """生成大量原始需求"""
        demands = []
        for i in range(size):
            origin = random.choice(self.edge_list)
            destination = random.choice(self.edge_list)
            while destination == origin:
                destination = random.choice(self.edge_list)

            # 基于时间分布生成出发时间
            departure_time = self._generate_departure_time()

            # 随机选择需求类型
            demand_type = random.choice(['private_car', 'ride_hailing'])

            demand = TripDemand(
                demand_id=f"demand_{i}",
                origin=origin,
                destination=destination,
                departure_time=departure_time,
                demand_type=demand_type
            )
            demands.append(demand)

        return demands

    def generate_simulation_demands(self, size: int, vehicle_composition: Dict) -> List[TripDemand]:
        """基于车辆构成生成仿真需求"""
        demands = []

        # 计算各类型需求比例
        total_vehicles = sum(vehicle_composition.values())
        private_ratio = vehicle_composition.get('private_car', 0) / total_vehicles
        ridehail_ratio = vehicle_composition.get('ride_hailing', 0) / total_vehicles

        private_count = int(size * private_ratio)
        ridehail_count = size - private_count

        # 生成私家车需求
        for i in range(private_count):
            demand = self._create_single_demand(f"private_{i}", 'private_car')
            demands.append(demand)

        # 生成网约车需求
        for i in range(ridehail_count):
            demand = self._create_single_demand(f"ridehail_{i}", 'ride_hailing')
            demands.append(demand)

        return demands

    def _create_single_demand(self, demand_id: str, demand_type: str) -> TripDemand:
        """创建单个需求"""
        origin = random.choice(self.edge_list)
        destination = random.choice(self.edge_list)
        while destination == origin:
            destination = random.choice(self.edge_list)

        # 网约车需求更倾向于早期时段
        if demand_type == 'ride_hailing':
            departure_time = random.uniform(0, 1800)  # 前30分钟
        else:
            departure_time = self._generate_departure_time()

        return TripDemand(
            demand_id=demand_id,
            origin=origin,
            destination=destination,
            departure_time=departure_time,
            demand_type=demand_type
        )

    def _generate_departure_time(self) -> float:
        """生成出发时间"""
        # 50%在前30分钟，50%分布在全天
        if random.random() < 0.5:
            return random.uniform(0, 1800)
        else:
            return random.uniform(1800, self.config.simulation_duration)


# =============================================================================
# 2. 车队优化接口
# =============================================================================
class IFleetOptimizer(ABC):
    """车队优化器接口"""

    @abstractmethod
    def optimize_fleet_size(self, demands: List[TripDemand]) -> Dict:
        """优化车队规模"""
        pass


class PaperBasedFleetOptimizer(IFleetOptimizer):
    """基于Nature论文的车队优化器"""

    def __init__(self, config):
        self.config = config
        self.max_connection_time = 900  # 15分钟
        self.avg_speed_kmh = 25.0

    def optimize_fleet_size(self, demands: List[TripDemand]) -> Dict:
        """执行车队规模优化"""
        print(f"执行基于论文的车队优化，需求数量: {len(demands)}")

        # 构建车辆共享网络
        network = self._build_shareability_network(demands)

        # 计算最小路径覆盖
        optimal_size = self._solve_minimum_path_cover(network)

        # 确定车辆构成
        composition = self._determine_vehicle_composition(demands, optimal_size)

        efficiency_ratio = optimal_size / len(demands)

        return {
            'original_demands': len(demands),
            'optimal_fleet_size': optimal_size,
            'vehicle_composition': composition,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_improvement': (1 - efficiency_ratio) * 100,
            'network_stats': {
                'nodes': len(network['nodes']),
                'edges': len(network['edges'])
            }
        }

    def _build_shareability_network(self, demands: List[TripDemand]) -> Dict:
        """构建车辆共享网络"""
        nodes = []
        edges = []

        # 每个需求对应一个节点
        for demand in demands:
            nodes.append({
                'id': demand.demand_id,
                'pickup_time': demand.departure_time,
                'dropoff_time': demand.departure_time + self._estimate_trip_duration(),
                'pickup_location': demand.origin,
                'dropoff_location': demand.destination
            })

        # 构建边：如果两个trips可以由同一车辆连续服务
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes):
                if i != j and self._can_serve_consecutively(node1, node2):
                    edges.append((node1['id'], node2['id']))

        print(f"网络构建完成: {len(nodes)} 个节点, {len(edges)} 条边")

        return {'nodes': nodes, 'edges': edges}

    def _can_serve_consecutively(self, trip1: Dict, trip2: Dict) -> bool:
        """判断两个trips是否可以连续服务"""
        # 计算行驶时间（基于平均速度）
        travel_time = self._estimate_travel_time(trip1['dropoff_location'], trip2['pickup_location'])

        # 检查时间约束
        if trip1['dropoff_time'] + travel_time > trip2['pickup_time']:
            return False

        # 检查连接时间约束
        connection_time = trip2['pickup_time'] - trip1['dropoff_time']
        if connection_time > self.max_connection_time:
            return False

        return True

    def _solve_minimum_path_cover(self, network: Dict) -> int:
        """求解最小路径覆盖问题"""
        nodes = network['nodes']
        edges = network['edges']

        if len(edges) == 0:
            # 没有边，每个节点需要一辆车
            return len(nodes)

        # 简化版：使用启发式算法
        # 实际应该使用二分图最大匹配算法
        covered_nodes = set()
        paths = []

        # 贪心算法构建路径
        for node in nodes:
            if node['id'] not in covered_nodes:
                path = self._build_path_from_node(node, edges, covered_nodes)
                paths.append(path)
                covered_nodes.update(path)

        print(f"构建了 {len(paths)} 条路径覆盖 {len(covered_nodes)} 个节点")

        return len(paths)

    def _build_path_from_node(self, start_node: Dict, edges: List, covered_nodes: set) -> List[str]:
        """从节点开始构建路径"""
        path = [start_node['id']]
        current_node = start_node['id']

        while True:
            # 找到可以连接的下一个节点
            next_node = None
            for edge in edges:
                if edge[0] == current_node and edge[1] not in covered_nodes:
                    next_node = edge[1]
                    break

            if next_node is None:
                break

            path.append(next_node)
            current_node = next_node

        return path

    def _determine_vehicle_composition(self, demands: List[TripDemand], optimal_size: int) -> Dict:
        """确定车辆构成"""
        # 统计需求类型
        private_demands = len([d for d in demands if d.demand_type == 'private_car'])
        ridehail_demands = len([d for d in demands if d.demand_type == 'ride_hailing'])

        total_demands = len(demands)

        # 按比例分配车辆
        private_vehicles = int(optimal_size * private_demands / total_demands)
        ridehail_vehicles = optimal_size - private_vehicles

        return {
            'private_car': private_vehicles,
            'ride_hailing': ridehail_vehicles
        }

    def _estimate_trip_duration(self) -> float:
        """估算行程持续时间"""
        return random.uniform(600, 2400)  # 10-40分钟

    def _estimate_travel_time(self, from_location: str, to_location: str) -> float:
        """估算行驶时间（基于平均速度）"""
        # 假设平均行程距离5公里
        assumed_distance_km = 5.0
        travel_time_seconds = (assumed_distance_km / self.avg_speed_kmh) * 3600
        return travel_time_seconds


# =============================================================================
# 3. 派单优化接口
# =============================================================================
class IDispatchOptimizer(ABC):
    """派单优化器接口"""

    @abstractmethod
    def dispatch_order(self, order: Order, available_vehicles: List[Vehicle]) -> Optional[str]:
        """为订单派遣车辆"""
        pass


class RealTimeDispatchOptimizer(IDispatchOptimizer):
    """实时派单优化器"""

    def __init__(self, config):
        self.config = config

    def dispatch_order(self, order: Order, available_vehicles: List[Vehicle]) -> Optional[str]:
        """实时派单算法"""
        # 筛选合适的车辆
        suitable_vehicles = [v for v in available_vehicles
                             if v.status == 'idle' and v.vehicle_type == 'ridehail']

        if not suitable_vehicles:
            return None

        # 选择最优车辆（简化版：最近距离）
        best_vehicle = self._find_best_vehicle(order, suitable_vehicles)

        return best_vehicle.vehicle_id if best_vehicle else None

    def _find_best_vehicle(self, order: Order, vehicles: List[Vehicle]) -> Optional[Vehicle]:
        """找到最优车辆"""
        # 简化版：随机选择
        return random.choice(vehicles) if vehicles else None


# =============================================================================
# 4. 停车优化接口
# =============================================================================
class IParkingOptimizer(ABC):
    """停车优化器接口"""

    @abstractmethod
    def decide_post_service_action(self, vehicle: Vehicle, location: str) -> Dict:
        """决定服务后行动"""
        pass


class SmartParkingOptimizer(IParkingOptimizer):
    """智能停车优化器"""

    def __init__(self, config):
        self.config = config
        self.demand_history = {}

    def decide_post_service_action(self, vehicle: Vehicle, location: str) -> Dict:
        """智能停车决策"""
        # 分析需求密度
        demand_score = self._analyze_demand_density(location)

        # 分析停车成本
        parking_cost = self._estimate_parking_cost(location)

        # 决策逻辑
        if demand_score > 0.7:
            return {
                'action': 'wait_here',
                'duration': 300,  # 5分钟
                'reason': 'high_demand_expected'
            }
        elif demand_score > 0.3 and parking_cost < 0.5:
            return {
                'action': 'park_nearby',
                'duration': random.randint(600, 1800),  # 10-30分钟
                'reason': 'moderate_demand_low_cost'
            }
        else:
            return {
                'action': 'relocate',
                'target_location': self._find_high_demand_area(),
                'reason': 'seek_better_opportunities'
            }

    def _analyze_demand_density(self, location: str) -> float:
        """分析需求密度"""
        # 简化版：返回随机值
        return random.random()

    def _estimate_parking_cost(self, location: str) -> float:
        """估算停车成本"""
        return random.random()

    def _find_high_demand_area(self) -> str:
        """寻找高需求区域"""
        return "high_demand_zone"


# =============================================================================
# 5. 集成控制器
# =============================================================================
class OptimizationController:
    """优化系统集成控制器"""

    def __init__(self, config, road_edges: Dict):
        self.config = config
        self.road_edges = road_edges

        # 初始化各个模块
        self.demand_generator = RealDemandGenerator(config, road_edges)
        self.fleet_optimizer = PaperBasedFleetOptimizer(config)
        self.dispatch_optimizer = RealTimeDispatchOptimizer(config)
        self.parking_optimizer = SmartParkingOptimizer(config)

        # 状态管理
        self.vehicles = {}
        self.active_orders = {}

    def run_complete_optimization(self) -> Dict:
        """执行完整优化流程"""
        print("\n=== 执行完整优化流程 ===")

        # Step 1: 生成大量需求用于分析
        print("Step 1: 生成分析用需求...")
        analysis_demands = self.demand_generator.generate_large_demand_set(10000)

        # Step 2: 车队规模优化
        print("Step 2: 车队规模优化...")
        optimization_result = self.fleet_optimizer.optimize_fleet_size(analysis_demands)

        # Step 3: 生成仿真用需求
        print("Step 3: 生成仿真用需求...")
        simulation_size = optimization_result['optimal_fleet_size'] * 2  # 每辆车2次出行
        simulation_demands = self.demand_generator.generate_simulation_demands(
            simulation_size, optimization_result['vehicle_composition']
        )

        print(f"\n优化结果摘要:")
        print(f"  原始分析需求: {optimization_result['original_demands']} 个")
        print(f"  最优车队规模: {optimization_result['optimal_fleet_size']} 辆")
        print(f"  效率提升: {optimization_result['efficiency_improvement']:.1f}%")
        print(f"  车辆构成: {optimization_result['vehicle_composition']}")
        print(f"  仿真需求数: {len(simulation_demands)} 个")

        return {
            'optimization_result': optimization_result,
            'simulation_demands': simulation_demands
        }

    def handle_realtime_dispatch(self, new_orders: List[Order]) -> List[Dict]:
        """处理实时派单"""
        dispatch_results = []

        available_vehicles = [v for v in self.vehicles.values() if v.status == 'idle']

        for order in new_orders:
            vehicle_id = self.dispatch_optimizer.dispatch_order(order, available_vehicles)
            if vehicle_id:
                dispatch_results.append({
                    'order_id': order.order_id,
                    'vehicle_id': vehicle_id,
                    'pickup_location': order.pickup_location,
                    'destination': order.destination
                })
                # 更新车辆状态
                self.vehicles[vehicle_id].status = 'busy'
                self.vehicles[vehicle_id].current_order_id = order.order_id

        return dispatch_results

    def handle_parking_decision(self, vehicle_id: str, location: str) -> Dict:
        """处理停车决策"""
        vehicle = self.vehicles.get(vehicle_id)
        if not vehicle:
            return {'action': 'error', 'reason': 'vehicle_not_found'}

        decision = self.parking_optimizer.decide_post_service_action(vehicle, location)

        # 更新车辆状态
        if decision['action'] == 'park_nearby':
            self.vehicles[vehicle_id].status = 'parked'
        elif decision['action'] == 'wait_here':
            self.vehicles[vehicle_id].status = 'idle'  # 等待状态视为空闲

        return decision
