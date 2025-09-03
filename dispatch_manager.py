# src/dispatch_manager.py

import json
import math
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import csv


@dataclass
class Driver:
    """司机信息"""
    driver_id: str
    lat: float
    lon: float
    status: str  # idle, busy, offline
    rating: float
    speed: float
    remaining_power: int
    rejection_list: List[str]
    car_plate: str


@dataclass
class Order:
    """订单信息"""
    order_id: str
    price: float
    origin_lat: float
    origin_lon: float
    destination_lat: float
    destination_lon: float
    preference_list: List[str]
    member_id: str
    order_pref: Dict


@dataclass
class MatchResult:
    """匹配结果"""
    order_id: str
    driver_id: str
    match_score: float
    distance_km: float
    eta_seconds: float
    match_reason: str


class DispatchManager:
    """网约车派单管理器"""

    def __init__(self, config):
        self.config = config
        self.drivers = {}  # driver_id -> Driver
        self.orders = {}  # order_id -> Order
        self.match_history = []
        self.dispatch_records = []

        # 匹配参数
        self.max_pickup_distance_km = 5.0  # 最大接单距离5公里
        self.max_pickup_time_minutes = 15  # 最大接单时间15分钟
        self.avg_speed_kmh = 30  # 平均车速30公里/小时

        print("派单管理器初始化完成")

    def load_input_data(self, json_file_path: str) -> bool:
        """从JSON文件加载司机和订单数据"""
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 加载司机数据
            drivers_data = data.get('drivers', [])
            for driver_data in drivers_data:
                driver = Driver(
                    driver_id=driver_data['driver_id'],
                    lat=driver_data['lat'],
                    lon=driver_data['lon'],
                    status=driver_data['status'],
                    rating=driver_data['rating'],
                    speed=driver_data['speed'],
                    remaining_power=driver_data['remaining_power'],
                    rejection_list=driver_data.get('rejection_list', []),
                    car_plate=driver_data['car_plate']
                )
                self.drivers[driver.driver_id] = driver

            # 加载订单数据
            orders_data = data.get('orders', [])
            for order_data in orders_data:
                order = Order(
                    order_id=order_data['order_id'],
                    price=order_data['price'],
                    origin_lat=order_data['origin_lat'],
                    origin_lon=order_data['origin_lon'],
                    destination_lat=order_data['destination_lat'],
                    destination_lon=order_data['destination_lon'],
                    preference_list=order_data.get('preference_list', []),
                    member_id=order_data['member_id'],
                    order_pref=order_data.get('order_pref', {})
                )
                self.orders[order.order_id] = order

            print(f"成功加载 {len(self.drivers)} 个司机, {len(self.orders)} 个订单")
            return True

        except Exception as e:
            print(f"加载输入数据失败: {e}")
            return False

    def calculate_distance_km(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """计算两点间距离（公里）"""
        # 使用Haversine公式计算球面距离
        R = 6371  # 地球半径(公里)

        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def calculate_eta_seconds(self, distance_km: float, current_speed_kmh: float = None) -> float:
        """计算预估到达时间（秒）"""
        speed = current_speed_kmh if current_speed_kmh and current_speed_kmh > 0 else self.avg_speed_kmh
        return (distance_km / speed) * 3600

    def get_available_drivers(self, order: Order) -> List[Driver]:
        """获取可接单的司机列表"""
        available_drivers = []

        for driver in self.drivers.values():
            # 基本筛选条件
            if driver.status != 'idle':
                continue

            # 检查是否在拒单列表中
            if order.order_id in driver.rejection_list:
                continue

            # 计算距离筛选
            distance_km = self.calculate_distance_km(
                driver.lat, driver.lon, order.origin_lat, order.origin_lon
            )

            if distance_km <= self.max_pickup_distance_km:
                available_drivers.append(driver)

        return available_drivers

    def calculate_match_score(self, driver: Driver, order: Order) -> float:
        """计算司机与订单的匹配分数（0-100）"""
        # TODO: 这里可以实现复杂的匹配算法
        # 目前使用简化版本，你可以后续优化

        # 距离因子（距离越近分数越高）
        distance_km = self.calculate_distance_km(
            driver.lat, driver.lon, order.origin_lat, order.origin_lon
        )
        distance_score = max(0, 100 - distance_km * 20)  # 每公里扣20分

        # 司机评分因子
        rating_score = driver.rating * 20  # 5星制转100分制

        # 偏好列表加分
        preference_bonus = 20 if driver.driver_id in order.preference_list else 0

        # 综合得分
        total_score = distance_score * 0.5 + rating_score * 0.3 + preference_bonus * 0.2

        return min(100, max(0, total_score))

    def find_best_match(self, order: Order) -> Optional[MatchResult]:
        """为订单找到最佳匹配司机"""
        available_drivers = self.get_available_drivers(order)

        if not available_drivers:
            return None

        best_match = None
        best_score = -1

        for driver in available_drivers:
            score = self.calculate_match_score(driver, order)
            distance_km = self.calculate_distance_km(
                driver.lat, driver.lon, order.origin_lat, order.origin_lon
            )
            eta_seconds = self.calculate_eta_seconds(distance_km, driver.speed)

            # 时间约束检查
            if eta_seconds > self.max_pickup_time_minutes * 60:
                continue

            if score > best_score:
                best_score = score
                best_match = MatchResult(
                    order_id=order.order_id,
                    driver_id=driver.driver_id,
                    match_score=score,
                    distance_km=distance_km,
                    eta_seconds=eta_seconds,
                    match_reason="best_score_match"
                )

        return best_match

    def dispatch_order(self, order_id: str) -> Optional[MatchResult]:
        """派单主函数"""
        if order_id not in self.orders:
            print(f"订单 {order_id} 不存在")
            return None

        order = self.orders[order_id]
        match_result = self.find_best_match(order)

        if match_result:
            # 更新司机状态
            if match_result.driver_id in self.drivers:
                self.drivers[match_result.driver_id].status = 'busy'

            # 记录派单历史
            self.match_history.append(match_result)

            # 记录派单详情
            self.dispatch_records.append({
                'timestamp': len(self.dispatch_records),
                'order_id': order_id,
                'driver_id': match_result.driver_id,
                'match_score': match_result.match_score,
                'pickup_distance_km': match_result.distance_km,
                'eta_seconds': match_result.eta_seconds,
                'driver_rating': self.drivers[match_result.driver_id].rating,
                'order_price': order.price,
                'match_reason': match_result.match_reason
            })

            print(f"订单 {order_id} 派给司机 {match_result.driver_id}, "
                  f"匹配分数: {match_result.match_score:.1f}, "
                  f"距离: {match_result.distance_km:.2f}km, "
                  f"ETA: {match_result.eta_seconds / 60:.1f}分钟")

            return match_result
        else:
            print(f"订单 {order_id} 未找到合适司机")
            return None

    def batch_dispatch(self) -> List[MatchResult]:
        """批量派单"""
        results = []

        print(f"\n开始批量派单，共 {len(self.orders)} 个订单...")

        for order_id in self.orders:
            result = self.dispatch_order(order_id)
            if result:
                results.append(result)

        print(f"批量派单完成，成功派单 {len(results)} 个")
        return results

    def simulate_driver_movement(self, step: int):
        """模拟司机位置变化（供仿真使用）"""
        # TODO: 集成到SUMO仿真中，根据仿真中的车辆位置更新司机位置
        # 目前使用随机移动模拟
        for driver in self.drivers.values():
            if driver.status == 'idle':
                # 随机小幅移动
                driver.lat += random.uniform(-0.001, 0.001)
                driver.lon += random.uniform(-0.001, 0.001)

    def update_driver_status(self, driver_id: str, new_status: str):
        """更新司机状态"""
        if driver_id in self.drivers:
            self.drivers[driver_id].status = new_status
            print(f"司机 {driver_id} 状态更新为 {new_status}")

    def add_dynamic_order(self, order_data: Dict) -> str:
        """动态添加新订单"""
        order = Order(
            order_id=order_data['order_id'],
            price=order_data.get('price', 0),
            origin_lat=order_data['origin_lat'],
            origin_lon=order_data['origin_lon'],
            destination_lat=order_data['destination_lat'],
            destination_lon=order_data['destination_lon'],
            preference_list=order_data.get('preference_list', []),
            member_id=order_data.get('member_id', ''),
            order_pref=order_data.get('order_pref', {})
        )

        self.orders[order.order_id] = order
        print(f"新增订单 {order.order_id}")

        return order.order_id

    def get_dispatch_statistics(self) -> Dict:
        """获取派单统计信息"""
        if not self.dispatch_records:
            return {}

        total_dispatches = len(self.dispatch_records)
        successful_matches = len([r for r in self.dispatch_records if r['match_score'] > 0])

        avg_match_score = sum(r['match_score'] for r in self.dispatch_records) / total_dispatches
        avg_pickup_distance = sum(r['pickup_distance_km'] for r in self.dispatch_records) / total_dispatches
        avg_eta = sum(r['eta_seconds'] for r in self.dispatch_records) / total_dispatches

        return {
            'total_orders': len(self.orders),
            'total_drivers': len(self.drivers),
            'total_dispatches': total_dispatches,
            'successful_matches': successful_matches,
            'match_rate': successful_matches / len(self.orders) if len(self.orders) > 0 else 0,
            'avg_match_score': avg_match_score,
            'avg_pickup_distance_km': avg_pickup_distance,
            'avg_eta_minutes': avg_eta / 60,
            'idle_drivers': len([d for d in self.drivers.values() if d.status == 'idle']),
            'busy_drivers': len([d for d in self.drivers.values() if d.status == 'busy'])
        }

    def save_dispatch_results(self, output_dir: str):
        """保存派单结果"""
        import os

        # 保存派单记录
        dispatch_file = os.path.join(output_dir, "dispatch_records.csv")
        if self.dispatch_records:
            with open(dispatch_file, 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['timestamp', 'order_id', 'driver_id', 'match_score',
                              'pickup_distance_km', 'eta_seconds', 'driver_rating',
                              'order_price', 'match_reason']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.dispatch_records)

        # 保存统计信息
        stats_file = os.path.join(output_dir, "dispatch_statistics.csv")
        stats = self.get_dispatch_statistics()
        if stats:
            with open(stats_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['指标', '数值', '单位'])
                for key, value in stats.items():
                    unit = ''
                    if 'rate' in key:
                        unit = '%'
                        value = f"{value * 100:.1f}"
                    elif 'distance' in key:
                        unit = 'km'
                        value = f"{value:.2f}"
                    elif 'minutes' in key or 'eta' in key:
                        unit = '分钟'
                        value = f"{value:.1f}"
                    elif isinstance(value, float):
                        value = f"{value:.2f}"

                    writer.writerow([key, value, unit])

        print(f"派单结果已保存到 {output_dir}")

    def ensure_ridehail_drivers(self, available_vehicles):
        """确保网约车辆都有对应的司机记录"""
        for veh_id in available_vehicles:
            if veh_id not in self.drivers:
                try:
                    # 获取车辆位置
                    x, y = traci.vehicle.getPosition(veh_id)
                    lat_center, lon_center = self.config.center_coord
                    lat = lat_center + (y / 111000)
                    lon = lon_center + (x / (111000 * 0.8))

                    # 创建司机记录
                    self.drivers[veh_id] = Driver(
                        driver_id=veh_id,
                        lat=lat,
                        lon=lon,
                        status='idle',
                        rating=round(random.uniform(4.0, 5.0), 1),
                        speed=traci.vehicle.getSpeed(veh_id) * 3.6,
                        remaining_power=random.choice([-1, random.randint(20, 100)]),
                        rejection_list=[],
                        car_plate=f"模拟{random.randint(10000, 99999)}"
                    )
                except:
                    continue

    def generate_dynamic_orders(self, current_step, available_vehicles):
        """根据当前仿真状态动态生成订单"""
        # 确保所有网约车都有司机记录
        self.ensure_ridehail_drivers(available_vehicles)

        # 根据时间段、空闲司机数量等生成适量订单
        num_orders = self.calculate_order_demand(current_step, len(available_vehicles))

        for i in range(num_orders):
            order = self.create_random_order(f"dynamic_order_{current_step}_{i}")
            self.add_dynamic_order(order)

            # 立即尝试派单
            self.dispatch_order(order['order_id'])

    def should_pickup_passenger(self, driver_id):
        """判断司机是否应该接客（到达接客地点）"""
        if driver_id not in self.drivers:
            return False

        driver = self.drivers[driver_id]
        if driver.status != 'busy':
            return False

        # 检查是否有当前派单且接近接客地点
        current_dispatch = self.get_current_dispatch(driver_id)
        if current_dispatch:
            pickup_distance = self.calculate_distance_km(
                driver.lat, driver.lon,
                current_dispatch['pickup_lat'], current_dispatch['pickup_lon']
            )
            return pickup_distance < 0.1  # 100米内算到达

        return False

    def calculate_order_demand(self, current_step, available_vehicles_count):
        """根据当前时间和可用车辆数量计算订单需求"""
        # 基于时间的需求波动
        hour = (current_step / 3600) % 24

        # 早晚高峰需求较高
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            demand_factor = 2.0
        elif 10 <= hour <= 16:
            demand_factor = 1.2
        else:
            demand_factor = 0.5

        # 基于可用司机数量调整
        base_demand = max(1, available_vehicles_count // 10)  # 每10个司机生成1个订单

        return max(1, int(base_demand * demand_factor))

    def create_random_order(self, order_id):
        """创建随机订单，使用实际道路边"""
        try:
            # 获取当前仿真中的道路边
            edge_list = traci.edge.getIDList()
            suitable_edges = [e for e in edge_list if not e.startswith(':')]

            if len(suitable_edges) < 2:
                # 回退到坐标生成
                return self._create_coordinate_based_order(order_id)

            origin_edge = random.choice(suitable_edges)
            destination_edge = random.choice(suitable_edges)

            while destination_edge == origin_edge and len(suitable_edges) > 1:
                destination_edge = random.choice(suitable_edges)

            # 获取边的坐标作为经纬度近似
            try:
                origin_pos = traci.edge.getPosition(origin_edge, 0)
                dest_pos = traci.edge.getPosition(destination_edge, 0)

                lat_center, lon_center = self.config.center_coord
                origin_lat = lat_center + (origin_pos[1] / 111000)
                origin_lon = lon_center + (origin_pos[0] / (111000 * 0.8))
                dest_lat = lat_center + (dest_pos[1] / 111000)
                dest_lon = lon_center + (dest_pos[0] / (111000 * 0.8))

            except:
                return self._create_coordinate_based_order(order_id)

            # 计算价格
            distance = ((dest_lat - origin_lat) ** 2 + (dest_lon - origin_lon) ** 2) ** 0.5 * 111000
            price = max(15, int(distance * 0.02) + random.randint(-5, 10))

            return {
                'order_id': order_id,
                'price': price,
                'origin_lat': origin_lat,
                'origin_lon': origin_lon,
                'destination_lat': dest_lat,
                'destination_lon': dest_lon,
                'pickup_lat': origin_lat,
                'pickup_lon': origin_lon,
                'origin_edge': origin_edge,
                'destination_edge': destination_edge,
                'preference_list': [],
                'member_id': f"dynamic_member_{len(self.orders)}",
                'order_pref': {
                    'car_type': random.choice(['', 'economy', 'comfort']),
                    'specified_driver': False,
                    'is_test': True
                }
            }
        except:
            return self._create_coordinate_based_order(order_id)

    def _create_coordinate_based_order(self, order_id):
        """基于坐标创建订单（回退方案）"""
        lat_center, lon_center = self.config.center_coord
        radius_deg = self.config.radius_meters / 111000

        origin_lat = lat_center + random.uniform(-radius_deg, radius_deg)
        origin_lon = lon_center + random.uniform(-radius_deg, radius_deg)
        dest_lat = lat_center + random.uniform(-radius_deg, radius_deg)
        dest_lon = lon_center + random.uniform(-radius_deg, radius_deg)

        distance = ((dest_lat - origin_lat) ** 2 + (dest_lon - origin_lon) ** 2) ** 0.5 * 111000
        price = max(15, int(distance * 0.02) + random.randint(-5, 10))

        return {
            'order_id': order_id,
            'price': price,
            'origin_lat': origin_lat,
            'origin_lon': origin_lon,
            'destination_lat': dest_lat,
            'destination_lon': dest_lon,
            'pickup_lat': origin_lat,
            'pickup_lon': origin_lon,
            'preference_list': [],
            'member_id': f"dynamic_member_{len(self.orders)}",
            'order_pref': {
                'car_type': random.choice(['', 'economy', 'comfort']),
                'specified_driver': False,
                'is_test': True
            }
        }
    def get_current_dispatch(self, driver_id):
        """获取司机当前的派单信息"""
        # 从dispatch_records中查找司机最近的派单
        for record in reversed(self.dispatch_records):
            if record['driver_id'] == driver_id:
                # 查找对应的订单
                order_id = record['order_id']
                if order_id in self.orders:
                    order = self.orders[order_id]
                    return {
                        'order_id': order_id,
                        'pickup_lat': order.origin_lat,
                        'pickup_lon': order.origin_lon,
                        'destination_lat': order.destination_lat,
                        'destination_lon': order.destination_lon
                    }
        return None
    # TODO: 以下是预留的优化接口，供后续扩展
    def optimize_matching_algorithm(self):
        """优化匹配算法接口 - 待实现"""
        pass

    def implement_surge_pricing(self):
        """实现动态定价接口 - 待实现"""
        pass

    def add_traffic_condition_factor(self):
        """添加交通状况因子接口 - 待实现"""
        pass

    def implement_driver_preference_learning(self):
        """实现司机偏好学习接口 - 待实现"""
        pass