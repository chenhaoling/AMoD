# src/ridehail_coordinator.py

import random
import traci
from dispatch_manager import DispatchManager
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class RideOrder:
    """网约车订单"""
    order_id: str
    pickup_edge: str
    destination_edge: str
    pickup_time: float
    passenger_count: int = 1
    status: str = 'waiting'  # waiting, assigned, completed
    assigned_vehicle: Optional[str] = None
    match_time: Optional[float] = None  # 新增
class OrderGenerator:
    """订单生成器"""

    def __init__(self, config):
        self.config = config
        self.order_counter = 0
        self.quiet_mode = True  # 可以用来控制 debug 输出

    def _is_edge_allowed_for_vtype(self, edge_id: str, vtype: str) -> bool:
        """检查某条edge是否允许指定vType出发"""
        try:
            lane_ids = traci.edge.getLaneIDs(edge_id)
            for lane_id in lane_ids:
                allowed = traci.lane.getAllowed(lane_id)
                disallowed = traci.lane.getDisallowed(lane_id)

                # 修改：空allowed表示所有类型可走
                if (not allowed) or (vtype in allowed):
                    if vtype not in disallowed:
                        return True
            return False
        except traci.TraCIException:
            return False

    def generate_random_order(self, current_step) -> Optional[RideOrder]:
        """生成随机订单 - 带合法性检查"""
        try:
            edge_list = traci.edge.getIDList()
            suitable_edges = [
                e for e in edge_list
                if not e.startswith(':')
                and len(e) > 3
                and self._is_edge_allowed_for_vtype(e, 'ridehail_car')
            ]

            if not suitable_edges or len(suitable_edges) < 2:
                if not self.quiet_mode:
                    print("[WARN] 没有足够的合法边可供生成订单")
                return None

            # 最多尝试10次找到可达路径
            for _ in range(10):
                pickup_edge = random.choice(suitable_edges)
                destination_edge = random.choice(suitable_edges)

                while destination_edge == pickup_edge and len(suitable_edges) > 1:
                    destination_edge = random.choice(suitable_edges)

                # 验证路径可达性
                try:
                    route = traci.simulation.findRoute(
                        pickup_edge, destination_edge, vType='ridehail_car'
                    )
                    if route.length > 0:
                        self.order_counter += 1
                        return RideOrder(
                            order_id=f"order_{current_step}_{self.order_counter}",
                            pickup_edge=pickup_edge,
                            destination_edge=destination_edge,
                            pickup_time=current_step,
                            passenger_count=random.randint(1, 2)
                        )
                except traci.TraCIException as e:
                    if not self.quiet_mode:
                        print(f"[WARN] 路径验证失败: {pickup_edge} -> {destination_edge}, {e}")
                    continue

            return None  # 10次尝试都失败
        except Exception as e:
            if not self.quiet_mode:
                print(f"[ERROR] 订单生成失败: {e}")
            return None


class ParkingDecisionMaker:
    """停车决策模块"""

    def __init__(self, config):
        self.config = config
        self.vehicle_decisions = {}

    def make_parking_decision(self, vehicle_id: str, current_edge: str) -> Dict:
        """智能停车决策"""
        # 简单决策逻辑
        decision_value = random.random()

        if decision_value < 0.4:
            # 40%概率就地等待
            return {
                'action': 'wait_here',
                'duration': random.randint(60, 300),
                'reason': 'wait_for_orders'
            }
        elif decision_value < 0.7:
            # 30%概率短暂停车
            return {
                'action': 'park_briefly',
                'duration': random.randint(30, 120),
                'reason': 'brief_rest'
            }
        else:
            # 30%概率巡游到其他区域
            return {
                'action': 'cruise_to_area',
                'target_location': self._select_target_area(),
                'reason': 'seek_demand'
            }

    def _select_target_area(self) -> str:
        """选择目标巡游区域"""
        try:
            edge_list = traci.edge.getIDList()
            suitable_edges = [e for e in edge_list if not e.startswith(':')]
            return random.choice(suitable_edges) if suitable_edges else "default_area"
        except:
            return "default_area"


class RidehailCoordinator:
    """网约车协调器 - 从预生成订单发出"""

    def __init__(self, config):
        self.config = config
        self.dispatch_manager = DispatchManager(config)
        self.parking_decision_maker = ParkingDecisionMaker(config)

        # 状态管理
        self.active_orders = {}
        self.ridehail_vehicles = {}
        self.vehicle_orders = {}
        # 统计
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_matches_made = 0

        # 等候时长追踪
        self.wait_times = []
        self.recent_wait_times = []
        self.max_recent_samples = 100


        self.debug_printed = False
        self.quiet_mode = True
        self.parking_detector = None

        # 新增：预生成订单队列
        self.scheduled_orders = []
        self.next_order_index = 0


    def load_scheduled_orders(self, orders_file):
        """加载预生成的订单"""
        try:
            import pickle
            with open(orders_file, 'rb') as f:
                demands = pickle.load(f)

            # 转换为RideOrder并按时间排序
            for demand in demands:
                order = RideOrder(
                    order_id=demand.demand_id,
                    pickup_edge=demand.origin_edge,
                    destination_edge=demand.destination_edge,
                    pickup_time=demand.departure_time,
                    passenger_count=1
                )
                self.scheduled_orders.append(order)

            self.scheduled_orders.sort(key=lambda x: x.pickup_time)
            print(f"加载了 {len(self.scheduled_orders)} 个预生成订单")
            return True
        except Exception as e:
            print(f"加载订单失败: {e}")
            return False

    def update_ridehail_vehicles(self):
        """更新网约车列表"""
        try:
            current_vehicles = traci.vehicle.getIDList()
            for veh_id in current_vehicles:
                veh_type = traci.vehicle.getTypeID(veh_id)
                is_ridehail = (
                        veh_type == 'ridehail_car' or
                        veh_id.startswith('ridehail_car_') or
                        'ridehail' in veh_id.lower()
                )
                if is_ridehail and veh_id not in self.ridehail_vehicles:
                    self.ridehail_vehicles[veh_id] = 'idle'
        except traci.TraCIException:
            pass

    def release_orders_for_step(self, current_step) -> List[RideOrder]:
        """按时间发出预生成的订单"""
        orders = []

        while (self.next_order_index < len(self.scheduled_orders) and
               self.scheduled_orders[self.next_order_index].pickup_time <= current_step):
            order = self.scheduled_orders[self.next_order_index]
            orders.append(order)
            self.active_orders[order.order_id] = order
            self.total_orders_generated += 1
            self.next_order_index += 1

        return orders

    def dispatch_orders(self, orders: List[RideOrder], current_step: float) -> int:  # 添加 current_step 参数
        """派发订单"""
        successful_matches = 0

        for order in orders:
            available_vehicles = [
                veh_id for veh_id, status in self.ridehail_vehicles.items()
                if status == 'idle'
            ]

            if available_vehicles:
                selected_vehicle = self._find_reachable_vehicle(order, available_vehicles)

                if selected_vehicle:
                    order.status = 'assigned'
                    order.assigned_vehicle = selected_vehicle
                    order.match_time = current_step  # 现在 current_step 已定义

                    self.ridehail_vehicles[selected_vehicle] = 'busy'
                    self.vehicle_orders[selected_vehicle] = order.order_id
                    successful_matches += 1
                    self.total_matches_made += 1

                    # 修正：等待时间 = 匹配时刻 - 订单生成时刻
                    wait_time = order.match_time - order.pickup_time
                    self.wait_times.append(wait_time)
                    self.recent_wait_times.append(wait_time)
                    if len(self.recent_wait_times) > self.max_recent_samples:
                        self.recent_wait_times.pop(0)

                    self._send_vehicle_to_pickup(selected_vehicle, order)
                else:
                    if order.order_id in self.active_orders:
                        del self.active_orders[order.order_id]
            else:
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]

        return successful_matches

    def _find_reachable_vehicle(self, order: RideOrder, available_vehicles: List[str]) -> Optional[str]:
        """找到能完成整个行程的车辆"""
        random.shuffle(available_vehicles)

        for vehicle_id in available_vehicles:
            try:
                if vehicle_id not in traci.vehicle.getIDList():
                    continue

                current_edge = traci.vehicle.getRoadID(vehicle_id)

                if current_edge != order.pickup_edge:
                    route1 = traci.simulation.findRoute(current_edge, order.pickup_edge, vType='ridehail_car')
                    if route1.length <= 0:
                        continue

                route2 = traci.simulation.findRoute(order.pickup_edge, order.destination_edge, vType='ridehail_car')
                if route2.length <= 0:
                    continue

                return vehicle_id

            except traci.TraCIException:
                continue

        return None

    def _send_vehicle_to_pickup(self, vehicle_id: str, order: RideOrder):
        """引导车辆到接客点"""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return False

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge == order.pickup_edge:
                return True

            traci.vehicle.changeTarget(vehicle_id, order.pickup_edge)
            return True

        except traci.TraCIException:
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            if vehicle_id in self.vehicle_orders:
                del self.vehicle_orders[vehicle_id]
            if vehicle_id in self.ridehail_vehicles:
                self.ridehail_vehicles[vehicle_id] = 'idle'
            return False

    def check_order_completion(self, current_step):
        """检查订单完成情况"""
        for vehicle_id, order_id in list(self.vehicle_orders.items()):
            if order_id in self.active_orders:
                order = self.active_orders[order_id]

                try:
                    if vehicle_id not in traci.vehicle.getIDList():
                        if vehicle_id in self.vehicle_orders:
                            del self.vehicle_orders[vehicle_id]
                        if vehicle_id in self.ridehail_vehicles:
                            del self.ridehail_vehicles[vehicle_id]
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                        continue

                    current_edge = traci.vehicle.getRoadID(vehicle_id)
                    speed = traci.vehicle.getSpeed(vehicle_id)
                    order_duration = current_step - order.pickup_time

                    completion_condition = (
                            current_edge == order.destination_edge or
                            order_duration > 300 or
                            (speed < 2.0 and random.random() < 0.4)
                    )

                    if completion_condition:
                        order.status = 'completed'
                        self.ridehail_vehicles[vehicle_id] = 'idle'
                        del self.vehicle_orders[vehicle_id]
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                        self.total_orders_completed += 1

                        if self.parking_detector:
                            self.parking_detector.trigger_ridehail_parking(vehicle_id, current_step)

                except traci.TraCIException:
                    if vehicle_id in self.vehicle_orders:
                        del self.vehicle_orders[vehicle_id]
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]

    def step_update(self, current_step):
        """每步更新"""
        self.update_ridehail_vehicles()
        new_orders = self.release_orders_for_step(current_step)
        if new_orders:
            self.dispatch_orders(new_orders, current_step)  # 传入 current_step
        self.check_order_completion(current_step)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        idle_count = len([v for v, s in self.ridehail_vehicles.items() if s == 'idle'])
        busy_count = len([v for v, s in self.ridehail_vehicles.items() if s == 'busy'])

        completion_rate = 0
        if self.total_orders_generated > 0:
            completion_rate = (self.total_orders_completed / self.total_orders_generated) * 100

        match_rate = 0
        if self.total_orders_generated > 0:
            match_rate = (self.total_matches_made / self.total_orders_generated) * 100

        overall_avg_wait = 0
        if self.wait_times:
            overall_avg_wait = sum(self.wait_times) / len(self.wait_times)

        recent_avg_wait = 0
        if self.recent_wait_times:
            recent_avg_wait = sum(self.recent_wait_times) / len(self.recent_wait_times)

        return {
            'total_ridehail_vehicles': len(self.ridehail_vehicles),
            'idle_vehicles': idle_count,
            'busy_vehicles': busy_count,
            'active_orders': len(self.active_orders),
            'total_orders_generated': self.total_orders_generated,
            'total_orders_completed': self.total_orders_completed,
            'total_matches_made': self.total_matches_made,
            'completion_rate': completion_rate,
            'match_rate': match_rate,
            'recent_avg_wait_time_minutes': recent_avg_wait ,
            'overall_avg_wait_time_minutes': overall_avg_wait
        }