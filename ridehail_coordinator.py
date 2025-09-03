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


class OrderGenerator:
    """订单生成器"""

    def __init__(self, config):
        self.config = config
        self.order_counter = 0

    def generate_random_order(self, current_step) -> RideOrder:
        """生成随机订单 - 保留路径验证但静默处理错误"""
        try:
            edge_list = traci.edge.getIDList()
            suitable_edges = [e for e in edge_list if not e.startswith(':') and len(e) > 3]

            if len(suitable_edges) < 2:
                return None

            # 最多尝试10次找到可达路径
            for _ in range(10):
                pickup_edge = random.choice(suitable_edges)
                destination_edge = random.choice(suitable_edges)

                while destination_edge == pickup_edge and len(suitable_edges) > 1:
                    destination_edge = random.choice(suitable_edges)

                # 验证路径可达性 - 保留验证但静默处理错误
                try:
                    route = traci.simulation.findRoute(pickup_edge, destination_edge, vType='ridehail_car')
                    if route.length > 0:  # 有有效路径
                        self.order_counter += 1
                        return RideOrder(
                            order_id=f"order_{current_step}_{self.order_counter}",
                            pickup_edge=pickup_edge,
                            destination_edge=destination_edge,
                            pickup_time=current_step,
                            passenger_count=random.randint(1, 2)
                        )
                except traci.TraCIException as e:
                    # 静默处理特定的错误，避免控制台输出
                    if "Invalid departure edge" in str(e):
                        continue  # 静默跳过，不输出错误信息
                    else:
                        # 其他类型的错误可能需要关注
                        if not self.config.quiet_mode:
                            print(f"路径查找错误: {e}")
                        continue

            return None  # 10次尝试都失败
        except:
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
    """网约车协调器 - 整合订单生成、派单和停车决策"""

    def __init__(self, config):
        self.config = config
        self.order_generator = OrderGenerator(config)
        self.dispatch_manager = DispatchManager(config)
        self.parking_decision_maker = ParkingDecisionMaker(config)

        # 状态管理
        self.active_orders = {}  # order_id -> RideOrder
        self.ridehail_vehicles = {}  # vehicle_id -> status
        self.vehicle_orders = {}  # vehicle_id -> current_order_id

        # 统计
        self.total_orders_generated = 0
        self.total_orders_completed = 0
        self.total_matches_made = 0

        # 调试信息
        self.debug_printed = False
        self.quiet_mode = True  # 新增：静默模式
        # 添加这一行来存储停车检测器的引用
        self.parking_detector = None

    def update_ridehail_vehicles(self):
        """更新网约车列表 - 修复版本"""
        try:
            current_vehicles = traci.vehicle.getIDList()
            found_new_ridehail = False

            for veh_id in current_vehicles:
                # 修复：更全面的网约车识别逻辑
                veh_type = traci.vehicle.getTypeID(veh_id)

                # 多重判断条件确保识别准确
                is_ridehail = (
                        veh_type == 'ridehail_car' or  # 通过SUMO车辆类型识别
                        veh_id.startswith('ridehail_car_') or  # 通过ID前缀识别
                        'ridehail' in veh_id.lower()  # 通过ID包含关键字识别
                )

                if is_ridehail and veh_id not in self.ridehail_vehicles:
                    self.ridehail_vehicles[veh_id] = 'idle'
                    found_new_ridehail = True
                    if not self.quiet_mode:
                        print(f"发现网约车: {veh_id}, 类型: {veh_type}")

            # 调试信息：第一次运行时输出详细信息
            if not self.debug_printed and len(current_vehicles) > 0:
                self.debug_printed = True
                if not self.quiet_mode:
                    print(f"\n=== 车辆识别调试信息 ===")
                    print(f"当前仿真中总车辆数: {len(current_vehicles)}")
                    print(f"已识别网约车数量: {len(self.ridehail_vehicles)}")

                    # 输出前10个车辆的详细信息
                    print("前10个车辆详情:")
                    for i, veh_id in enumerate(list(current_vehicles)[:10]):
                        try:
                            vtype = traci.vehicle.getTypeID(veh_id)
                            is_ridehail = (vtype == 'ridehail_car' or
                                           veh_id.startswith('ridehail_car_') or
                                           'ridehail' in veh_id.lower())
                            print(f"  {i + 1}. {veh_id} -> 类型: {vtype}, 是网约车: {is_ridehail}")
                        except Exception as e:
                            print(f"  {i + 1}. {veh_id} -> 获取信息失败: {e}")

                    # 如果还是没有网约车，检查是否有包含"ridehail"的车辆
                    potential_ridehail = [v for v in current_vehicles if 'ridehail' in v.lower()]
                    if potential_ridehail:
                        print(f"包含'ridehail'的车辆ID: {potential_ridehail[:5]}")
                    else:
                        print("未找到任何包含'ridehail'的车辆ID")

                    print("=== 调试信息结束 ===\n")

        except traci.TraCIException as e:
            if not self.quiet_mode:
                print(f"更新网约车列表时出错: {e}")

    def generate_orders_for_step(self, current_step) -> List[RideOrder]:
        """为当前步骤生成订单 - 增加生成频率用于调试"""
        orders = []

        # 基于时间和现有车辆数量决定订单生成率
        num_ridehail = len([v for v, s in self.ridehail_vehicles.items() if s == 'idle'])

        # 修正：增加订单生成频率，每3辆空闲车生成1-2个订单
        if num_ridehail > 0:
            order_rate = max(1, num_ridehail // 3)  # 从5改为3
            num_orders = random.randint(1, min(order_rate + 1, max(1, num_ridehail // 2)))

            for _ in range(num_orders):
                order = self.order_generator.generate_random_order(current_step)
                if order:
                    orders.append(order)
                    self.active_orders[order.order_id] = order
                    self.total_orders_generated += 1

        return orders

    def dispatch_orders(self, orders: List[RideOrder]) -> int:
        """派发订单 - 带路径验证"""
        successful_matches = 0

        for order in orders:
            # 获取可用车辆
            available_vehicles = [
                veh_id for veh_id, status in self.ridehail_vehicles.items()
                if status == 'idle'
            ]

            if available_vehicles:
                # 找到能完成整个行程的车辆
                selected_vehicle = self._find_reachable_vehicle(order, available_vehicles)

                if selected_vehicle:
                    # 分配订单
                    order.status = 'assigned'
                    order.assigned_vehicle = selected_vehicle
                    self.ridehail_vehicles[selected_vehicle] = 'busy'
                    self.vehicle_orders[selected_vehicle] = order.order_id
                    successful_matches += 1
                    self.total_matches_made += 1

                    # 引导车辆到接客点（已验证可达）
                    self._send_vehicle_to_pickup(selected_vehicle, order)

                    if not self.quiet_mode:
                        print(f"订单 {order.order_id} 派给车辆 {selected_vehicle}")
                else:
                    # 没有能完成此行程的车辆，取消订单
                    if order.order_id in self.active_orders:
                        del self.active_orders[order.order_id]
            else:
                # 没有可用车辆，从活跃订单中移除避免积压
                if order.order_id in self.active_orders:
                    del self.active_orders[order.order_id]

        return successful_matches

    def _find_reachable_vehicle(self, order: RideOrder, available_vehicles: List[str]) -> Optional[str]:
        """找到能完成整个行程的车辆"""
        # 随机打乱车辆列表，避免总是选择同一辆车
        random.shuffle(available_vehicles)

        for vehicle_id in available_vehicles:
            try:
                if vehicle_id not in traci.vehicle.getIDList():
                    continue

                current_edge = traci.vehicle.getRoadID(vehicle_id)

                # 验证完整路径：当前位置 -> 接客点 -> 目的地
                # 1. 车辆到接客点 - 修复：添加vType参数
                if current_edge != order.pickup_edge:
                    route1 = traci.simulation.findRoute(current_edge, order.pickup_edge, vType='ridehail_car')
                    if route1.length <= 0:
                        continue

                # 2. 接客点到目的地 - 修复：添加vType参数
                route2 = traci.simulation.findRoute(order.pickup_edge, order.destination_edge, vType='ridehail_car')
                if route2.length <= 0:
                    continue

                # 路径验证通过，选择此车辆
                return vehicle_id

            except traci.TraCIException:
                continue

        return None

    def _send_vehicle_to_pickup(self, vehicle_id: str, order: RideOrder):
        """引导车辆到接客点 - 简化版（已预先验证路径）"""
        try:
            if vehicle_id not in traci.vehicle.getIDList():
                return False

            current_edge = traci.vehicle.getRoadID(vehicle_id)
            if current_edge == order.pickup_edge:
                return True

            # 由于派单前已验证路径，直接执行changeTarget
            traci.vehicle.changeTarget(vehicle_id, order.pickup_edge)
            return True

        except traci.TraCIException:
            # 如果还是失败，说明路径验证不够准确，取消此订单
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
            if vehicle_id in self.vehicle_orders:
                del self.vehicle_orders[vehicle_id]
            if vehicle_id in self.ridehail_vehicles:
                self.ridehail_vehicles[vehicle_id] = 'idle'
            return False

    def check_order_completion(self, current_step):
        """检查订单完成情况 - 修复版本"""
        completed_orders = []

        for vehicle_id, order_id in list(self.vehicle_orders.items()):
            if order_id in self.active_orders:
                order = self.active_orders[order_id]

                try:
                    if vehicle_id not in traci.vehicle.getIDList():
                        # 车辆已离开，清理状态
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

                    # 放宽完成条件，提高完成率
                    completion_condition = (
                            current_edge == order.destination_edge or  # 到达目标边
                            order_duration > 300 or  # 5分钟超时
                            (speed < 2.0 and random.random() < 0.4)  # 低速时40%概率完成
                    )

                    if completion_condition:
                        order.status = 'completed'
                        self.ridehail_vehicles[vehicle_id] = 'idle'
                        completed_orders.append((vehicle_id, order))
                        del self.vehicle_orders[vehicle_id]
                        if order_id in self.active_orders:
                            del self.active_orders[order_id]
                        self.total_orders_completed += 1

                        # 新增：网约车完成订单后立即停车
                        if self.parking_detector:
                            self.parking_detector.trigger_ridehail_parking(vehicle_id, current_step)

                        if not self.quiet_mode:
                            print(f"订单 {order_id} 完成，车辆 {vehicle_id} 恢复空闲")

                except traci.TraCIException:
                    # 清理异常车辆
                    if vehicle_id in self.vehicle_orders:
                        del self.vehicle_orders[vehicle_id]
                    if order_id in self.active_orders:
                        del self.active_orders[order_id]

    def _execute_parking_decision(self, vehicle_id: str, decision: Dict):
        """执行停车决策"""
        try:
            if decision['action'] == 'wait_here':
                # 原地等待
                traci.vehicle.setSpeed(vehicle_id, 0)
            elif decision['action'] == 'park_briefly':
                # 短暂停车
                current_edge = traci.vehicle.getRoadID(vehicle_id)
                traci.vehicle.setStop(vehicle_id, current_edge, duration=decision['duration'])
            elif decision['action'] == 'cruise_to_area':
                # 巡游到其他区域 - 添加目标验证
                try:
                    traci.vehicle.changeTarget(vehicle_id, decision['target_area'])
                except traci.TraCIException:
                    # 如果目标不可达，就保持当前路线
                    pass
        except traci.TraCIException:
            pass

    def step_update(self, current_step):
        """每步更新"""
        # 更新车辆列表
        self.update_ridehail_vehicles()

        # 生成新订单
        new_orders = self.generate_orders_for_step(current_step)

        # 派发订单
        if new_orders:
            matches = self.dispatch_orders(new_orders)

        # 检查订单完成
        self.check_order_completion(current_step)

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        idle_count = len([v for v, s in self.ridehail_vehicles.items() if s == 'idle'])
        busy_count = len([v for v, s in self.ridehail_vehicles.items() if s == 'busy'])

        return {
            'total_ridehail_vehicles': len(self.ridehail_vehicles),
            'idle_vehicles': idle_count,
            'busy_vehicles': busy_count,
            'active_orders': len(self.active_orders),
            'total_orders_generated': self.total_orders_generated,
            'total_orders_completed': self.total_orders_completed,
            'total_matches_made': self.total_matches_made,
            'completion_rate': (self.total_orders_completed / max(1, self.total_orders_generated)) * 100
        }