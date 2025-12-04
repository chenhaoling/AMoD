
# src/terminal_parking.py
"""私家车终点停车场模块 - 管理私家车在目的地的停车位"""

import traci
import traci.constants
import random
import math
from typing import Dict, List, Set, Tuple, Optional


class TerminalParkingManager:
    """私家车终点停车场管理器"""

    def __init__(self, config):
        self.config = config
        self.parked_vehicles = set()  # 已停入停车场的车辆
        self.searching_vehicles: Dict[str, str] = {}  # 车辆ID -> 目标停车场ID
        self.sumo_parking_areas: Dict[str, Dict] = {}  # 存储由simulation_setup生成的SUMO停车场信息
        self.parking_spot_numbers: Dict[str, int] = {}  # 停车位ID -> 编号
        self.parking_records: Dict[int, List[str]] = {}  # 停车位编号 -> 停过的车辆列表

    def set_sumo_parking_areas(self, parking_info: Dict):
        """从主程序接收SUMO停车场信息"""
        self.sumo_parking_areas = parking_info

        # 按道路和位置对停车位编号
        edge_spots = {}
        for spot_id, info in parking_info.items():
            edge_id = info['edge_id']
            position = info['position']
            if edge_id not in edge_spots:
                edge_spots[edge_id] = []
            edge_spots[edge_id].append((position, spot_id))

        # 每条道路上的停车位按位置排序并编号
        global_number = 1
        for edge_id in sorted(edge_spots.keys()):
            spots = sorted(edge_spots[edge_id])
            for _, spot_id in spots:
                self.parking_spot_numbers[spot_id] = global_number
                self.parking_records[global_number] = []
                global_number += 1

        print(f"✓ 终点停车管理器已接收 {len(parking_info)} 个停车位的信息。")
        print(f"✓ 停车位已按道路位置编号（1-{global_number - 1}）")

    def find_and_assign_parking(self, veh_id: str, destination_edge: str):
        """为车辆寻找并分配一个停车位"""
        if veh_id in self.searching_vehicles or veh_id in self.parked_vehicles:
            return

        # 1. 优先在目的地道路上寻找停车位
        spot_id = self._find_parking_on_edge(destination_edge)

        # 2. 如果目的地没有,则在附近寻找最近的
        if not spot_id:
            spot_id = self._find_closest_available_spot(veh_id)

        if spot_id:
            target_edge = self.sumo_parking_areas[spot_id]['edge_id']
            try:
                # 新增：可达性预检查
                curr_edge = traci.vehicle.getRoadID(veh_id)
                vtype = traci.vehicle.getTypeID(veh_id)
                route = traci.simulation.findRoute(curr_edge, target_edge, vType=vtype)

                if route.length > 0:
                    # 路径可达,执行changeTarget
                    traci.vehicle.changeTarget(veh_id, target_edge)
                    self.searching_vehicles[veh_id] = spot_id
                    traci.vehicle.setColor(veh_id, (255, 255, 0, 255))
                else:
                    # 不可达,放弃该车位
                    return
            except traci.TraCIException:
                return

    def _find_parking_on_edge(self, edge_id: str) -> Optional[str]:
        """在指定的道路上寻找一个可用的停车位"""
        spots_on_edge = [
            pid for pid, info in self.sumo_parking_areas.items() if info['edge_id'] == edge_id
        ]
        random.shuffle(spots_on_edge)  # 随机化选择，避免拥堵

        for spot_id in spots_on_edge:
            try:
                info = self.sumo_parking_areas[spot_id]
                if traci.parkingarea.getVehicleCount(spot_id) < info['capacity']:
                    return spot_id
            except traci.TraCIException:
                continue
        return None

    def _find_closest_available_spot(self, veh_id: str, search_radius: float = 1000.0) -> Optional[str]:
        """在车辆附近寻找最近的可用停车位"""
        try:
            x, y = traci.vehicle.getPosition(veh_id)
        except traci.TraCIException:
            return None

        available_spots = []
        for spot_id, info in self.sumo_parking_areas.items():
            try:
                if traci.parkingarea.getVehicleCount(spot_id) < info['capacity']:
                    # 简化距离计算，使用停车场的第一个坐标点
                    spot_pos = traci.lane.getShape(info['lane_id'])[0]
                    dist = math.sqrt((x - spot_pos[0]) ** 2 + (y - spot_pos[1]) ** 2)
                    if dist < search_radius:
                        available_spots.append((dist, spot_id))
            except traci.TraCIException:
                continue

        if not available_spots:
            return None

        # 返回距离最近的那个
        available_spots.sort(key=lambda item: item[0])
        return available_spots[0][1]

    def execute_parking_for_arrived_vehicle(self, veh_id: str):
        """当车辆到达目标停车场道路后,执行停车"""
        if veh_id not in self.searching_vehicles:
            return

        target_spot_id = self.searching_vehicles[veh_id]

        try:
            current_edge = traci.vehicle.getRoadID(veh_id)
            target_info = self.sumo_parking_areas.get(target_spot_id)

            if not target_info:
                return

            target_edge = target_info['edge_id']

            # 确认车辆已在正确的道路上
            if current_edge == target_edge:
                # === 新增：检测是否距离过近导致无法刹车 (Too close to brake) ===
                current_pos = traci.vehicle.getLanePosition(veh_id)
                current_speed = traci.vehicle.getSpeed(veh_id)
                try:
                    decel = traci.vehicle.getDecel(veh_id)
                except:
                    decel = 4.0  # 默认值

                parking_start_pos = target_info['position']

                # 计算需要的刹车距离: d = v^2 / (2a)
                required_braking_dist = (current_speed ** 2) / (2 * decel) if decel > 0 else 0

                # 如果当前位置 + 刹车距离 > 停车位起始位置，说明来不及刹车
                # 预留 5 米的安全缓冲
                if current_pos + required_braking_dist + 5.0 > parking_start_pos:
                    # 距离不足，放弃本次停车操作（车辆会驶过，稍后可能会重新寻找或循环）
                    # 移除寻找状态，避免每帧报错
                    del self.searching_vehicles[veh_id]
                    return
                # =========================================================

                # 新增：预减速避免刹停距离不足
                try:
                    if current_speed > 3.0:
                        traci.vehicle.slowDown(veh_id, 3.0, 2)
                    else:
                        traci.vehicle.slowDown(veh_id, 0.5, 2)
                except traci.TraCIException:
                    pass

                traci.vehicle.setParkingAreaStop(
                    vehID=veh_id,
                    stopID=target_spot_id,
                    duration=self.config.terminal_parking_duration
                )
                # 记录停车
                spot_number = self.parking_spot_numbers.get(target_spot_id)
                if spot_number:
                    self.parking_records[spot_number].append(veh_id)

                # 更新状态
                self.parked_vehicles.add(veh_id)
                del self.searching_vehicles[veh_id]

        except traci.TraCIException as e:
            # 如果出错,例如车位被抢占或距离计算边缘,则放弃
            if veh_id in self.searching_vehicles:
                del self.searching_vehicles[veh_id]

    def get_parking_lot_statistics(self) -> Dict:
        """获取停车场统计信息"""
        occupied_spots = 0
        total_spots = sum(info.get('capacity', 0) for info in self.sumo_parking_areas.values())

        try:
            if traci.isLoaded():
                for parking_id in self.sumo_parking_areas:
                    occupied_spots += traci.parkingarea.getVehicleCount(parking_id)
        except traci.TraCIException:
            # 仿真结束时可能查询失败，使用内部记录估算
            occupied_spots = len(self.parked_vehicles)

        # 统计停车记录
        used_spots = sum(1 for vehicles in self.parking_records.values() if len(vehicles) > 0)

        return {
            'total_parking_spots': total_spots,
            'occupied_parking_spots': occupied_spots,
            'parked_in_lots': len(self.parked_vehicles),
            'searching_vehicles': len(self.searching_vehicles),
            'available_spots': max(0, total_spots - occupied_spots),
            'used_spots_count': used_spots
        }