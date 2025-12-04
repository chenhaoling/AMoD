# -*- coding: utf-8 -*-
# src/detector.py

import traci
import random
import csv
import os
import time
import sys
from collections import defaultdict
from emissions import EmissionCalculator

# 修复编码问题
if sys.stdout.encoding != 'utf-8':
    import codecs

    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)


class SafeParkingValidator:
    """安全停车位置验证器"""

    def __init__(self, config):
        self.config = config

    def find_safe_parking_position(self, veh_id, edge_id):
        """寻找安全的停车位置 - 改进制动距离计算"""
        try:
            # 获取车辆当前车道
            current_lane = traci.vehicle.getLaneID(veh_id)
            if not current_lane:
                return None

            # 获取车道长度
            edge_length = traci.lane.getLength(current_lane)
            current_pos = traci.vehicle.getLanePosition(veh_id)

            # 改进的安全距离计算
            speed = traci.vehicle.getSpeed(veh_id)

            # 制动距离 = 反应距离 + 制动距离 + 安全缓冲
            reaction_distance = speed * 1.5  # 1.5秒反应时间
            braking_distance = (speed * speed) / (2 * 4.0)  # 4m/s²减速度
            safety_buffer = 50  # 50米安全缓冲

            safe_distance = reaction_distance + braking_distance + safety_buffer
            safe_distance = max(100, safe_distance)  # 最小100米

            # 寻找合适位置
            min_pos = max(50, current_pos + safe_distance)
            max_pos = edge_length - 50

            if min_pos >= max_pos:
                return None

            return random.uniform(min_pos, max_pos)

        except traci.TraCIException:
            return None


class DynamicParkingDetector:
    """动态路边停车检测器（含排放计算）"""

    def __init__(self, config, parking_manager=None):
        self.config = config
        self.parking_manager = parking_manager
        self.emission_calculator = EmissionCalculator()
        self.parking_validator = SafeParkingValidator(config)  # 新增验证器

        # 车辆状态管理
        self.vehicle_states = {}
        self.planned_parking_spots = {}
        self.parked_vehicles = {}
        self.vehicle_types = {}

        # 停车候选和颜色管理
        self.parking_candidates = {}
        self.original_colors = {}
        self.parking_attempts = defaultdict(int)  # 记录停车尝试次数

        # 数据记录
        self.parking_events = []
        self.emission_records = []
        self.traffic_impacts = []

        # 调试模式
        self.debug_mode = False  # 默认关闭以减少输出

    def assign_vehicle_emission_type(self, veh_id):
        """为车辆分配排放类型"""
        if veh_id not in self.vehicle_types:
            self.vehicle_types[veh_id] = self.emission_calculator.assign_vehicle_type(veh_id)
        return self.vehicle_types[veh_id]

    def should_vehicle_park(self, veh_id):
        """随机决定车辆是否有停车需求"""
        if veh_id in self.parking_candidates or veh_id in self.parked_vehicles:
            return False
        return random.random() < (self.config.roadside_parking_ratio)

    def store_original_color(self, veh_id):
        """存储车辆原始颜色"""
        if veh_id not in self.original_colors:
            self.original_colors[veh_id] = (100, 149, 237, 255)  # 蓝色
            try:
                traci.vehicle.setColor(veh_id, self.original_colors[veh_id])
            except traci.TraCIException:
                pass

    def restore_original_color(self, veh_id):
        """恢复车辆原始颜色"""
        try:
            if veh_id in self.original_colors:
                traci.vehicle.setColor(veh_id, self.original_colors[veh_id])
        except traci.TraCIException:
            pass

    def execute_safe_parking(self, veh_id, step):
        """安全执行停车命令 - 只修改制动距离部分"""
        try:
            # 检查车辆是否还在仿真中
            if veh_id not in traci.vehicle.getIDList():
                return False

            current_edge = traci.vehicle.getRoadID(veh_id)
            current_lane = traci.vehicle.getLaneID(veh_id)

            # 使用改进的验证器寻找安全的停车位置
            target_pos = self.parking_validator.find_safe_parking_position(veh_id, current_edge)

            if target_pos is None:
                if self.debug_mode:
                    print(f"车辆 {veh_id} 无法找到安全停车位置")
                return False

            # 计算停车时长
            duration = random.randint(
                self.config.parking_duration_min,
                self.config.parking_duration_max
            )

            lane_index = int(current_lane.split('_')[-1]) if '_' in current_lane else 0

            # 存储原始颜色
            self.store_original_color(veh_id)

            # 渐进减速避免急刹车
            try:
                current_speed = traci.vehicle.getSpeed(veh_id)
                if current_speed > 3:
                    # 分步减速，先减到中等速度
                    traci.vehicle.slowDown(veh_id, 3.0, 2)
                    time.sleep(0.05)
                    # 再减到低速
                    traci.vehicle.slowDown(veh_id, 0.5, 2)
                else:
                    traci.vehicle.slowDown(veh_id, 0.5, 1)
            except traci.TraCIException:
                pass

            # 执行停车命令
            traci.vehicle.setStop(
                vehID=veh_id,
                edgeID=current_edge,
                pos=target_pos,
                laneIndex=lane_index,
                duration=duration,
                flags=0
            )

            # 变色为停车状态
            traci.vehicle.setColor(veh_id, (255, 165, 0, 255))  # 橙色

            # 记录停车信息
            self.parked_vehicles[veh_id] = {
                'start_time': step,
                'end_time': step + duration,
                'lane_id': current_lane,
                'edge_id': current_edge,
                'position': target_pos,
                'duration': duration
            }

            # 记录停车事件
            self.parking_events.append({
                'timestamp': step,
                'vehicle_id': veh_id,
                'edge_id': current_edge,
                'lane_id': current_lane,
                'position': target_pos,
                'event_type': 'roadside_parking_start',
                'duration': duration,
                'vehicle_type': traci.vehicle.getTypeID(veh_id),
                'emission_type': self.assign_vehicle_emission_type(veh_id)
            })

            if self.debug_mode:
                print(f"车辆 {veh_id} 成功开始停车，持续 {duration} 秒")

            return True

        except traci.TraCIException as e:
            if self.debug_mode:
                print(f"车辆 {veh_id} 停车失败: {e}")
            return False

    def manage_dynamic_parking(self, step):
        """管理动态停车 - 保持原逻辑不变"""
        try:
            current_vehicles = set(traci.vehicle.getIDList())

            # 1. 新车辆的停车决策
            for veh_id in current_vehicles:
                if (veh_id not in self.parking_candidates and
                        veh_id not in self.parked_vehicles and
                        self.parking_attempts[veh_id] < 2):  # 限制尝试次数

                    if self.should_vehicle_park(veh_id):
                        self.parking_candidates[veh_id] = step
                        if self.debug_mode:
                            print(f"车辆 {veh_id} 被标记为停车候选")

            # 2. 处理停车候选
            for veh_id in list(self.parking_candidates.keys()):
                if veh_id in current_vehicles:
                    # 等待一段随机时间后尝试停车
                    wait_time = step - self.parking_candidates[veh_id]
                    if wait_time >= random.randint(10, 30):
                        if self.execute_safe_parking(veh_id, step):
                            del self.parking_candidates[veh_id]
                        else:
                            # 停车失败，增加尝试次数
                            self.parking_attempts[veh_id] += 1
                            if self.parking_attempts[veh_id] >= 2:
                                del self.parking_candidates[veh_id]
                            else:
                                # 重新设置候选时间
                                self.parking_candidates[veh_id] = step
                else:
                    # 车辆已离开
                    if veh_id in self.parking_candidates:
                        del self.parking_candidates[veh_id]

            # 3. 处理停车结束
            finished_vehicles = []
            for veh_id, info in self.parked_vehicles.items():
                if step >= info['end_time'] or veh_id not in current_vehicles:
                    finished_vehicles.append(veh_id)

            for veh_id in finished_vehicles:
                self.end_parking(veh_id, step)

        except Exception as e:
            print(f"管理动态停车时出错: {e}")

    def trigger_ridehail_parking(self, veh_id, step):
        """触发网约车强制停车（完成订单后）"""
        if veh_id in self.parked_vehicles or veh_id in self.parking_candidates:
            return False

        # 直接执行停车，不需要随机判断
        if self.execute_safe_parking(veh_id, step):
            if self.debug_mode:
                print(f"网约车 {veh_id} 完成订单后开始停车")
            return True
        return False

    def end_parking(self, veh_id, step):
        """结束停车"""
        if veh_id in self.parked_vehicles:
            parking_info = self.parked_vehicles.pop(veh_id)
            actual_duration = step - parking_info['start_time']

            # 恢复原始颜色
            self.restore_original_color(veh_id)

            # 如果车辆还在仿真中，恢复行驶
            if veh_id in traci.vehicle.getIDList():
                try:
                    # 检查车辆是否真的在停车状态
                    if traci.vehicle.isStopped(veh_id):
                        # 先尝试移除停车命令，再resume
                        traci.vehicle.setSpeed(veh_id, -1)  # 恢复正常速度
                        traci.vehicle.resume(veh_id)
                except traci.TraCIException as e:
                    # 如果resume失败，尝试其他方法
                    try:
                        traci.vehicle.setSpeed(veh_id, -1)  # 只恢复速度
                    except traci.TraCIException:
                        pass

            # 记录停车结束事件
            self.parking_events.append({
                'timestamp': step,
                'vehicle_id': veh_id,
                'edge_id': parking_info['edge_id'],
                'lane_id': parking_info['lane_id'],
                'position': parking_info['position'],
                'event_type': 'roadside_parking_end',
                'duration': actual_duration,
                'vehicle_type': traci.vehicle.getTypeID(veh_id) if veh_id in traci.vehicle.getIDList() else 'unknown',
                'emission_type': self.vehicle_types.get(veh_id, 'unknown')
            })

            if self.debug_mode:
                print(f"车辆 {veh_id} 结束停车，实际停车 {actual_duration} 秒")

            # 清理状态
            self.cleanup_vehicle_state(veh_id)

    def cleanup_vehicle_state(self, veh_id):
        """清理车辆状态"""
        if veh_id in self.vehicle_states:
            del self.vehicle_states[veh_id]
        if veh_id in self.original_colors:
            del self.original_colors[veh_id]
        if veh_id in self.parking_attempts:
            del self.parking_attempts[veh_id]

    def calculate_vehicle_emissions(self, step):
        """计算车辆排放"""
        try:
            current_vehicles = traci.vehicle.getIDList()

            for veh_id in current_vehicles:
                try:
                    speed = traci.vehicle.getSpeed(veh_id)
                    edge_id = traci.vehicle.getRoadID(veh_id)

                    # 计算加速度
                    prev_speed = self.vehicle_states.get(veh_id, {}).get('prev_speed', speed)
                    acceleration = speed - prev_speed

                    # 更新车辆状态
                    if veh_id not in self.vehicle_states:
                        self.vehicle_states[veh_id] = {}
                    self.vehicle_states[veh_id]['prev_speed'] = speed

                    # 计算距离
                    distance_km = (speed * 1.0) / 1000

                    # 确定驾驶模式
                    vehicle_emission_type = self.assign_vehicle_emission_type(veh_id)
                    is_parking = veh_id in self.parked_vehicles
                    is_congested = speed < 2.78

                    driving_mode = self.emission_calculator.determine_driving_mode(
                        speed, acceleration, is_parking, is_congested
                    )

                    # 计算排放
                    emissions = self.emission_calculator.calculate_emissions(
                        vehicle_emission_type, distance_km, driving_mode
                    )

                    co2_equivalent = self.emission_calculator.calculate_carbon_equivalent(emissions)

                    # 记录排放数据
                    self.emission_records.append({
                        'timestamp': step,
                        'vehicle_id': veh_id,
                        'vehicle_type': traci.vehicle.getTypeID(veh_id),
                        'emission_type': vehicle_emission_type,
                        'edge_id': edge_id,
                        'speed_ms': speed,
                        'speed_kmh': speed * 3.6,
                        'acceleration': acceleration,
                        'distance_km': distance_km,
                        'driving_mode': driving_mode,
                        'is_parking': is_parking,
                        'is_congested': is_congested,
                        'co2_g': emissions['CO2'],
                        'co_g': emissions['CO'],
                        'nox_g': emissions['NOx'],
                        'hc_g': emissions['HC'],
                        'pm_g': emissions['PM'],
                        'co2_equivalent_kg': co2_equivalent
                    })

                except traci.TraCIException:
                    continue

        except Exception as e:
            print(f"计算排放时出错: {e}")

    def analyze_traffic_impact(self, step):
        """分析交通影响"""
        try:
            current_vehicles = traci.vehicle.getIDList()
            if not current_vehicles:
                return

            speeds = []
            for v in current_vehicles:
                try:
                    speeds.append(traci.vehicle.getSpeed(v))
                except traci.TraCIException:
                    continue

            if speeds:
                self.traffic_impacts.append({
                    'timestamp': step,
                    'total_vehicles': len(current_vehicles),
                    'parking_vehicles': len(self.parked_vehicles),
                    'normal_vehicles': len(current_vehicles) - len(self.parked_vehicles),
                    'avg_speed_ms': sum(speeds) / len(speeds),
                    'avg_speed_kmh': sum(speeds) / len(speeds) * 3.6,
                    'stopped_vehicles': sum(1 for s in speeds if s < 0.1),
                    'currently_parking': len(self.parked_vehicles)
                })
        except Exception as e:
            print(f"分析交通影响时出错: {e}")

    def plan_parking_spots_for_edge(self, edge_id, edge_length):
        """为道路规划停车位（保留接口兼容性）"""
        # 简化版本，不预先规划具体位置
        if edge_length > 100:
            self.planned_parking_spots[edge_id] = True

    def detect_events(self, step):
        """主检测函数"""
        self.manage_dynamic_parking(step)
        # 每5步计算一次排放，而不是每步
        if step % 5 == 0:
            self.calculate_vehicle_emissions(step)

        if step % 300 == 0:  # 每分钟分析一次交通影响
            self.analyze_traffic_impact(step)

    def get_summary(self):
        """获取统计摘要"""
        # 明确统计路边停车
        roadside_parking_starts = len([e for e in self.parking_events
                                       if e['event_type'] == 'roadside_parking_start'])

        total_co2e = 0
        total_emissions = {}
        emission_by_type = {}
        vehicle_type_counts = {'private_car': 0, 'ridehail_car': 0, 'passenger_car': 0}

        try:
            current_vehicles = traci.vehicle.getIDList()
            for veh_id in current_vehicles:
                vtype = traci.vehicle.getTypeID(veh_id)
                if vtype in vehicle_type_counts:
                    vehicle_type_counts[vtype] += 1
        except traci.TraCIException:
            # 仿真结束时可能查询失败
            pass

        if self.emission_records:
            total_co2e = sum(r['co2_equivalent_kg'] for r in self.emission_records)

            for pollutant in ['co2_g', 'co_g', 'nox_g', 'hc_g', 'pm_g']:
                total_emissions[pollutant] = sum(r[pollutant] for r in self.emission_records)

            for record in self.emission_records:
                vtype = record['emission_type']
                if vtype not in emission_by_type:
                    emission_by_type[vtype] = {'count': 0, 'co2e': 0}
                emission_by_type[vtype]['count'] += 1
                emission_by_type[vtype]['co2e'] += record['co2_equivalent_kg']

        return {
            'roadside_parking_starts': roadside_parking_starts,
            'currently_roadside_parking': len(self.parked_vehicles),
            'parking_candidates': len(self.parking_candidates),
            'total_co2_equivalent_kg': total_co2e,
            'total_emissions': total_emissions,
            'emission_by_vehicle_type': emission_by_type,
            'current_vehicle_types': vehicle_type_counts,
        }

    def save_results(self, paths):
        """保存结果文件"""
        try:
            # 保存停车事件
            if self.parking_events:
                with open(paths.parking_events_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['timestamp', 'vehicle_id', 'event_type', 'vehicle_type',
                                  'emission_type', 'edge_id', 'lane_id', 'position', 'duration']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.parking_events)

            # 保存排放详情
            if self.emission_records:
                with open(paths.emission_details_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['timestamp', 'vehicle_id', 'vehicle_type', 'emission_type',
                                  'edge_id', 'speed_ms', 'speed_kmh', 'acceleration', 'distance_km',
                                  'driving_mode', 'is_parking', 'is_congested', 'co2_g', 'co_g',
                                  'nox_g', 'hc_g', 'pm_g', 'co2_equivalent_kg']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.emission_records)

            # 保存统计摘要
            summary = self.get_summary()
            with open(paths.emission_summary_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value', 'Unit'])
                writer.writerow(['Total_CO2_Equivalent', f"{summary['total_co2_equivalent_kg']:.3f}", 'kg'])

                if summary['total_emissions']:
                    for pollutant, value in summary['total_emissions'].items():
                        writer.writerow([pollutant, f"{value:.3f}", 'g'])

                writer.writerow(['', '', ''])
                writer.writerow(['Vehicle_Type', 'CO2e_kg', 'Records'])
                for vtype, data in summary['emission_by_vehicle_type'].items():
                    writer.writerow([vtype, f"{data['co2e']:.3f}", data['count']])

            # 保存交通影响
            if self.traffic_impacts:
                with open(paths.traffic_impact_file, 'w', newline='', encoding='utf-8') as f:
                    fieldnames = ['timestamp', 'total_vehicles', 'parking_vehicles',
                                  'normal_vehicles', 'avg_speed_ms', 'avg_speed_kmh',
                                  'stopped_vehicles', 'currently_parking']
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(self.traffic_impacts)

            # 输出摘要信息
            print("\nSimulation Results Summary")
            print(f"Total roadside parking events: {summary['roadside_parking_starts']}")
            print(f"Total CO2 equivalent: {summary['total_co2_equivalent_kg']:.3f} kg")

            if summary['total_emissions']:
                print("Main emissions:")
                for pollutant, value in summary['total_emissions'].items():
                    print(f"  {pollutant}: {value:.3f} g")

            print(f"\nFiles saved to: {paths.output_dir}")

        except Exception as e:
            print(f"Error saving results: {e}")