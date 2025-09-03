# src/od_demand_generator.py

import random
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET


@dataclass
class TravelDemand:
    """出行需求"""
    demand_id: str
    origin_zone: str
    destination_zone: str
    departure_time: float
    demand_type: str  # 'private_car' 或 'ride_hailing'
    origin_edge: str = None
    destination_edge: str = None
    served: bool = False


@dataclass
class Zone:
    """交通小区"""
    zone_id: str
    center_coord: Tuple[float, float]
    edges: List[str]
    population_weight: float
    commercial_weight: float
    residential_weight: float


class ODDemandGenerator:
    """基于OD的交通需求生成器 - 适配北京全域"""

    def __init__(self, config):
        self.config = config
        self.zones = {}
        self.zone_edges = defaultdict(list)

        # 北京实际出行方式分担比例
        if config.simulation_type == "beijing":
            self.mode_split = {
                'private_car': 0.65,  # 私家车35%
                'ride_hailing': 0.35,  # 网约车12%

            }
        else:
            self.mode_split = {
                'private_car': 0.60,
                'ride_hailing': 0.40,

            }

        # 调整为只考虑私家车和网约车的比例
        total_car_ratio = self.mode_split['private_car'] + self.mode_split['ride_hailing']
        self.car_mode_split = {
            'private_car': self.mode_split['private_car'] / total_car_ratio,
            'ride_hailing': self.mode_split['ride_hailing'] / total_car_ratio
        }

        print(f"车辆出行分担比例: 私家车 {self.car_mode_split['private_car']:.1%}, "
              f"网约车 {self.car_mode_split['ride_hailing']:.1%}")

    def initialize_zones_from_network(self, road_edges: Dict):
        """从路网初始化交通小区 - 北京全域适配"""
        print("初始化交通小区...")

        if self.config.simulation_type == "beijing":
            # 北京全域：创建基于行政区划的小区
            self._create_beijing_districts(road_edges)
        else:
            # 局部区域：基于密度创建小区
            self._create_density_based_zones(road_edges)

        print(f"创建了 {len(self.zones)} 个交通小区")

        # 为小区分配边
        self._assign_edges_to_zones(road_edges)

        # 计算小区权重
        self._calculate_zone_weights()

    def _create_beijing_districts(self, road_edges: Dict):
        """创建北京市行政区划小区"""
        # 北京市主要区域及其大致中心坐标
        beijing_districts = {
            'dongcheng': (39.928, 116.418),  # 东城区
            'xicheng': (39.915, 116.380),  # 西城区
            'chaoyang': (39.921, 116.486),  # 朝阳区
            'fengtai': (39.863, 116.287),  # 丰台区
            'shijingshan': (39.906, 116.195),  # 石景山区
            'haidian': (39.959, 116.298),  # 海淀区
            'tongzhou': (39.665, 116.658),  # 通州区
            'shunyi': (40.128, 116.655),  # 顺义区
            'daxing': (39.729, 116.338),  # 大兴区
            'changping': (40.218, 116.231),  # 昌平区
            'fangshan': (39.742, 115.993),  # 房山区
            'mentougou': (39.937, 116.102),  # 门头沟区
            'pinggu': (40.144, 117.112),  # 平谷区
            'miyun': (40.372, 116.843),  # 密云区
            'yanqing': (40.465, 115.972),  # 延庆区
            'huairou': (40.324, 116.637)  # 怀柔区
        }

        for district_id, center_coord in beijing_districts.items():
            lat, lon = center_coord

            # 根据区域特点设置权重
            if district_id in ['dongcheng', 'xicheng', 'chaoyang']:
                # 城区：商业权重高
                commercial_weight = 0.9
                residential_weight = 0.7
                population_weight = 0.8
            elif district_id in ['haidian', 'fengtai', 'shijingshan']:
                # 主城区：居住和商业并重
                commercial_weight = 0.7
                residential_weight = 0.8
                population_weight = 0.75
            else:
                # 远郊区：居住权重高
                commercial_weight = 0.3
                residential_weight = 0.9
                population_weight = 0.6

            self.zones[district_id] = Zone(
                zone_id=district_id,
                center_coord=center_coord,
                edges=[],
                population_weight=population_weight,
                commercial_weight=commercial_weight,
                residential_weight=residential_weight
            )

    def _create_density_based_zones(self, road_edges: Dict, num_zones: int = 12):
        """基于道路密度创建小区（适用于局部仿真）"""
        # 简化版，在中心点周围创建同心圆小区
        lat_center, lon_center = self.config.center_coord
        radius_deg = self.config.radius_meters / 111000

        # 创建同心圆小区
        for ring in range(3):  # 3个环
            for sector in range(4):  # 每环4个扇区
                zone_id = f"zone_ring{ring}_sector{sector}"

                # 计算小区中心
                ring_radius = (ring + 1) * radius_deg / 3
                sector_angle = sector * np.pi / 2

                zone_lat = lat_center + ring_radius * np.cos(sector_angle)
                zone_lon = lon_center + ring_radius * np.sin(sector_angle)

                # 内环商业权重高，外环住宅权重高
                commercial_weight = max(0.2, 1.0 - ring * 0.3)
                residential_weight = min(1.0, 0.3 + ring * 0.3)
                population_weight = (commercial_weight + residential_weight) / 2

                self.zones[zone_id] = Zone(
                    zone_id=zone_id,
                    center_coord=(zone_lat, zone_lon),
                    edges=[],
                    population_weight=population_weight,
                    commercial_weight=commercial_weight,
                    residential_weight=residential_weight
                )

    def _assign_edges_to_zones(self, road_edges: Dict):
        """将道路边分配到最近的交通小区"""
        print("分配道路边到交通小区...")

        # 简化，为每个小区随机分配一些边
        edges_list = list(road_edges.keys())
        edges_per_zone = max(1, len(edges_list) // len(self.zones))

        random.shuffle(edges_list)

        for i, (zone_id, zone) in enumerate(self.zones.items()):
            start_idx = i * edges_per_zone
            end_idx = min(len(edges_list), start_idx + edges_per_zone)

            zone.edges = edges_list[start_idx:end_idx]

            for edge_id in zone.edges:
                self.zone_edges[zone_id].append(edge_id)

    def _calculate_zone_weights(self):
        """计算小区吸引和发生权重"""
        # 标准化权重
        total_pop = sum(zone.population_weight for zone in self.zones.values())
        total_comm = sum(zone.commercial_weight for zone in self.zones.values())
        total_res = sum(zone.residential_weight for zone in self.zones.values())

        if total_pop > 0 and total_comm > 0 and total_res > 0:
            for zone in self.zones.values():
                zone.population_weight /= total_pop
                zone.commercial_weight /= total_comm
                zone.residential_weight /= total_res

    def generate_od_matrix(self, total_trips: int) -> Dict[Tuple[str, str], int]:
        """生成OD矩阵 - 适配北京全域"""
        print(f"生成OD矩阵，总出行量: {total_trips}")

        od_matrix = defaultdict(int)
        zone_list = list(self.zones.keys())

        # 北京市出行特点：
        # 1. 城区之间出行频繁
        # 2. 郊区到城区通勤出行多
        # 3. 同区域内部出行占一定比例

        for _ in range(total_trips):
            # 根据人口权重选择起点
            origin_weights = [self.zones[z].population_weight for z in zone_list]
            origin_zone = np.random.choice(zone_list, p=self._normalize_weights(origin_weights))

            # 北京出行距离衰减函数
            destination_weights = []
            for dest_zone in zone_list:
                if dest_zone == origin_zone:
                    # 区内出行基础权重
                    weight = 0.3
                else:
                    # 区间出行：考虑距离和吸引力
                    dest_attraction = (self.zones[dest_zone].commercial_weight +
                                       self.zones[dest_zone].residential_weight) / 2

                    # 简化距离计算
                    origin_coord = self.zones[origin_zone].center_coord
                    dest_coord = self.zones[dest_zone].center_coord
                    distance = ((origin_coord[0] - dest_coord[0]) ** 2 +
                                (origin_coord[1] - dest_coord[1]) ** 2) ** 0.5

                    # 距离衰减函数
                    distance_decay = max(0.1, 1.0 / (1 + distance * 10))
                    weight = dest_attraction * distance_decay

                destination_weights.append(weight)

            destination_zone = np.random.choice(zone_list, p=self._normalize_weights(destination_weights))
            od_matrix[(origin_zone, destination_zone)] += 1

        return dict(od_matrix)

    def _normalize_weights(self, weights: List[float]) -> List[float]:
        """标准化权重为概率"""
        total = sum(weights)
        return [w / total if total > 0 else 1.0 / len(weights) for w in weights]

    def generate_demand_from_od(self, od_matrix: Dict[Tuple[str, str], int]) -> List[TravelDemand]:
        """从OD矩阵生成具体的出行需求"""
        print("从OD矩阵生成出行需求...")

        demands = []
        demand_counter = 0

        for (origin_zone, destination_zone), trip_count in od_matrix.items():
            for _ in range(trip_count):
                # 随机选择出行方式
                mode = np.random.choice(
                    ['private_car', 'ride_hailing'],
                    p=[self.car_mode_split['private_car'], self.car_mode_split['ride_hailing']]
                )

                # 根据车型生成不同的出发时间
                if mode == 'ride_hailing':
                    # **[修改]** 网约车出发时间分布更均匀，分布在整个仿真时长的前80%
                    departure_time = random.uniform(0, self.config.simulation_duration)
                else:
                    # 私家车使用原有分布
                    departure_time = self._generate_departure_time()

                # 随机选择起终点边
                origin_edge = random.choice(self.zones[origin_zone].edges) if self.zones[origin_zone].edges else None
                destination_edge = random.choice(self.zones[destination_zone].edges) if self.zones[
                    destination_zone].edges else None

                if origin_edge and destination_edge:
                    demand = TravelDemand(
                        demand_id=f"demand_{demand_counter}",
                        origin_zone=origin_zone,
                        destination_zone=destination_zone,
                        departure_time=departure_time,
                        demand_type=mode,
                        origin_edge=origin_edge,
                        destination_edge=destination_edge
                    )
                    demands.append(demand)
                    demand_counter += 1

        print(f"生成了 {len(demands)} 个出行需求")
        print(f"  - 私家车需求: {sum(1 for d in demands if d.demand_type == 'private_car')}")
        print(f"  - 网约车需求: {sum(1 for d in demands if d.demand_type == 'ride_hailing')}")

        return sorted(demands, key=lambda x: x.departure_time)

    # 这是正确的版本，确保出发时间在仿真时长之内
    def _generate_departure_time(self) -> float:
        """生成符合仿真时长的出发时间"""
        # 修正：确保所有出发时间都在仿真时长内
        # 模拟一个早高峰：60%的车辆在前半段出发，40%在后半段
        if random.random() < 0.6:
            # 前半段（高峰期）
            return random.uniform(0, self.config.simulation_duration * 0.5)
        else:
            # 后半段（平峰期）
            return random.uniform(self.config.simulation_duration * 0.5, self.config.simulation_duration)

    def save_demands_to_files(self, demands: List[TravelDemand], output_dir: str):
        """保存需求到文件"""
        # 分别保存私家车和网约车的trips文件
        private_trips = [d for d in demands if d.demand_type == 'private_car']
        ridehail_demands = [d for d in demands if d.demand_type == 'ride_hailing']

        # 保存私家车trips.xml
        if private_trips:
            self._save_trips_xml(private_trips, f"{output_dir}/private_car_trips.xml")

        # 保存网约车需求为独立文件（供调度模块使用）
        if ridehail_demands:
            self._save_ridehail_demands(ridehail_demands, f"{output_dir}/ridehail_demands.csv")

        print(f"需求文件已保存到 {output_dir}")

    def _save_trips_xml(self, trips: List[TravelDemand], filename: str):
        """保存私家车trips为SUMO XML格式"""
        root = ET.Element('trips')

        for trip in trips:
            trip_elem = ET.SubElement(root, 'trip')
            trip_elem.set('id', trip.demand_id)
            trip_elem.set('depart', f'{trip.departure_time:.1f}')
            trip_elem.set('from', trip.origin_edge)
            trip_elem.set('to', trip.destination_edge)
            trip_elem.set('type', 'private_car')

        ET.ElementTree(root).write(filename, encoding='utf-8', xml_declaration=True)

    def _save_ridehail_demands(self, demands: List[TravelDemand], filename: str):
        """保存网约车需求为CSV格式"""
        import csv

        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['demand_id', 'departure_time', 'origin_zone', 'destination_zone',
                             'origin_edge', 'destination_edge', 'served'])

            for demand in demands:
                writer.writerow([
                    demand.demand_id, demand.departure_time, demand.origin_zone,
                    demand.destination_zone, demand.origin_edge, demand.destination_edge,
                    demand.served
                ])

    def get_zone_info(self) -> Dict:
        """获取小区信息摘要"""
        return {
            'total_zones': len(self.zones),
            'total_edges_assigned': sum(len(zone.edges) for zone in self.zones.values()),
            'mode_split': self.car_mode_split,
            'zones': {zid: {
                'center': zone.center_coord,
                'edges_count': len(zone.edges),
                'weights': {
                    'population': zone.population_weight,
                    'commercial': zone.commercial_weight,
                    'residential': zone.residential_weight
                }
            } for zid, zone in self.zones.items()}
        }