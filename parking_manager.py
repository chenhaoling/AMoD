import random


class ImprovedParkingDataManager:
    """简化的停车位数据管理器 - 专注路边停车"""

    def __init__(self, config):
        self.config = config
        self.road_parking_density = {}

    def generate_roadside_parking_density(self, road_edges):
        """为路边停车生成密度分布"""
        print("生成路边停车密度分布...")
        road_type_weights = {
            'highway.residential': 4.0,  # 住宅区停车频繁
            'highway.tertiary': 3.5,  # 三级道路
            'highway.secondary': 3.0,  # 二级道路
            'highway.unclassified': 2.8,
            'highway.service': 2.0,  # 服务道路
            'highway.primary': 1.5,  # 主干道停车较少
            'default': 1.0
        }

        for edge_id, edge_info in road_edges.items():
            road_type = edge_info.get('type', 'default')
            base_weight = road_type_weights.get(road_type, road_type_weights['default'])
            length = edge_info.get('length', 100)

            # 长度因子
            length_factor = min(2.0, length / 200)
            # 随机因子
            random_factor = random.uniform(0.6, 1.8)

            final_weight = base_weight * length_factor * random_factor
            parking_per_km = min(150, max(10, int(final_weight * 25)))

            self.road_parking_density[edge_id] = {
                'road_type': road_type,
                'length': length,
                'parking_per_km': parking_per_km,
                'total_estimated_spots': int((length / 1000) * parking_per_km),
                'density_level': 'high' if parking_per_km > 80 else 'medium' if parking_per_km > 40 else 'low'
            }

        high_density_roads = sum(1 for v in self.road_parking_density.values() if v['density_level'] == 'high')
        medium_density_roads = sum(1 for v in self.road_parking_density.values() if v['density_level'] == 'medium')
        total_estimated_spots = sum(v['total_estimated_spots'] for v in self.road_parking_density.values())

        print(f"生成路边停车密度:")
        print(f"   - 高密度道路: {high_density_roads} 条")
        print(f"   - 中密度道路: {medium_density_roads} 条")
        print(f"   - 估计总停车位: {total_estimated_spots} 个")
        return True