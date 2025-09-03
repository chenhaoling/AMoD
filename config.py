# src/config.py

class SimulationConfig:
    """仿真配置管理"""

    def __init__(self, simulation_type="local"):
        self.simulation_type = simulation_type  # "local" 或 "beijing"

        if simulation_type == "local":
            # 五道口局部仿真参数
            self.area_name = "wudaokou"
            self.center_coord = (39.993, 116.336)
            self.radius_meters = 4000
            self.output_dir_name = "../sumo_parking_simulation/wudaokou_roadside_parking"
            self.trip_period = 2
            self.vehicle_density_multiplier = 3
            self.simulation_duration = 3600
        else:
            # 北京全域仿真参数 - 基于行政区划边界，不使用半径
            self.area_name = "beijing"
            self.center_coord = (39.904, 116.407)  # 仅作参考，实际使用行政边界
            self.radius_meters = None  # 北京全域不使用半径限制
            self.output_dir_name = "../sumo_parking_simulation/beijing_roadside_parking"
            self.trip_period = 1  # 缩短生成间隔
            self.simulation_duration = 7200  # 2小时仿真
            self.vehicle_density_multiplier = 2  # 适中的密度

        # 修改：路边停车参数
        self.roadside_parking_ratio = 0.15  # 有停车需求的车辆比例
        self.parking_duration_min = 30  # 最短停车30秒
        self.parking_duration_max = 120  # 最长停车120秒
        self.color_change_duration = 10  # 变色持续时间（秒）

        # 物理参数
        self.reaction_time = 1.5
        self.deceleration = 4.0
        self.search_distance = 50.0
        self.min_safe_distance = 20.0

        # 道路类型 - 适配北京全域，增加更多道路类型
        self.allowed_road_parking_types = {
            'highway.residential', 'highway.tertiary', 'highway.secondary',
            'highway.unclassified', 'highway.service', 'highway.primary',
            'highway.trunk_link', 'highway.primary_link', 'highway.secondary_link',
            'highway.living_street', 'highway.trunk', 'highway.motorway_link'  # 添加更多类型
        }

        # 车队优化参数
        self.max_connection_time = 900  # 15分钟最大连接时间
        self.optimal_fleet_size = None  # 将由优化器设置
        self.vehicle_composition = None  # 车辆类型组成