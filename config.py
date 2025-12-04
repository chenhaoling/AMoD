# src/config.py

class SimulationConfig:
    """仿真配置管理 - 统一版本（局部和全局使用相同逻辑）"""

    def __init__(self, simulation_type="local"):
        self.simulation_type = simulation_type

        if simulation_type == "local":
            # 五道口局部仿真参数
            self.area_name = "wudaokou"
            self.center_coord = (39.993, 116.336)
            self.radius_meters = 4000
            self.bbox = None  # 局部不使用bbox
            self.output_dir_name = "../../sumo_parking_simulation/sumo_parking_simulation/wudaokou_roadside_parking"
            self.trip_period = 2
            self.vehicle_density_multiplier = 3
            self.simulation_duration = 3600
            self.default_use_gui = None  # None表示询问用户

        else:
            # 北京全域仿真参数 - 使用矩形边界框
            self.area_name = "beijing"
            self.center_coord = (39.904, 116.407)
            self.radius_meters = None
            # 北京市矩形边界框 (minlat, minlon, maxlat, maxlon)
            self.bbox = (39.445, 115.485313, 40.763304, 117.6997)
            self.output_dir_name = "../../sumo_parking_simulation/sumo_parking_simulationAV/beijing_roadside_parking"
            self.trip_period = 1
            self.simulation_duration = 7200
            self.vehicle_density_multiplier = 2
            # 大规模仿真默认使用无GUI模式（可以在代码中覆盖）
            self.default_use_gui = False
            # 北京全域车辆总数（私家车:网约车 = 6:4）
            self.total_vehicles = 2000000
            self.private_car_ratio = 0.6
            self.ridehail_ratio = 0.4

        # ===== 通用参数（所有仿真类型共享）=====

        # 路边停车参数
        self.roadside_parking_ratio = 0.15
        self.parking_duration_min = 30
        self.parking_duration_max = 120
        self.color_change_duration = 10

        # 物理参数
        self.reaction_time = 1.5
        self.deceleration = 4.0
        self.search_distance = 50.0
        self.min_safe_distance = 20.0

        # 道路类型（所有仿真类型使用相同的道路过滤逻辑）
        self.allowed_road_parking_types = {
            'highway.residential', 'highway.tertiary', 'highway.secondary',
            'highway.unclassified', 'highway.service', 'highway.primary',
            'highway.trunk_link', 'highway.primary_link', 'highway.secondary_link',
            'highway.living_street', 'highway.trunk', 'highway.motorway_link'
        }

        # 车队优化参数
        self.max_connection_time = 900
        self.optimal_fleet_size = None
        self.vehicle_composition = None

        # 停车场参数（终点停车场生成规则统一）
        self.parking_lot_generation_interval = 500
        self.parking_spaces_min = 10
        self.parking_spaces_max = 30
        self.terminal_parking_duration = 3600

        # ===== 性能优化参数（可根据规模调整）=====
        # 大规模仿真可以增加这些值以提高性能
        if simulation_type != "local":
            # 北京全域或其他大规模仿真的优化参数
            self.progress_interval = 500  # 更新间隔更长
            self.teleport_time = 300  # 传送时间保持合理
        else:
            # 局部仿真默认参数
            self.progress_interval = 200
            self.teleport_time = 300