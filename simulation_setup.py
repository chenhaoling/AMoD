# src/simulation_setup.py

import os
import sys
import xml.etree.ElementTree as ET
import random
from utils import run_command
from od_demand_generator import ODDemandGenerator
from fleet_optimizer import VehicleShareabilityOptimizer


class TripGenerator:
    """手动行程生成器"""

    def __init__(self, config):
        self.config = config

    def extract_edges_from_network(self, net_file):
        try:
            tree = ET.parse(net_file)
            root = tree.getroot()
            edges = []

            for edge in root.findall('edge'):
                if edge.get('function') != 'internal':
                    edge_info = {
                        'id': edge.get('id'),
                        'from': edge.get('from'),
                        'to': edge.get('to'),
                        'type': edge.get('type', ''),
                        'lanes': len(edge.findall('lane'))
                    }

                    lanes = edge.findall('lane')
                    if lanes:
                        edge_info['length'] = float(lanes[0].get('length', 0))
                    else:
                        edge_info['length'] = 0

                    edges.append(edge_info)

            return edges
        except Exception as e:
            print(f"✗ 提取边信息失败: {e}")
            return []

    def generate_trips_manually(self, net_file, trips_file):
        print("手动生成行程文件...")
        edges = self.extract_edges_from_network(net_file)
        if not edges:
            return False

        suitable_edges = [e for e in edges if e['length'] > 50 and not e['id'].startswith(':')]
        if len(suitable_edges) < 10:
            print("可用边数量不足")
            return False

        print(f"找到 {len(suitable_edges)} 条可用道路")

        trips_root = ET.Element('trips')
        base_trips = int(self.config.simulation_duration / self.config.trip_period)

        if self.config.simulation_type == "beijing":
            num_trips = base_trips * 5
        else:
            num_trips = base_trips

        print(f"生成 {num_trips} 个行程...")

        for i in range(num_trips):
            from_edge = random.choice(suitable_edges)
            to_edge = random.choice(suitable_edges)

            while to_edge['id'] == from_edge['id'] and len(suitable_edges) > 1:
                to_edge = random.choice(suitable_edges)

            if i < num_trips * 0.6:
                depart_time = random.uniform(0, self.config.simulation_duration * 0.33)
            else:
                depart_time = random.uniform(0, self.config.simulation_duration * 0.9)

            trip = ET.SubElement(trips_root, 'trip')
            trip.set('id', f'trip_{i}')
            trip.set('depart', f'{depart_time:.1f}')
            trip.set('from', from_edge['id'])
            trip.set('to', to_edge['id'])

        if sys.version_info >= (3, 9):
            ET.indent(trips_root, space="    ")

        ET.ElementTree(trips_root).write(trips_file, encoding='utf-8', xml_declaration=True)
        print(f"行程文件已生成")
        return True


def download_osm_data(config, output_file):
    """下载OSM数据 - 基于行政区划边界"""
    print(f"\n--- 步骤: 下载{config.area_name}OSM数据 ---")
    try:
        import requests
    except ImportError:
        print("需要安装requests库: pip install requests")
        return False

    if config.simulation_type == "beijing":
        # 北京市行政区划边界查询
        query = '''[out:xml][timeout:900];
        (
          relation["name:zh"~"北京市"]["admin_level"="4"]["type"="boundary"];
          relation["name:en"~"Beijing"]["admin_level"="4"]["type"="boundary"];
          relation["name"~"北京"]["admin_level"="4"]["type"="boundary"];
        )->.beijing_boundary;
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|living_street)$"](area.beijing_boundary);
        );
        (._;>;);
        out meta;'''

        print("正在下载北京市行政区划内完整道路网络...")
        print("基于行政边界，不使用半径限制...")
    else:
        # 局部区域仍使用半径查询
        lat, lon = config.center_coord
        radius = config.radius_meters
        query = f'[out:xml][timeout:300];(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service)$"](around:{radius},{lat},{lon}););(._;>;);out meta;'
        print(f"正在查询 {config.area_name} 道路数据...")

    try:
        # 使用备用服务器列表
        servers = [
            "http://overpass-api.de/api/interpreter",
            "https://lz4.overpass-api.de/api/interpreter",
            "https://z.overpass-api.de/api/interpreter"
        ]

        response = None
        for server in servers:
            try:
                print(f"尝试服务器: {server}")
                response = requests.post(server, data=query, timeout=900)
                response.raise_for_status()

                # 检查返回数据大小
                data_size_mb = len(response.content) / (1024 * 1024)
                print(f"下载数据大小: {data_size_mb:.1f} MB")

                if data_size_mb > 0.5:  # 北京全域至少需要500KB数据
                    break
                else:
                    print("数据量太小，尝试下一个服务器")
                    response = None
            except Exception as e:
                print(f"服务器 {server} 失败: {e}")
                continue

        if not response or len(response.content) < 1000:
            print("标准查询失败，尝试简化查询")
            # 备用方案：直接使用北京市边界框
            fallback_query = '''[out:xml][timeout:600];
            (
              way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service)$"](bbox:39.4,115.4,40.9,117.5);
            );
            (._;>;);
            out meta;'''

            response = requests.post("http://overpass-api.de/api/interpreter", data=fallback_query, timeout=600)
            response.raise_for_status()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"OSM数据已保存到 {output_file}")

        # 验证数据
        if "way" in response.text and "highway" in response.text:
            way_count = response.text.count('<way')
            node_count = response.text.count('<node')
            print(f"数据验证通过，包含 {way_count} 条道路, {node_count} 个节点")
            return True
        else:
            print("数据验证失败")
            return False

    except Exception as e:
        print(f"下载失败: {e}")
        return False


def extract_road_info_from_network(net_file):
    """从SUMO网络文件提取道路信息"""
    print("提取道路信息...")
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
        road_edges = {}

        for edge in root.findall('edge'):
            if edge.get('function') != 'internal':
                lanes = edge.findall('lane')
                if lanes:
                    road_edges[edge.get('id')] = {
                        'id': edge.get('id'),
                        'type': edge.get('type', ''),
                        'from_node': edge.get('from'),
                        'to_node': edge.get('to'),
                        'lanes': len(lanes),
                        'length': float(lanes[0].get('length', 0))
                    }

        print(f"提取到 {len(road_edges)} 条道路")
        return road_edges
    except Exception as e:
        print(f"道路信息提取失败: {e}")
        return {}


def create_routes_with_vehicle_types(net_file, trips_file, rou_file, config, paths):
    """基于OD需求创建包含车辆类型的路由 - 修复版本"""
    print(f"\n--- 步骤: 基于OD需求生成路由 ---")

    # 读取道路网络信息
    road_edges = extract_road_info_from_network(net_file)
    if not road_edges:
        print("无法读取道路网络信息")
        return False

    # 初始化OD需求生成器
    od_generator = ODDemandGenerator(config)
    od_generator.initialize_zones_from_network(road_edges)

    # 根据仿真类型调整规模
    if config.simulation_type == "beijing":
        large_scale_demands = 50000
        simulation_scale_factor = 1  # 北京全域使用2%的规模进行仿真
    else:
        large_scale_demands = 10000
        simulation_scale_factor = 1  # 局部仿真使用10%的规模

    print(f"生成大规模需求用于车队优化分析...")

    # 生成OD矩阵和需求
    od_matrix = od_generator.generate_od_matrix(large_scale_demands)
    demands = od_generator.generate_demand_from_od(od_matrix)
    od_generator.save_demands_to_files(demands, paths.output_dir)

    print(f"\n--- 车队规模优化 ---")
    optimizer = VehicleShareabilityOptimizer(config)

    # 添加需求到优化器
    for demand in demands:
        optimizer.add_trip(
            trip_id=demand.demand_id,
            pickup_time=demand.departure_time,
            dropoff_time=demand.departure_time + 600,
            pickup_location=demand.origin_edge,
            dropoff_location=demand.destination_edge,
            trip_type=demand.demand_type
        )

    # 执行优化
    optimization_result = optimizer.optimize_fleet()
    optimal_vehicles = optimization_result['total_vehicles']
    vehicle_composition = optimization_result['composition']

    print(f"原始trips数量: {len(demands)}")
    print(f"优化后车队规模: {optimal_vehicles}")
    print(f"效率提升: {(len(demands) - optimal_vehicles) / len(demands) * 100:.1f}%")
    print(f"车辆构成: {vehicle_composition}")

    # 将优化结果保存到config中
    config.optimal_fleet_size = optimal_vehicles
    config.vehicle_composition = vehicle_composition

    # 修复：按优化比例生成仿真车辆，而不是各占一半
    simulation_total = int(optimal_vehicles * simulation_scale_factor)
    private_ratio = vehicle_composition['private_car'] / optimal_vehicles
    ridehail_ratio = vehicle_composition.get('ride_hailing', 0) / optimal_vehicles

    simulation_private_count = int(simulation_total * private_ratio)
    simulation_ridehail_count = simulation_total - simulation_private_count

    print(f"仿真规模: 总计{simulation_total}辆 (私家车{simulation_private_count}, 网约车{simulation_ridehail_count})")

    # 按比例选择trips
    private_trips = [d for d in demands if d.demand_type == 'private_car'][:simulation_private_count]
    ridehail_trips = [d for d in demands if d.demand_type == 'ride_hailing'][:simulation_ridehail_count]

    routes_root = ET.Element('routes')

    # 定义车辆类型
    ET.SubElement(routes_root, 'vType', {
        'id': 'private_car',
        'vClass': 'passenger',
        'color': '70,130,180',
        'lcStrategic': '1.0',
        'lcCooperative': '1.0',
        'speedFactor': '1.0'
    })

    # 修改车辆类型定义，确保兼容性
    ET.SubElement(routes_root, 'vType', {
        'id': 'ridehail_car',
        'vClass': 'passenger',  # 保持为passenger类
        'color': '34,139,34',
        'lcStrategic': '1.0',
        'lcCooperative': '1.0',
        'speedFactor': '1.0',
        'allow': 'all'  # 明确允许所有道路类型
    })

    # 添加trips (修正：为车辆ID添加类型前缀，确保唯一性和可识别性)
    for i, demand in enumerate(private_trips):
        trip_elem = ET.SubElement(routes_root, 'trip')
        trip_elem.set('id', f"private_car_{i}")  # <-- ID 修正
        trip_elem.set('depart', f'{demand.departure_time:.1f}')
        trip_elem.set('from', demand.origin_edge)
        trip_elem.set('to', demand.destination_edge)
        trip_elem.set('type', 'private_car')

    for i, demand in enumerate(ridehail_trips):
        trip_elem = ET.SubElement(routes_root, 'trip')
        trip_elem.set('id', f"ridehail_car_{i}")  # <-- ID 修正
        trip_elem.set('depart', f'{demand.departure_time:.1f}')
        trip_elem.set('from', demand.origin_edge)
        trip_elem.set('to', demand.destination_edge)
        trip_elem.set('type', 'ridehail_car')

    print(f"处理 {len(private_trips)} 个私家车行程")
    print(f"处理 {len(ridehail_trips)} 个网约车行程")

    # **[修改]** 添加duarouter步骤
    # 1. 将生成的trips保存到一个临时的trips.xml文件
    temp_trips_file = rou_file.replace('.rou.xml', '.trips.xml')
    if sys.version_info >= (3, 9):
        ET.indent(routes_root, space="    ")
    ET.ElementTree(routes_root).write(temp_trips_file, encoding='utf-8', xml_declaration=True)

    # 2. 调用duarouter将trips文件转换为routes文件
    print("调用duarouter转换trips为routes...")
    duarouter_cmd = [
        os.path.join(paths.sumo_home, 'bin', 'duarouter.exe' if os.name == 'nt' else 'duarouter'),
        '--net-file', net_file,
        '--trip-files', temp_trips_file,
        '--output-file', rou_file,
        '--ignore-errors', 'true',
        '--repair', 'true'
    ]

    if not run_command(duarouter_cmd, "转换trips为routes"):
        print("duarouter调用失败，请检查SUMO环境变量和文件路径。")
        return False

    print(f"路由文件 {rou_file} 已成功生成。")
    return True


def create_sumo_config(cfg_file, net_file, rou_file, config):
    """创建SUMO配置文件"""
    print(f"\n--- 步骤: 创建SUMO配置文件 ---")
    config_root = ET.Element('configuration')
    input_sec = ET.SubElement(config_root, 'input')
    ET.SubElement(input_sec, 'net-file', value=os.path.basename(net_file))
    ET.SubElement(input_sec, 'route-files', value=os.path.basename(rou_file))
    time_sec = ET.SubElement(config_root, 'time')
    ET.SubElement(time_sec, 'begin', value='0')
    ET.SubElement(time_sec, 'end', value=str(config.simulation_duration))
    processing_sec = ET.SubElement(config_root, 'processing')
    ET.SubElement(processing_sec, 'ignore-route-errors', value='true')
    gui_sec = ET.SubElement(config_root, 'gui_only')
    ET.SubElement(gui_sec, 'gui-settings-file', value='gui.settings.xml')
    if sys.version_info >= (3, 9):
        ET.indent(config_root, space="    ")
    ET.ElementTree(config_root).write(cfg_file, encoding='utf-8', xml_declaration=True)
    print(f"配置文件已生成")
    return True


def create_gui_settings(gui_settings_file):
    """创建GUI设置文件"""
    gui_root = ET.Element('viewsettings')
    ET.SubElement(gui_root, 'viewport', zoom='150')
    ET.SubElement(gui_root, 'vehicles', {
        'vehicleQuality': '2', 'minVehicleSize': '5', 'vehicleExaggeration': '2.5', 'vehicleColorer': 'given'
    })
    if sys.version_info >= (3, 9):
        ET.indent(gui_root, space="    ")
    ET.ElementTree(gui_root).write(gui_settings_file, encoding='utf-8', xml_declaration=True)


def test_generated_files(paths):
    """测试生成的文件"""
    print(f"\n--- 测试生成的文件 ---")
    all_ok = True
    files_to_test = [(paths.net_file, "网络"), (paths.rou_file, "路由"), (paths.cfg_file, "配置")]
    for file_path, name in files_to_test:
        try:
            if os.path.exists(file_path):
                ET.parse(file_path)
                print(f"{name}文件正常")
            else:
                print(f"{name}文件不存在")
                all_ok = False
        except Exception as e:
            print(f"{name}文件错误: {e}")
            all_ok = False
    return all_ok