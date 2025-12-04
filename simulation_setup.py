# src/simulation_setup.py

import os
import sys
import xml.etree.ElementTree as ET
import random
from utils import run_command
from od_demand_generator import ODDemandGenerator
from fleet_optimizer import VehicleShareabilityOptimizer
import subprocess


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
    """下载OSM数据 - 基于矩形边界框"""
    print(f"\n--- 步骤: 下载{config.area_name}OSM数据 ---")
    try:
        import requests
    except ImportError:
        print("需要安装requests库: pip install requests")
        return False

    if config.simulation_type == "beijing":
        bbox = config.bbox
        query = f'''[out:xml][timeout:900];
        (
          way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service|living_street)$"](bbox:{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
        );
        (._;>;);
        out meta;'''

        print(f"正在下载北京市路网数据 (矩形边界框: {bbox})...")
    else:
        lat, lon = config.center_coord
        radius = config.radius_meters
        query = f'[out:xml][timeout:300];(way["highway"~"^(motorway|trunk|primary|secondary|tertiary|residential|unclassified|service)$"](around:{radius},{lat},{lon}););(._;>;);out meta;'
        print(f"正在查询 {config.area_name} 路段数据...")

    try:
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

                data_size_mb = len(response.content) / (1024 * 1024)
                print(f"下载数据大小: {data_size_mb:.1f} MB")

                if data_size_mb > 0.5:
                    break
                else:
                    print("数据量太小，尝试下一个服务器")
                    response = None
            except Exception as e:
                print(f"服务器 {server} 失败: {e}")
                continue

        if not response or len(response.content) < 1000:
            print("下载失败")
            return False

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"OSM数据已保存到 {output_file}")

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


def process_routes_in_batches(net_file, trips_file, rou_file, batch_size=10000):  # CHANGED: 20000 -> 10000
    """分批处理大规模路由计算"""
    print(f"\n=== 分批处理模式 ===")
    print(f"批次大小: {batch_size}辆/批")

    # 读取trips文件
    tree = ET.parse(trips_file)
    root = tree.getroot()

    # 保存vType定义
    vtypes = list(root.findall('vType'))
    all_trips = list(root.findall('trip'))
    total_trips = len(all_trips)

    if total_trips <= batch_size:
        # 车辆数少，不需要分批
        return False

    num_batches = (total_trips + batch_size - 1) // batch_size
    print(f"总计 {total_trips} 辆车，分 {num_batches} 批处理\n")

    batch_files = []

    # 分批处理
    for i in range(0, total_trips, batch_size):
        batch_num = i // batch_size + 1
        batch_trips = all_trips[i:i + batch_size]

        print(f"[批次 {batch_num}/{num_batches}] 处理 {len(batch_trips)} 辆车...")

        # 创建批次trips文件
        batch_trips_file = trips_file.replace('.trips.xml', f'_batch{batch_num}.trips.xml')
        batch_root = ET.Element('routes')

        # 添加 vType
        for vtype in vtypes:
            batch_root.append(vtype)

        for trip in batch_trips:
            batch_root.append(trip)

        ET.ElementTree(batch_root).write(batch_trips_file, encoding='utf-8', xml_declaration=True)

        # 为该批次生成路由
        batch_rou_file = rou_file.replace('.rou.xml', f'_batch{batch_num}.rou.xml')

        # CHANGED: 使用绝对路径避免 PATH 问题
        sumo_bin = os.environ.get('SUMO_BIN', '')
        duarouter_exe = os.path.join(sumo_bin, 'duarouter.exe' if os.name == 'nt' else 'duarouter')
        if not os.path.exists(duarouter_exe):
            duarouter_exe = 'duarouter'

        duarouter_cmd = [
            duarouter_exe,
            '--net-file', net_file,
            '--trip-files', batch_trips_file,
            '--output-file', batch_rou_file,
            '--ignore-errors', 'true',
            '--repair', 'true',
            '--routing-threads', '16',  # CHANGED: '8' -> '16'
            '--no-warnings', 'true'
        ]

        try:
            subprocess.run(duarouter_cmd, check=True, capture_output=True, timeout=None)
            print(f"[批次 {batch_num}/{num_batches}] ✓ 完成\n")
            batch_files.append(batch_rou_file)
        except subprocess.CalledProcessError as e:
            print(f"[批次 {batch_num}/{num_batches}] ✗ 失败")
            return False

    # 合并所有批次
    print("=== 合并所有批次 ===")
    merge_route_files(batch_files, rou_file, vtypes)

    # 清理临时文件
    print("清理临时文件...")
    for f in batch_files:
        try:
            os.remove(f)
            temp_trips = f.replace('.rou.xml', '.trips.xml').replace('_batch', '_batch')
            if os.path.exists(temp_trips):
                os.remove(temp_trips)
        except:
            pass

    return True


def merge_route_files(batch_files, output_file, vtypes):
    """合并多个路由文件"""
    final_root = ET.Element('routes')

    # 添加 vType定义
    for vtype in vtypes:
        final_root.append(vtype)

    # 合并所有vehicle/trip
    total_vehicles = 0
    for batch_file in batch_files:
        tree = ET.parse(batch_file)
        root = tree.getroot()

        for elem in root:
            if elem.tag in ['vehicle', 'trip']:
                final_root.append(elem)
                total_vehicles += 1

    if sys.version_info >= (3, 9):
        ET.indent(final_root, space="    ")

    ET.ElementTree(final_root).write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"✓ 合并完成: {total_vehicles} 辆车 -> {output_file}\n")


def create_routes_with_vehicle_types(net_file, trips_file, rou_file, config, paths):
    """基于OD需求创建包含车辆类型的路由 - 北京全域直接使用固定规模"""
    print(f"\n--- 步骤: 基于OD需求生成路由 ---")

    road_edges = extract_road_info_from_network(net_file)
    if not road_edges:
        print("无法读取道路网络信息")
        return False

    od_generator = ODDemandGenerator(config)
    od_generator.initialize_zones_from_network(road_edges)

    if config.simulation_type == "beijing":
        # # 北京全域：直接使用固定规模，不进行优化计算
        # print("北京全域模式：使用固定车队规模")
        #
        # # 固定规模：100万订单对应20万网约车
        # total_orders = 1000000
        # ridehail_vehicles = 200000
        # private_vehicles = int(ridehail_vehicles * 0.01 / 0.99)  # 1:99比例
        #
        # optimal_vehicles = ridehail_vehicles + private_vehicles
        #
        # vehicle_composition = {
        #     'private_car': private_vehicles,
        #     'ride_hailing': ridehail_vehicles
        # }
        #
        # # 仿真规模：使用100%
        # simulation_scale_factor = 1
        # simulation_total = int(optimal_vehicles * simulation_scale_factor)
        # simulation_private_count = int(simulation_total * 0.01)
        # simulation_ridehail_count = simulation_total - simulation_private_count
        #
        # print(f"固定车队规模: {optimal_vehicles} 辆 (私家车{private_vehicles}, 网约车{ridehail_vehicles})")
        # print(f"仿真规模 : {simulation_total}辆 (私家车{simulation_private_count}, 网约车{simulation_ridehail_count})")
        #
        # # 生成对应数量的demands
        # large_scale_demands = int(total_orders * simulation_scale_factor)
        order_file = os.path.join(paths.output_dir, 'ttr_02_v3_sample_order_details_1.txt')

        if os.path.exists(order_file):
            print(f"发现订单文件: {order_file}")
            try:
                # 读取订单数据
                orders_df = pd.read_csv(order_file)
                print(f"成功读取订单文件，共 {len(orders_df)} 个订单")

                # 解析订单为demands
                ridehail_trips = []
                suitable_edges = [e for e in road_edges.keys() if not e.startswith(':')]

                base_date = pd.to_datetime(f"{orders_df.iloc[0]['tdate']} 00:00:00", format='%Y%m%d %H:%M:%S')

                for idx, row in orders_df.iterrows():
                    try:
                        begin_time = pd.to_datetime(row['begin_time'])
                        departure_time = (begin_time - base_date).total_seconds()

                        origin_edge = random.choice(suitable_edges)
                        dest_edge = random.choice(suitable_edges)
                        while dest_edge == origin_edge:
                            dest_edge = random.choice(suitable_edges)

                        from od_demand_generator import TravelDemand
                        demand = TravelDemand(
                            demand_id=f"order_{row['order_id']}",
                            origin_zone='real_order',
                            destination_zone='real_order',
                            departure_time=max(0, departure_time),
                            demand_type='ride_hailing',
                            origin_edge=origin_edge,
                            destination_edge=dest_edge
                        )
                        ridehail_trips.append(demand)
                    except:
                        continue

                print(f"成功解析 {len(ridehail_trips)} 个真实订单")

                ridehail_vehicles = len(ridehail_trips)
                private_vehicles = int(ridehail_vehicles * (config.private_car_ratio / config.ridehail_ratio))
                optimal_vehicles = ridehail_vehicles + private_vehicles

                vehicle_composition = {
                    'private_car': private_vehicles,
                    'ride_hailing': ridehail_vehicles
                }

                print(f"\n车队规模: {optimal_vehicles} 辆 (私家车{private_vehicles}, 网约车{ridehail_vehicles})")

            except Exception as e:
                print(f"订单文件处理失败: {e}，使用随机生成模式")
                order_file = None  # 标记失败，进入else逻辑

        if not os.path.exists(order_file) or order_file is None:
            # 使用config中的参数
            print(f"未找到订单文件，使用配置参数")

            # 从config获取参数
            if hasattr(config, 'total_vehicles'):
                optimal_vehicles = config.total_vehicles
                private_ratio = config.private_car_ratio
                ridehail_ratio = config.ridehail_ratio
            else:
                # 兼容旧配置
                optimal_vehicles = 100000
                private_ratio = 0.6
                ridehail_ratio = 0.4

            private_vehicles = int(optimal_vehicles * private_ratio)
            ridehail_vehicles = int(optimal_vehicles * ridehail_ratio)

            vehicle_composition = {
                'private_car': private_vehicles,
                'ride_hailing': ridehail_vehicles
            }

            print(f"\n车队规模: {optimal_vehicles} 辆 (私家车{private_vehicles}, 网约车{ridehail_vehicles})")

            # 生成随机demands
            total_orders = 1000000
            large_scale_demands = int(total_orders)

        # 仿真规模设置（两种模式共用）
        simulation_scale_factor = 1
        simulation_total = int(optimal_vehicles * simulation_scale_factor)
        simulation_private_count = int(optimal_vehicles * (vehicle_composition['private_car'] / optimal_vehicles))
        simulation_ridehail_count = simulation_total - simulation_private_count

        print(f"仿真规模: {simulation_total}辆 (私家车{simulation_private_count}, 网约车{simulation_ridehail_count})")
        # ========== 北京全域修改部分结束 ==========
    else:
        # 局部仿真保持原有优化逻辑
        large_scale_demands = 10000
        simulation_scale_factor = 1

    print(f"生成大规模需求用于车队分配...")

    od_matrix = od_generator.generate_od_matrix(large_scale_demands)
    demands = od_generator.generate_demand_from_od(od_matrix)
    od_generator.save_demands_to_files(demands, paths.output_dir)

    if config.simulation_type != "beijing":
        # 局部仿真才进行优化
        print(f"\n--- 车队规模优化 ---")
        optimizer = VehicleShareabilityOptimizer(config)

        for demand in demands:
            optimizer.add_trip(
                trip_id=demand.demand_id,
                pickup_time=demand.departure_time,
                dropoff_time=demand.departure_time + 600,
                pickup_location=demand.origin_edge,
                dropoff_location=demand.destination_edge,
                trip_type=demand.demand_type
            )

        optimization_result = optimizer.optimize_fleet()
        optimal_vehicles = optimization_result['total_vehicles']
        vehicle_composition = optimization_result['composition']

        print(f"原始trips数量: {len(demands)}")
        print(f"优化后车队规模: {optimal_vehicles}")
        print(f"效率提升: {(len(demands) - optimal_vehicles) / len(demands) * 100:.1f}%")
        print(f"车辆构成: {vehicle_composition}")

        simulation_total = int(optimal_vehicles * simulation_scale_factor)
        private_ratio = vehicle_composition['private_car'] / optimal_vehicles
        ridehail_ratio = vehicle_composition.get('ride_hailing', 0) / optimal_vehicles

        simulation_private_count = int(simulation_total * private_ratio)
        simulation_ridehail_count = simulation_total - simulation_private_count

    config.optimal_fleet_size = optimal_vehicles
    config.vehicle_composition = vehicle_composition

    ridehail_demands = [d for d in demands if d.demand_type == 'ride_hailing']
    if ridehail_demands:
        import pickle
        orders_file = os.path.join(paths.output_dir, 'ridehail_orders.pkl')
        with open(orders_file, 'wb') as f:
            pickle.dump(ridehail_demands, f)
        print(f"保存了 {len(ridehail_demands)} 个预生成网约车订单")

    print(f"仿真规模: 总计{simulation_total}辆 (私家车{simulation_private_count}, 网约车{simulation_ridehail_count})")

    private_trips = sorted([d for d in demands if d.demand_type == 'private_car'],
                           key=lambda d: d.departure_time)[:simulation_private_count]
    ridehail_trips = sorted([d for d in demands if d.demand_type == 'ride_hailing'],
                            key=lambda d: d.departure_time)[:simulation_ridehail_count]
    routes_root = ET.Element('routes')

    ET.SubElement(routes_root, 'vType', {
        'id': 'private_car',
        'vClass': 'passenger',
        'color': '70,130,180',
        'lcStrategic': '1.0',
        'lcCooperative': '1.0',
        'speedFactor': '1.0'
    })

    ET.SubElement(routes_root, 'vType', {
        'id': 'ridehail_car',
        'vClass': 'passenger',
        'color': '34,139,34',
        'speedFactor': '1.0',
        'lcStrategic': '100.0',
        'lcSpeedGain': '0.1',
        'lcAssertive': '0.1',
        'lcCooperative': '1.0',
        'lcImpatience': '0.0',
        'lcPushy': '0.1',
        'lcKeepRight': '10.0',
        'carFollowModel': 'IDM',
        'tau': '0.5',
        'sigma': '0.0',
        'emergencyDecel': '9.0',
        'accel': '2.0',
        'decel': '3.0',
    })

    for i, demand in enumerate(private_trips):
        trip_elem = ET.SubElement(routes_root, 'trip')
        trip_elem.set('id', f"private_car_{i}")
        trip_elem.set('depart', f'{demand.departure_time:.1f}')
        trip_elem.set('from', demand.origin_edge)
        trip_elem.set('to', demand.destination_edge)
        trip_elem.set('type', 'private_car')

    for i, demand in enumerate(ridehail_trips):
        trip_elem = ET.SubElement(routes_root, 'trip')
        trip_elem.set('id', f"ridehail_car_{i}")
        trip_elem.set('depart', f'{demand.departure_time:.1f}')
        trip_elem.set('from', demand.origin_edge)
        trip_elem.set('to', demand.destination_edge)
        trip_elem.set('type', 'ridehail_car')

    print(f"处理 {len(private_trips)} 个私家车行程")
    print(f"处理 {len(ridehail_trips)} 个网约车行程")

    temp_trips_file = rou_file.replace('.rou.xml', '.trips.xml')
    if sys.version_info >= (3, 9):
        ET.indent(routes_root, space="    ")
    ET.ElementTree(routes_root).write(temp_trips_file, encoding='utf-8', xml_declaration=True)

    print("调用duarouter转换trips为routes...")

    # 尝试分批处理（对大规模数据）
    if process_routes_in_batches(net_file, temp_trips_file, rou_file, batch_size=10000):
        print("✓ 分批处理完成")
        return True

    # 否则使用标准处理（对小规模数据）
    duarouter_cmd = [
        os.path.join(paths.sumo_home, 'bin', 'duarouter.exe' if os.name == 'nt' else 'duarouter'),
        '--net-file', net_file,
        '--trip-files', temp_trips_file,
        '--output-file', rou_file,
        '--ignore-errors', 'true',
        '--repair', 'true',
        '--routing-threads', '8',
        '--no-warnings', 'true'
    ]

    try:
        result = subprocess.run(duarouter_cmd, check=True, capture_output=True, timeout=None)
        print(f"✓ 路由文件 {rou_file} 已成功生成。")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ duarouter失败: {e.stderr.decode('utf-8', errors='ignore')}")
        return False


def create_parking_facilities(net_file, config, paths):
    """创建停车位 - 只在高需求区生成"""
    try:
        tree = ET.parse(net_file)
        root = tree.getroot()
    except (ET.ParseError, FileNotFoundError) as e:
        print(f"✗ 解析网络文件失败: {e}")
        return {}

    additional_root = ET.Element('additional')
    parking_info = {}
    spot_counter = 0
    max_spots = 3000  # 总量控制

    # 定义高需求道路类型
    high_demand_types = ['highway.residential', 'highway.tertiary', 'highway.secondary']

    for edge in root.findall('edge'):
        if spot_counter >= max_spots:  # 达到上限就停止
            break

        if edge.get('function') == 'internal':
            continue

        edge_id = edge.get('id')
        edge_type = edge.get('type', '')

        # 只在高需求道路类型上创建停车位
        if not any(demand_type in edge_type for demand_type in high_demand_types):
            continue

        lanes = edge.findall('lane')
        if not lanes:
            continue

        lane = lanes[0]
        lane_id = lane.get('id')
        length = float(lane.get('length', 0))

        if length < 150:  # 提高最小长度要求
            continue

        # 每200米1个停车位
        num_spots = min(3, max(1, int(length / 200)))
        spot_length = 7.0

        for i in range(num_spots):
            if spot_counter >= max_spots:
                break

            spot_id = f"parkingSpot_{edge_id}_{i}"
            pos = 50 + i * 100  # 每100米一个

            if pos + spot_length < length - 20:
                parking = ET.SubElement(additional_root, 'parkingArea')
                parking.set('id', spot_id)
                parking.set('lane', lane_id)
                parking.set('startPos', str(pos))
                parking.set('endPos', str(pos + spot_length))
                parking.set('roadsideCapacity', '1')
                spot_counter += 1

                parking_info[spot_id] = {
                    'parking_id': spot_id,
                    'edge_id': edge_id,
                    'lane_id': lane_id,
                    'position': pos,
                    'capacity': 1
                }

    if sys.version_info >= (3, 9):
        ET.indent(additional_root, space="    ")
    ET.ElementTree(additional_root).write(paths.additional_file, encoding='utf-8', xml_declaration=True)
    print(f"创建了 {spot_counter} 个停车位（仅在高需求区域）")
    return parking_info


def create_sumo_config(cfg_file, net_file, rou_file, config, paths):
    """创建SUMO配置文件"""
    print(f"\n--- 步骤: 创建SUMO配置文件 ---")
    config_root = ET.Element('configuration')
    input_sec = ET.SubElement(config_root, 'input')
    ET.SubElement(input_sec, 'net-file', value=os.path.basename(net_file))
    ET.SubElement(input_sec, 'route-files', value=os.path.basename(rou_file))

    if os.path.exists(paths.additional_file) and os.path.getsize(paths.additional_file) > 0:
        ET.SubElement(input_sec, 'additional-files', value=os.path.basename(paths.additional_file))

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
    print(f"✓ 配置文件已生成")
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
    files_to_test = [(paths.net_file, "网络"), (paths.rou_file, "路由"), (paths.cfg_file, "配置"),
                     (paths.additional_file, "停车设施")]
    for file_path, name in files_to_test:
        try:
            if os.path.exists(file_path):
                if os.path.getsize(file_path) > 0:
                    ET.parse(file_path)
                    print(f"✓ {name}文件正常")
                else:
                    if name == "停车设施":
                        print(f"✓ {name}文件为空，跳过")
                    else:
                        print(f"✗ {name}文件为空")
                        all_ok = False
            else:
                print(f"✗ {name}文件不存在")
                all_ok = False
        except Exception as e:
            print(f"✗ {name}文件错误: {e}")
            all_ok = False
    return all_ok