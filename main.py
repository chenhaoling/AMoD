# src/main.py

import os
import sys
import time
import signal
import traci
import csv

from config import SimulationConfig
from paths import PathManager
from traci_manager import TraCIManager
from parking_manager import ImprovedParkingDataManager
from detector import DynamicParkingDetector
from ridehail_coordinator import RidehailCoordinator
from utils import run_command
from simulation_setup import (
    download_osm_data,
    extract_road_info_from_network,
    create_routes_with_vehicle_types,
    create_parking_facilities,
    create_sumo_config,
    create_gui_settings,
    test_generated_files
)
from terminal_parking import TerminalParkingManager
from fleet_optimizer import VehicleShareabilityOptimizer
from emission_visualizer import EmissionVisualizer


def main():
    """主程序"""
    print("=" * 80)
    print("  改进版北京道路停车综合仿真系统（含碳排放计算）")
    print("  Improved Beijing Road Parking Simulation with Emission Calculation")
    print("=" * 80)

    print("\n选择仿真类型:")
    print("1. 局部仿真（五道口地区，快速测试）")
    print("2. 北京全域仿真（基于道路类型估算停车密度）")
    print("3. 自定义区域仿真")

    choice = input("请选择 (1/2/3): ").strip()

    if choice == '1':
        config = SimulationConfig("local")
        use_estimated_data = False
    elif choice == '2':
        config = SimulationConfig("beijing")
        use_estimated_data = True
    elif choice == '3':
        config = SimulationConfig("local")
        print("\n自定义区域配置:")
        config.area_name = input("区域名称 (默认: custom): ").strip() or "custom"
        try:
            lat = float(input("中心纬度 (默认: 39.993): ") or "39.993")
            lon = float(input("中心经度 (默认: 116.336): ") or "116.336")
            config.center_coord = (lat, lon)
            radius = int(input("半径(米) (默认: 2000): ") or "2000")
            config.radius_meters = radius
            config.output_dir_name = f"{config.area_name}_roadside_parking"
        except ValueError:
            print("输入格式错误，使用默认值")
        use_estimated_data = input("是否使用基于道路类型的停车密度估算? (y/n): ").strip().lower() == 'y'
    else:
        print("无效选择")
        return

    paths = PathManager(config)
    traci_manager = TraCIManager()
    parking_manager = None

    print(f"\n工作目录: {paths.output_dir}")
    print(f"仿真类型: {config.simulation_type}")
    print(f"仿真区域: {config.area_name}")
    print(f"中心坐标: {config.center_coord}")
    if config.radius_meters:
        print(f"半径: {config.radius_meters} 米")
    else:
        print(f"边界框: {config.bbox}")
    print(f"路边停车比例: {config.roadside_parking_ratio:.1%}")

    if use_estimated_data:
        print(f"\n--- 初始化停车密度估算 ---")
        parking_manager = ImprovedParkingDataManager(config)

    if not os.path.exists(paths.osm_file) or os.path.getsize(paths.osm_file) < 1024:
        if not download_osm_data(config, paths.osm_file):
            return
    else:
        print("✓ 使用已存在的OSM数据")

    # 修改为：只有在 .net.xml 不存在时才需要下载 OSM 和生成路网
    if not os.path.exists(paths.net_file):
        if not os.path.exists(paths.osm_file) or os.path.getsize(paths.osm_file) < 1024:
            if not download_osm_data(config, paths.osm_file):
                return

        cmd_netconvert = [
            paths.netconvert, "--osm-files", paths.osm_file, "-o", paths.net_file,
            "--geometry.remove", "--roundabouts.guess", "--ramps.guess", "--junctions.join",
            "--tls.guess-signals",
            "--ignore-errors", "true",
            "--remove-edges.isolated",
            "--keep-edges.by-vclass", "passenger",
            "--remove-edges.by-vclass", "hov,taxi,bus,delivery,truck,bicycle,pedestrian"
        ]
        if not run_command(cmd_netconvert, "生成SUMO路网"):
            return
    else:
        print("✓ 使用已存在的SUMO路网")

    road_edges = extract_road_info_from_network(paths.net_file)
    if not road_edges:
        return

    if not os.path.exists(paths.cfg_file):
        print("\n配置文件不存在，正在生成必要文件...")

        # 生成停车设施
        if not os.path.exists(paths.additional_file):
            print("生成停车设施...")
            parking_info = create_parking_facilities(paths.net_file, config, paths)

        # 生成配置文件
        if not create_sumo_config(paths.cfg_file, paths.net_file, paths.rou_file, config, paths):
            print("✗ 配置文件生成失败")
            return

        # 生成GUI设置
        create_gui_settings(os.path.join(paths.output_dir, 'gui.settings.xml'))
        print("✓ 所有配置文件已生成")
    else:
        print("✓ 使用已存在的配置文件")

    detector = DynamicParkingDetector(config, parking_manager)
    detector.debug_mode = False
    terminal_parking_manager = TerminalParkingManager(config)

    # 初始化碳排放可视化器
    emission_visualizer = EmissionVisualizer(config.simulation_duration)

    parking_info = create_parking_facilities(paths.net_file, config, paths)
    terminal_parking_manager.set_sumo_parking_areas(parking_info)

    if use_estimated_data and parking_manager:
        print(f"\n--- 生成停车密度估算 ---")
        parking_manager.generate_roadside_parking_density(road_edges)

    ridehail_coordinator = RidehailCoordinator(config)
    ridehail_coordinator.parking_detector = detector
    for edge_id, edge_info in road_edges.items():
        detector.plan_parking_spots_for_edge(edge_id, edge_info['length'])

    # 智能判断是否需要生成路由：只有当路由文件或订单文件不存在时才生成
    orders_file = os.path.join(paths.output_dir, 'ridehail_orders.pkl')
    need_regenerate = not os.path.exists(paths.rou_file) or not os.path.exists(orders_file)

    if need_regenerate:
        print(f"\n--- 生成路由文件 ---")
        if os.path.exists(paths.trips_file):
            os.remove(paths.trips_file)
        if os.path.exists(paths.rou_file):
            os.remove(paths.rou_file)

        if not create_routes_with_vehicle_types(paths.net_file, paths.trips_file, paths.rou_file, config, paths):
            print("✗ 路由生成失败")
            return
        print("✓ 路由文件生成完成")
    else:
        print(f"\n✓ 使用已存在的路由文件和订单文件")

    # === 确保配置文件存在 ===
    if not os.path.exists(paths.cfg_file):
        print("配置文件不存在，正在生成...")
        if not os.path.exists(paths.additional_file):
            print("附加文件不存在，正在生成停车设施...")
            parking_info = create_parking_facilities(paths.net_file, config, paths)
            terminal_parking_manager.set_sumo_parking_areas(parking_info)

        if not create_sumo_config(paths.cfg_file, paths.net_file, paths.rou_file, config, paths):
            print("✗ 配置文件生成失败")
            return

        create_gui_settings(os.path.join(paths.output_dir, 'gui.settings.xml'))
        print("✓ 配置文件已生成")

    # ===== 修改：统一运行模式选择逻辑，不再硬编码判断 =====
    # 根据配置中的默认GUI设置决定是否提示用户
    if hasattr(config, 'default_use_gui'):
        use_gui = config.default_use_gui
        if not use_gui:
            print(f"\n{config.area_name}仿真使用配置默认模式（无GUI）...")
    else:
        # 没有默认配置，询问用户
        print(f"\n选择运行模式:\n1. 无GUI模式（快速数据分析）\n2. GUI模式（可视化观察）\n3. 仅生成文件")
        run_choice = input("请选择 (1/2/3): ").strip()

        if run_choice == '3':
            print("文件已生成完毕！")
            return

        use_gui = run_choice == '2'

    sumo_cmd = [
        paths.sumo_gui if use_gui else paths.sumo,
        "-c", paths.cfg_file,
        "--step-length", "1",
        "--time-to-teleport", "300",  # 5分钟后传送卡住的车
        "--no-warnings"
    ]

    if not traci_manager.start_connection(sumo_cmd):
        return

    data_source = "基于道路类型估算" if use_estimated_data and parking_manager else "默认配置"
    print(f"\n仿真开始 ({data_source})...")
    print("正在计算碳排放...")
    print("私家车到达目的地后将寻找预设停车位")

    orders_file = os.path.join(paths.output_dir, 'ridehail_orders.pkl')
    if os.path.exists(orders_file):
        ridehail_coordinator.load_scheduled_orders(orders_file)
    else:
        print("警告：未找到预生成订单文件，网约车将不会运营")

    step = 0
    start_time = time.time()
    progress_interval = 200

    # 用集合来跟踪哪些车辆已经处理过终点停车，避免重复调用
    vehicles_at_destination = set()

    while step < config.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()

        detector.detect_events(step)
        ridehail_coordinator.step_update(step)

        # ===== 修改：记录碳排放增量（修复累积图问题）=====
        current_summary = detector.get_summary()
        if step > 0 and 'total_co2_equivalent_kg' in current_summary:
            # 直接传入累积值，visualizer内部会自动计算增量
            emission_visualizer.add_emission_data(step, current_summary['total_co2_equivalent_kg'])

        # --- 终点停车逻辑（统一适用于所有仿真类型）---
        current_vehicles = traci.vehicle.getIDList()
        for veh_id in current_vehicles:
            # 只处理私家车
            try:
                if traci.vehicle.getTypeID(veh_id) != 'private_car':
                    continue
            except traci.TraCIException:
                continue

            # 1. 检查车辆是否到达终点，并触发停车寻找
            if veh_id not in vehicles_at_destination:
                try:
                    route = traci.vehicle.getRoute(veh_id)
                    if not route:
                        continue

                    # 到达路径的最后一条边
                    if traci.vehicle.getRouteIndex(veh_id) == len(route) - 1:
                        destination_edge = route[-1]
                        terminal_parking_manager.find_and_assign_parking(veh_id, destination_edge)
                        vehicles_at_destination.add(veh_id)  # 标记为已处理
                except traci.TraCIException:
                    continue

        # 2. 对所有正在寻找车位的车辆，检查是否可以执行停车
        searching_now = list(terminal_parking_manager.searching_vehicles.keys())
        for veh_id in searching_now:
            try:
                if veh_id in traci.vehicle.getIDList():
                    terminal_parking_manager.execute_parking_for_arrived_vehicle(veh_id)
            except traci.TraCIException:
                continue

        # 进度显示
        if step > 0 and step % progress_interval == 0:
            roadside_summary = detector.get_summary()
            parking_lot_stats = terminal_parking_manager.get_parking_lot_statistics()

            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            progress = (step / config.simulation_duration) * 100

            vehicle_types = roadside_summary.get('current_vehicle_types', {})
            vehicle_info = f"私家车: {vehicle_types.get('private_car', 0)} | " \
                           f"网约车: {vehicle_types.get('ridehail_car', 0)} | " \
                           f"普通车: {vehicle_types.get('passenger_car', 0)}"

            print(f"\n[进度 {progress:.1f}%] 步长 {step}/{config.simulation_duration}")
            print(f"   速度: {sps:.1f} 步/秒 | {vehicle_info}")

            print(f"   当前停车: 路边 {roadside_summary.get('currently_roadside_parking', 0)} 辆 | "
                  f"停车场 {parking_lot_stats['occupied_parking_spots']} 辆 (寻找中: {parking_lot_stats['searching_vehicles']})")
            print(f"   累计: 路边临停 {roadside_summary.get('roadside_parking_starts', 0)} 次 | "
                  f"终点停车 {parking_lot_stats['parked_in_lots']} 辆")
            print(f"   停车场信息: 总车位 {parking_lot_stats['total_parking_spots']} 个 | "
                  f"剩余 {parking_lot_stats['available_spots']} 个")
            print(f"   CO2当量: {roadside_summary['total_co2_equivalent_kg']:.3f} kg")

            ridehail_current_stats = ridehail_coordinator.get_statistics()
            recent_wait = ridehail_current_stats.get('recent_avg_wait_time_minutes', 0)
            overall_wait = ridehail_current_stats.get('overall_avg_wait_time_minutes', 0)

            print(f"   网约车详情: 总数{ridehail_current_stats['total_ridehail_vehicles']}辆 | "
                  f"空闲{ridehail_current_stats['idle_vehicles']}辆 | "
                  f"接单中{ridehail_current_stats['busy_vehicles']}辆")
            print(f"   订单详情: 当前订单{ridehail_current_stats['active_orders']}个 | "
                  f"累计生成{ridehail_current_stats['total_orders_generated']}个 | "
                  f"成功派单{ridehail_current_stats['total_matches_made']}个 | "
                  f"完成订单{ridehail_current_stats['total_orders_completed']}个")
            print(f"   等候时长: 最近平均 {recent_wait:.2f} 步长 | 总体平均 {overall_wait:.2f} 步长")

            # 添加私家车停车统计
            try:
                all_vehicles = traci.vehicle.getIDList()
                private_cars = [v for v in all_vehicles if traci.vehicle.getTypeID(v) == 'private_car']

                parked_count = len(terminal_parking_manager.parked_vehicles)
                searching_count = len(terminal_parking_manager.searching_vehicles)
                active_private_cars = len(private_cars) - parked_count - searching_count

                total_arrived = len(vehicles_at_destination)
                success_rate = (parked_count / max(1, total_arrived)) * 100

                print(f"   私家车停车详情:")
                print(f"     - 行驶中: {active_private_cars} 辆")
                print(f"     - 已停车: {parked_count} 辆")
                print(f"     - 寻找中: {searching_count} 辆")
                print(f"     - 到达终点车辆: {total_arrived} 辆")
                print(f"     - 停车成功率: {success_rate:.1f}%")

            except Exception as e:
                print(f"   停车统计获取失败: {e}")

        step += 1

    print("\n仿真结束。")
    final_roadside_summary = detector.get_summary()
    final_parking_lot_stats = terminal_parking_manager.get_parking_lot_statistics()
    detector.save_results(paths)

    # 生成碳排放可视化图
    emission_plot_path = os.path.join(paths.output_dir, "emission_24h_distribution.png")
    emission_visualizer.plot_24h_emissions(emission_plot_path,
                                           f"24小时碳排放分布 - {config.area_name}")
    emission_stats = emission_visualizer.get_statistics()
    print(f"\n✓ 碳排放统计:")
    print(f"  - 总排放: {emission_stats['total_emission_kg']:.2f} kg CO2")
    print(f"  - 峰值时段: {emission_stats['peak_hour']:02d}:00")
    print(f"  - 峰值排放: {emission_stats['peak_emission_kg']:.2f} kg")
    print(f"  - 平均时排放: {emission_stats['average_hourly_kg']:.2f} kg")

    traci_manager.close_connection()

    print(f"\n仿真完成！")
    print(f"最终统计:")
    print(f"   - 路边临停总次数: {final_roadside_summary['roadside_parking_starts']} 次")
    print(f"   - 终点停车总数: {final_parking_lot_stats['parked_in_lots']} 辆")
    print(f"   - 未找到车位数 (结束时): {final_parking_lot_stats['searching_vehicles']} 辆")
    print(
        f"   - 停车场占用率: {final_parking_lot_stats['occupied_parking_spots'] / final_parking_lot_stats['total_parking_spots']:.1%}"
        if final_parking_lot_stats['total_parking_spots'] > 0 else "N/A")
    print(f"   - 总CO2当量排放: {final_roadside_summary['total_co2_equivalent_kg']:.3f} kg")

    ridehail_stats = ridehail_coordinator.get_statistics()
    print(f"   - 网约车运营情况:")
    print(f"     * 总车辆数: {ridehail_stats['total_ridehail_vehicles']} 辆")
    print(f"     * 订单完成率: {ridehail_stats['completion_rate']:.1f}%")
    print(f"     * 匹配率: {ridehail_stats['match_rate']:.1f}%")
    print(f"     * 平均等候时长: {ridehail_stats['overall_avg_wait_time_minutes']:.2f} 步长")

    if 'total_emissions' in final_roadside_summary:
        emissions = final_roadside_summary['total_emissions']
        print(f"   - 主要污染物排放 (CO2): {emissions.get('co2_g', 0):.1f} g")

    print(f"\n所有结果文件保存在: {paths.output_dir}/")
    print("\n路边停车与排放仿真完成！")


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print(f"\n收到中断信号，正在安全退出...")
        try:
            if traci.isLoaded():
                traci.close()
        except:
            pass
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()