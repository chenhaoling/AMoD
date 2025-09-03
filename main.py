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
    create_sumo_config,
    create_gui_settings,
    test_generated_files
)
from fleet_optimizer import VehicleShareabilityOptimizer


def main():
    """主程序"""
    print("=" * 80)
    print("  改进版北京道路停车综合仿真系统（含碳排放计算）")
    print("  Improved Beijing Road Parking Simulation with Emission Calculation")
    print("=" * 80)

    print("\n 选择仿真类型:")
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
        print("\n 自定义区域配置:")
        config.area_name = input("区域名称 (默认: custom): ").strip() or "custom"
        try:
            lat = float(input("中心纬度 (默认: 39.993): ") or "39.993")
            lon = float(input("中心经度 (默认: 116.336): ") or "116.336")
            config.center_coord = (lat, lon)
            radius = int(input("半径(米) (默认: 2000): ") or "2000")
            config.radius_meters = radius
            config.output_dir_name = f"{config.area_name}_roadside_parking"
        except ValueError:
            print(" 输入格式错误，使用默认值")
        use_estimated_data = input("是否使用基于道路类型的停车密度估算? (y/n): ").strip().lower() == 'y'
    else:
        print(" 无效选择")
        return

    paths = PathManager(config)
    traci_manager = TraCIManager()
    parking_manager = None

    print(f"\n工作目录: {paths.output_dir}")
    print(f"仿真类型: {config.simulation_type}")
    print(f"仿真区域: {config.area_name}")
    print(f"中心坐标: {config.center_coord}")
    print(f"半径: {config.radius_meters} 米")
    print(f"路边停车比例: {config.roadside_parking_ratio:.1%}")

    if use_estimated_data:
        print(f"\n--- 初始化停车密度估算 ---")
        parking_manager = ImprovedParkingDataManager(config)

    if not os.path.exists(paths.osm_file) or os.path.getsize(paths.osm_file) < 1024:
        if not download_osm_data(config, paths.osm_file): return
    else:
        print("✓ 使用已存在的OSM数据")

    cmd_netconvert = [
        paths.netconvert, "--osm-files", paths.osm_file, "-o", paths.net_file,
        "--geometry.remove", "--roundabouts.guess", "--ramps.guess", "--junctions.join",
        "--tls.guess-signals",

        "--ignore-errors", "true" , # 忽略错误
        "--default.allow", "all",  # 默认允许所有车辆类型
        "--remove-edges.isolated"
    ]
    if not run_command(cmd_netconvert, "生成SUMO路网"): return

    road_edges = extract_road_info_from_network(paths.net_file)
    if not road_edges: return

    if use_estimated_data and parking_manager:
        print(f"\n--- 生成停车密度估算 ---")
        parking_manager.generate_roadside_parking_density(road_edges)

    # 关闭调试模式以减少输出
    detector = DynamicParkingDetector(config, parking_manager)
    detector.debug_mode = False  # 关闭调试输出

    # 初始化网约车协调器
    ridehail_coordinator = RidehailCoordinator(config)
    ridehail_coordinator.parking_detector = detector
    for edge_id, edge_info in road_edges.items():
        detector.plan_parking_spots_for_edge(edge_id, edge_info['length'])

    if not create_routes_with_vehicle_types(paths.net_file, paths.trips_file, paths.rou_file, config, paths): return
    if not create_sumo_config(paths.cfg_file, paths.net_file, paths.rou_file, config): return
    create_gui_settings(os.path.join(paths.output_dir, 'gui.settings.xml'))
    if not test_generated_files(paths): return

    print("\n 文件准备完成!")
    print(f"\n 选择运行模式:\n1. 无GUI模式（快速数据分析）\n2. GUI模式（可视化观察）\n3. 仅生成文件")
    run_choice = input("请选择 (1/2/3): ").strip()

    if run_choice == '3':
        print(" 文件已生成完毕！")
        return

    use_gui = run_choice == '2'
    sumo_cmd = [
        paths.sumo_gui if use_gui else paths.sumo, "-c", paths.cfg_file,
        "--step-length", "1", "--time-to-teleport", "-1", "--no-warnings"
    ]
    if not traci_manager.start_connection(sumo_cmd): return

    data_source = "基于道路类型估算" if use_estimated_data and parking_manager else "默认配置"
    print(f"\n 仿真开始 ({data_source})...")
    print(" 正在计算碳排放...")
    print("车辆将随机决定是否需要路边停车")

    step = 0
    start_time = time.time()

    # 减少常规输出频率
    progress_interval = 200  # 每10分钟输出一次进度

    while step < config.simulation_duration and traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        detector.detect_events(step)

        # 网约车协调更新
        ridehail_coordinator.step_update(step)

        # 减少常规进度输出频率
        if step > 0 and step % progress_interval == 0:
            summary = detector.get_summary()
            elapsed = time.time() - start_time
            sps = step / elapsed if elapsed > 0 else 0
            progress = (step / config.simulation_duration) * 100
            # 新增车辆类型信息
            vehicle_types = summary.get('current_vehicle_types', {})
            vehicle_info = f"私家车: {vehicle_types.get('private_car', 0)} | " \
                           f"网约车: {vehicle_types.get('ridehail_car', 0)} | " \
                           f"普通车: {vehicle_types.get('passenger_car', 0)}"

            print(f"\n [进度 {progress:.1f}%] 步长 {step}/{config.simulation_duration}")
            print(f"   速度: {sps:.1f} 步/秒 | {vehicle_info}")
            print(f"   路边停车: {summary['currently_parking']} 辆")
            print(f"   累计停车: {summary['total_parking_starts']} 次 | CO2当量: {summary['total_co2_equivalent_kg']:.3f} kg")

            # 添加网约车实时状况
            # 添加网约车详细状况
            ridehail_current_stats = ridehail_coordinator.get_statistics()
            print(f"   网约车详情: 总数{ridehail_current_stats['total_ridehail_vehicles']}辆 | "
                  f"空闲{ridehail_current_stats['idle_vehicles']}辆 | "
                  f"接单中{ridehail_current_stats['busy_vehicles']}辆")
            print(f"   订单详情: 当前订单{ridehail_current_stats['active_orders']}个 | "
                  f"累计生成{ridehail_current_stats['total_orders_generated']}个 | "
                  f"成功派单{ridehail_current_stats['total_matches_made']}个 | "
                  f"完成订单{ridehail_current_stats['total_orders_completed']}个")

            # 调试信息：检查车辆状态
            if ridehail_current_stats['total_ridehail_vehicles'] == 0:
                print("   DEBUG: 未发现网约车，检查车辆类型识别")
            elif ridehail_current_stats['total_orders_generated'] == 0:
                print("   DEBUG: 未生成订单，检查订单生成逻辑")
            elif ridehail_current_stats['total_matches_made'] == 0:
                print("   DEBUG: 未成功派单，检查派单匹配逻辑")
        step += 1

    print("\n 仿真结束。")
    final_summary = detector.get_summary()
    detector.save_results(paths)

    traci_manager.close_connection()


    print(f"\n 仿真完成！")
    print(f" 最终统计:")
    print(f"   - 路边停车总次数: {final_summary['total_parking_starts']} 次")
    print(f"   - 总CO2当量排放: {final_summary['total_co2_equivalent_kg']:.3f} kg")

    # 输出详细网约车统计
    ridehail_stats = ridehail_coordinator.get_statistics()
    print(f"   - 网约车运营情况:")
    print(f"     * 总车辆数: {ridehail_stats['total_ridehail_vehicles']} 辆")
    print(f"     * 订单生成: {ridehail_stats['total_orders_generated']} 个")
    print(f"     * 成功派单: {ridehail_stats['total_matches_made']} 个")
    print(f"     * 完成订单: {ridehail_stats['total_orders_completed']} 个")
    print(f"     * 订单完成率: {ridehail_stats['completion_rate']:.1f}%")
    print(f"     * 最终状态: 空闲{ridehail_stats['idle_vehicles']}辆, 服务中{ridehail_stats['busy_vehicles']}辆")

    if 'total_emissions' in final_summary:
        emissions = final_summary['total_emissions']
        print(f"   - 主要污染物排放:")
        print(f"     * CO2: {emissions.get('co2_g', 0):.1f} g")
        print(f"     * CO: {emissions.get('co_g', 0):.1f} g")
        print(f"     * NOx: {emissions.get('nox_g', 0):.3f} g")

    print(f"\n 所有结果文件保存在: {paths.output_dir}/")

    print(f"\n 环境影响评估:")
    total_co2e = final_summary['total_co2_equivalent_kg']
    simulation_hours = config.simulation_duration / 3600
    hourly_emission = total_co2e / simulation_hours
    print(f"每小时CO2当量排放: {hourly_emission:.3f} kg")
    daily_emission = hourly_emission * 24
    print(f"估算日CO2当量排放: {daily_emission:.1f} kg")

    print("\n 路边停车与排放仿真完成！")


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print(f"\n 收到中断信号，正在安全退出...")
        try:
            traci.close()
        except:
            pass
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()