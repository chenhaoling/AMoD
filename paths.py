# src/paths.py

import os
import sys


class PathManager:
    """路径管理器"""

    def __init__(self, config):
        self.config = config
        # 让输出目录在项目根目录，而不是src目录
        self.script_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
        self.output_dir = os.path.join(self.script_dir, config.output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # 文件路径
        self.osm_file = os.path.join(self.output_dir, f"{config.area_name}.osm")
        self.net_file = os.path.join(self.output_dir, f"{config.area_name}.net.xml")
        self.trips_file = os.path.join(self.output_dir, f"{config.area_name}.trips.xml")
        self.rou_file = os.path.join(self.output_dir, f"{config.area_name}.rou.xml")
        self.cfg_file = os.path.join(self.output_dir, f"{config.area_name}.sumo.cfg")

        # 输出文件
        self.parking_events_file = os.path.join(self.output_dir, "roadside_parking_events.csv")
        self.emission_details_file = os.path.join(self.output_dir, "emission_details.csv")
        self.emission_summary_file = os.path.join(self.output_dir, "emission_summary.csv")
        self.traffic_impact_file = os.path.join(self.output_dir, "traffic_impact.csv")

        # SUMO工具路径
        self.sumo_home = self._find_sumo_home()
        if not self.sumo_home:
            print("错误: 找不到SUMO安装目录。")
            sys.exit(1)

        exe_suffix = '.exe' if os.name == 'nt' else ''
        self.netconvert = os.path.join(self.sumo_home, 'bin', f'netconvert{exe_suffix}')
        self.sumo = os.path.join(self.sumo_home, 'bin', f'sumo{exe_suffix}')
        self.sumo_gui = os.path.join(self.sumo_home, 'bin', f'sumo-gui{exe_suffix}')

        # 查找randomTrips.py
        possible_randomtrips_paths = [
            os.path.join(self.sumo_home, 'tools', 'randomTrips.py'),
            os.path.join(self.sumo_home, 'tools', 'trip', 'randomTrips.py'),
            'randomTrips.py'
        ]

        self.random_trips_script = None
        for path in possible_randomtrips_paths:
            if os.path.exists(path):
                self.random_trips_script = path
                break

        if not self.random_trips_script:
            print(f"⚠ 找不到randomTrips.py，将手动生成行程")
        else:
            print(f"✓ 找到randomTrips.py: {self.random_trips_script}")

        self.python_path = sys.executable

    def _find_sumo_home(self):
        """查找SUMO安装目录"""
        if 'SUMO_HOME' in os.environ:
            return os.environ['SUMO_HOME']

        possible_paths = [
            'C:/Program Files (x86)/Eclipse/Sumo',
            'C:/Program Files/Eclipse/Sumo',
            '/usr/share/sumo',
            '/opt/homebrew/share/sumo'
        ]

        for path in possible_paths:
            if os.path.exists(path):
                return path
        return None