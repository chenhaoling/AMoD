# src/paths.py
import os
import sys
import glob


class PathManager:
    """路径管理器"""

    def __init__(self, config):
        self.config = config
        self.script_dir = os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))) if '__file__' in locals() else os.getcwd()
        self.output_dir = os.path.join(self.script_dir, config.output_dir_name)
        os.makedirs(self.output_dir, exist_ok=True)

        # 核心输出文件
        self.osm_file = os.path.join(self.output_dir, f"{config.area_name}.osm")
        self.net_file = os.path.join(self.output_dir, f"{config.area_name}.net.xml")
        self.trips_file = os.path.join(self.output_dir, f"{config.area_name}.trips.xml")
        self.rou_file = os.path.join(self.output_dir, f"{config.area_name}.rou.xml")
        self.cfg_file = os.path.join(self.output_dir, f"{config.area_name}.sumo.cfg")
        self.additional_file = os.path.join(self.output_dir, f"{config.area_name}.add.xml")

        # 结果输出
        self.parking_events_file = os.path.join(self.output_dir, "roadside_parking_events.csv")
        self.emission_details_file = os.path.join(self.output_dir, "emission_details_2.csv")
        self.emission_summary_file = os.path.join(self.output_dir, "emission_summary.csv")
        self.traffic_impact_file = os.path.join(self.output_dir, "traffic_impact.csv")

        # 发现 SUMO_HOME
        self.sumo_home = self._find_sumo_home()
        if not self.sumo_home:
            print("错误: 找不到SUMO安装目录。请先设置 SUMO_HOME 或从源码/包管理器安装。")
            sys.exit(1)

        # 选一个真实存在的 sumo bin 目录（优先与 SUMO_HOME 同来源）
        candidates = []
        prefix = self.sumo_home
        if prefix.endswith("/share/sumo"):
            prefix = prefix[:-len("/share/sumo")]
        candidates.append(os.path.join(prefix, "bin"))  # 源码/包安装常见结构

        # macOS Homebrew 常见路径（可选）
        if sys.platform == "darwin":
            candidates.append("/opt/homebrew/opt/sumo/bin")

        # 兜底：PATH 里能找到 duarouter 的目录
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if p and os.path.exists(os.path.join(p, "duarouter")):
                candidates.append(p)

        sumo_bin = None
        for c in candidates:
            if c and os.path.exists(os.path.join(c, "duarouter")):
                sumo_bin = os.path.realpath(c)
                break
        if not sumo_bin:
            # 最后兜底：从 SUMO_HOME 反推
            sumo_bin = os.path.join(self.sumo_home.replace('/share/sumo', ''), 'bin')

        # 环境变量：把选中的 bin 放 PATH 最前面
        os.environ["SUMO_HOME"] = self.sumo_home
        os.environ["SUMO_BIN"] = sumo_bin
        os.environ["PATH"] = sumo_bin + os.pathsep + os.environ.get("PATH", "")

        # ---- 找 PROJ 的 proj.db 并设置 PROJ_LIB ----
        def _ensure_proj_lib():
            cur = os.environ.get("PROJ_LIB")
            if cur and os.path.exists(os.path.join(cur, "proj.db")):
                print(f"✓ PROJ_LIB 已设置: {cur}")
                return

            candidates = [
                "/opt/homebrew/share/proj",  # Homebrew 默认
                "/usr/local/share/proj",  # Intel Mac/Homebrew 旧位
            ]

            # 当前 conda 环境常见位置
            import sys as _sys
            _pyver = f"python{_sys.version_info.major}.{_sys.version_info.minor}"
            _prefix = _sys.prefix
            candidates += [
                os.path.join(_prefix, "share", "proj"),
                os.path.join(_prefix, "lib", _pyver, "site-packages", "pyproj", "proj_dir", "share", "proj"),
                os.path.join(_prefix, "lib", _pyver, "site-packages", "pyogrio", "proj_data"),
            ]

            # 你源码编译的 SUMO（极少数构建带 share/proj）
            candidates.append(os.path.join(self.sumo_home, "share", "proj"))

            for c in candidates:
                if os.path.exists(os.path.join(c, "proj.db")):
                    os.environ["PROJ_LIB"] = c
                    print(f"✓ 设置 PROJ_LIB: {c}")
                    return
            print("⚠ 未找到 proj.db，可能仍会出现坐标转换警告。")

        _ensure_proj_lib()

        exe = '.exe' if os.name == 'nt' else ''
        self.netconvert = os.path.join(sumo_bin, f'netconvert{exe}')
        self.sumo = os.path.join(sumo_bin, f'sumo{exe}')
        self.sumo_gui = os.path.join(sumo_bin, f'sumo-gui{exe}')
        self.duarouter = os.path.join(sumo_bin, f'duarouter{exe}')

        # 查找 randomTrips.py
        possible_randomtrips_paths = [
            os.path.join(self.sumo_home, 'tools', 'randomTrips.py'),
            os.path.join(self.sumo_home, 'tools', 'trip', 'randomTrips.py'),
            'randomTrips.py'
        ]
        self.random_trips_script = next((p for p in possible_randomtrips_paths if os.path.exists(p)), None)
        if not self.random_trips_script:
            print("⚠ 找不到 randomTrips.py，将手动生成行程")
        else:
            print(f"✓ 找到 randomTrips.py: {self.random_trips_script}")

        self.python_path = sys.executable

    def _find_sumo_home(self):
        """查找SUMO安装目录"""
        if 'SUMO_HOME' in os.environ and os.path.exists(os.environ['SUMO_HOME']):
            return os.environ['SUMO_HOME']

        # 常见路径
        windows_paths = [
            'C:/Program Files (x86)/Eclipse/Sumo',
            'C:/Program Files/Eclipse/Sumo',
        ]
        linux_paths = [
            '/usr/share/sumo',
            '/opt/homebrew/share/sumo',
        ]
        mac_paths = [
            '/Applications/sumo',
            '/usr/local/opt/sumo/share/sumo',
            '/opt/homebrew/opt/sumo/share/sumo',
            os.path.expanduser('~/sumo'),
        ]

        all_paths = windows_paths + linux_paths

        for pattern in mac_paths:
            for p in glob.glob(pattern):
                all_paths.append(p)

        for p  in all_paths:
            if os.path.exists(p):
                return p
        return None