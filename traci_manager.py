# src/traci_manager.py

import traci
import time
import random
import atexit


class TraCIManager:
    """TraCI连接管理"""

    def __init__(self):
        self.is_connected = False
        atexit.register(self.cleanup)

    def cleanup(self):
        if self.is_connected:
            try:
                traci.close()
            except Exception:
                pass
            finally:
                self.is_connected = False

    def start_connection(self, sumo_cmd, max_retries=3):
        try:
            traci.close()
        except Exception:
            pass

        for attempt in range(max_retries):
            try:
                traci.start(sumo_cmd, label=f"sim_{random.randint(1000, 9999)}")
                self.is_connected = True
                print(f"✓ 成功启动TraCI连接 (尝试 {attempt + 1})")
                return True
            except Exception as e:
                print(f"⚠ TraCI启动失败 (尝试 {attempt + 1}): {e}")
                time.sleep(2)

        print("❌ 所有TraCI连接尝试均失败")
        return False

    def close_connection(self):
        if self.is_connected:
            try:
                traci.close()
                self.is_connected = False
                print("✓ TraCI连接已关闭")
            except Exception as e:
                print(f"⚠ 关闭连接时出现警告: {e}")