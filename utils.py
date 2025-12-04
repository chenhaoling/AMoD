# utils.py
import subprocess
import os


def run_command(command, step_name):
    """执行命令"""
    print(f"\n--- 步骤: {step_name} ---")
    try:
        # 改进的duarouter判定逻辑
        exe = os.path.basename(str(command[0]))
        cmd_str = " ".join(map(str, command))
        is_duarouter = ('duarouter' in exe.lower()) or ('duarouter' in cmd_str.lower())
        is_netconvert = ('netconvert' in exe.lower()) or ('netconvert' in cmd_str.lower())

        # duarouter永不超时，netconvert 30分钟，其他5分钟
        timeout = None if is_duarouter else (1800 if is_netconvert else 300)

        result = subprocess.run(command, check=True, capture_output=True, timeout=timeout)
        try:
            stdout = result.stdout.decode('utf-8')
            stderr = result.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stdout = result.stdout.decode('gbk', errors='ignore')
            stderr = result.stderr.decode('gbk', errors='ignore')
        print(f"✓ {step_name} 成功。")
        return True
    except subprocess.CalledProcessError as e:
        try:
            stderr_decoded = e.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stderr_decoded = e.stderr.decode('gbk', errors='ignore')
        print(f"✗ {step_name} 失败:\n{stderr_decoded}")
        return False
    except subprocess.TimeoutExpired:
        print(f"✗ {step_name} 超时。")
        return False