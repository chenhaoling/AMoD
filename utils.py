# src/utils.py

import subprocess

def run_command(command, step_name):
    """执行命令"""
    print(f"\n--- 步骤: {step_name} ---")
    try:
        result = subprocess.run(command, check=True, capture_output=True, timeout=300)
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