# src/emission_visualizer.py

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
import os


class EmissionVisualizer:
    """碳排放可视化模块 - 24小时柱状图（修复版：记录增量而非累积值）"""

    def __init__(self, simulation_duration: int = 3600):
        self.simulation_duration = simulation_duration
        self.hourly_emissions = [0.0] * 24  # 每小时的排放增量
        self.last_total_emission = 0.0  # 记录上一次的累积排放量

    def add_emission_data(self, timestamp: float, total_co2_cumulative: float):
        """
        添加碳排放数据（自动计算增量）

        Args:
            timestamp: 当前仿真时间（秒）
            total_co2_cumulative: 累积的CO2排放总量（kg）
        """
        # 计算本次的增量
        increment = total_co2_cumulative - self.last_total_emission
        self.last_total_emission = total_co2_cumulative

        # 如果增量为负（不应该发生），跳过
        if increment < 0:
            return

        # 将仿真时间映射到24小时
        # 假设simulation_duration是仿真总时长（秒），要映射到24小时
        hour_idx = int((timestamp / self.simulation_duration) * 24) % 24

        # 累加到对应小时
        self.hourly_emissions[hour_idx] += increment

    def plot_24h_emissions(self, output_path: str, title: str = "24小时碳排放分布"):
        """绘制24小时碳排放柱状图"""
        plt.figure(figsize=(14, 6))

        hours = list(range(24))
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, 24))

        bars = plt.bar(hours, self.hourly_emissions, color=colors, edgecolor='black', linewidth=0.5)

        plt.xlabel('小时 (Hour)', fontsize=12, fontweight='bold')
        plt.ylabel('CO2排放量 (kg)', fontsize=12, fontweight='bold')
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xticks(hours, [f'{h:02d}:00' for h in hours], rotation=45)
        plt.grid(axis='y', alpha=0.3, linestyle='--')

        # 添加数值标签
        max_height = max(self.hourly_emissions) if self.hourly_emissions else 1
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width() / 2., height,
                         f'{height:.1f}',
                         ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 碳排放可视化图已保存: {output_path}")

    def get_statistics(self) -> Dict:
        """获取统计信息"""
        total = sum(self.hourly_emissions)
        peak_hour = np.argmax(self.hourly_emissions) if self.hourly_emissions else 0
        peak_emission = self.hourly_emissions[peak_hour] if self.hourly_emissions else 0

        return {
            'total_emission_kg': total,
            'peak_hour': peak_hour,
            'peak_emission_kg': peak_emission,
            'average_hourly_kg': total / 24 if total > 0 else 0
        }