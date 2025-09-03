# src/emissions.py

import random


class EmissionCalculator:
    """碳排放计算模块 - 基于HBEFA4模型"""

    def __init__(self):
        # 车辆类型分布
        self.vehicle_distribution = {
            'gasoline': 0.65,
            'diesel': 0.25,
            'electric': 0.10
        }

        # 排放因子 (g/km)
        self.emission_factors = {
            'gasoline': {
                'CO2': 130.0,
                'CO': 0.5,
                'NOx': 0.06,
                'HC': 0.05,
                'PM': 0.005
            },
            'diesel': {
                'CO2': 120.0,
                'CO': 0.5,
                'NOx': 0.08,
                'HC': 0.05,
                'PM': 0.045
            },
            'electric': {
                'CO2': 0.0,
                'CO': 0.0,
                'NOx': 0.0,
                'HC': 0.0,
                'PM': 0.0
            }
        }

        # 北京电网CO2排放因子
        self.beijing_grid_factor = 0.5815
        self.electric_consumption = 0.15

        # CO2当量换算因子
        self.co2_equivalent_factors = {
            'CO2': 1.0,
            'CO': 1.0,
            'NOx': 298.0,
            'HC': 25.0,
            'PM': 1.0
        }

        # 驾驶模式乘数
        self.driving_mode_factors = {
            'normal': 1.0,
            'high_speed': 1.44,
            'acceleration': 1.30,
            'idle': 0.8,
            'congestion': 1.5
        }

    def assign_vehicle_type(self, vehicle_id):
        """为车辆分配类型"""
        rand = random.random()
        if rand < self.vehicle_distribution['gasoline']:
            return 'gasoline'
        elif rand < self.vehicle_distribution['gasoline'] + self.vehicle_distribution['diesel']:
            return 'diesel'
        else:
            return 'electric'

    def determine_driving_mode(self, speed, acceleration, is_parking=False, is_congested=False):
        """确定驾驶模式"""
        if is_parking:
            return 'idle'
        elif is_congested:
            return 'congestion'
        elif speed > 22.2:  # 80 km/h
            return 'high_speed'
        elif abs(acceleration) > 1.0:
            return 'acceleration'
        else:
            return 'normal'

    def calculate_emissions(self, vehicle_type, distance_km, driving_mode='normal'):
        """计算排放量"""
        if distance_km <= 0:
            return {pollutant: 0.0 for pollutant in self.emission_factors['gasoline'].keys()}

        base_factors = self.emission_factors[vehicle_type]
        mode_factor = self.driving_mode_factors[driving_mode]

        emissions = {}
        for pollutant, factor in base_factors.items():
            emissions[pollutant] = factor * distance_km * mode_factor

        # 电动车间接排放
        if vehicle_type == 'electric':
            indirect_co2 = distance_km * self.electric_consumption * self.beijing_grid_factor * 1000
            emissions['CO2'] = indirect_co2

        return emissions

    def calculate_carbon_equivalent(self, emissions):
        """计算CO2当量"""
        total_co2e = 0.0
        for pollutant, amount in emissions.items():
            if pollutant in self.co2_equivalent_factors:
                amount_kg = amount / 1000
                co2e = amount_kg * self.co2_equivalent_factors[pollutant]
                total_co2e += co2e
        return total_co2e