'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:36:48
'''
from tshub.utils.init_log import set_logger
from tshub.utils.get_abs_path import get_abs_path
from tshub.sumo_tools.generate_routes import generate_route

# 初始化日志
current_file_path = get_abs_path(__file__)
set_logger(current_file_path('./'), file_log_level='WARNING', terminal_log_level='INFO')

# 开启仿真 --> 指定 net 文件
sumo_net = current_file_path("./networks/normal.net.xml")

traffic_flow_configs = {
    # 1. 稳定低密度 (X~0.35)
    "low_density": {
        '417937574#1.74': [6, 5, 6, 4, 5],
        '417937497#0.2105': [6, 4, 5, 6, 5],
        '339537541#3': [5, 6, 6, 4, 5],
        '339537367#2.7': [6, 4, 5, 5, 6],
    },

    # 2. 波动通勤 (时段错峰, X~0.55)
    "fluctuating_commuter": {
        '339537541#3': [4, 6, 8, 10, 12],
        '339537367#2.7': [5, 7, 11, 6, 5],
        '417937574#1.74': [6, 6, 6, 6, 6],
        '417937497#0.2105': [15, 13, 11, 8, 6],
    },

    # 3. 饱和高密度 (最难但可解, X~0.72)
    "high_density": {
        '339537541#3': [11, 14, 12, 13, 13],
        '339537367#2.7': [11, 12, 12, 10, 12],
        '417937574#1.74': [10, 10, 11, 8, 10],
        '417937497#0.2105': [13, 10, 12, 12, 12],
    },

    # 4. 随机扰动 (随机尖峰, X~0.68)
    "random_perturbation": {
        '339537541#3': [14, 7, 7, 7, 7],
        '339537367#2.7': [8, 8, 17, 8, 8],
        '417937574#1.74': [7, 14, 7, 7, 7],
        '417937497#0.2105': [8, 8, 8, 16, 7],
    },

    # 5. 递增需求 (主干道主导+递增, 支路≈主向40%)
    "increasing_demand": {
        '339537541#3': [7, 10, 13, 15, 17],
        '417937574#1.74': [6, 9, 11, 14, 15],
        '339537367#2.7': [4, 5, 5, 5, 6],
        '417937497#0.2105': [4, 5, 5, 5, 5],
    },

}

for config_id, config_info in traffic_flow_configs.items():
    generate_route(
        sumo_net=sumo_net,
        interval=[2,2,2,2,2], # 共有 10 min
        edge_flow_per_minute=config_info,
        edge_turndef={},
        veh_type={
            'background': {'color':'220,220,220', 'length': 5, 'probability':1},
        },
        output_trip=current_file_path('./testflow.trip.xml'),
        output_turndef=current_file_path('./testflow.turndefs.xml'),
        output_route=current_file_path(f'./routes/{config_id}.rou.xml'),
        interpolate_flow=False,
        interpolate_turndef=False,
    )
