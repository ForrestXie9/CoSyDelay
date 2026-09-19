'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: Please set LastEditors
Description: 车辆 Route 生成
LastEditTime: 2026-02-27 13:33:34
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
        '84355055#1': [5, 4, 4, 3, 6],
        '741602126#2.93': [6, 5, 6, 4, 6],
        '739536522.212': [6, 4, 6, 6, 5],
        '387284606#0.817': [6, 6, 5, 6, 4],
    },

    # 2. 波动通勤 (时段错峰, X~0.55)
    "fluctuating_commuter": {
        '84355055#1': [10, 10, 10, 10, 10],
        '741602126#2.93': [12, 10, 9, 6, 4],
        '739536522.212': [4, 5, 6, 8, 10],
        '387284606#0.817': [5, 9, 13, 9, 5],
    },

    # 3. 饱和高密度 (最难但可解, X~0.72)
    "high_density": {
        '84355055#1': [13, 11, 12, 14, 12],
        '741602126#2.93': [10, 12, 11, 13, 12],
        '739536522.212': [11, 13, 12, 10, 12],
        '387284606#0.817': [12, 11, 13, 12, 14],
    },

    # 4. 随机扰动 (随机尖峰, X~0.68)
    "random_perturbation": {
        '84355055#1': [7, 7, 7, 15, 7],
        '741602126#2.93': [16, 7, 7, 7, 7],
        '739536522.212': [7, 7, 14, 7, 6],
        '387284606#0.817': [7, 7, 7, 7, 16],
    },

    # 5. 递增需求 (主干道主导+递增, 支路≈主向40%)
    "increasing_demand": {
        '84355055#1': [9, 12, 15, 18, 20],
        '739536522.212': [8, 11, 13, 16, 18],
        '741602126#2.93': [4, 6, 6, 7, 8],
        '387284606#0.817': [4, 5, 6, 6, 7],
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
