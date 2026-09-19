'''
Author: WANG Maonan
Date: 2025-07-17 13:01:58
LastEditors: WMN7 18811371255@163.com
Description: 车辆 Route 生成
LastEditTime: 2026-04-13 16:03:31
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
        '-885943869#2.897': [5, 2, 4, 4, 5],
        '170446483#0.356': [5, 4, 6, 6, 2],
        '885943869#0.263': [6, 4, 6, 4, 5],
        '806740830#0.118': [5, 4, 4, 2, 6],
    },

    # 2. 波动通勤 (时段错峰, X~0.55)
    "fluctuating_commuter": {
        '-885943869#2.897': [10, 10, 10, 10, 9],
        '806740830#0.118': [4, 8, 11, 8, 4],
        '170446483#0.356': [11, 9, 7, 4, 3],
        '885943869#0.263': [3, 6, 8, 10, 11],
    },

    # 3. 饱和高密度 (最难但可解, X~0.72)
    "high_density": {
        '-885943869#2.897': [13, 10, 12, 12, 12],
        '806740830#0.118': [12, 12, 12, 12, 10],
        '170446483#0.356': [10, 12, 12, 11, 11],
        '885943869#0.263': [12, 12, 12, 10, 13],
    },

    # 4. 随机扰动 (随机尖峰, X~0.68)
    "random_perturbation": {
        '-885943869#2.897': [9, 9, 19, 8, 8],
        '806740830#0.118': [7, 7, 7, 16, 7],
        '170446483#0.356': [7, 7, 7, 7, 17],
        '885943869#0.263': [7, 16, 7, 7, 6],
    },

    # 5. 递增需求 (主干道主导+递增, 支路≈主向40%)
    "increasing_demand": {
        '-885943869#2.897': [8, 11, 14, 16, 19],
        '885943869#0.263': [8, 10, 12, 14, 16],
        '806740830#0.118': [3, 5, 5, 7, 8],
        '170446483#0.356': [3, 4, 5, 5, 7],
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
