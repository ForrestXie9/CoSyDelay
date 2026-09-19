# CoSyDelay 项目数据集与实验数据清单

更新时间：2026-09-14

本文档把目前工作中实际使用的数据分成三层：

1. **原始延误建模数据**：用于 CoSyDelay/V22 和数据驱动 baseline 的训练、验证和测试。
2. **同一批六个路口的过饱和扩展数据**：用于检验冻结模型在高需求状态下的泛化。
3. **真实路口布局的 SUMO 信号优化场景**：用于离线信号配时（TSO）比较 V22 与 Webster、HCM、Akcelik 及其优化变体。

此外，文末单独记录 X1/X2 外部交叉口延误迁移数据。它们是预测泛化实验，不是 TSO 场景，也不应与主实验的六个路口混合统计。

## 1. 数据总览

| 数据层 | 数据来源 | 用途 | 是否用于拟合/选择 | 规模 |
|---|---|---|---|---:|
| 原始 JSONL | `Final_cosy_delay/jsonl_files/` | 初始延误建模、主实验和 baseline | 是；测试集不参与选择 | I1–I9 共 14,531 条 |
| 锁定三分割 | `08_strict_three_way_comparison/frozen_splits_seed20260712/` | 当前正式 Training/Validation/Test | Training 拟合、Validation 选模型、Test 最后一次打开 | I1–I6 共 10,274 条 |
| AASUMO 扩展 CSV | `Z:\CoSydelay\AASumo-I1to6_AllSites-14220.csv` | 同布局需求/配时压力测试 | 否，V22 冻结后才评估 | 原始 14,219 行；严格新增 3,940 行 |
| 过饱和冻结子集 | `17_v22_vs_classics_generalization/oversaturated_frozen_v22/` | `x >= 1` 的过饱和泛化 | 否 | 3,644 行最终评估集 |
| X1 外部数据 | NGSIM Peachtree | I1–I6 → 外部路口预测迁移 | 零样本或只用目标 Training 校准 | 136 个周期记录，83/21/32 |
| X2 外部数据 | DLR Urban Traffic AIM intersection | I1–I6 → 外部路口预测迁移 | 零样本或只用目标 Training 校准 | 1,553 条保留记录，941/348/264 |
| TSO 场景 | `SingleTSCBaselines` 开源 SUMO 路网 | 离线信号方案选择 | 不用目标 Test/Validation 标签 | 3 路口 × 5 需求模式 × 20 seeds |

派生数据不能简单相加：锁定三分割来自原始 JSONL，过饱和数据来自 AASUMO CSV，TSO 数据则是独立的路网/路线文件和 SUMO 仿真输出。

## 2. 原始延误建模数据（I1–I9）

### 2.1 文件与规模

原始文件位于 [`Final_cosy_delay/jsonl_files/`](../../../Final_cosy_delay/jsonl_files/)，每行是一个严格 JSON 对象：

```json
{"dialogue": "traffic volumes + phase green times", "summary": "[Delay_S, Delay_E, Delay_N, Delay_W]"}
```

审计清单见 [`dataset_manifest.csv`](00_data_protocol_audit/outputs/dataset_manifest.csv)，其 SHA-256、JSON/schema 检查结果也记录在同一目录。

| 路口 | Train | Test | 合计 |
|---:|---:|---:|---:|
| I1 | 1,251 | 503 | 1,754 |
| I2 | 1,250 | 503 | 1,753 |
| I3 | 1,218 | 438 | 1,656 |
| I4 | 1,251 | 501 | 1,752 |
| I5 | 1,206 | 412 | 1,618 |
| I6 | 1,245 | 496 | 1,741 |
| I7 | 1,250 | 503 | 1,753 |
| I8 | 1,250 | 503 | 1,753 |
| I9 | 503 | 248 | 751 |
| **总计** | **10,424** | **4,107** | **14,531** |

I7–I9 保留在原始数据审计中；当前正式 V22 主比较和 baseline 公平比较使用 I1–I6。

### 2.2 每条记录包含什么

- 原始交通量：各进口道左转、直行、右转流量（veh/h；没有对应运动时为 0 或按该路口配置缺省）。
- 信号输入：各相位绿灯时间（s）。
- 目标：四个进口道的 approach-level delay（s），存储在 `summary` 中。
- 因而 baseline 的原始输入接口通常是 12 个 movement 流量 + 4 个相位绿灯（共 16 个数值）；CoSyDelay 的 `q/s`、`g/C`、`x` 等是由这些原始量确定性派生的，不是额外观测。
- 由加载器派生的量：
  - 饱和流率 `s = 1800 veh/h`；
  - 周期 `C = active green 总和 + 路口固定 cycle offset`；
  - 流量比 `q/s`；
  - 有效绿灯比 `g/C`；
  - 饱和度 `x = (q/s)/(g/C)`。

路口的相位拓扑、movement 映射和 cycle offset（12、18 或 24 s）记录在 [`schema.json`](00_data_protocol_audit/outputs/schema.json) 中。

### 2.3 当前正式三分割

三分割清单为 [`FROZEN_SPLIT_MANIFEST.json`](08_strict_three_way_comparison/frozen_splits_seed20260712/FROZEN_SPLIT_MANIFEST.json)，分割种子为 `20260712`。它把原始 Train 的 20% 独立划为 Validation，原始 Test 原样保留：

| 路口 | Training | Validation | Test |
|---:|---:|---:|---:|
| I1 | 1,009 | 242 | 503 |
| I2 | 1,027 | 223 | 503 |
| I3 | 934 | 284 | 438 |
| I4 | 987 | 264 | 501 |
| I5 | 966 | 240 | 412 |
| I6 | 986 | 259 | 496 |
| **合计** | **5,909** | **1,512** | **2,853** |

每个路口的 Training/Validation/Test 输入哈希交集均为 0。V22 的表达式/参数选择只看 Training 和 Validation；Test 只在模型冻结后评估。所有数据驱动 baseline 也应使用这一相同划分。

### 2.4 数据来源边界

`00_data_protocol_audit` 已确认 JSONL 的结构、行数、哈希和泄漏检查，但原始生成器、SUMO 版本、完整仿真设置及再发布许可在当前仓库中仍标为 **unknown**。因此，这一层应称为“本地 SUMO 来源的延误数据”，不能在论文中写成已验证的现场观测数据。

## 3. AASUMO 同布局扩展与过饱和数据

### 3.1 原始扩展文件

来源文件：`Z:\CoSydelay\AASumo-I1to6_AllSites-14220.csv`  
SHA-256：`b5f4d44188c3f0f397b497ad7545e1a701b7ace02779f31f3780edbf14d9f234`

审计和派生文件见 [`aasumo_all_sites_extension/`](17_v22_vs_classics_generalization/aasumo_all_sites_extension/)。

- 1 个表头 + 14,219 条数据，57 列；
- 无缺失值、无完整行重复、无非有限数值；
- 流量和绿灯派生字段与原始 movement 输入的一致性误差仅为浮点舍入量级；
- 数据仍是 I1–I6 六个既有路口布局，不是新的路口。

### 3.2 去重和冻结规则

为避免把主实验样本重新评估成“泛化”样本：

1. 先删除与锁定 Training + Validation + Test 输入完全相同的 10,274 行；
2. 再在模型可见空间（active `q/1800`、active `g/C`、`C/180`）中删除 Chebyshev 距离 `<= 0.02` 的 5 条近重复；
3. 得到 3,940 条严格新增输入；V22 表达式和系数在此之前已经冻结，没有重新拟合、选式或调阈值。

完整规则见 [`manifest.json`](17_v22_vs_classics_generalization/aasumo_all_sites_extension/manifest.json)。

### 3.3 过饱和子集

过饱和定义为一个样本中最大 movement 饱和度 `x >= 1.0`；`x >= 1.3` 作为更严重的过饱和标记。扩展审计中有 3,648 条 `x >= 1.0` 候选，经过最终执行有效性筛选后，冻结评估集为 **3,644 条**。各路口最终行数如下：

| 路口 | 过饱和行数 | 其中 `x >= 1.3` |
|---:|---:|---:|
| I1 | 749 | 674 |
| I2 | 748 | 691 |
| I3 | 514 | 492 |
| I4 | 678 | 590 |
| I5 | 420 | 410 |
| I6 | 535 | 376 |
| **合计** | **3,644** | **3,233** |

冻结协议和结果见 [`oversaturated_frozen_v22/README.md`](17_v22_vs_classics_generalization/oversaturated_frozen_v22/README.md)。这是“同一六个布局上的高需求压力测试”，不是 unseen-intersection 或外部现场验证；主报告应优先使用逐路口指标和 macro 结果，pooled R² 只作补充。

另有 [`oversaturation_same_cross_site/`](17_v22_vs_classics_generalization/oversaturation_same_cross_site/)：它复用这批过饱和行，比较 target-site frozen V22 与跨路口 source-skeleton/target-coefficient 的转移效果，不是另一套原始数据。

## 4. X1/X2 外部交叉口延误迁移数据

这部分对应“ I1–I6 训练/冻结结构 → 外部路口测试”，与主 I1–I6 排名和 TSO 仿真分开统计。协议见 [`18_external_cross_site_transfer_i1_i6_to_x1_x2/run_20260913_125940/REPORT.md`](18_external_cross_site_transfer_i1_i6_to_x1_x2/run_20260913_125940/REPORT.md)。

### X1：NGSIM Peachtree

- 来源：NGSIM Peachtree 轨迹数据；项目将轨迹和重建的信号放行信息整理为周期级 approach-delay 记录。
- 4 个进口道：E、N、S、W；共 136 个周期记录。
- 按 15 min time block、种子 `20260829` 固定为：Training 83、Validation 21、Test 32。
- Test 只在 source 选择和（如有）target Training 校准后打开；没有把 Test 用于选式。
- 该数据的绿灯是从车辆 stop-to-go 放行重建的，并非原始官方 SPaT 文件，因此应明确写成“公开 NGSIM 轨迹构造的外部验证集”。

锁定清单：[`NGSIMPT FROZEN_SPLIT_MANIFEST.json`](10_open_real_scenarios/runs/ngsim_official_peachtree_20260829/ngsimpt_split/FROZEN_SPLIT_MANIFEST.json)。

### X2：DLR Urban Traffic AIM Research Intersection

- 来源：DLR Urban Traffic Dataset v1.1.0 的轨迹、交通灯和几何信息。
- 目标采用 E、S、W 三个入口；原始周期记录 1,995 条，按预先声明的入口覆盖规则剔除 432 条，并排除校准时间窗 10 条。
- 最终冻结为 Training 941、Validation 348、Test 264，共 1,553 条。
- 目标是 approach-level stopped delay；零样本模式不使用 X2 Training/Validation，structure-transfer 模式只用 X2 Training 校准系数。

数据文件和分割：[`external_validation_v2_split_frozen/`](../external_data/dlr_ut_v1_1_verified/external_validation_v2_split_frozen/)。这部分是外部预测泛化，不是信号方案控制实验。

## 5. 真实路口布局的 SUMO 信号优化场景（SingleTSCBaselines）

### 5.1 来源与实际使用范围

来源是开源仓库 [`Traffic-Alpha/SingleTSCBaselines`](https://github.com/Traffic-Alpha/SingleTSCBaselines)，本地副本在 [`20_singletscbaselines_tso/source/SingleTSCBaselines/`](20_singletscbaselines_tso/source/SingleTSCBaselines/)，固定版本为 `3147aa9aef16e5c78f83a557d2935a41c1fde079`。仓库包含 12 个真实路口场景目录；为满足当前 V22 movement 映射（4 个保护相位、12 条入口 movement），正式实验选用了以下 3 个：

- `Beijing_Gaojiaoyuan`
- `Chengdu_Guanghua`
- `Tianjin_zhijingdao`

每个路口包含 SUMO `normal.net.xml` 网络、5 个 `.rou.xml` 需求文件及对应 `.sumocfg`：

| 需求模式 | 含义 |
|---|---|
| `low_density` | 低需求 |
| `high_density` | 高需求 |
| `fluctuating_commuter` | 通勤波动 |
| `increasing_demand` | 随时间增加的需求 |
| `random_perturbation` | 随机扰动需求 |

### 5.2 TSO 实验设计

正式批次见 [`20_singletscbaselines_tso/runs/batch_20seed_20260913_215849/`](20_singletscbaselines_tso/runs/batch_20seed_20260913_215849/)，协议状态为 `complete`：

- 7 个控制器：冻结 CoSyDelay-V22、Webster、HCM、Akcelik，以及三者各自的 optimized 变体；
- 3 路口 × 5 需求模式 × 20 个 SUMO seeds；
- 每个控制器 300 个配对条件，共 2,100 次 SUMO 运行；
- 每个场景先由 route-derived demand 生成一个固定配时方案，再用不同 seed 做随机仿真复现；
- 评价为 SUMO 中的平均 vehicle total delay（s），并按 `(junction, regime, seed)` 配对做统计；
- V22 使用冻结的 [`symbolic_lane_model.json`](../signal_optimization_experiment/models/symbolic_lane_model.json)，没有在这些目标路口重新训练；
- 目标路口的 Test/Validation 延误标签没有参与方案选择。

这里的“真实路口”指真实路口布局/开源路网场景；车辆轨迹、排队和延误标签由 SUMO 仿真产生，不是现场观测值。该区分必须在论文和回复审稿人时保留。

运行明细、网络/路线哈希、配时方案和结果分别见 [`MANIFEST.json`](20_singletscbaselines_tso/runs/batch_20seed_20260913_215849/MANIFEST.json)、[`scenario_audit.json`](20_singletscbaselines_tso/runs/batch_20seed_20260913_215849/scenario_audit.json) 和 [`README.md`](20_singletscbaselines_tso/runs/batch_20seed_20260913_215849/README.md)。

## 6. 不要混用的目录和口径

- `Final_cosy_delay/` 下的大量历史模型、历史 prompt 和旧 JSONL 副本只用于复现/审计；不能当作当前正式结果的新测试集。
- `05_generalization_and_stress_tests/` 中仍有 `implemented-not-run` 的协议/演练脚本；没有完成的实验不应写成已验证数据。
- X1/X2 是外部延误预测迁移数据；SingleTSCBaselines 是信号方案 SUMO 仿真数据；二者的指标和统计单位不同。
- 过饱和扩展与原始 I1–I6 共享六个路口布局，但经过输入去重，不能称为 unseen-layout 数据。
- 现有审计把原始 JSONL 的生成器、SUMO 设置和许可证记为 unknown；对外发布前应补齐来源和许可说明。

## 7. 推荐的论文表述

> We evaluate CoSyDelay on nine locally maintained intersection-specific delay datasets. The formal comparison uses six intersections (I1–I6) with a frozen, leakage-free Training/Validation/Test split. A separate AASUMO extension provides input-deduplicated same-layout oversaturated stress cases (`x >= 1.0`) and is evaluated only after the V22 model is frozen. For offline signal timing optimization, we use three real-world intersection layouts from the open-source SingleTSCBaselines benchmark; traffic trajectories and delays in this part are generated by SUMO under five demand regimes and twenty random seeds, rather than measured field labels.
