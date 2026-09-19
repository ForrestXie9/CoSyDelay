# Prospective: L-BFGS-B r10 parallel3 + analytic Jacobian

该目录不会修改已经冻结的正式方法。它保留相同的 L-BFGS-B、10 组随机
初值、边界、MSE 目标函数和三 approach 并行，只将 SciPy 的有限差分梯度
替换为同一表达式的解析 Jacobian。

动机来自正式 I6 审计：300 个 restart 共执行约 747,579 次目标函数求值，
而 L-BFGS-B 迭代数为 33,300；有限差分是明显的数值开销来源。

晋升前必须依次通过：

1. 中心差分梯度校验；
2. 固定 seed 后 RNG 消耗一致；
3. I6 高难度/易难度小型 pilot；
4. 完整十表达式的验证集配对测试；
5. 10/10 物理规则通过，且验证指标不劣于正式基线。

I6 测试集已经在历史流程中被观察，本实验禁止载入 test，也不能被解释为
新的 locked-test 证据。

## 完整十次结果

预冻结的完整 I6 验证集配对已经完成，全部晋升门槛通过：

- 30/30 实际 approach 梯度通过中心差分校验，最大误差 `5.23e-6`；
- 10/10 拟合后 RNG 状态与正式版本一致；
- 10/10 通过原物理验证器；
- 函数求值从 `747,579` 降至 `71,363`，减少 `90.45%`；
- 平均 wall time 从 `9.75s` 降至 `6.24s`，ratio-of-means 为 `1.56x`；
- 验证 R² `+0.00145`、RMSE `-0.01109`、MAE `-0.01213`、
  MAPE `-0.08988`。

速度在 8/10 个 run 中更快；验证 R²、RMSE、MAE、MAPE 分别在
6/10、5/10、6/10、8/10 个 run 中改善。准确率均值向好，但除 MAPE 的
Wilcoxon 检验外未形成一致的统计显著性，因此不得表述为已证明的泛化提升。

当前决策是：作为 future search 和全新 holdout 的首选工程候选保留；不替换
已经归档的论文测试结果。完整证据位于
`experiments/i6_paired_full10/summary.json`。

## 在后续搜索中使用

```python
from methods.prospective_lbfgsb_jacobian_v1 import (
    install_population_evolution_jacobian_fitter,
)

with install_population_evolution_jacobian_fitter():
    result = evolve_universal_lane_expression(...)
```

上下文结束后会关闭进程池，并恢复正式 baseline fitter。
