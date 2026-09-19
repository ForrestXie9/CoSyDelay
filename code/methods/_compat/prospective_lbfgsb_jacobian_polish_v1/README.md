# Preferred engineering candidate: Jacobian r10 parallel3 + polish200

该目录在已验证的解析 Jacobian 版本之后，对每个 approach 的最佳 r10
系数向量追加一次确定性的 L-BFGS-B polish。polish 不增加随机重启，也不
消耗 RNG；只有训练 MSE 更低的结果才会被接受。

## 验证集选择

在查看三预算结果前，候选固定为额外 `50/100/200` 次迭代，并冻结物理、
RMSE 非劣和 wall-time 门槛。三个候选全部合格，按最低验证 macro-RMSE
选中 `polish=200`：

- 10/10 通过物理验证；
- 10/10 run 的训练目标下降；
- 26/30 个 approach 接受了更优系数；
- 额外平均耗时 `0.355s`；
- 组合平均耗时约 `6.59s`，相对正式受控 baseline 仍为 `1.48x`；
- 相对未 polish Jacobian：R² `+0.00119`、RMSE `-0.00791`、
  MAE `-0.00412`、MAPE `-0.02685`。

相对最初正式 r10 parallel3，最终验证均值为：

| 指标 | 正式 baseline | Jacobian + polish200 | 变化 |
|---|---:|---:|---:|
| R² | 0.78656 | 0.78920 | +0.00264 |
| RMSE | 2.63053 | 2.61154 | -0.01900 |
| MAE | 1.61325 | 1.59700 | -0.01625 |
| MAPE | 9.42132 | 9.30459 | -0.11674 |

这些是验证集上的方向性提升，除 MAPE 的 Wilcoxon 结果外未达到一致统计
显著性，不能替代已经归档的论文 test 结果。完整调优证据位于
`experiments/i6_polish_tuning_v1/summary.json`。

## 后续搜索使用

```python
from methods.prospective_lbfgsb_jacobian_polish_v1 import (
    install_population_evolution_polished_fitter,
)

with install_population_evolution_polished_fitter():
    result = evolve_universal_lane_expression(...)
```

默认冻结为 `polish_maxiter=200`。上下文退出时会关闭进程池并恢复原 fitter。
