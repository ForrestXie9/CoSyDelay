# CoSyDelay 正式系数拟合器：L-BFGS-B r10 parallel3

这个目录正式保留当前方法的系数拟合配置：

- 单一优化器 `L-BFGS-B`；
- 每个独立 approach 使用 10 次重启；
- `maxiter=200`、`maxfun=20000`；
- 最多 3 个 approach 并行，每个 approach 内的 10 次重启仍串行；
- 在主进程中按旧代码完全相同的次序生成随机初值。

这是一项实现层加速，不是新的建模方法。表达式、系数边界、目标函数、
流量加权聚合、数据划分和物理验证器均未改变，因此不要求修改论文的
方法描述。Sobol、warm start、selective-log 和 hybrid optimizer 均未纳入
本正式版本。

## 为什么可以并行

一个 approach 的预测不含其他 approach 的系数。旧目标函数是各 approach
MSE 的和，所以三个系数块可独立求解。实现先在父进程生成全部重启点，
然后才分发任务，避免并行调度改变随机数序列。每个 worker 将 numexpr
限制为 1 个线程，以免形成 `3 × CPU线程数` 的嵌套过度并行。

## 直接拟合

在 `Parameters_sensitive/gmini` 目录运行代码：

```python
import numpy as np

from methods.cosydelay_lbfgsb_r10_parallel3 import (
    fit_lane_parameters_to_approaches_parallel,
)

diagnostics = {}
parameters = fit_lane_parameters_to_approaches_parallel(
    universal_expr=expression,
    df=fit_data,
    lanes=lanes,
    lane_to_approach=lane_to_approach,
    approach_targets=fit_targets,
    intersection_id=intersection_id,
    rng=np.random.default_rng(refit_seed),
    diagnostics=diagnostics,
)
```

函数默认值就是冻结配置，无需再传 `n_restarts`、`maxiter` 或 `maxfun`。

## 用于完整进化搜索

完整搜索会反复拟合候选表达式，应复用常驻进程池：

```python
from methods.cosydelay_lbfgsb_r10_parallel3 import (
    install_population_evolution_fitter,
)

with install_population_evolution_fitter():
    result = evolve_universal_lane_expression(...)
```

上下文结束时会关闭进程池并恢复原 fitter，不会永久修改
`population_evolution_lane.py`。

## 验证

快速回归测试：

```powershell
python -m unittest methods.cosydelay_lbfgsb_r10_parallel3.test_parallel_fitter
```

I6 十个冻结表达式的验证集等价性复核：

```powershell
python methods/cosydelay_lbfgsb_r10_parallel3/verify_i6_equivalence.py
```

复核脚本只载入原 fit/validation 训练文件及已归档的 r10 结果，不载入 test。
结果写入 `verification/i6_archived_r10_equivalence/`。

正式 I6 复核结果为 10/10 等价且 10/10 通过物理验证：

- 最大系数差：`0`；
- 最大验证指标差：`0`；
- 最大目标函数差：`7.11e-15`，低于 `1e-12` 验收阈值；
- 并行平均 wall time：`9.47 s`；已归档串行平均为 `20.21 s`。

最后一项约为 `2.13×` 的历史时间比，但两组时间并非同一时刻的受控配对
测量，只作为效率背景；此前串并行受控 pilot 的平均配对加速约为 `1.35×`。

后续改进的边界和优先级见 [NEXT_STEPS.md](NEXT_STEPS.md)。
