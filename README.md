# SCR_Monitoring_Static
A package for SCR field construction based on limited monitoring points

这个仓库目前主要围绕静态 SCR 的 PINN / Hybrid PINN 建模做实验。

我现在重点保留和推进的是 3V 路线，即：

- 状态变量取 `theta(s), T(s), M(s)`
- 几何 `x(s), z(s)` 由 `theta(s)` 积分重建
- `Hybrid` 思路不是纯 PDE，也不是纯 data，而是把 `data + PDE + BC` 放在同一个训练框架里

当前这个仓库里，和这一条路线最直接相关的脚本是：

- [scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py](./scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py)

## 1. 这个版本在做什么

`scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py` 是一个**单文件可运行**的 parametric hybrid PINN。

它做的事情是：

- 输入参数取 `Us, Ub, p, Dx, ht`
- 网络学习一个条件化 decoder
- decoder 输入是 `[s, Us, Ub, p, Dx, ht]`
- decoder 输出是 `theta(s), M(s), T(s)`
- 在参数空间中：
  - 对大量随机 case，只施加 `PDE + BC`
  - 对少量离散 data cases，施加**整场** `data loss`

这里所谓的 sparse，不是 `s` 上稀疏，而是**参数空间里稀疏**。

换句话说：

- `physics batch` 负责覆盖参数域
- `data case bank` 负责在参数空间里提供少量锚点
- 每个 data case 上仍然做全场监督，思路和 single-case hybrid v1_0 一致

## 2. 文件结构

目前没有刻意整理成 package，主要还是脚本驱动。和这个版本直接相关的文件有：

- [scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py](./scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py)
- [scr_exact_bvp_solver.py](./scr_exact_bvp_solver.py)
- [scr_exact_bvp_solver_thetaMHV_v1.py](./scr_exact_bvp_solver_thetaMHV_v1.py)

其中：

- `parametric_sparse_v1_0` 已经是 standalone，不依赖旧的 `scr_static_pinn_3V_Hybrid_v1_0.py`
- exact 参考优先走 `scr_exact_bvp_solver.py`
- 如果主 solver 导入失败，会自动尝试 fallback solver

## 3. 模型定义

### 输入

网络输入固定为 6 维：

```text
[s, Us, Ub, p, Dx, ht]
```

其中：

- `s` 是弧长坐标，归一化到 `[0,1]`
- 其余参数按各自范围归一化到 `[-1,1]`

### 输出

网络输出是 3 个量：

```text
theta(s), M(s), T(s)
```

其中：

- `theta(0)=0` 通过输出锚定实现
- `M(0)=M(L)=0` 通过 endpoint elimination 精确满足
- `T(s)=T0+T_res(s)`，其中 `T0` 是全局可学习量

### 派生量

由输出进一步得到：

- `Q(s) = dM/ds`
- `x(s), z(s)` 由 `theta(s)` 积分重建
- `qt(s), qn(s)` 由全局载荷投影得到

## 4. Hybrid 损失

这个版本里总损失仍然保持 hybrid 结构：

```text
L = w_data * L_data + w_pde * L_pde + lambda_bc * L_bc
```

其中：

- `L_data`：只在稀疏参数 case 集上施加
- `L_pde`：在每步随机采样的 physics cases 上施加
- `L_bc`：同样在 physics cases 上施加

`L_data` 仍然是整场监督，包含：

- `x`
- `z`
- `theta`
- `T`
- `M`
- `Q`

`L_pde` 包含：

- `theta` 方程强形式
- `T` 方程强形式
- `M` 方程弱形式

这点和之前的 3V hybrid v1_0 保持一致，没有另起炉灶。

## 5. 参数空间与采样

默认参数范围写在 `ParameterRanges` 中：

```python
Us_min, Us_max = 0.5, 2.5
Ub_min, Ub_max = 0.0, 0.8
p_min,  p_max  = 1/7, 1/3
Dx_min, Dx_max = 1400.0, 2100.0
ht_min, ht_max = 0.0, 100.0
```

采样时目前做了两层约束：

1. `Us >= Ub`
2. 几何可行性过滤

几何可行性这里用的是当前 2D 单调 SCR 构型下的基本筛选：

```text
sqrt(Dx^2 + (water_depth - ht)^2) <= L
L <= Dx + (water_depth - ht)
```

这不是最终的工程可行域定义，但足够把明显无效的样本先剔掉。

## 6. exact data case bank

如果 `reference_source='exact'`，data case bank 的构造方式是：

- 在参数域中随机采样 case
- 对每个 case 调 exact solver
- 只有 exact 成功的 case 才保留
- 一直重复，直到收集够 `num_cases`

所以：

- exact 并不是“导入能成功就所有 case 都能解”
- parametric 版里 exact 失败，更多时候是采样到的 case 对 exact solver 不友好
- 这不是单文件脚本的 bug，而是参数域和 exact solver 可解域之间的关系问题

如果只是 smoke test，直接把 `reference_source` 改成 `catenary` 即可。

## 7. 运行方式

最直接的跑法：

```bash
python scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.py
```

脚本内部的默认配置在几个 dataclass 里：

- `NetworkConfig`
- `HybridConfig`
- `FullStageConfig`
- `SparseParametricDataConfig`
- `ParametricTrainingConfig`

如果只想做 smoke test，建议直接在脚本外写一个小入口，手动缩小：

- `hidden_dim`
- `num_hidden_layers`
- `n_nodes`
- `physics_batch_size`
- `num_cases`
- `adam_steps`

## 8. 当前建议的调试顺序

我现在实际采用的顺序是：

1. 先用 `catenary` 跑 smoke test，确认训练循环、autograd、batch 维都正常
2. 再切到 `exact`
3. 再去讨论正式训练的默认参数

这样可以把“代码路径错误”和“exact case 不可解”这两类问题分开。

## 9. 输出内容

训练完成后，输出目录中通常会包含：

- `scr_static_pinn_3V_Hybrid_parametric_sparse_v1_0.pth`
- `history_hybrid.json`
- `config.json`
- `sparse_case_bank.json`
- `training_history.png`
- `final_prediction_anchor_case.png`
- `final_anchor_curves.npz`

这里的可视化默认只画一个 anchor case，用来快速看训练是否跑偏。

## 10. 当前限制

目前这版还是 `v1_0`，我自己把它定位成“先把 parametric hybrid 主干立住”的版本，所以有几个限制是明确存在的：

- 还没有引入 encoder / inverse 模块
- 还没有做更系统的参数域设计
- data case 采样策略目前还是随机 + 过滤，不是最优设计
- 对 exact solver 的成功域还没有单独建模
- README 也只覆盖这一条主线，不覆盖仓库中所有历史脚本

## 11. 后续方向

这版站稳以后，后面的工作就比较清楚了：

1. 先把 parametric decoder 训稳
2. 再把 sparse observations -> parameter inference 的模块 A 接上
3. 最后做 A + B 的联合微调

也就是说，这个文件的角色不是最终系统，而是双模块系统里的 decoder 基座。
