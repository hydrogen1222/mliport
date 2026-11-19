# RMSD 和 RMSF

`rmsd` 任务计算考虑周期性的均方根偏差（RMSD）和均方根涨落（RMSF）——快速诊断晶体在 MD
过程中是否保持了结构。

```bash
mliport analyze runs/md rmsd
mliport analyze runs/md rmsd --species Li
mliport analyze runs/md rmsd --drift-reference nonmobile
```

## 物理背景

### RMSD

RMSD 度量每帧中原子相对于初始位置的位移，对所有选定原子取平均：

$$
\text{RMSD}(t) = \sqrt{ \frac{1}{N} \sum_{i=1}^{N} |\Delta \mathbf{r}_i(t)|^2 }
$$

其中 Δ**r**ᵢ(t) 是原子 *i* 从初始位置到第 *t* 帧的最近映像位移。与分子 RMSD 不同，
这里**不做旋转对齐**——位移是纯平移的，周期性边界条件通过 ASE 的最近映像约定处理。

稳定晶体的 RMSD 有界（热振动）。无界增长意味着熔化、相变或结构坍塌。

### RMSF

RMSF 度量每个原子围绕其平均位置的涨落：

$$
\text{RMSF}_i = \sqrt{ \frac{1}{T} \sum_{t=1}^{T} |\Delta \mathbf{r}_i(t) - \langle \Delta \mathbf{r}_i \rangle|^2 }
$$

RMSF 高的原子最活跃。逐原子 RMSF 有助于识别可移动亚晶格或缺陷位点。

## 输出文件

| 文件 | 内容 |
|---|---|
| `rmsd.csv` | `time_ps`, `periodic_displacement_rmsd_A` |
| `rmsf.csv` | `atom_index`, `periodic_displacement_rmsf_A` |

## 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--species` | 全部 | 限定元素（如 `Li`、`O`） |
| `--drift-reference` | `none` | 漂移校正参考系：`none`、`nonmobile`、`indices` |
| `--drift-indices` | — | 漂移校正的原子索引（配合 `--drift-reference indices`） |
| `--include-equilibration` | 关 | 包含平衡帧 |
| `--start-frame` | — | 起始帧索引 |
| `--stop-frame` | — | 终止帧索引（不包含） |

### 漂移校正

默认情况下（`--drift-reference none`）不应用漂移校正。对于固态电解质，使用
`--drift-reference nonmobile` 可以扣除不动亚晶格的漂移，将可移动离子的位移从缓慢的
骨架漂移中分离出来。

## 提示

- RMSD 平台期确认晶体稳定；同时运行 `thermo` 验证温度也稳定。
- 比较不同元素的 RMSF 以识别哪个亚晶格是可移动的——这有助于为 MSD 和 transport
  任务选择 `--mobile`。
- 对于非晶或液态体系，RMSD 无界增长；请改用 MSD 进行扩散的定量分析。
