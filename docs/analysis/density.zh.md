# 密度图

`density` 任务为选定的可移动物种计算周期性三维占据概率和数密度网格，揭示固态电解质中的
优选扩散路径和位点占据。

```bash
mliport analyze runs/md density --mobile Li --spacing 0.2 --stride 5
```

## 输出

输出为 NumPy `.npz` 归档文件（`density.npz`），包含：

| 数组 | 形状 | 说明 |
|---|---|---|
| `occupancy_probability` | 网格 | 每个体素被占据的时间分数（总和为 1） |
| `number_density_A3` | 网格 | 体素的数密度（Å⁻³） |
| `counts_per_voxel` | 网格 | 原始直方图计数 |
| `cell_A` | (3, 3) | 模拟晶胞（Å） |

网格形状由晶胞尺寸和所请求的间距决定。

### 加载输出

```python
import numpy as np

data = np.load("density.npz")
probability = data["occupancy_probability"]
density = data["number_density_A3"]
cell = data["cell_A"]
```

## 选项

| 选项 | 默认值 | 说明 |
|---|---|---|
| `--mobile SPECIES` | — | 要映射的元素（如 `Li`）。**必需。** |
| `--spacing` | 0.25 | 体素边长（Å）。更小的值产生更精细的网格。 |
| `--smoothing-sigma-A` | — | 高斯平滑宽度（Å）。周期性应用。 |
| `--stride` | 1 | 每 *n* 帧取一帧（减少长轨迹的内存占用） |
| `--include-equilibration` | 关 | 包含平衡帧 |
| `--start-frame` | — | 起始帧索引 |
| `--stop-frame` | — | 终止帧索引（不包含） |

## 要求

- 三维周期性边界条件
- 固定晶胞（可变晶胞轨迹被拒绝）

这与 RDF 门控相同，外加固定晶胞约束。

## 物理细节

原子位置折叠为分数坐标，在覆盖单胞的三维网格上做直方图统计。每个晶格矢量方向的网格维度为
⌈*L* / spacing⌉，每个方向至少 2 个格点。

**占据概率** 是归一化的直方图（总和为 1）——随机选择的快照中某体素有可移动原子的概率。

**数密度** 是每帧计数除以体素体积，得到空间密度（Å⁻³）。

设置 `--smoothing-sigma-A` 时，在倒空间应用各向同性高斯平滑并周期性卷绕，使密度在
晶胞边界处平滑过渡。这有助于可视化，但会展宽尖锐特征。

## 提示

- 从默认间距（0.25 Å）开始，按需细化。非常精细的网格（< 0.1 Å）会产生大文件，
  可能需要更多帧才能收敛。
- 使用 `--stride` 减少多帧轨迹的内存占用。
- `.npz` 文件可以通过简短的 Python 脚本后处理为体积格式（Cube、CHGCAR、
  XDATCAR 兼容网格），在 VESTA 或 OVITO 中可视化。
