# 轨迹验证

`validate` 任务检查轨迹目录，报告数据是否满足各下游分析的要求。建议在分析前先运行一次，
尽早发现问题。

```bash
mliport analyze runs/md validate
```

输出为 JSON 文件（`validate.json`），包含所有检查结果，便于程序化解析。

## 检查内容

### 结构完整性

| 检查项 | 说明 |
|---|---|
| 有限位置 | 所有原子坐标为有限值（无 NaN / Inf） |
| 有限晶胞 | 所有晶格矢量为有限值 |
| 晶胞秩 | 每帧晶胞非退化（秩为 3） |
| 固定晶胞 | 所有帧共享同一晶胞——MSD 和 transport 要求 |
| PBC | 三个方向均有周期性边界条件 |

### 时间轴

| 检查项 | 说明 |
|---|---|
| 可用 | 存在显式或声明的时间轴 |
| 有限 | 时间间隔无 NaN 或 Inf |
| 严格递增 | 时间戳从不回退 |
| 均匀 | 帧间距恒定——FFT MSD、VACF 和 kinisi transport 要求 |

如果时间轴均匀，报告会包含帧间隔（fs）以及频谱分析的 Nyquist 限（THz 和 cm⁻¹）。

### 产出帧与运行状态

- 帧被划分为 **平衡**（equilibration）和 **产出**（production）阶段。
  MSD、transport 和 VACF 仅使用产出帧。
- 记录源运行的状态（`completed`、`aborted` 等）。未完成的运行会产生警告并阻止
  MSD/transport。

### 速度

轨迹中是否保存了速度。VACF 和振动谱需要速度；其他任务不需要。

### 位置约定

位置记录为 `wrapped`（折叠）还是 `unwrapped`（展开）。需要计算位移的任务（MSD、transport）
需要已知约定以重建连续原子轨迹。

## 资格标志

报告包含四个布尔标志，汇总轨迹支持哪些任务：

| 标志 | 所需条件 |
|---|---|
| `eligible_for_rdf` | 有限位置/晶胞、秩-3 晶胞、3D PBC、≥ 1 帧 |
| `eligible_for_msd` | 以上全部，加：固定晶胞、均匀时间、已知位置约定、≥ 4 产出帧、完整运行 |
| `eligible_for_transport` | 与 MSD 相同（transport 继承 MSD 门控） |
| `eligible_for_vacf` | 有限位置/晶胞、秩-3 晶胞、存储速度、均匀时间、≥ 4 帧 |

如果某任务**不**合格，`errors` 列表会给出具体原因。`warnings` 列表记录非致命问题，
如非均匀采样或未验证的 PBC。

## 输出示例（简略）

```json
{
  "n_frames": 5001,
  "n_atoms": 64,
  "duration_ps": 50.0,
  "three_dimensional_pbc": true,
  "fixed_cell": true,
  "velocities_stored": false,
  "positions_convention": "wrapped",
  "run_status": "completed",
  "production_frames": 5001,
  "eligible_for_rdf": true,
  "eligible_for_msd": true,
  "eligible_for_transport": true,
  "eligible_for_vacf": false,
  "errors": [],
  "warnings": []
}
```

此处 `eligible_for_vacf` 为 `false`，因为未保存速度。其他任务均合格。

## 提示

- 每次 MD 运行后、分析前先运行 `validate`。
- 如果 `eligible_for_msd` 为 `false`，检查 `errors` 数组——常见原因包括可变晶胞
  （NPT 轨迹用于需要固定晶胞的结构任务）、非均匀时间采样和未完成的运行。
- 对于外部轨迹（非 mliport 产生），位置约定可能为 `unknown`，会阻止 MSD/transport。
  请使用无损的 ASE 轨迹格式以确保约定追踪可靠。
