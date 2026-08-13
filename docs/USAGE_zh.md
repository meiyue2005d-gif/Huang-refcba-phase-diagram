# 中文使用说明：refCBA 三相图工作流

这套流程是在原有 refCBA/HOOMD 代码上扩展的。势函数仍由
`huang_md/potential.py` 计算，单状态仍依次调用原来的导出、HOOMD、团簇和
动力学脚本。新增代码只管理网格、续跑、证据合并、复核选点和相图。

## 1. 环境

建议在 Linux + NVIDIA GPU 上使用两个 Conda 环境：

```bash
conda env create -f environment.yml
conda env create -f environment-hoomd.yml
```

记下两个 Python 路径，例如：

```bash
ANALYSIS_PY=$HOME/miniconda3/envs/huang-refcba/bin/python
HOOMD_PY=$HOME/miniconda3/envs/huang-refcba-hoomd/bin/python
```

## 2. 生成 448 点粗筛清单

默认配置是 `configs/phase_scan.yaml`：7 个 pH、8 个 NaCl、8 个浓度和 1
个粗筛种子，总计 448 个状态。

```bash
$ANALYSIS_PY scripts/generate_phase_scan_manifest.py \
  --output manifests/refcba_full_grid_v3.tsv
```

修改扫描点、种子或输出目录时，只编辑该 YAML，不需要改脚本。

## 3. 运行粗筛

```bash
$ANALYSIS_PY scripts/run_phase_scan.py \
  --manifest manifests/refcba_full_grid_v3.tsv \
  --analysis-python "$ANALYSIS_PY" \
  --hoomd-python "$HOOMD_PY"
```

脚本按状态续跑：已经存在 `dynamics_analysis/dynamics_summary.json` 的状态
会跳过。调试时可用 `--task-index 0 --dry-run`。集群作业数组可把数组编号
传给 `--task-index`。

## 4. 分类并生成复核清单

```bash
$ANALYSIS_PY scripts/summarize_phase_scan.py \
  --manifest manifests/refcba_full_grid_v3.tsv
```

重要输出：

- `phase_state_table.csv`：粗筛证据和保守/操作性标签；
- `phase_boundary_intervals.csv`：相邻采样点夹出的边界区间；
- `long_run_manifest.tsv`：边界、低置信和未决点的 3 种子长程清单；
- `direct_coexistence_manifest.tsv`：均相 coarsening 候选的 slab 清单。

均相 NVT 中看到大团簇或增长只能生成 `llps_candidate`，不能直接确认
LLPS。可移动有限团簇属于可溶的 equilibrium cluster fluid；只有冻结有限
团簇或持续贯通网络才判聚集。

## 5. 运行 30 ns 复核

```bash
$ANALYSIS_PY scripts/run_phase_scan.py \
  --manifest results/refcba_full_grid_v3/summary/long_run_manifest.tsv \
  --analysis-python "$ANALYSIS_PY" \
  --hoomd-python "$HOOMD_PY" \
  --equil-steps 500000 \
  --prod-steps 30000000 \
  --report-interval 10000
```

## 6. 运行 direct-coexistence

```bash
$ANALYSIS_PY scripts/run_direct_coexistence_manifest.py \
  --manifest results/refcba_full_grid_v3/summary/direct_coexistence_manifest.tsv \
  --analysis-python "$ANALYSIS_PY" \
  --hoomd-python "$HOOMD_PY"
```

slab 导出器现在会把同一个 `state_config` 和 `md_config` 传到底层，避免均
相和共存模拟使用不同 Hamiltonian。

## 7. 输出最终三相图

```bash
$ANALYSIS_PY scripts/finalize_phase_scan.py
```

默认至少需要两个独立种子一致。输出包括：

- `final_phase_state_table.csv`；
- `final_phase_boundary_intervals.csv`；
- `phase_diagram_concentration_slices.png`；
- `phase_diagram_concentration_slices.pdf`。

相图默认画 5、10、20 mg/mL 三张 pH-NaCl 切片。可单独重画：

```bash
$ANALYSIS_PY scripts/plot_phase_diagrams.py \
  --input results/refcba_full_grid_v3/final/final_phase_state_table.csv \
  --output-dir results/refcba_full_grid_v3/final \
  --phase-column phase_final \
  --concentrations 1 5 10 20
```

## 模型解释边界

- `configs/refcba_state_model.yaml` 是 Huang A1 参数锚定的 refCBA 灵敏度模
  型，不是 refCBA 的实验定量标定。
- Huang 的 pH 外推范围和缓冲液固定；0--500 mM NaCl 是 Gouy-Chapman
  扩展，所有加盐状态都会在 metadata 中标为 extrapolation。
- pH 跨越 refCBA 序列等电点也是外推。
- 旧 v1/v2 轨迹可以作为旧 Hamiltonian 的历史证据，但不能混进 v3 的最
  终相图。需要 v3 结果时必须重跑对应点。
