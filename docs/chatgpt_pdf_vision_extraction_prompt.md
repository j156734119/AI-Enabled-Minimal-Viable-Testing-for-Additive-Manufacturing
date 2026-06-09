# ChatGPT PDF Vision Extraction Prompt

请分析我上传的增材制造期刊论文，并将论文中明确报告的实验数据提取为结构化 CSV。

目标：
提取可用于增材制造机械性能建模的“逐试样、逐实验条件或逐表格行”数据。不要总结代替数据，不要推断缺失值，不要根据图像趋势估算数值。

提取字段：

```text
source_title
doi
publication_year
journal
page_or_section
table_or_figure
evidence_text
material
alloy
alloy_family
am_process
machine_model
powder_feedstock
laser_power_W
scan_speed_mm_s
hatch_spacing_mm
layer_thickness_mm
energy_density_J_mm3
build_orientation
heat_treatment
post_processing
surface_condition
porosity_percent
relative_density_percent
defect_type
specimen_geometry
test_type
test_standard
test_temperature_C
strain_rate_s
stress_ratio_R
frequency_Hz
stress_amplitude_MPa
max_stress_MPa
runout_cycles
ultimate_tensile_strength_MPa
yield_strength_MPa
elongation_percent
elastic_modulus_GPa
fatigue_life_cycles
hardness
failure_mode
fracture_origin
confidence
needs_human_check
extraction_notes
```

严格规则：

1. 每一行只能代表一个明确的试样、实验条件或论文表格行。
2. 只提取正文、表格、图题或补充材料中明确报告的值。
3. 禁止根据领域知识补全或推断缺失值；缺失字段留空。
4. 禁止从曲线、柱状图或图片中目测估算数值。
5. 单位必须转换为字段要求的单位，并在 `extraction_notes` 记录原始单位和换算。
6. 不要把范围、平均值和单个试样值混为一行。
7. 如果论文只报告组平均值，仍可提取，但在 `extraction_notes` 标记 `group_mean`。
8. `evidence_text` 仅保留支持该行数据的最短必要原文片段，不要复制大段正文。
9. `page_or_section` 和 `table_or_figure` 必须填写，便于人工复查。
10. `failure_mode`、`fracture_origin` 和 `defect_type` 只能在论文明确说明时填写，禁止自行判断。
11. 如果表格结构、单位、试样对应关系或 OCR 文字不清楚：
    - `confidence` 填写 `low`；
    - `needs_human_check` 填写 `true`；
    - `extraction_notes` 说明具体疑点。
12. 如果某篇论文没有可提取的逐条件数值，输出空 CSV，并单独说明原因。
13. 不要输出论文全文、完整 OCR 文本或长篇摘要。

输出要求：

1. 每篇论文分别输出一个 CSV 代码块。
2. CSV 必须包含完整表头，即使部分列全部为空。
3. 使用 UTF-8 兼容内容。
4. 字段中含逗号、引号或换行时使用标准 CSV 双引号转义。
5. CSV 后给出简短统计：
   - 提取行数
   - 涉及的目标性能
   - 数据来自哪些表格或页面
   - 需要人工检查的行数
6. 最后再输出一份合并 CSV。
7. 不要使用 Markdown 表格代替 CSV。

请确保 `source_title`、`doi`、`page_or_section`、`evidence_text`、`confidence` 和 `needs_human_check` 不为空；缺少可验证证据的记录不要纳入最终 CSV。
