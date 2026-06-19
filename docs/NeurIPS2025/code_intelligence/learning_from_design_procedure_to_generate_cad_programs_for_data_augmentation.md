---
title: >-
  [论文解读] Learning From Design Procedure To Generate CAD Programs for Data Augmentation
description: >-
  [NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)][代码智能][CAD程序生成] 提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。
tags:
  - "NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)"
  - "代码智能"
  - "CAD程序生成"
  - "数据增强"
  - "LLM提示策略"
  - "B-Spline几何"
  - "工业设计"
---

# Learning From Design Procedure To Generate CAD Programs for Data Augmentation

**会议**: NeurIPS 2025 (Workshop: Deep Learning for Code in the Agentic Era)  
**arXiv**: [2603.06894](https://arxiv.org/abs/2603.06894)  
**代码**: 无  
**领域**: 代码智能  
**关键词**: CAD程序生成, 数据增强, LLM提示策略, B-Spline几何, 工业设计

## 一句话总结
提出一种受工业设计流程启发的CAD程序数据增强范式，通过向LLM提供参考曲面程序和设计流程描述来引导生成包含B-Spline有机形状的CAD程序，显著缩小了公开CAD数据集与工业级设计在几何复杂度上的差距。

## 研究背景与动机

**领域现状**：CAD程序生成（如DeepCAD、GenCAD、CAD-MLLM）是3D设计自动化的重要方向，近期借助Python脚本库（CadQuery、Build123d）与LLM结合的趋势加速了发展。

**现有痛点**：当前公开CAD程序数据集（DeepCAD、ABC等）几乎仅包含sketch-extrude等基础操作，生成的几何体以平面、圆柱等标准基元为主，缺少B-Spline自由曲面，与工业级CAD设计相差甚远。

**核心矛盾**：CAD数据标注需要领域专家、数据封闭在不同CAD工具中难以规模化；即使用图像做条件生成（GenCAD、CAD-MLLM），条件图像本身也来自简单数据集，无法突破几何分布的天花板。

**本文目标**：如何让LLM生成包含复杂有机形状（B-Spline曲面/曲线）的CAD程序，实现高质量数据增强？

**切入角度**：工业设计师的实际设计流程——先选择一个参考曲面（通常是B-Spline自由曲面），然后在此基础上依次建模，所有后续操作的几何特征都受参考曲面曲率影响。

**核心idea**：用Python脚本表示的B-Spline参考曲面程序 + 自然语言设计流程描述，作为LLM的提示输入，引导生成具有工业级几何复杂度的CAD程序。

## 方法详解

### 整体框架
系统分为四个阶段：(1) 设计流程提示构建——将设计描述和参考曲面程序组合为提示输入；(2) LLM程序生成——用off-the-shelf LLM（OpenAI o3）生成CadQuery Python脚本；(3) 程序验证——执行程序并检查是否可转为B-rep STEP文件；(4) 结构验证——检查生成的B-rep是否为水密实体且结构可行。

### 关键设计

1. **设计流程提示 (Design Procedure Prompting)**:

    - 功能：构建一个多部分组合的文本提示，模拟工业设计师从参考曲面出发的建模流程
    - 核心思路：提示模板 = [前置系统提示, 设计描述, 设计上下文, 后置系统提示]。前置提示指定用CadQuery库编写CAD程序；设计描述提供目标形状的文字描述（如"一个带两个圆孔的矩形支架"）；设计上下文要求生成物体需匹配参考曲面的曲率，完成后移除参考曲面；后置提示要求水密实体并指定输出路径
    - 设计动机：工业设计中，设计师常以一个自由曲面作为起始约束（如壁面、外壳曲面），后续操作自然产生B-Spline几何。这种"从参考曲面出发"的流程是工业标准做法，但在现有数据增强方法中从未被利用

2. **参考曲面程序 (Reference Surface Program)**:

    - 功能：用Python脚本（而非图像或点云）表示参考曲面，作为提示的一部分输入LLM
    - 核心思路：准备四种B-Spline曲面类型——高斯曲面、鞍面、波浪曲面和涟漪曲面，每种用CadQuery脚本参数化表示。通过变化参数（如鞍面曲率从浅到深）进一步增加多样性。每次生成时随机选择一个参考曲面与设计描述配对
    - 设计动机：脚本形式的参考曲面对LLM来说是精确且可理解的参数化几何描述，避免了图像/点云等跨模态输入带来的几何不准确问题

3. **程序生成与迭代验证**:

    - 功能：生成的程序经过两阶段验证，不通过则反馈错误给LLM进行自修正
    - 核心思路：程序验证——执行Python脚本检查是否能导出B-rep STEP文件；结构验证——使用DTGBrepGen的有效性检查验证水密性和结构可行性。验证失败的错误信息被回注提示中进行迭代修正
    - 设计动机：参考曲面的引入增加了程序复杂度，需要反馈驱动的迭代生成来保证成功率

### 评价指标

使用B-Spline比率 $\beta_i$ 衡量每个CAD对象中有机形状的占比：

$$\beta_i = \frac{1}{2}\left(\frac{f_i^b}{f_i} + \frac{e_i^b}{e_i}\right)$$

其中 $f_i$ 为面数，$f_i^b$ 为B-Spline面数，$e_i$ 为曲线数，$e_i^b$ 为B-Spline曲线数。值越高表示有机形状占比越大。

## 实验关键数据

### 主实验

以bracket（支架）为目标类别，与商业工业数据集和三个公开CAD程序数据集对比：

| 数据集 | 平均STEP行数 | 平均面数 | 平均曲线数 | 含B-Spline面 | 含B-Spline曲线 | B-Spline比率 |
|--------|-------------|---------|-----------|-------------|---------------|-------------|
| Industry (商业) | 10099 | 82.91 | 259.4 | 100% | 100% | 0.535 |
| DeepCAD-b | 1783 | 21.48 | 50.30 | 0% | 1% | 0.0004 |
| GenCAD* | 991 | 12.55 | 27.86 | 0% | 1% | 0.0009 |
| CAD-MLLM* | 2402 | 28.65 | 69.26 | 0% | 1% | 0.0008 |
| **Ours** | **4494** | **26.57** | **67.57** | **77%** | **89%** | **0.2217** |

核心发现：本文方法生成的CAD对象中77%含B-Spline面、89%含B-Spline曲线，而DeepCAD/GenCAD/CAD-MLLM基本为0%。B-Spline比率从不到0.001提升到0.22，大幅接近工业级的0.54。

### 消融实验

测试移除参考曲面提示的影响：

| 方法 | 平均STEP行数 | 平均面数 | 含B-Spline面 | 含B-Spline曲线 | B-Spline比率 |
|------|-------------|---------|-------------|---------------|-------------|
| Ours(-RT) 无设计上下文+无参考曲面 | 1225 | 14.86 | 2% | 6% | 0.0085 |
| Ours(-R) 文本引导"smooth organic" | 2992 | 34.56 | 18% | 27% | 0.0478 |
| **Ours 完整方法** | **4494** | **26.57** | **77%** | **89%** | **0.2217** |

### 关键发现
- 参考曲面程序是生成B-Spline几何的关键因素：移除后B-Spline比率从0.22骤降到0.008
- 仅用文本引导"smooth and organic"只能微弱改善（0.048），说明自然语言对复杂几何的引导能力有限
- 有趣的是，Ours(-R)的平均面数和曲线数反而更高——LLM倾向于用更多标准基元来"模拟"圆滑形状，而非使用B-Spline简洁表示
- 参考曲面引入的复杂性导致21%的样本需要>5次迭代修正，高于无参考曲面的12%

## 亮点与洞察
- **脚本即提示**：用可执行的Python脚本而非图像/点云作为参考曲面的表示方式。LLM在文本域内处理几何信息远比跨模态更精确，这个思路可迁移到任何需要精确几何约束的代码生成场景
- **设计流程知识注入**：将工业设计师的隐性知识（从参考曲面出发的建模流程）编码为提示策略，本质上是一种domain-aware的提示工程
- **无需多模态模型**：与GenCAD/CAD-MLLM需要图像条件不同，本方法纯文本输入即可，兼容任何代码生成LLM，部署门槛更低

## 局限与展望
- 参考曲面只有4种类型（高斯、鞍面、波浪、涟漪），多样性靠参数变化，可通过程序化扩展更多曲面族进一步增加
- 无法精确控制支架的哪个部分与参考曲面对齐（有时在底座，有时在支脚），精确控制需要更细粒度的CAD参数化理解
- 目前仅在bracket和wheel两个类别上验证，泛化到更多工业设计类别尚待探索
- 评价指标偏向代理（B-Spline比率等），未评估曲率连续性、制造容差等真实工业质量

## 相关工作与启发
- **vs DeepCAD/GenCAD/CAD-MLLM**: 它们依赖有限操作集和简单数据集，几何复杂度低；本文从数据增强角度突破，生成更接近工业水平的训练数据
- **vs CAD-Recode**: 同为基于Python脚本+LLM的方向，但CAD-Recode是点云→CAD的条件生成，本文是无条件数据增强，切入角度不同
- **vs 图像条件方法**: GenCAD/CAD-MLLM用渲染图像做提示，受限于源数据集的几何分布；本文用脚本参考曲面避免了这一瓶颈

## 评分
- 新颖性: ⭐⭐⭐⭐ 将工业设计流程编码为LLM提示策略的idea既简单又有效，但方法本身不涉及新的模型架构
- 实验充分度: ⭐⭐⭐⭐ 与多个基线和工业数据集对比，消融研究充分，但仅限bracket类别的定量评估
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰、动机链完整，设计流程的解释直观易懂
- 价值: ⭐⭐⭐⭐ 对CAD程序生成社区有直接实用价值，生成的数据可直接用于训练下游模型

## 与相关工作的对比

## 启发与关联

## 评分

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Automated Multi-Agent Workflows for RTL Design](automated_multi-agent_workflows_for_rtl_design.md)
- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[ICLR 2026\] Sharing State Between Prompts and Programs](../../ICLR2026/code_intelligence/sharing_state_between_prompts_and_programs.md)
- [\[ICLR 2026\] Behavioral Embeddings of Programs: A Quasi-Dynamic Approach for Optimization Prediction](../../ICLR2026/code_intelligence/behavioral_embeddings_of_programs_a_quasi-dynamic_approach_for_optimization_pred.md)
- [\[NeurIPS 2025\] Learning to Solve Complex Problems via Dataset Decomposition](learning_to_solve_complex_problems_via_dataset_decomposition.md)

</div>

<!-- RELATED:END -->
