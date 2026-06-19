---
title: >-
  [论文解读] Context is Key: A Benchmark for Forecasting with Essential Textual Information
description: >-
  [ICML 2025][多模态VLM][上下文辅助预测] 提出 Context is Key（CiK）基准——71个手工设计的预测任务横跨7个领域，每个任务必须结合数值历史和自然语言上下文才能准确预测，同时提出 RCRPS 评估指标和 Direct Prompt 方法，发现 Llama-3.1-405B 的简单提示方法（RCRPS=0.159）大幅领先所有统计模型和时序基础模型。
tags:
  - "ICML 2025"
  - "多模态VLM"
  - "上下文辅助预测"
  - "时间序列基准"
  - "LLM预测器"
  - "RCRPS"
  - "多模态"
---

# Context is Key: A Benchmark for Forecasting with Essential Textual Information

**会议**: ICML 2025  
**arXiv**: [2410.18959](https://arxiv.org/abs/2410.18959)  
**代码**: [GitHub](https://github.com/ServiceNow/context-is-key-forecasting)  
**领域**: 时间序列 / 多模态预测  
**关键词**: 上下文辅助预测, 时间序列基准, LLM预测器, RCRPS, 多模态

## 一句话总结

提出 Context is Key（CiK）基准——71个手工设计的预测任务横跨7个领域，每个任务必须结合数值历史和自然语言上下文才能准确预测，同时提出 RCRPS 评估指标和 Direct Prompt 方法，发现 Llama-3.1-405B 的简单提示方法（RCRPS=0.159）大幅领先所有统计模型和时序基础模型。

## 研究背景与动机

**领域现状**：时间序列预测传统上仅依赖历史数值数据。近年出现两类新趋势：(1) 时序基础模型（如Chronos、Moirai、Lag-Llama）学习跨领域的通用预测能力；(2) LLM适配为预测器（如LLMP、UniTime）利用自然语言整合边信息。

**现有痛点**：尽管LLM预测器号称能整合文本信息，但没有基准能系统评估这一能力。现有上下文辅助基准（如Zhang et al. 2023、Merrill et al. 2024）的文本上下文不保证对预测有用——模型可能仅靠数值就能做出好预测，无法区分是否真正利用了文本。

**核心矛盾**：评估多模态预测能力需要一个"上下文是必要条件"的基准：仅看数值无法准确预测，必须理解文本信息才行。

**本文目标** (1) 创建每个任务都需要文本上下文的预测基准；(2) 系统评估不同类型模型在上下文辅助下的表现差异；(3) 提出适合评估的评分规则。

**切入角度**：手工精心设计每个任务的文本上下文和数值数据，确保上下文包含预测所需的关键信息。例如光伏发电量的预测需要知道"这是太阳能板"才能推断夜间为零。95%任务通过人类验证确认上下文有效。

**核心 idea**：预测不仅需要看"数字怎么变"，还需要理解"发生了什么事"——CiK基准精确衡量模型是否具备这一能力。

## 方法详解

### 整体框架

CiK基准由71个预测任务组成，涵盖气候学、经济学、能源、力学、公共安全、交通、零售7个领域，使用2644条真实时序数据。每个任务包含：数值历史 $\mathbf{X}_H$、自然语言上下文 $\mathbf{C}$、需要预测的未来 $\mathbf{X}_F$。目标是估计 $P(\mathbf{X}_F|\mathbf{X}_H, \mathbf{C})$。关键：每个任务经过人类和LLM评估面板验证上下文的必要性（95%实例确认有效）。

### 关键设计

1. **五类文本上下文分类体系**

    - 功能：系统化定义上下文信息类型，确保基准覆盖实际场景中的各种辅助信息
    - 核心思路：(1) **时间不变信息** $\mathbf{c}_I$：过程描述、变量性质（如"太阳能发电"推断夜间为零）；(2) **未来信息** $\mathbf{c}_F$：未来事件或情景（如"ATM将停机两天"推断零取款）；(3) **历史信息** $\mathbf{c}_H$：超出可见历史的过去统计（如"去年同期峰值100"）；(4) **协变量信息** $\mathbf{c}_{cov}$：关联变量的数值行为；(5) **因果信息** $\mathbf{c}_{causal}$：协变量和目标变量间的因果关系
    - 设计动机：不同类型的上下文对模型能力要求不同——理解时间不变信息需要常识，理解因果信息需要推理能力，这样可以细粒度分析模型在不同类型上下文上的表现

2. **Region of Interest CRPS（RCRPS）评分规则**

    - 功能：专为上下文辅助概率预测设计的评估指标
    - 核心思路：在标准CRPS基础上扩展：(1) **兴趣区域RoI**：对上下文最相关的时间窗口赋予更高权重；(2) **约束满足惩罚** $\beta \cdot \text{CRPS}(v_\mathbf{C}(\tilde{\mathbf{X}}_F), 0)$：当预测违反上下文约束时额外惩罚（$\beta=10$）；(3) **归一化** $\alpha$：使不同量级任务可公平聚合。RCRPS将RoI内外的CRPS各占1/2权重加总
    - 设计动机：标准CRPS平等对待所有时间步，但上下文辅助预测的关键在于模型是否在"上下文相关时间窗口"做对了

3. **Direct Prompt Forecasters**

    - 功能：将LLM直接用作上下文辅助预测器的简单但强力的基线
    - 核心思路：将数值时序格式化为文本（如表格形式），与自然语言上下文拼接成prompt，直接让LLM输出未来数值预测。通过多次采样获得预测分布。不需要微调、不需要特殊tokenization，直接利用LLM的in-context learning能力
    - 设计动机：相比LLMP等需要训练的方法，Direct Prompt测试了LLM"原生"整合数值和文本的能力

### 防记忆污染策略

使用持续更新的活数据源、衍生序列（如事故日志→时序）、轻微变换（加噪/时间偏移）来降低LLM数据污染风险。

## 实验关键数据

### 主实验 — 加权平均 RCRPS↓

| 模型类别 | 模型 | Avg RCRPS↓ | Avg Rank↓ |
|---------|------|-----------|-----------|
| Direct Prompt | **Llama-3.1-405B** | **0.159** | **4.516** |
| Direct Prompt | GPT-4o | 0.274 | 4.381 |
| LLMP | Llama-3-70B (base) | 0.236 | 6.522 |
| 多模态 | UniTime | 0.370 | 14.675 |
| 时序基础模型* | Chronos-Large | 0.326 | 12.298 |
| 时序基础模型* | Moirai-Large | 0.520 | 12.873 |
| 统计模型* | ARIMA | 0.475 | 12.721 |
| 统计模型* | ETS | 0.530 | 15.001 |

### 按上下文类型分析（Direct Prompt RCRPS↓）

| 模型 | 时间不变 | 历史信息 | 未来信息 | 协变量 | 因果 |
|------|---------|---------|---------|-------|------|
| Llama-405B | 0.174 | 0.146 | **0.075** | 0.164 | 0.398 |
| GPT-4o | 0.218 | **0.118** | 0.121 | 0.250 | 0.858 |
| Llama-70B | 0.336 | 0.180 | 0.194 | 0.228 | 0.629 |
| Qwen-2.5-7B | 0.290 | 0.176 | 0.287 | 0.240 | 0.525 |

### 关键发现

- **Direct Prompt Llama-3.1-405B全面领先**：RCRPS 0.159远低于最优时序基础模型Chronos的0.326（改善51%）
- **文本上下文是决定性因素**：时序基础模型无法利用上下文，在需要上下文的任务上系统性落后
- **模型规模关键**：Llama-405B→70B→Qwen-0.5B，RCRPS从0.159到0.290到0.463——更大LLM显著更善于利用上下文
- **因果信息最难**：所有模型在因果推理类任务上表现最差（0.360~0.858），说明因果推理仍是LLM瓶颈
- **Direct Prompt优于LLMP**：无需训练的直接提示反而优于需训练的LLMP，说明预训练LLM的in-context能力足够强

## 亮点与洞察

- **"上下文是关键"直击时序基础模型的盲区**：光伏发电夜间为零的例子极其直观——不知道是太阳能板就无法预测。这揭示了纯数值时序模型的根本局限：它们无法编码领域知识和事件约束。CiK为多模态时序预测领域设定了明确的评估标准
- **RCRPS设计精巧**：通过兴趣区域加权+约束违反惩罚，精确衡量了模型是否"理解并使用了上下文"，避免了标准CRPS对上下文相关时间步的低估

## 局限与展望

- 71个任务全部手工设计，规模有限，难以覆盖所有真实场景
- LLM的数值理解能力仍有局限——涉及精确数值计算的任务（如因果推理）表现较差
- 未测试端到端训练的多模态时序模型（如专门的数值+文本联合训练架构）
- 对数据污染的缓解策略有效性未完全验证
- 评估代价较高（5实例×25次采样×71任务）

## 相关工作与启发

- **vs TimesFM/Chronos等时序基础模型**：它们在纯数值预测上很强，但CiK揭示了无法利用文本上下文是结构性短板
- **vs LLMP (Requeima et al., 2024)**：LLMP需训练适配层，但Direct Prompt的零样本方法反而更好——预训练LLM已包含足够的预测能力
- **vs 现有基准 (Zhang et al., 2023; Merrill et al., 2024)**：这些基准不保证上下文有用，CiK通过手工设计+人类验证确保上下文是必要的
- **启发**：未来时序模型应原生支持多模态输入——将自然语言上下文视为一等公民

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 首个要求上下文必不可少的时序预测基准，填补了重要空白
- 实验充分度: ⭐⭐⭐⭐ 20+模型在71个任务上全面评估，按上下文类型的细粒度分析深入
- 写作质量: ⭐⭐⭐⭐⭐ 动机直观、例子生动、RCRPS设计思路清晰
- 价值: ⭐⭐⭐⭐⭐ 对时序预测社区有重要方向性指引

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] What Limits Virtual Agent Application? OmniBench: A Scalable Multi-Dimensional Benchmark for Essential Virtual Agent Capabilities](what_limits_virtual_agent_application_omnibench_a_scalable_multi-dimensional_ben.md)
- [\[ICML 2026\] FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs](../../ICML2026/multimodal_vlm/futureomni_evaluating_future_forecasting_from_omni-modal_context_for_multimodal_.md)
- [\[ICCV 2025\] MC-Bench: A Benchmark for Multi-Context Visual Grounding in the Era of MLLMs](../../ICCV2025/multimodal_vlm/mc-bench_a_benchmark_for_multi-context_visual_grounding_in_the_era_of_mllms.md)
- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[ICML 2025\] CoCoA-Mix: Confusion-and-Confidence-Aware Mixture Model for Context Optimization](cocoa-mix_confusion-and-confidence-aware_mixture_model_for_context_optimization.md)

</div>

<!-- RELATED:END -->
