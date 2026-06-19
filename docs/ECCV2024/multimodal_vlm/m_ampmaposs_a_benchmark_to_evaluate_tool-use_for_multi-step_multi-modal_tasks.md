---
title: >-
  [论文解读] m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks
description: >-
  [ECCV 2024][多模态VLM][工具使用] 提出 m&m's 基准，包含 4K+ 多步骤多模态任务和 33 个可执行工具，系统评估 10 个 LLM 在不同规划策略（多步 vs 逐步）、计划格式（JSON vs 代码）和反馈类型（解析/验证/执行）下的工具使用能力，发现多步JSON规划配合反馈是当前最优设计。
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "工具使用"
  - "LLM规划"
  - "多步骤多模态"
  - "基准评测"
  - "规划策略"
---

# m&m's: A Benchmark to Evaluate Tool-Use for Multi-step Multi-modal Tasks

**会议**: ECCV 2024  
**arXiv**: [2403.11085](https://arxiv.org/abs/2403.11085)  
**代码**: [GitHub](https://github.com/RAIVNLab/mms)  
**领域**: 多模态VLM  
**关键词**: 工具使用, LLM规划, 多步骤多模态, 基准评测, 规划策略

## 一句话总结

提出 m&m's 基准，包含 4K+ 多步骤多模态任务和 33 个可执行工具，系统评估 10 个 LLM 在不同规划策略（多步 vs 逐步）、计划格式（JSON vs 代码）和反馈类型（解析/验证/执行）下的工具使用能力，发现多步JSON规划配合反馈是当前最优设计。

## 研究背景与动机

现实世界的多模态问题很少能由单一模型解决，通常需要组合多个模型的多步计算方案。工具增强的LLM在自动生成这类计划方面前景巨大，但缺乏标准化基准来评估LLM作为多步多模态任务规划器的设计决策：

**规划策略**：LLM应该一次生成完整计划还是逐步生成？现有工作各自使用不同策略但从未系统比较

**计划格式**：应该用Python代码还是JSON等结构化格式？不同格式对可执行性有何影响

**反馈机制**：解析反馈、规则验证反馈、执行反馈各自如何影响规划质量

现有基准的关键不足：
- **TaskBench**：不提供工具实现，用占位符文件名（如"example.png"），无法执行
- **ToolEmu**：用LLM模拟工具执行而非真实执行，无法研究执行反馈
- **ToolBench/ToolFormer**：不支持多模态或缺乏ground truth计划

核心矛盾：规划智能体的设计空间组合爆炸式增长，但没有一个基准能支持沿所有维度进行公平评估。m&m's 首次提供了支持真实多模态输入、真实工具执行、人工验证计划、以及多种评估维度的统一基准。

## 方法详解

### 整体框架

m&m's 包含三个核心组件：
1. **基准数据集**：4427个原始任务，1565个人工验证任务，882个工具平衡子集
2. **33个可执行工具**：13个ML模型 + 11个图像处理模块 + 9个公共API
3. **模块化规划系统**：LLM + 解析器 + 验证器 + 执行器，基于AutoGen框架

### 关键设计

1. **数据生成流水线**:

    - 功能：自动生成多样化的查询-计划对
    - 核心思路：五步流程——
        - **工具图构建与采样**：将33个工具构建为有向图（边表示输出-输入类型匹配的合法连接），采样子图作为工具序列
        - **输入示例采样**：从ImageNet、SQuAD、Visual Genome、MagicBrush、LibriSpeech等11个数据集的验证集中采样真实输入
        - **查询生成**：用GPT-4根据工具序列和输入示例生成自然语言用户查询
        - **计划生成**：用规则程序（非GPT-4）生成ground truth计划，包括工具名、参数名和参数值
        - **人工验证**：3名标注员验证每个查询-计划对
    - 设计动机：使用真实输入（而非"example.jpg"占位符）确保查询的真实性和计划的可执行性；使用规则程序而非GPT-4生成计划，消除幻觉和错误计划的可能

2. **工具图设计**:

    - 功能：定义33个工具之间的合法连接关系
    - 核心思路：有向图中每个工具是节点，边表示源工具的输出类型匹配目标工具的输入类型。例如 image_classification → wikipedia_simple_search 是合法的（文本标签 → 搜索查询）
    - 工具分三类：
        - **ML模型**（13个）：文本生成、图像分类、目标检测、VQA等（HuggingFace）
        - **图像处理**（11个）：裁剪、分割、计数、背景模糊、Emoji叠加等（来自VisProg）
        - **公共API**（9个）：天气、地理位置、数学事实、维基百科搜索等（RapidAPI）

3. **模块化规划智能体**:

    - 功能：灵活组合不同规划策略、格式和反馈类型
    - 核心思路：
        - **多步规划**：LLM一次生成完整计划（所有工具步骤）
        - **逐步规划**：LLM每次只预测一个动作，获取反馈后预测下一个
    - 三种反馈机制：
        - **解析反馈**：LLM输出能否被解析为有效JSON/代码
        - **验证反馈**：工具是否存在、连接是否合法、参数名是否正确
        - **执行反馈**：实际执行工具，返回输出或执行错误信息
    - 设计动机：模块化设计使得每个维度可独立变化，支持组合式实验

4. **评估指标设计**:

    - 功能：从工具选择和工具调用两个维度评估规划质量
    - 核心思路：三个主要指标——
        - **tool-F1**：工具名预测的F1分数（衡量工具选择）
        - **argname-F1**：参数名预测的F1分数（衡量工具调用）
        - **pass rate**：无执行错误的预测比例（衡量可执行性）
    - 设计动机：分离工具选择和工具调用的评估，避免将规划错误与执行错误混为一谈

5. **替代计划生成**:

    - 功能：为每个查询生成多个合法替代计划，避免仅靠单一ground truth评估
    - 核心思路：为每个工具生成语法合法（输入输出类型匹配）和语义合法（功能等价）的替代工具，组合所有位置的替代工具得到完整替代计划集
    - 影响：考虑替代计划后，plan accuracy可恢复1-5%的误判

### 损失函数 / 训练策略

本文不涉及模型训练，而是评估现有LLM的零样本规划能力。评估覆盖：
- 5个开源LLM：Llama-2-7B/13B/70B、Mixtral-8x7B、Llama-3-70B
- 2个代码LLM：CodeLlama-34B/70B
- 3个商业LLM：GPT-3.5-turbo、GPT-4、GPT-4o
- 总计 10个模型 × 2种策略 × 2种格式 × 4种反馈组合

## 实验关键数据

### 主实验：规划策略比较（Tool-F1）

| 模型 | 逐步规划 | 多步规划 | 差值 |
|------|---------|---------|------|
| Llama-2-7B | ~20 | 29.78 | +~10 |
| Llama-2-13B | ~32 | 42.27 | +~10 |
| GPT-3.5-turbo | ~59 | 80.52 | +21.8 |
| GPT-4 | ~83 | 88.46 | +~5 |
| GPT-4o | ~85 | 89.28 | +~4 |

### 主实验：反馈类型对工具调用的影响

| 模型 | 基线P（pass rate） | +V Δ | +E Δ | +VE Δ |
|------|---------|------|------|-------|
| Llama-2-7B | 28.23 | +18.14 | +10.32 | +13.72 |
| Llama-2-13B | 38.10 | +29.93 | +32.99 | +23.92 |
| Mixtral-8x7B | 75.74 | +10.32 | +8.96 | +10.77 |
| GPT-3.5-turbo | 89.46 | +6.69 | +7.26 | +6.92 |
| GPT-4 | 97.73 | +1.13 | -1.25 | +2.15 |

### 消融实验：JSON vs 代码格式

| 模型 | JSON tool-F1 | Code tool-F1 | JSON pass rate | Code pass rate |
|------|-------------|-------------|---------------|---------------|
| Llama-2-13B | 42.27 | ~40 | 38.10 | ~15 |
| Mixtral-8x7B | 66.79 | ~65 | 75.74 | ~40 |
| GPT-3.5-turbo | 80.52 | ~79 | 89.46 | ~55 |
| GPT-4 | 88.46 | ~87 | 97.73 | ~75 |

### 关键发现

- **多步规划一致优于逐步规划**：所有模型的tool-F1在多步规划下都更高。逐步规划的模型在收到正面反馈后倾向于提前终止，遗漏必要的工具步骤
- **反馈大幅提升可执行性但可能轻微损害工具选择**：验证和执行反馈可将pass rate提升20-30%，但tool-F1可能下降<5%，因为模型有时在修复错误时错误地替换了正确的工具
- **JSON格式远优于代码生成的可执行性**：两者在工具选择上差异不大（<3%），但代码生成的pass rate大幅下降。代码生成中常见错误是无法正确访问上一步工具的输出
- **验证反馈比执行反馈更有针对性**：验证器能精确指出错误位置，而执行器返回的错误信息可能模糊和令人困惑
- **能力越强的模型对反馈的边际收益越小**：GPT-4已接近满分pass rate，反馈改善有限

## 亮点与洞察

- **首个支持真实输入和真实执行的多步多模态工具使用基准**：填补了ToolEmu/TaskBench的关键空白
- **系统化的设计空间探索**：10个模型 × 2×2×4 的组合实验提供了全面的实证结论
- **规则程序生成ground truth计划**：避免GPT-4幻觉导致的计划错误，保证数据质量
- **替代计划机制**：承认同一任务可有多种正确解法，使评估更公平
- **对工具图的形式化定义**：为后续工具使用研究提供了规范的问题建模

## 局限与展望

- 仅考虑顺序计划，未支持动态/条件分支计划（如"如果图像分类结果是A则执行步骤X，否则执行步骤Y"）
- 未探索更复杂的prompt策略（如Tree-of-Thoughts），仅用直接和ReACT风格
- 部分工具是非确定性的（如图像生成），导致执行结果评估困难
- 仅评估纯文本LLM规划器，未测试多模态LLM（如GPT-4V直接看图规划）
- 33个工具的规模相对较小（ToolBench有3451个），可能不足以测试大规模工具选择能力
- 人工验证的一致率为75.75%，说明任务的正确性判定存在一定主观性

## 相关工作与启发

- **VisProg/ViperGPT**（Gupta et al. / Surís et al.）：用Python代码格式的多步视觉推理，是多模态工具使用的早期探索
- **HuggingGPT**（Shen et al., 2023）：多步规划但不执行计划，仅评估计划准确性
- **ToolFormer**（Schick et al., 2024）：让语言模型自学工具使用，但用自然语言而非代码接口
- **TaskBench**（Shen et al., 2023）：最相关的并发工作，但使用占位符输入且用GPT-4生成答案
- **AutoGen**（Wu et al., 2023）：提供了多智能体对话框架，被m&m's用作规划系统的基础
- 启发：**结构化格式（JSON）的约束性反而是优势**——刚性结构减少了输出格式错误，使计划更可执行。未来的工具使用系统应在灵活性和可靠性之间寻找平衡

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个支持完整执行链的多模态工具使用基准，设计空间的系统化探索
- 实验充分度: ⭐⭐⭐⭐⭐ 10个模型×多维度交叉实验，15+评估指标，极为全面
- 写作质量: ⭐⭐⭐⭐ 问题陈述清晰，实验组织系统，发现总结务实
- 价值: ⭐⭐⭐⭐ 为LLM工具使用研究提供了标准化评测平台和实证设计指导

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Multi-modal Data Spectrum: Multi-modal Datasets are Multi-dimensional](../../ICLR2026/multimodal_vlm/multi-modal_data_spectrum_multi-modal_datasets_are_multi-dimensional.md)
- [\[ECCV 2024\] ShareGPT4V: Improving Large Multi-Modal Models with Better Captions](sharegpt4v_improving_large_multi-modal_models_with_better_captions.md)
- [\[ACL 2026\] AdaTooler-V: Adaptive Tool-Use for Images and Videos](../../ACL2026/multimodal_vlm/adatooler-v_adaptive_tool-use_for_images_and_videos.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ECCV 2024\] MathVerse: Does Your Multi-modal LLM Truly See the Diagrams in Visual Math Problems?](mathverse_does_your_multi-modal_llm_truly_see_the_diagrams_in_visual_math_proble.md)

</div>

<!-- RELATED:END -->
