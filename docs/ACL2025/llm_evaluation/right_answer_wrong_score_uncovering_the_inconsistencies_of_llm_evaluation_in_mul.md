---
title: >-
  [论文解读] Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA
description: >-
  [ACL 2025][LLM评测][多选问答评估] 系统揭示了LLM在多选问答(MCQA)评估中的不一致性——不同评估策略(RegEx/Logprobs/xFinder)和提示设置(约束/自由生成)组合会导致模型性能报告产生显著差异，且即使是SOTA的LLM-based答案提取器也无法可靠识别推理矛盾，呼吁建立标准化评估协议。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "多选问答评估"
  - "LLM评测一致性"
  - "答案提取策略"
  - "CoT推理"
  - "对抗性评估"
---

# Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in Multiple-Choice QA

**会议**: ACL 2025  
**arXiv**: [2503.14996](https://arxiv.org/abs/2503.14996)  
**作者**: Francesco Maria Molfese, Luca Moroni, Luca Gioffré, Alessandro Scirè, Simone Conia, Roberto Navigli (Sapienza University of Rome / Babelscape)
**代码**: [github.com/SapienzaNLP/mcqa-eval](https://github.com/SapienzaNLP/mcqa-eval)  
**领域**: LLM评测  
**关键词**: 多选问答评估, LLM评测一致性, 答案提取策略, CoT推理, 对抗性评估

## 一句话总结

系统揭示了LLM在多选问答(MCQA)评估中的不一致性——不同评估策略(RegEx/Logprobs/xFinder)和提示设置(约束/自由生成)组合会导致模型性能报告产生显著差异，且即使是SOTA的LLM-based答案提取器也无法可靠识别推理矛盾，呼吁建立标准化评估协议。

## 研究背景与动机

### 问题背景
MCQA是评估LLM最广泛使用的任务之一，覆盖常识推理、科学知识、多领域挑战等方面。其评估看似简单——模型从预定义选项中选择答案即可——但随着CoT等技术的引入，模型在给出最终答案前会生成大量自由文本进行推理，这使得"从模型输出中提取意图答案"变得复杂且不可靠。

### 已有工作的不足
- 以往研究关注的是选项顺序偏置、标签绑定偏差等"任务格式"问题，较少系统分析**评估策略本身**的可靠性
- RegEx方法在模型生成复杂推理链时miss率很高；Logprobs方法无法处理自由文本生成
- 已有LLM-based答案提取器(如xFinder)虽表现更好，但其系统性失败模式尚未被深入研究
- 缺乏将自动评估与**人工判断**进行全面对齐分析的工作

### 核心动机
随着模型越来越多地使用test-time scaling和"thinking"机制(如DeepSeek-R1)，自由生成推理文本成为趋势。评估策略的可靠性直接影响模型间的公平比较。本文旨在回答：现有评估方法到底有多可靠？它们与人工判断的差距在哪里？

## 方法详解

### 整体框架
本文设计了一套三维度分析框架，控制变量地考察**评估策略 × 提示设置 × 基准领域**对报告性能的影响：

- **评估策略**：RegEx（18种正则模式）、Logprobs（首token概率）、xFinder-Llama(8B)/xFinder-Qwen(500M)
- **提示设置**：Zero-Shot(ZS)、Zero-Shot CoT(ZS-CoT)、Zero-Shot Constrained(ZS-Const)、Few-Shot(FS)
- **基准数据集**：MMLU-Redux(5700题/57领域)、OpenBookQA、ARC-Challenge
- **被评估模型**：8个LLM，1B-8B参数规模(Llama系列、Phi系列、Qwen、Mistral、SmolLM2)

### 关键设计1：人工标注与一致性分析
从MMLU-Redux上8个模型×4种提示设置的全部输出中随机抽取1000条实例，由4名标注员提取意图答案。标注选项包括A-D标签或"[No valid answer]"（用于推理矛盾、标签绑定错误、拒答等情况）。200条共享样本用于计算标注员间一致性(Cohen's kappa=98.5)，建立可靠的gold标准。

### 关键设计2：对抗性数据集MMLU-Adversarial
针对xFinder的两种典型失败模式，构建了专门的对抗性数据集：
1. **推理不一致**：推理链支持答案C但结论写"Answer: A"——用Gemini-1.5-Flash保留原始推理并替换最终答案
2. **多答案冲突**：模型对每个选项都给出"正确"的论证——让模型生成对多个选项的合理论证

每种模式各1000条高质量人工审核实例，测试答案提取器的鲁棒性。

### 关键设计3：答案提取器"解题"倾向检测
设计3种对抗性提示，故意制造模糊输出，测试xFinder是否会绕过答案提取任务去直接"解题"。定义两个指标：
- **对抗率(Adversarial Rate)**：xFinder分配有效标签而非"[No valid answer]"的比例
- **相对准确率(Relative Accuracy)**：xFinder选择的标签恰好匹配ground truth的比例

## 实验关键数据

### 表1：评估策略与人工判断的一致性(Cohen's kappa)

| 评估策略 | ZS | ZS-CoT | ZS-Const | FS | 平均 |
|---------|-----|--------|----------|-----|------|
| RegEx | 90.7 | 84.3 | 97.9 | 97.3 | 92.5 |
| Logprobs | 74.7 | — | 94.1 | 90.4 | 86.4 |
| xFinder-Llama | 95.8 | 89.7 | 98.4 | 97.3 | 95.3 |
| xFinder-Qwen | 94.8 | 90.3 | 98.4 | 97.3 | 95.2 |
| Human | 98.2 | 97.0 | 98.7 | 100.0 | 98.5 |

关键发现：从约束提示(ZS-Const)切换到自由生成(ZS)，所有策略一致性大幅下降；xFinder-Llama下降2.6%，RegEx下降7.2%，Logprobs下降19.2%。而人工标注仅下降0.5%。

### 表2：xFinder在对抗性数据集上的检测能力

| 失败模式 | xFinder-Llama正确标记率 | xFinder-Qwen正确标记率 |
|---------|----------------------|----------------------|
| 推理不一致 | 1.9% | 0.6% |
| 多答案冲突 | 10.9% | 0.9% |

xFinder几乎完全无法识别推理矛盾和多答案冲突，说明其本质是在做"最终答案匹配"而非理解推理链。

### 表3：xFinder的"解题"倾向(MMLU-Redux)

| 对抗提示 | xFinder-Llama 对抗率/相对准确率 | xFinder-Qwen 对抗率/相对准确率 |
|---------|------------------------------|------------------------------|
| Prompt A | 58.9% / 68.2% | 45.7% / 29.3% |
| Prompt B | 54.0% / 69.6% | 43.2% / 28.0% |
| Prompt C | 15.3% / 74.8% | 96.9% / 23.1% |

在OBQA上xFinder-Llama的相对准确率高达89.9%，强烈暗示其在"提取答案"时可能偷偷利用了基座模型的知识来"解题"。

## 关键发现

1. **约束提示vs自由生成存在根本性权衡**：约束提示(ZS-Const/FS)让答案提取更可靠但可能抑制推理；自由生成(ZS/ZS-CoT)提升模型真实能力但让评估变得不可靠
2. **领域敏感性**：STEM类任务在自由生成设置下获益最大(ZS比ZS-Const高2.5个准确率点)；人文类任务在不同设置下表现稳定，更依赖事实记忆而非推理
3. **答案偏移量递减效应**：模型输出越长(答案偏移量越大)，准确率越高，但从$10^2$到$10^3$字符仅提升0.6%，边际收益递减
4. **xFinder不是纯粹的答案提取器**：在对抗设置下，它会绕过提取功能直接解题，继承了基座模型的偏置

## 亮点与洞察

- **首次全面对齐分析**：将4种自动评估策略、4种提示设置与人工判断进行系统交叉对比，1000条人工标注提供了可靠基线
- **问题定义精准**：不是笼统地说"MCQA评估有问题"，而是精确识别了失败模式（推理不一致、多答案冲突）并构建对抗数据集量化
- **发现具有实际影响力**：当前排行榜和论文中的MCQA分数因评估策略差异可能存在系统性偏差，直接影响模型间公平比较
- **MMLU-Adversarial数据集**：为未来研究提供了可直接使用的benchmark，推动更鲁棒的答案提取方法研发
- **揭示LLM-based评估器的"角色混淆"**：答案提取器不应该具备解题能力，否则会引入系统性偏差——这是一个深刻的设计层面的洞见

## 局限性

- **仅限英文**：未涉及多语言或跨语言场景，而多语言LLM的答案提取可能面临更大挑战
- **模型规模有限**：仅评估1B-8B参数模型，未覆盖>8B的大模型（如Llama-70B、GPT-4等）
- **基准数量有限**：仅覆盖3个MCQA数据集，未涉及更复杂的推理任务或知识密集型任务
- **对抗数据集依赖Gemini生成**：MMLU-Adversarial的"推理不一致"模式由Gemini-1.5-Flash生成，可能引入生成模型自身的偏置
- **未评估最新reasoning模型**：如DeepSeek-R1、o1等使用test-time scaling的模型，其输出可能更长更复杂

## 相关工作与启发

- **Robinson et al. (2023)**：研究了MCQA中符号绑定问题，即模型在不同选项排列下表现不一致。本文进一步发现这影响了答案提取的可靠性
- **Zheng et al. (2024)**：记录了LLM的位置偏置。本文聚焦评估端而非模型端
- **Wang et al. (2024a)**：发现首token概率与文本答案不匹配。本文系统量化了这种不匹配的规模和影响
- **Yu et al. (2024) xFinder**：本文的核心对比对象。xFinder虽是SOTA答案提取器，但本文揭示了其在对抗场景下的脆弱性和"解题"倾向
- **启发**：评估方法本身需要被评估——这是一个元层面的研究方向。未来可能需要引入推理链一致性检测、多答案冲突识别等机制

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统性地将MCQA评估策略与人工判断对齐，并构建对抗数据集量化失败模式
- 实验充分度: ⭐⭐⭐⭐ — 8模型×4提示×3数据集×4策略的全面交叉实验，加上1000条人工标注
- 写作质量: ⭐⭐⭐⭐⭐ — 结构以4个RQ层层递进，图表数据翔实，论证逻辑清晰
- 价值: ⭐⭐⭐⭐ — 对当前LLM评估实践有直接警示意义，MMLU-Adversarial可推动后续研究

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] WiCkeD: A Simple Method to Make Multiple Choice Benchmarks More Challenging](wicked_a_simple_method_to_make_multiple_choice_benchmarks_more_challenging.md)
- [\[ACL 2026\] BenchMarker: An Education-Inspired Toolkit for Highlighting Flaws in Multiple-Choice Benchmarks](../../ACL2026/llm_evaluation/benchmarker_an_education-inspired_toolkit_for_highlighting_flaws_in_multiple-cho.md)
- [\[NeurIPS 2025\] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation](../../NeurIPS2025/llm_evaluation/beyond_the_singular_revealing_the_value_of_multiple_generations_in_benchmark_eva.md)
- [\[ACL 2025\] ELABORATION: A Comprehensive Benchmark on Human-LLM Competitive Programming](elaboration_competitive_programming.md)
- [\[ACL 2025\] HomeBench: Evaluating LLMs in Smart Homes with Valid and Invalid Instructions Across Single and Multiple Devices](homebench_evaluating_llms_in_smart_homes_with_valid_and_invalid_instructions_acr.md)

</div>

<!-- RELATED:END -->
