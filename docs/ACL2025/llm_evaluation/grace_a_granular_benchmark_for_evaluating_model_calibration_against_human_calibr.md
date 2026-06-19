---
title: >-
  [论文解读] GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration
description: >-
  [ACL 2025][LLM评测][模型校准] 提出GRACE基准，通过渐进式增量问答和真人vs模型竞赛收集1749个数据点，以人类校准表现为参照评估LLM校准能力，并引入CalScore指标发现：虽然人类准确率可能低于模型，但人类在校准方面普遍优于SOTA模型——模型在不确定时过度自信、在正确时又信心不足。
tags:
  - "ACL 2025"
  - "LLM评测"
  - "模型校准"
  - "人类校准"
  - "增量问答"
  - "对抗性基准"
  - "置信度评估"
  - "CalScore"
---

# GRACE: A Granular Benchmark for Evaluating Model Calibration Against Human Calibration

**会议**: ACL 2025  
**arXiv**: [2502.19684](https://arxiv.org/abs/2502.19684)  
**代码**: [GitHub](https://github.com/yysung/advcalibration)  
**领域**: LLM评测  
**关键词**: 模型校准, 人类校准, 增量问答, 对抗性基准, 置信度评估, CalScore

## 一句话总结

提出GRACE基准，通过渐进式增量问答和真人vs模型竞赛收集1749个数据点，以人类校准表现为参照评估LLM校准能力，并引入CalScore指标发现：虽然人类准确率可能低于模型，但人类在校准方面普遍优于SOTA模型——模型在不确定时过度自信、在正确时又信心不足。

## 研究背景与动机

LLM常常"信心满满地犯错"——模型的置信度与实际准确率之间存在严重不匹配，这导致用户过度信任模型，甚至否定自己的正确判断。

现有校准研究存在三个核心缺陷：

**缺乏与人类校准的对比**：现有工作只进行模型间的校准比较，没有纳入人类校准表现作为参照。但用户期望模型至少与人类一样校准——当模型的校准表现比人类更差时，用户往往没有准备应对这些错误。

**粒度不足**：传统校准评估只计算整个数据集的聚合指标（如ECE），无法识别具体哪些问题上模型的校准表现特别差。

**现有基准过于简单**：现有增量问答数据集对现代模型来说太容易——GPT-4在TrickMe数据集上仅看60%线索就有80%准确率，无法有效测试校准能力。

GRACE通过三个创新解决这些问题：（1）专家创作的对抗性增量问题；（2）真人vs模型现场竞赛收集人类校准数据；（3）融入人类表现的CalScore指标。

## 方法详解

### 整体框架

GRACE的构建流程分为三个阶段：

1. **问题创作**：专家出题+对抗性编辑界面
2. **数据收集**：离线模型置信度 + 在线人机竞赛
3. **校准评估**：CalScore指标

### 关键设计

#### 1. 增量对抗性问题设计

每个问题包含至少5个线索句，按**难度递减**排列，所有线索指向同一答案。关键特征：
- **对抗性**：使用人-AI协作界面创作。出题者实时看到GPT-3.5对每个线索的猜测和置信度，据此调整线索使其对模型更难但对人类仍可解答
- **增量性**：每个线索都是一个决策点——参与者可以选择回答或等待更多线索
- **质量控制**：6名出题者+10名编辑，每题经过作者、类别编辑和总编三轮审核

最终数据集包含243个问答对，共1236个线索句，覆盖科学、历史、文学等6个类别。

#### 2. 人机竞赛——真人buzzpoint收集

模拟知识竞赛形式：主持人朗读问题，人类队伍和模型同时竞答——
- **模型buzzpoint**：预先计算。模型对每个线索计算置信度，超过阈值时"抢答"
- **人类buzzpoint**：现场实时记录。玩家按物理抢答器，主持人验证答案
- **竞赛规模**：3场竞赛，17支人类队伍 vs 3个LLM（GPT-4o、GPT-4、Mistral-7b），共93场比赛，1749个数据点

#### 3. CalScore——人类基准校准指标

**模型校准误差（MCE）**：

$$\text{MCE} = 1 - r(\mathbb{E}_t[g_t c_t])$$

其中 $g_t$ 是线索 $t$ 处猜测的正确性（1或-1），$c_t$ 是模型置信度，$r(\cdot)$ 归一化到[0,1]。

**CalScore（人类调整后的校准误差）**：

$$\text{CalScore}_q = 1 - r(\mathbb{E}_t[(1-h_t) g_t c_t])$$

其中 $h_t$ 是到线索 $t$ 时人类正确回答的累积概率。关键设计：
- 用 $(1-h_t)$ 加权：**在人类还不知道答案时模型的表现权重更高**
- 模型在人类还不确定时自信且正确 → 高奖励
- 模型在人类还不确定时自信但错误 → 高惩罚（因为人类无法纠错）
- 支持逐问题评估，不仅是聚合指标

### 置信度获取方式

两种方式并行评估：
- **Logit-based**：token logit概率的指数平均值
- **Verbalized**：提示模型直接表达置信度

## 实验关键数据

### 校准指标对比

**Verbalized置信度**：

| 模型 | Brier Score | ECE | MCE | CalScore |
|------|:---:|:---:|:---:|:---:|
| GPT-4 | 0.274 | 0.259 | 0.584 | 0.588 |
| GPT-4o | 0.266 | 0.224 | 0.601 | 0.604 |
| Llama-3.1-70B | 0.373 | 0.392 | 0.685 | 0.719 |
| Llama-2-70b | 0.490 | 0.570 | 0.739 | 0.803 |
| Llama-3.1-8B | 0.623 | 0.693 | 0.774 | 0.843 |
| Mistral-7b | 0.716 | 0.784 | 0.790 | 0.881 |

**Logit-based置信度**：

| 模型 | Brier Score | ECE | MCE | CalScore |
|------|:---:|:---:|:---:|:---:|
| GPT-4o | 0.341 | 0.353 | 0.654 | 0.661 |
| Llama-3.1-70B | 0.323 | 0.339 | 0.651 | 0.679 |
| GPT-4 | 0.380 | 0.388 | 0.672 | 0.684 |
| Llama-3.1-8B | 0.302 | 0.397 | 0.675 | 0.718 |

### 人类vs模型抢答表现

- **Top四分之一人类队伍**的累积正确抢答率峰值超过最佳模型的两倍
- GPT-4和GPT-4o的错误抢答率远高于所有人类队伍
- 模型在问题早期（线索少、信息不足时）尤其过度自信
- 随着线索增多，人类在正确时更倾向抢答（信心增强），而模型反而更不倾向抢答——与直觉相反

### 条件概率分析

- **P(buzz|correct)**：人类>50%，模型<45%。模型在正确时信心不足
- **P(buzz|incorrect)**：GPT-4最高。模型（尤其GPT-4）在错误时过度自信
- 人类看到更多线索后正确时抢答概率增加，模型则下降

### 关键发现

1. **CalScore > MCE（所有模型）**：纳入人类表现后校准误差一致增大，揭示了传统指标低估的校准缺陷
2. **弱模型的CalScore/MCE差距更大**：弱模型在考虑人类表现后暴露出更多校准问题
3. **模型强于检索，弱于抽象推理**：模型在涉及具体专名的问题上表现好（如含有"TRF2蛋白"的端粒问题），但在抽象描述、需要多步推理的问题上严重错校准
4. **模型的错误答案"不合理"**：人类给出的错误答案与正确答案在领域内相关，但模型可能给出完全不相关的答案（如哲学问题答"费马小定理"），显示出更根本的校准缺陷
5. **人类专家间校准差异大**：即使是经验丰富的知识竞赛选手，校准表现也存在显著差异。最强的人类显著优于顶级模型，但并非所有人类都如此

## 亮点与洞察

1. **知识竞赛形式的巧妙设计**：将校准评估"游戏化"——抢答机制天然要求参与者在准确性和速度之间权衡，这恰是校准能力的核心。模型和人类在同一任务上直接对比，消除了以往研究中比较框架的不一致性
2. **CalScore的人类基准思想**：不是问"模型校准好不好"，而是问"模型校准比人类好不好"——这才是实际部署中用户关心的问题。在人类还不确定时模型信心满满地犯错，远比两者都确定时的校准误差更有实际危害
3. **对抗性出题的有效性**：GPT-4在GRACE上直到看到90%+的线索才达到50%准确率，而在TrickMe等旧基准上60%线索就有80%准确率。这为校准评估提供了更具区分度的测试场景
4. **社区参与的附带价值**：现场竞赛不仅收集了高质量数据，还激发了非研究者对AI的兴趣和参与

## 局限与展望

- **任务形式局限**：GRACE仅涵盖事实性问答场景，未扩展到开放式生成、多轮对话等更广泛的NLP任务
- **CalScore不够全面**：作为基线指标，CalScore未捕捉所有形式的不确定性和错校准
- **数据规模有限**：243个问题规模较小，可能不足以进行统计上稳健的跨类别分析
- **人类参与者偏向专家**：所有竞赛参与者都是有经验的知识竞赛选手，无法代表普通用户的校准表现
- **缺乏校准改进方法**：GRACE主要是诊断工具，未提出如何改进模型校准的方法

## 相关工作与启发

- **增量问答（Incremental QA）**：Boyd-Graber等人的早期工作。GRACE在此基础上加入了对抗性创作和置信度评估
- **对AI辅助决策的启发**：校准不良的模型在人类-AI协作中可能造成负面影响——当模型自信但错误且人类不确知答案时，人类更可能被误导
- **对校准改进研究的启发**：GRACE可作为校准方法的评测平台，评估verbalized confidence改进、个性化abstention策略等
- **Watson/Jeopardy!的学术化**：GRACE某种程度上是IBM Watson知识竞赛挑战的现代学术版，但加入了校准维度和多模型比较

## 评分

- **新颖性**: ⭐⭐⭐⭐⭐ 首个将人类校准纳入模型校准评估的基准+知识竞赛形式的数据收集+CalScore指标，三重创新
- **实验充分度**: ⭐⭐⭐⭐ 6个LLM+17支人类队伍+93场比赛+多种校准指标对比+定性分析+玩家反馈
- **写作质量**: ⭐⭐⭐⭐⭐ 问题定义清晰、实验设计严谨、图表丰富且信息量大
- **价值**: ⭐⭐⭐⭐ 作为基准和诊断工具价值高，CalScore为校准评估提供了新范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] A Conformal Risk Control Framework for Granular Word Assessment and Uncertainty Calibration of CLIPScore Quality Estimates](a_conformal_risk_control_framework_for_granular_word_assessment_and_uncertainty_.md)
- [\[ACL 2025\] PATCH: Psychometrics-Assisted Benchmarking of LLMs Against Human Populations](patch_psychometrics-assisted_benchmarking_of_large_language_models_against_human.md)
- [\[ACL 2025\] Atomic Calibration of LLMs in Long-Form Generations](atomic_calibration_of_llms_in_long-form_generations.md)
- [\[ACL 2025\] Influences on LLM Calibration: A Study of Response Agreement, Loss Functions, and Prompt Styles](influences_on_llm_calibration_a_study_of_response_agreement_loss_functions_and_p.md)
- [\[NeurIPS 2025\] On the Entropy Calibration of Language Models](../../NeurIPS2025/llm_evaluation/on_the_entropy_calibration_of_language_models.md)

</div>

<!-- RELATED:END -->
