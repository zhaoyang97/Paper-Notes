---
title: >-
  [论文解读] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model
description: >-
  [ACL 2025][多模态VLM][多模态奖励模型] 基于InternLM-XComposer2.5构建判别式多模态奖励模型IXC-2.5-Reward，通过精心构建跨文本/图像/视频的多领域偏好数据集训练，在多模态奖励基准VL-RewardBench上以70.0% Macro Acc超越GPT-4o（62.4%），并展示了RL训练、Best-of-N测试时缩放和数据清洗三大应用。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态奖励模型"
  - "RLHF"
  - "偏好对齐"
  - "PPO"
  - "测试时缩放"
  - "数据清洗"
---

# InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model

**会议**: ACL 2025  
**arXiv**: [2501.12368](https://arxiv.org/abs/2501.12368)  
**作者**: Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao 等 (上海AI实验室, 港中文等)
**代码**: [GitHub](https://github.com/InternLM/InternLM-XComposer)  
**领域**: 多模态VLM  
**关键词**: 多模态奖励模型, RLHF, 偏好对齐, PPO, 测试时缩放, 数据清洗

## 一句话总结

基于InternLM-XComposer2.5构建判别式多模态奖励模型IXC-2.5-Reward，通过精心构建跨文本/图像/视频的多领域偏好数据集训练，在多模态奖励基准VL-RewardBench上以70.0% Macro Acc超越GPT-4o（62.4%），并展示了RL训练、Best-of-N测试时缩放和数据清洗三大应用。

## 研究背景与动机

### 问题背景
奖励模型（Reward Model）在LLM领域已被广泛研究，是RLHF训练和测试时缩放（test-time scaling）的关键组件。然而在多模态大模型（LVLM）领域，公开可用的多模态奖励模型极为稀缺，闭源模型的实现细节也不透明。

### 已有工作的不足
- **领域受限**：已有多模态RM（如RLAIF-V、LLaVA-Critic）大多局限于减少幻觉等特定领域，缺乏对指令跟随、安全性、推理等多领域的覆盖
- **基座模型弱**：部分工作使用较弱的基座模型，导致多模态RM显著落后于纯语言RM
- **偏好数据匮乏**：现有偏好数据以文本为主，图像偏好数据有限，视频偏好数据几乎为零
- **生成式RM局限**：通过提示LVLM生成评价的方式（generative RM）在区分能力上不如判别式RM

### 核心动机
构建一个跨模态（文本+图像+视频）、跨领域（指令跟随、通用理解、文档理解、数学推理、视频理解）的通用判别式多模态奖励模型，同时验证其在RL训练、测试时缩放和数据清洗中的实际价值。

## 方法详解

### 整体框架
IXC-2.5-Reward基于InternLM-XComposer2.5（IXC-2.5）的SFT模型构建。保留预训练的视觉编码器和MLP投影层不动，将最后的线性层替换为一个**评分头（score head）**，将所有token的平均隐藏状态特征映射为一个标量奖励分数 $r(x, y)$。训练时冻结视觉编码器和投影层，仅训练LLM（InternLM2）和评分头。

### 关键设计1：多领域多模态偏好数据构建

数据来源分两部分：

**开源数据**（文本为主）：包括Tulu-3指令跟随数据、UltraFeedback通用反馈、HHH/PKU-Safe等安全数据、WildVision-Battle多模态对话、LLaVA-Critic/VL-Feedback/RLAIF-V等图像偏好数据。

**新收集数据**（重点补全图像和视频）：
- 图像：覆盖指令跟随（MM-IFDPO-23k）、知识VQA（KVQA、A-OKVQA、PMC-VQA）、文档理解（AI2D、ChartQA、DVQA等）、数学推理（GeoQA、CLEVR-Math、TabMWP等）
- 视频：TrafficQA、FunQA、MiraData

新数据的偏好标注流程：用IXC-2.5 SFT模型对每个prompt生成多个回复，然后用**GPT-4o进行成对评估**（通用/文档类）或**验证器函数对比ground-truth**（数学/指令跟随类）来确定chosen和rejected。

### 关键设计2：长度约束去偏

训练偏好数据时，移除chosen回复显著长于rejected回复的数据对，防止奖励模型将"长度"等同于"质量"。这是因为LLM-as-Judge评估存在严重的长度偏好——GPT-4o倾向于给更长的回复更高的分数。

消融实验表明，去掉长度约束后WildVision分数从74.6升至76.2（因为Judge偏好长回复），但平均token长度从274暴增至361，实际用户体验下降。作者选择保留长度约束以优化真实用户体验而非刷榜。

### 关键设计3：三大下游应用

**应用1 — PPO强化学习训练**：用IXC-2.5-Reward作为奖励信号，通过PPO算法训练策略模型IXC-2.5-Chat。Critic模型从IXC-2.5-Reward初始化，使用GAE估计优势函数，PPO clipped目标更新策略。训练数据重点覆盖指令跟随和开放对话。

**应用2 — Best-of-N测试时缩放**：对每个prompt用IXC-2.5-Chat生成N个不同回复，用IXC-2.5-Reward打分并选择最高分回复。N=4时即可获得显著提升。

**应用3 — 数据清洗**：低奖励分数与问题样本强相关，包括幻觉、空回答、图文不匹配等。可用于过滤预训练和后训练数据中的噪声样本。

### 训练细节
- 奖励模型学习率：1e-5，batch size 256
- PPO策略模型学习率：5e-5，batch size 256
- PPO超参：$\gamma=0.99$, $\beta=0.95$, $\epsilon=0.2$
- 损失函数：Bradley-Terry偏好损失 $\mathcal{L}_{\text{RM}} = -\mathbb{E}[\log\sigma(r(x, y_w) - r(x, y_l))]$

## 实验关键数据

### 表1：VL-RewardBench多模态奖励基准结果

| 模型 | 参数量 | General | Hallucination | Reasoning | Overall Acc | Macro Acc |
|------|--------|---------|---------------|-----------|-------------|-----------|
| GPT-4o | - | 49.1 | 67.6 | 70.5 | 65.8 | 62.4 |
| Gemini-1.5-Pro | - | 50.8 | 72.5 | 64.2 | 67.2 | 62.5 |
| LLaVA-Critic-8B | 8B | 54.6 | 38.3 | 59.1 | 41.2 | 44.0 |
| InternVL2-8B | 8B | 35.6 | 41.1 | 59.0 | 44.5 | 45.2 |
| Llama-3.2-90B | 90B | 42.6 | 57.3 | 61.7 | 56.2 | 53.9 |
| **IXC-2.5-Reward** | **7B** | **84.7** | **62.5** | **62.9** | **65.8** | **70.0** |

IXC-2.5-Reward以7B参数量在Macro Acc上超越所有模型（包括闭源），General类别以84.7%大幅领先。

### 表2：IXC-2.5-Chat指令跟随与对话评测（≤10B开源模型）

| 基准 | 类型 | 闭源SOTA | 开源SOTA | IXC-2.5 (SFT) | IXC-2.5-Chat (PPO) |
|------|------|---------|---------|---------------|-------------------|
| WildVision | Open | 89.2 (GPT-4o) | 67.3 | 37.5 | **74.6** |
| MIA-bench | Open | 88.6 (GPT-4o) | 80.7 | 80.4 | **84.0** |
| MM-MT | Open | 7.72 (GPT-4o) | 5.45 | 3.85 | **5.70** |
| MM-Vet v2 | Open | 71.8 | 58.1 | 45.8 | **54.8** |

PPO训练后在指令跟随和开放对话上获得大幅提升（WildVision从37.5→74.6），同时知识、推理、文档理解等能力未下降。

### 表3：Best-of-N测试时缩放效果

| 设定 | Avg Tokens | WildVision | MIA | MM-MT | MM-Vet v2 |
|------|-----------|------------|-----|-------|-----------|
| IXC-2.5-Chat | 274 | 74.6 | 84.0 | 5.70 | 54.8 |
| IXC-2.5-Chat + BoN (N=4) | 283 | **77.7** | **87.3** | **6.03** | **56.3** |

BoN采样在PPO基础上进一步提升性能，且平均token长度仅从274增至283，说明提升来自回复质量而非长度hack。

### 表4：长度约束消融

| 设置 | Avg Tokens | WildVision | MIA | MM-MT | MM-Vet v2 |
|------|-----------|------------|-----|-------|-----------|
| 去除长度约束 | 361 | 76.2 | 87.0 | 5.86 | 56.6 |
| 保留长度约束（最终版） | 274 | 74.6 | 84.0 | 5.70 | 54.8 |

## 关键发现

1. **判别式RM远优于生成式RM**：在VL-RewardBench上，7B的判别式IXC-2.5-Reward（70.0%）大幅超过90B的Llama-3.2生成式RM（53.9%），说明专门训练的评分头比prompting方式更有效
2. **多模态RM可保持语言能力**：IXC-2.5-Reward在纯文本RewardBench上达88.6%，RM-Bench上达68.8%，接近专门的纯语言RM（如InternLM2-7B-Reward的87.6%）
3. **长度偏好是多模态评估的系统性问题**：不仅LLM评估存在长度偏好，多模态VQA评估同样如此，去掉长度约束后分数反而更高但用户体验更差
4. **PPO训练不损害其他能力**：在知识（MMBench、MMMU）、推理（MathVista）、文档理解（ChartQA）等方面，PPO训练后的Chat模型与SFT模型持平

## 亮点与洞察

- **简洁有效的架构**：不引入复杂的多模态对齐模块，仅在已对齐的LVLM上加一个评分头并训练偏好数据，思路简单但效果显著
- **数据为王的设计理念**：通过系统性地补全图像/视频/文档/推理等领域的偏好数据，而非改进模型架构，实现了跨领域泛化
- **务实的长度去偏策略**：明确指出"刷榜分数高 ≠ 用户体验好"，主动放弃更高的benchmark分数以换取更好的实际对话质量
- **三位一体的应用验证**：不仅展示RM本身的评分准确性，还通过RL训练、BoN采样和数据清洗证明了实际应用价值
- **General类别的压倒性优势**：84.7% vs 第二名54.6%，说明判别式RM在区分"平手"判断上有独特优势

## 局限性

- **英语单语缺陷**：训练数据以英文为主，多语言能力受限，在非英语场景中可能表现不佳
- **仅ORM无PRM**：仅做结果级评分（Outcome RM），未涉及过程级奖励（Process RM），对数学推理等需要逐步验证的场景支持不足
- **评分头设计简单**：用所有token平均隐藏状态做评分，未探索注意力池化、最后token等替代方案
- **PPO效率问题**：需要同时维护策略、参考、奖励、Critic四个模型，计算开销大；未对比GRPO等更高效的RL算法
- **数据清洗缺乏定量验证**：仅展示低分样本的定性可视化，未做"清洗后重训LVLM"的定量消融
- **视频偏好数据仍较少**：视频数据仅来自3个来源，领域覆盖和规模有限

## 相关工作与启发

- **LLaVA-Critic**：用LVLM做生成式评估，但在VL-RewardBench上仅44.0%，说明提示词方式的天花板较低
- **RLAIF-V**：用AI反馈替代人类反馈做视觉对齐，本文直接将其作为训练数据的一部分
- **Tulu-3**：在LLM领域，through PPO+RM的范式已被验证有效，本文将其完整迁移到多模态
- **启发**：多模态RM的核心瓶颈在偏好数据而非模型架构，构建高质量多领域偏好数据集是关键投入点；长度偏好问题在多模态评估中同样严重，需要社区级别的评估协议改进

## 评分

- 新颖性: ⭐⭐⭐ — 方法本身是成熟技术的组合迁移，核心创新在数据构建和工程实践
- 实验充分度: ⭐⭐⭐⭐ — 覆盖3个RM基准和12+个下游任务，消融和应用验证完整
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，动机阐述充分，表格和可视化丰富
- 价值: ⭐⭐⭐⭐ — 填补了开源多模态RM的空白，为社区提供了实用的基础工具和数据构建范式

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[NeurIPS 2025\] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](../../NeurIPS2025/multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[ICLR 2026\] BaseReward: A Strong Baseline for Multimodal Reward Model](../../ICLR2026/multimodal_vlm/basereward_a_strong_baseline_for_multimodal_reward_model.md)

</div>

<!-- RELATED:END -->
