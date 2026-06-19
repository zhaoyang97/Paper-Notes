---
title: >-
  [论文解读] Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization
description: >-
  [AAAI2026][幻觉检测][MLLM hallucination] 针对多模态大模型（MLLM）在强化学习训练中产生幻觉的三大根因——视觉误解、探索多样性不足、样本冲突——分别提出 Caption Reward、奖励方差引导的样本选择、以及基于 NTK 相似度的 InfoNCE 正则化，在多个基准上显著降低幻觉率。
tags:
  - "AAAI2026"
  - "幻觉检测"
  - "MLLM hallucination"
  - "reinforcement-learning"
  - "GRPO"
  - "caption reward"
  - "NTK similarity"
  - "InfoNCE"
---

# Ground What You See: Hallucination-Resistant MLLMs via Caption Feedback, Diversity-Aware Sampling, and Conflict Regularization

**会议**: AAAI2026  
**arXiv**: [2601.06224](https://arxiv.org/abs/2601.06224)  
**代码**: [ZJU-OmniAI/OMNEX-VL](https://github.com/ZJU-OmniAI/OMNEX-VL)  
**领域**: 幻觉检测  
**关键词**: MLLM hallucination, reinforcement-learning, GRPO, caption reward, NTK similarity, InfoNCE

## 一句话总结

针对多模态大模型（MLLM）在强化学习训练中产生幻觉的三大根因——视觉误解、探索多样性不足、样本冲突——分别提出 Caption Reward、奖励方差引导的样本选择、以及基于 NTK 相似度的 InfoNCE 正则化，在多个基准上显著降低幻觉率。

## 背景与动机

多模态大模型在视觉问答、视频理解等任务上表现优异，但 **幻觉问题**（生成流畅但与视觉输入事实不一致的回答）严重制约了实际部署，尤其在安全关键场景。近期大量工作将强化学习（RL）引入 MLLM 以提升推理能力（如 DeepSeek-R1 风格的 GRPO 训练），但 RL 训练反而会 **加剧幻觉**：模型陷入语义冗余推理循环、思考过程与最终答案不匹配等问题。

作者系统分析了 RL 训练中幻觉的三大根因：

1. **视觉误解（Visual Misinterpretation）**：模型在推理链早期产生不准确的视觉描述，后续推理被锚定到错误信息上；或者模型笼统关注输入，生成冗余无关推理
2. **探索多样性不足（Limited Exploration）**：策略优化阶段采样多样性不够，导致输出过度自信、过拟合
3. **样本冲突（Sample Conflict）**：对一个样本的梯度更新无意中破坏模型对其他无关样本的预测，引入虚假关联

## 核心问题

如何在 MLLM 的 RL 训练过程中同时解决视觉锚定不准确、策略探索不充分、以及样本间梯度干扰这三类幻觉来源？

## 方法详解

整体框架包含三个模块，在 GRPO 基础上联合使用：

### 1. Visual-Grounded Reasoning Enhancement（视觉锚定推理增强）

**重新定义推理范式**：在标准的 thinking → answer 流程之前，插入两个新阶段：

- **Planning 阶段**：提前定位与问题相关的视觉区域
- **Caption 阶段**：对这些区域生成简洁的文本描述，作为后续推理的中间锚点

完整流程变为：planning → caption → thinking → answer。

**Caption Reward**：提取模型生成的 caption，将其与问题（不含图像）输入一个独立的 LLM。若该 LLM 仅凭 caption 即可正确回答问题，则给予正奖励，否则奖励为零。这确保了 caption 准确反映视觉内容，避免错误描述在推理链中传播。

### 2. Reward Variance-Guided Sample Selection（奖励方差引导样本选择）

通过对 GRPO 损失关于 logits 求梯度，论文推导出关键结论：

- 正 advantage（$A_{i,t} > 0$）→ 分布变尖锐（sharpening）：强化正确答案
- 负 advantage（$A_{i,t} < 0$）→ 分布变平坦（flattening）：削弱错误高置信输出

据此，论文将样本分为三类：

| 类型 | 特征 | 训练效果 |
|------|------|----------|
| Easy | 高均值、低方差 | 过拟合风险，分布过度尖锐 |
| Hard | 低均值、低方差 | 模型无法学习，分布持续平坦 |
| **Medium** | **高方差** | **最有价值**：先探索后收敛，理想学习轨迹 |

具体做法：对每个输入生成 64 个响应，计算奖励的均值和方差，仅保留方差最高的 50% 样本用于 RL 训练。

### 3. Conflict-Aware Regularization（冲突感知正则化）

论文基于 Neural Tangent Kernel（NTK）分析样本间的梯度干扰：

$$\Delta\log\pi^t(\mathbf{y}_u|\mathbf{x}_o) \propto \eta \cdot \mathcal{A}^t(x_o) \cdot \mathcal{K}^t(x_o, x_u) \cdot \nabla_z\log\pi^t(y_u|x_u) \cdot A_{u,t}$$

其中 $\mathcal{K}^t(x_o, x_u)$ 是 NTK 相似度。当 NTK 相似度过高时，对一个样本的更新会大幅改变模型对另一个不相关样本的预测。

**关键洞察**：不能简单最小化 NTK 相似度（会抑制有益交互），而应 **调节到合理范围**。

具体做法：

- 用最后一层 log-probability 梯度的余弦相似度近似 NTK 相似度
- 基于阈值 $\tau$ 将样本对分为正对（相似度过低，需拉近）和负对（相似度过高，需推远）
- 应用 InfoNCE 损失调整：$\mathcal{L} = -\frac{1}{B}\sum_{i=1}^{B}\log\frac{\sum_{j\in\mathcal{P}(i)}\exp(\text{sim}(f_i,f_j))}{\sum_{k=1}^{B}\mathbb{I}_{[k\neq i]}\exp(\text{sim}(f_i,f_k))}$
- 实验确定最优阈值 $\tau = 0.54$

## 实验关键数据

基于 Qwen-VL-2.5-7B，在多个基准上对比：

| 模型 | MMVU | VideoHallucer | POPE | MMBench |
|------|------|---------------|------|---------|
| Qwen-VL-2.5-7B（基线） | 57.6 | 46.5 | 84.4 | 86.3 |
| + SFT | 62.7 | 43.5 | 82.2 | 83.9 |
| + GRPO | 62.1 | **42.3↓** | 83.6 | 86.8 |
| + **Ours** | **65.6** | **50.8** | **88.7** | **88.6** |
| GPT-4o | 75.4 | 53.3 | 86.9 | 83.4 |

关键发现：

- 标准 GRPO 训练 **降低** 了 VideoHallucer 分数（46.5→42.3），证实 RL 确实会引入幻觉
- 本文方法在 POPE 上达到 88.7%（超过 GPT-4o 的 86.9%），在 MMBench 上达到 88.6%（超过 GPT-4o 的 83.4%）
- MMVU 上 65.6% 为所有开源模型最高

消融实验：

- 移除 Caption + Caption Reward：MMVU 65.6→62.6，POPE 88.7→85.2
- 仅用 Easy/Hard 样本训练：性能均低于 Medium 样本
- 移除 InfoNCE Loss：MMVU 65.6→63.8，POPE 88.7→86.8

## 亮点

1. **问题分析深入**：从理论层面推导了 RL 训练中正负 advantage 对输出分布的影响，为样本选择提供了有说服力的理论支撑
2. **NTK 视角新颖**：首次将 NTK 相似度与 MLLM 训练中的样本冲突联系起来，并提出"调节而非消除"的思路
3. **Caption Reward 设计巧妙**：通过"仅凭 caption 能否回答问题"来间接评估视觉锚定质量，无需额外标注
4. **三个模块正交互补**：分别针对推理链入口（caption）、样本选择（方差）、优化过程（NTK 正则）三个层面

## 局限与展望

1. **领域分类有误**：论文主题是 MLLM 幻觉缓解，归类到 object_detection 不太合适，应属于 multimodal_vlm
2. **计算开销大**：每个样本生成 64 个响应来估计方差，再加上 NTK 相似度计算和 InfoNCE 损失，训练成本显著增加
3. **阈值 $\tau$ 需调优**：NTK 相似度阈值 0.54 是通过实验搜索得到的，不同数据集可能需要不同值
4. **基准有限**：主要在视频理解和幻觉检测基准上验证，缺少更广泛的视觉推理任务（如 visual grounding、referring expression）
5. **与 GPT-4o 差距**：在 MMVU（65.6 vs 75.4）和 VideoHallucer（50.8 vs 53.3）上仍有差距

## 与相关工作的对比

| 方法 | 核心思路 | 与本文差异 |
|------|----------|-----------|
| Vision-R1 | 渐进思维抑制训练 + 解耦奖励 | 未处理视觉描述错误传播问题 |
| R1-VL | StepGRPO 逐步一致性 | 未考虑样本间梯度冲突 |
| Video-R1 | 时序一致性奖励 | 仅针对视频时序，未处理一般性幻觉 |
| GRPO 标准训练 | 组相对策略优化 | 本文实验表明标准 GRPO 会加剧幻觉 |

本文的核心差异在于 **同时** 从推理范式、采样策略、优化正则化三个层面系统性地解决幻觉问题。

## 启发与关联

- **Caption Reward 的思路可泛化**：用独立模型验证中间推理产物的质量，可推广到其他需要过程监督的场景
- **方差引导选择与 curriculum learning 有联系**：medium 样本本质上是模型处于学习边界的样本，类似于 zone of proximal development
- **NTK 正则化可用于其他多任务/ multi-sample 训练**：任何需要减少样本间梯度干扰的场景都可能受益

## 评分

- 新颖性: ⭐⭐⭐⭐ — NTK 视角和 Caption Reward 设计有新意
- 实验充分度: ⭐⭐⭐⭐ — 消融完整，但基准范围可更广
- 写作质量: ⭐⭐⭐⭐ — 问题分析清晰，三模块逻辑连贯
- 价值: ⭐⭐⭐⭐ — 系统性解决 RL 训练中的 MLLM 幻觉问题，实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] TAG: Tangential Amplifying Guidance for Hallucination-Resistant Sampling](../../ICML2026/hallucination/tag_tangential_amplifying_guidance_for_hallucination-resistant_sampling.md)
- [\[CVPR 2026\] COPO: Causal-Oriented Policy Optimization for Hallucinations of MLLMs](../../CVPR2026/hallucination/copo_causal-oriented_policy_optimization_for_hallucinations_of_mllms.md)
- [\[ICML 2026\] Building Reliable Long-Form Generation via Hallucination Rejection Sampling](../../ICML2026/hallucination/building_reliable_long-form_generation_via_hallucination_rejection_sampling.md)
- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](../../ACL2025/hallucination/on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[CVPR 2026\] FINER: MLLMs Hallucinate under Fine-grained Negative Queries](../../CVPR2026/hallucination/finer_mllms_hallucinate_under_fine-grained_negative_queries.md)

</div>

<!-- RELATED:END -->
