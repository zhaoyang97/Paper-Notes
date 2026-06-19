---
title: >-
  [论文解读] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning
description: >-
  [NeurIPS 2025][LLM推理][视频推理] 提出 VideoRFT，通过认知启发的多专家 CoT 数据构建流水线和新颖的语义一致性奖励，将强化微调（RFT）范式扩展到视频推理领域，分别构建 VideoRFT-CoT-102K（SFT 用）和 VideoRFT-RL-310K（RL 用）两个数据集，在 6 个视频推理基准上达到 SOTA。
tags:
  - "NeurIPS 2025"
  - "LLM推理"
  - "视频推理"
  - "强化微调"
  - "思维链"
  - "多模态大模型"
  - "语义一致性奖励"
---

# VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning

**会议**: NeurIPS 2025  
**arXiv**: [2505.12434](https://arxiv.org/abs/2505.12434)  
**代码**: [GitHub](https://github.com/QiWang98/VideoRFT)  
**领域**: 多模态VLM  
**关键词**: 视频推理, 强化微调, 思维链, 多模态大模型, 语义一致性奖励

## 一句话总结

提出 VideoRFT，通过认知启发的多专家 CoT 数据构建流水线和新颖的语义一致性奖励，将强化微调（RFT）范式扩展到视频推理领域，分别构建 VideoRFT-CoT-102K（SFT 用）和 VideoRFT-RL-310K（RL 用）两个数据集，在 6 个视频推理基准上达到 SOTA。

## 研究背景与动机

多模态大语言模型（MLLM）在视频理解任务上取得了显著进展，但现有模型主要是"答案驱动"的——直接产出答案而不展示推理过程。随着 OpenAI-o1 和 DeepSeek-R1 等"先思考再回答"模型的成功，强化微调（RFT）已在图像领域展现出潜力（Visual-RFT、Vision-R1 等），但在视频领域面临三个核心挑战：

**高质量视频 CoT 数据稀缺**：视频的时间复杂性和因果结构使得生成高质量推理链远比图像困难。现有方法如 Video-R1 仅通过在回答中插入"let me think"等表面词触发 CoT，缺乏真正的深度推理。VideoEspresso 用少量关键帧提示 GPT-4o 生成 CoT，但稀疏视觉上下文容易产生幻觉。

**推理缺乏视觉锚定**：现有 RFT 方法的奖励函数（格式奖励 + 准确性奖励）无法保证推理输出忠实于视觉证据——模型可能产出"看似合理但视觉上不一致"的推理文本。

**认知过程缺失**：模板式推理（如 VoT 的五步固定流程）与人类灵活的认知方式相悖——人类的推理过程是基于感知输入自适应调整的。

## 方法详解

### 整体框架

VideoRFT 遵循标准的两阶段 RFT 方案：
- **阶段一（SFT 热启动）**：用 VideoRFT-CoT-102K 数据集进行监督微调，让模型学会结构化推理
- **阶段二（RL 强化）**：用 VideoRFT-RL-310K 数据集和三重奖励函数（格式奖励 + 准确性奖励 + 语义一致性奖励）通过 GRPO 算法进行强化学习

### 关键设计

1. **认知启发的 CoT 生成流水线**：分三步构建高质量视频 CoT 数据。

   **第一步：结构化视频表示**。使用 GPT-4o-mini 对每个视频均匀采样的帧生成结构化文本描述，包括高层摘要和每帧的 JSON 格式元数据（时间戳标注、关键视觉元素如物体/动作/场景/空间关系/潜在交互等）。

   **第二步：认知启发的盲推理 CoT 生成**。将结构化表示 $S_v$ 和问题 $q$ 输入推理型 LLM（如 DeepSeek-R1），通过精心设计的复合提示 $P_{\text{cog}} = [p_s, p_t, p_a, p_v, p_r]$ 生成初始 CoT：$\text{CoT}_v^{(0)} = \text{LLM}(q, S_v, P_{\text{cog}})$。五个子提示模拟人类认知过程：(i) 模拟观看——形成整体理解；(ii) 任务理解——推断问题类型；(iii) 选择性聚焦——定位相关时间段；(iv) 视觉推理——基于物体/动作/时空关系的分析；(v) 反思性回答——推导并自我验证答案。

   **第三步：跨模态 CoT 精修**。初始 CoT 因仅基于文本描述而可能存在视觉幻觉。使用 Qwen2.5-VL 将原始视频与初始 CoT 一起输入，通过对比提示 $P_{\text{cross}}$ 识别视觉-文本不一致并修正：$\text{CoT}_v = \text{MLLM}(v, \text{CoT}_v^{(0)}, P_{\text{cross}})$。然后通过答案正确性（结构化任务）和语义一致性（开放任务，CLIP 评分）过滤，从 310K 中筛选 102K 高置信样本。

2. **语义一致性奖励**：核心观察是 MLLM 的推理轨迹通常包含三个部分：问题解析（question parsing）、视频描述（video describing）和抽象推理（abstract reasoning）。其中视频描述部分应当与实际视频内容对齐。

   奖励计算：用正则表达式从推理文本中定位第一个句号后的 $M$ 个 token 作为视频描述段 $t_{[i,i+M]}$，通过 SigLIP 的文本编码器获取其表示；同时均匀采样 $F$ 帧通过 SigLIP 图像编码器获取平均视频表示 $\boldsymbol{v}$。语义一致性奖励为：

    $R_s = \min(1, w \times \max(\cos(\boldsymbol{t}_{[i,i+M]}, \boldsymbol{v}), 0))$

   其中 $w=2$ 为缩放常数。$\max(\cdot, 0)$ 保证非负，$\min(\cdot, 1)$ 限制上界稳定训练。仅在准确性奖励 $R_a > 0$ 时激活，避免强化语义合理但事实错误的推理。

3. **GRPO 训练框架**：VideoRFT 使用 GRPO（Group Relative Policy Optimization）作为 RL 算法。对每个查询 $q$ 生成 $K$ 个候选回答，评估获得奖励后计算群体优势 $A_i = \frac{r_i - \text{mean}(\{r_k\})}{\text{std}(\{r_k\})}$，通过 clipped 目标和 KL 正则化更新策略。总奖励为 $R = R_f + R_a + \mathbb{1}[R_a > 0] \cdot R_s$。

### 训练策略

- 基础模型：Qwen2.5-VL-7B
- 硬件：8× NVIDIA A800 (80GB)
- 训练时视频输入 16 帧，分辨率 $128 \times 28 \times 28$
- 推理时增加到 32 帧，分辨率 $256 \times 28 \times 28$
- SFT 阶段训练 1 epoch，RL 阶段训练 1K 步
- 语义一致性奖励使用轻量 SigLIP (400M) 计算

## 实验关键数据

### 主实验

**6 个视频推理/理解基准（准确率 %）**

| 模型 | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|------|----------|----------|------|---------|------------|---------|
| GPT-4o | 34.0 | 61.2 | 75.4 | - | - | 71.9 |
| Qwen2.5-VL-7B (基座) | 31.8 | 47.4 | 61.3 | 59.4 | 69.2 | 52.8 |
| Video-R1 | 35.8 | 52.3 | 63.8 | 63.9 | 73.2 | 59.3 |
| LLaVA-OneVision-7B | 32.4 | 33.8 | 49.2 | 56.7 | - | 58.2 |
| **VideoRFT** | **36.8** | 51.1 | **68.5** | 62.1 | **73.7** | **59.8** |

**相对于基座模型 Qwen2.5-VL-7B 的提升**

| 指标 | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|------|----------|----------|------|---------|------------|---------|
| 提升幅度 | +5.0 | +3.7 | **+7.2** | +2.7 | +4.5 | **+7.0** |

### 消融实验

| 配置 | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|------|----------|----------|------|---------|------------|---------|
| w/o CoT 精修 | 34.5 | 48.1 | 64.8 | 58.3 | 72.4 | 52.8 |
| SFT only | 31.7 | 48.5 | 60.5 | 57.0 | 68.4 | 54.1 |
| RL only | 32.1 | 47.4 | 63.5 | 59.2 | 70.8 | 51.9 |
| $R=R_f+R_a$ | 33.2 | 49.1 | 66.4 | 61.1 | 72.4 | 58.5 |
| $R=R_f+R_a+R_s$ (无门控) | 34.6 | 50.2 | 65.2 | 61.4 | 73.9 | 56.3 |
| **VideoRFT (完整)** | **36.8** | **51.1** | **68.5** | **62.1** | **73.7** | **59.8** |

### 关键发现

- **跨模态 CoT 精修至关重要**：去掉精修后 MMVU -3.7%、VideoMME -7.0%，证明盲推理 CoT 中的视觉幻觉问题严重。
- **SFT + RL 互补**：RL only 在 4/6 数据集上优于 SFT only（说明 RL 的泛化能力更强），但 SFT+RL 在所有 6 个数据集上最优（SFT 提供稳定初始化）。
- **语义一致性奖励的门控激活机制有效**：直接加 $R_s$（无门控）在 MMVU 和 VideoMME 上反而更差，因为会强化"语义合理但事实错误"的推理。通过 $\mathbb{1}[R_a > 0]$ 门控后仅在正确答案上激励视觉锚定。
- **Aha Moment 出现**：VideoRFT 展现出类人的"等等，让我再检查一下"的自我反思行为，表明 RL 训练确实激发了内在反馈循环而非简单的模式匹配。
- **超越 GPT-4o**：在 VSI-Bench 上 VideoRFT 超过 GPT-4o 2.8%，证明 7B 模型通过 RFT 可在空间推理任务上接近或超越闭源大模型。

## 亮点与洞察

- CoT 构建流水线的"先盲推理再视觉校准"思路值得借鉴：利用推理型 LLM 的强推理能力生成初始 CoT，再用 MLLM 的视觉能力修正幻觉，实现能力互补。
- 语义一致性奖励的设计理念——**选择性对齐**——非常优雅：仅对推理中与视觉内容直接对应的部分计算一致性，既不限制抽象推理也不惩罚问题解析，只要求视频描述要忠实。
- CoT 数据的对比分析有说服力：VideoRFT-CoT-102K 的 token 分布比 Video-R1 更长且更动态化（高频词为"video"、"happen"、"first"等时序词汇），反映了更深层的视频理解。

## 局限与展望

- 语义一致性奖励依赖 SigLIP 的文本-视觉对齐质量，对于 SigLIP 覆盖不好的领域（如科学图表）效果可能有限。
- 通过正则表达式定位"视频描述段"过于启发式——假设第一个句号后就是视频描述，对不规则推理输出可能失效。
- 训练和推理使用不同帧数和分辨率（16帧→32帧，128→256 patch），引入了不一致性但未充分分析其影响。
- 102K CoT 的过滤率（310K→102K）约 67% 较高，大量数据被丢弃，流水线效率可提升。
- 缺少与其他 RL 算法（如 PPO）的比较，GRPO 的选择合理性仅通过引用 DeepSeek-R1 支撑。

## 相关工作与启发

- 与 Video-R1 的核心区别在数据质量：Video-R1 用简单提示词触发 CoT，VideoRFT 用多专家认知流水线构建高质量 CoT。数据质量差异直接反映在性能上。
- 与 Vision-R1、R1-OneVision 等图像 RFT 工作相比，VideoRFT 的独特贡献是语义一致性奖励——图像领域通常不需要显式的视觉锚定奖励。
- 认知启发提示策略（模拟观看→任务理解→选择性聚焦→视觉推理→反思回答）可复用于其他需要结构化推理的多模态任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ CoT 构建流水线和语义一致性奖励均有实质创新
- 实验充分度: ⭐⭐⭐⭐ 6 个基准、完整消融，但缺少更多 RL 算法对比
- 写作质量: ⭐⭐⭐⭐ 结构清晰，流水线图示直观
- 价值: ⭐⭐⭐⭐⭐ 高质量 CoT 数据集和视觉锚定奖励对视频推理 MLLM 有广泛适用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/llm_reasoning/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] WebThinker: Empowering Large Reasoning Models with Deep Research Capability](webthinker_empowering_large_reasoning_models_with_deep_research_capability.md)
- [\[ACL 2025\] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning](../../ACL2025/llm_reasoning/enhancing_chain-of-thought_reasoning_with_critical_representation_fine-tuning.md)
- [\[ICML 2026\] Blending Supervised and Reinforcement Fine-Tuning with Prefix Sampling](../../ICML2026/llm_reasoning/blending_supervised_and_reinforcement_fine-tuning_with_prefix_sampling.md)
- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](../../ACL2025/llm_reasoning/tract_regression_cot.md)

</div>

<!-- RELATED:END -->
