---
title: >-
  [论文解读] Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models
description: >-
  [AAAI2026][推荐系统][recommender systems] 提出 LLMHNI 框架，利用 LLM 产生的语义相关性和逻辑相关性两类辅助信号，解决推荐系统中困难样本与噪声样本难以区分的问题，显著提升去噪推荐性能。 领域现状 领域现状：推荐系统通常依赖隐式反馈（点击、购买等）训练，默认将有交互标记为正样本、无交…
tags:
  - "AAAI2026"
  - "推荐系统"
  - "recommender systems"
  - "去噪"
  - "hard sample"
  - "LLM"
  - "对比学习"
---

# Hard vs. Noise: Resolving Hard-Noisy Sample Confusion in Recommender Systems via Large Language Models

**会议**: AAAI2026  
**arXiv**: [2511.07295](https://arxiv.org/abs/2511.07295)  
**代码**: [GitHub](https://github.com/TianRui-Song717/LLMHNI)  
**领域**: 图像复原  
**关键词**: recommender systems, denoising, hard sample, LLM, contrastive learning

## 一句话总结

提出 LLMHNI 框架，利用 LLM 产生的语义相关性和逻辑相关性两类辅助信号，解决推荐系统中困难样本与噪声样本难以区分的问题，显著提升去噪推荐性能。

## 背景与动机

### 领域现状

**领域现状**：推荐系统通常依赖隐式反馈（点击、购买等）训练，默认将有交互标记为正样本、无交互标记为负样本。然而这种标注方式存在两类噪声：

1. **假正例噪声**：用户误点击或受位置偏差影响产生的虚假正反馈
2. **假负例噪声**：用户实际感兴趣但因未曝光等原因没有交互的项目被错误标为负样本

现有去噪方法（样本丢弃、样本重加权）依赖 loss 值、预测分数、梯度等数值模式区分噪声样本和干净样本。但作者发现一个关键问题：**困难样本（hard samples）与噪声样本（noisy samples）在 loss 值和预测分数上呈现高度相似的分布模式**，导致去噪方法无法有效区分二者。这种混淆非常有害，因为困难样本对建模用户偏好至关重要，误将其当作噪声丢弃会严重损害推荐质量。

### 解决思路

**本文目标**：仅依赖交互数据的数值模式无法区分困难样本与噪声样本（hard-noisy confusion），需要引入超越协同过滤信号的辅助信息。本文利用 LLM 提供两类互补的相关性信号来解决这一问题：

- **语义相关性（Semantic Relevance）**：利用 LLM 编码的用户/项目文本嵌入的相似度
- **逻辑相关性（Logical Relevance）**：利用 LLM 的推理能力判断用户-项目之间的逻辑关联

同时需克服两个挑战：LLM 嵌入的目标不匹配问题（为语言任务而非推荐任务优化）和 LLM 幻觉导致的不可靠交互推断。

## 方法详解

LLMHNI 包含两个核心模块：

### 模块一：语义相关性引导的困难负采样（Semantic Relevance Guided Hard Negative Mining）

**（1）目标对齐的嵌入生成：** 使用 LLM 编码模型（text-embedding-ada-002）对用户和项目的文本画像编码，再通过 MLP 投影到低维推荐表示空间。训练 MLP 时构建伪标签：同时满足高文本嵌入相似度且有实际交互的项目作为可靠正样本，使用 InfoNCE 对比损失 $\mathcal{L}_{al}$ 训练投影器，使对齐后的嵌入更适合推荐场景的相关性建模。

**（2）语义引导负采样：** 维护一个动态困难负样本池 $\mathbf{HN}_u^-$，每轮从中根据推荐模型预测分来更新候选集，然后选择语义相似度最低的负样本作为训练用的困难负例。核心思路：预测分高的负样本可能是困难负样本也可能是假负例，通过选择语义相似度低的来过滤掉假负例（假负例通常语义相似度也高），保留真正的困难负样本。

### 模块二：逻辑相关性引导的交互去噪（Logical Relevance Guided Interaction Denoising）

**（1）逻辑相关性推断：** 先用预训练推荐模型筛选候选交互对（高分负样本 + 低分正样本），然后从两个角度调用 LLM 评分：
- 用户视角评分：基于用户画像判断用户与目标项目的逻辑相关性
- 项目视角评分：基于用户已交互的高分项目特征判断相关性
评分结果为 High / Mid / Low 三级。只有两个视角都评为 High 的才认定为困难样本 $\mathcal{C}_H$，其余为噪声样本 $\mathcal{C}_N$。

**（2）跨图对比对齐：** 构建增强交互图 $G'$（原图去掉噪声边、加入困难样本边），与原图 $G$ 一起训练。使用跨图对比损失 $\mathcal{L}_{de}$ 对齐两个图上的用户/项目表示：在两图上都一致的交互会被增强，不一致的被抑制。

**（3）抗幻觉对比学习：** 对 $G'$ 和 $G$ 分别做随机边丢弃产生增强视图，通过图对比学习损失 $\mathcal{L}_{hal}$ 对齐两个视图的表示。随机丢边能概率性地屏蔽 LLM 幻觉引入的不可靠边，使模型对幻觉噪声具有鲁棒性。

**联合优化目标：** $\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{de} + \lambda_2 \mathcal{L}_{hal}$

## 实验关键数据

- **数据集**：Amazon-Books、Yelp、Steam
- **骨干模型**：NGCF、LightGCN
- **基线**：实例级（WBPR、T-CE、BOD）、表示级（SGL、SimGCL、XSimGCL）、LLM 增强（RLMRec、LLaRD）

**主要结果（LightGCN 骨干）：**


### 主实验

| 数据集 | 指标 | LLMHNI | LLaRD（次优） | 提升 |
|--------|------|--------|--------------|------|
| Amazon-Books | R@20 | 0.2040 | 0.2028 | +0.6% |
| Amazon-Books | N@10 | 0.1168 | 0.1126 | +3.7% |
| Yelp | N@10 | 0.0837 | 0.0809 | +3.5% |
| Steam | N@10 | 0.0893 | 0.0868 | +2.9% |

- 相对原始 LightGCN 骨干平均提升 **45.31%**，相对原始 NGCF 提升 **46.55%**
- 相比传统去噪方法（T-CE、BOD 等）提升 11.78%–37.73%
- 相比 LLM 增强基线（RLMRec、LLaRD）提升 2.47%–33.86%

**噪声鲁棒性**：在 5%–20% 噪声注入实验中，LLMHNI 性能下降速率最为平稳，始终优于所有基线

**消融实验**（LightGCN/Amazon-Books）移除各模块后 R@20 均下降：完整模型 0.2040 → 移除语义负采样 0.1799、移除目标对齐 0.1848、移除交互去噪 0.1772、移除抗幻觉 0.1854

## 亮点与洞察

1. **问题洞察精准**：首次明确指出并系统分析了推荐系统去噪中困难样本与噪声样本的混淆问题，填补了该领域的认知空白
2. **双信号互补设计**：语义相关性用于负采样阶段（连续值），逻辑相关性用于交互图去噪（离散判断），两者在不同粒度上协同工作
3. **工程实践友好**：LLM 推断在训练前离线完成，不增加推荐模型的在线训练开销；框架兼容不同 GNN 骨干
4. **抗幻觉设计合理**：通过随机边丢弃+对比学习的方式处理 LLM 幻觉问题，比简单过滤更优雅

## 局限与展望

1. **LLM 成本高**：需要调用 GPT-4o 对候选交互对逐一评分，大规模数据集的 API 成本和延迟不可忽视
2. **分类过于简化**：困难/噪声的二分类标准（两个维度都为 High 才算困难）可能过于保守，中间状态的样本处理不够精细
3. **仅验证 GNN 骨干**：实验只在 NGCF 和 LightGCN 上验证，缺少对序列推荐模型（如 SASRec）或非图模型的泛化性验证
4. **LLM 嵌入模型依赖**：使用 OpenAI 的 text-embedding-ada-002，换用开源 LLM 时效果是否稳定未知
5. **候选对筛选依赖预训练模型**：逻辑相关性推断的候选集依赖预训练推荐模型的质量，若预训练模型本身不佳会传播误差

## 相关工作与启发

| 方法类别 | 代表方法 | 核心差异 |
|---------|---------|---------|
| 实例级去噪 | T-CE, BOD | 依赖 loss/预测分模式区分噪声，无法处理 hard-noisy 混淆 |
| 表示级去噪 | SGL, SimGCL | 通过数据增强提升鲁棒性，但未针对性区分困难与噪声 |
| LLM 增强推荐 | RLMRec | 利用 LLM 嵌入增强表示，但缺乏困难样本识别能力 |
| LLM 增强去噪 | LLaRD | 用 LLM 辅助去噪但未充分利用语义+逻辑双重信号 |
| **LLMHNI（本文）** | — | 同时利用语义和逻辑相关性，解决 hard-noisy 混淆，含抗幻觉机制 |

## 相关工作与启发

- LLM 在推荐系统中不仅可以增强表示，还可以作为"裁判"来判断交互的可靠性，这一思路可推广到其他需要噪声判别的场景
- 目标对齐策略（aligning LLM embeddings to task-specific space）具有通用价值，任何将 LLM 嵌入用于下游任务的场景都可借鉴
- 跨图对比对齐的思路——用不同信号源构建的图相互监督——可迁移到知识图谱补全、社交网络分析等领域

## 评分
- 新颖性: 8/10 — 首次系统解决 hard-noisy 混淆问题，双信号设计有新意
- 实验充分度: 8/10 — 三数据集两骨干、消融完整、噪声鲁棒性验证充分，但缺非 GNN 骨干验证
- 写作质量: 7/10 — 动机阐述清晰，但公式符号较多、部分表述冗长
- 价值: 7/10 — 解决实际问题但 LLM 成本限制了大规模部署的实用性

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](../../NeurIPS2025/recommender/r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](../../NeurIPS2025/recommender/inference-time_reward_hacking_in_large_language_models.md)
- [\[AAAI 2026\] RecToM: A Benchmark for Evaluating Machine Theory of Mind in LLM-based Conversational Recommender Systems](rectom_a_benchmark_for_evaluating_machine_theory_of_mind_in_llm-based_conversati.md)

</div>

<!-- RELATED:END -->
