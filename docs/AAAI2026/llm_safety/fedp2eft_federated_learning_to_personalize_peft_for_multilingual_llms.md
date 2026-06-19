---
title: >-
  [论文解读] FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs
description: >-
  [AAAI2026][LLM安全][联邦学习] 提出FedP²EFT，通过联邦学习协作训练一个Personalization Strategy Generator (PSG)，为每个客户端自动生成个性化的LoRA rank结构，在多语言LLM微调中大幅超越手工设计的PEFT配置和现有FL个性化方法。
tags:
  - "AAAI2026"
  - "LLM安全"
  - "联邦学习"
  - "个性化PEFT"
  - "LoRA rank选择"
  - "多语言LLM"
  - "Bayesian稀疏选择"
---

# FedP²EFT: Federated Learning to Personalize PEFT for Multilingual LLMs

**会议**: AAAI2026  
**arXiv**: [2502.04387](https://arxiv.org/abs/2502.04387)  
**作者**: Royson Lee, Minyoung Kim, Fady Rezk, Rui Li, Stylianos I. Venieris, Timothy Hospedales (Samsung AI / Univ. of Edinburgh)  
**代码**: [GitHub](https://github.com/SamsungLabs/fedp2eft)  
**领域**: 优化  
**关键词**: 联邦学习, 个性化PEFT, LoRA rank选择, 多语言LLM, Bayesian稀疏选择  

## 一句话总结

提出FedP²EFT，通过联邦学习协作训练一个Personalization Strategy Generator (PSG)，为每个客户端自动生成个性化的LoRA rank结构，在多语言LLM微调中大幅超越手工设计的PEFT配置和现有FL个性化方法。

## 背景与动机

### 多语言LLM的联邦学习困境
联邦学习使多语言LLM能够利用分散在不同地域的低资源语言数据进行训练，同时满足GDPR等隐私法规。然而现有方法面临三大挑战：

1. **Curse of Multilinguality**：随着语言数量增加，单一全局模型的性能递减
2. **Negative Interference**：不同语言之间竞争有限的模型容量
3. **个性化策略缺失**：现有方法使用手工设计的统一PEFT配置，忽视了不同客户端对个性化的差异化需求

### 为什么个性化LoRA rank比学习率更重要？
现有FL超参数优化（如FedL2P）主要学习个性化学习率，但LLM通常使用Adam等自适应优化器，对学习率鲁棒。相反，PEFT adapter的结构（在哪些层加LoRA、使用什么rank）对跨语言transfer learning的影响更为关键。

## 核心问题

如何在联邦学习设定下，自动为每个客户端学习最优的个性化LoRA rank配置，同时避免客户端数据量少导致的过拟合？

## 方法详解

### BayesTune-LoRA (BT-LoRA)

受BayesTune启发，为每个LoRA矩阵引入rank-wise的隐变量$\lambda \in \mathbb{R}^r, \lambda_i > 0$，修改LoRA为$B\lambda A$。优化目标为：

$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{CE}}(\theta; D) + \frac{\alpha_s}{N}\mathcal{L}_s(\boldsymbol{\lambda}, \boldsymbol{B}) + \frac{\alpha_p}{N}\mathcal{L}_p(\boldsymbol{\lambda})$$

其中$\mathcal{L}_s$为Laplace先验的对数（鼓励对重要rank保持较大$\lambda$）：

$$\mathcal{L}_s(\boldsymbol{\lambda}, \boldsymbol{B}) = \sum_l^L \sum_i^r \frac{\|B_{l,i}\|_1}{\lambda_{l,i}}$$

$\mathcal{L}_p$为Gamma超先验的对数（鼓励$\lambda$整体较小以实现稀疏）：

$$\mathcal{L}_p(\boldsymbol{\lambda}) = \sum_l^L \sum_i^r (\log \lambda_{l,i} + 100 \cdot \lambda_{l,i})$$

直觉：$\mathcal{L}_p$推动$\lambda$趋近于零（稀疏化），$\mathcal{L}_s$推动对更新量大的rank保持较大$\lambda$。两者博弈使重要rank保留、不重要rank被剪枝。

### PSG: Personalization Strategy Generator

使用单隐层MLP作为PSG，输入客户端元数据（base model各层特征的均值和标准差），输出个性化的$\hat{\boldsymbol{\lambda}}$：

$$\hat{\boldsymbol{\lambda}} = \text{MLP}(\phi;\; E(h_0), SD(h_0), E(h_1), SD(h_1), \ldots, E(h_{L-1}), SD(h_{L-1}))$$

### 联邦训练流程

每轮联邦训练中，每个被采样的客户端$i$执行：

1. 从服务器接收PSG参数$\phi$，前向传播获取$\hat{\boldsymbol{\lambda}}^i$
2. **Stage 1**：将$\hat{\boldsymbol{\lambda}}^i$插入BT-LoRA，按上述目标函数微调$s$步，得到优化后的$\hat{\boldsymbol{\lambda}}^{i,s}$
3. **Stage 2**：以$\hat{\boldsymbol{\lambda}}^{i,s}$为回归目标，用L1 loss训练MLP
4. 将更新后的$\phi$发回服务器进行FedAvg聚合

### 推理阶段

部署时，新客户端（包括训练中未见过的）通过PSG生成$\boldsymbol{\lambda}$，按资源预算$r \cdot L$取top-$(r \cdot L)$最大的rank，冻结$\boldsymbol{\lambda}$后标准微调。训练一次PSG即可适配所有$\leq r_{\text{max target}}$的rank预算。

## 实验关键数据

### MasakhaNEWS文本分类（16种非洲语言，Seen客户端，$r=2$）

| 语言 | LoRA | AdaLoRA | BT-LoRA | FedL2P | **FedP²EFT** |
|------|------|---------|---------|--------|-------------|
| eng | 90.4 | 89.9 | 89.9 | 90.7 | **92.0** |
| amh | 45.7 | 45.2 | 45.2 | 45.7 | **52.0** |
| tir | 44.9 | 44.9 | 44.9 | 45.3 | **63.5** |
| orm | 64.2 | 64.0 | 64.0 | 64.4 | **72.2** |
| fra | 88.6 | 88.6 | 88.6 | 89.1 | **93.5** |

低资源语言（tir, amh, orm）上FedP²EFT的优势尤为显著，如Tigrinya从44.9%提升至63.5%（+18.6pp）。

### Unseen客户端泛化性

| 语言 | LoRA | FedL2P | **FedP²EFT** |
|------|------|--------|-------------|
| xho | 64.2 | 64.4 | **78.5** |
| tir | 41.9 | 41.9 | **58.3** |
| orm | 62.0 | 62.2 | **73.0** |
| run | 82.0 | 82.6 | **88.4** |

在完全未参与训练的客户端上，FedP²EFT同样大幅领先，验证了PSG的泛化能力。

### XNLI + FedDPA-T个性化FL兼容性

| 语言 | LoRA | FedL2P | **FedP²EFT** |
|------|------|--------|-------------|
| ur | 41.9 | 44.8 | **63.7** |
| bg | 45.8 | 47.5 | **64.4** |
| hi | 42.8 | 44.5 | **57.8** |

FedP²EFT可无缝集成到现有pFL方法（FedDPA-T、DEPT等），进一步提升个性化性能。

## 亮点

- **首创联邦LoRA rank个性化**：将PEFT结构选择问题纳入联邦学习框架，避免了客户端独立训练的过拟合问题
- **一次训练适配所有rank预算**：BT-LoRA的稀疏选择特性使PSG训练为一次性成本
- **广泛兼容性**：可插入到Standard FL、FedDPA-T、DEPT等不同FL方法之上
- **低资源语言的巨大增益**：在Tigrinya、Amharic等极低资源语言上提升高达18.6pp
- **理论连接清晰**：从Bayesian稀疏模型选择推导出LoRA rank的稀疏先验

## 局限与展望

- **PSG输入仅用统计量**：均值和标准差可能丢失分布细节，更丰富的元数据提取或许能进一步提升
- **仅验证LoRA**：未探索对其他PEFT方法（Adapter、Prefix Tuning、IA³）的适用性
- **FedAvg聚合**：未对比FedProx、SCAFFOLD等更先进的聚合策略
- **Stage 1步数$s$的选择**：过大可能过拟合客户端数据，过小则$\hat{\boldsymbol{\lambda}}^{i,s}$质量不足，缺乏自适应调节
- **实验规模有限**：指令微调仅在MobileLLaMA-1.4B和Llama-3.2-3B上验证

## 与相关工作的对比

- **FedL2P**：学习个性化学习率，需二阶优化且在LLM+Adam设定下效果有限；FedP²EFT学习rank结构，更直接且避免二阶计算
- **AdaLoRA**：基于SVD的rank分配在少数据FL环境下易过拟合；FedP²EFT通过联邦协作缓解数据不足
- **BT-LoRA（独立版）**：每个客户端独立优化$\boldsymbol{\lambda}$会过拟合；FedP²EFT联邦训练PSG后再生成$\boldsymbol{\lambda}$泛化更好
- **DEPT / FedDPA-T**：手工设计个性化层（embedding/LoRA），不自动适配客户端需求；FedP²EFT可作为它们的补充

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将Bayesian稀疏rank选择与联邦元学习结合的思路新颖
- 实验充分度: ⭐⭐⭐⭐ — 覆盖文本分类和指令微调、seen/unseen客户端、多种FL基座，但模型规模偏小
- 写作质量: ⭐⭐⭐⭐ — 动机清晰，方法推导完整，图示直观
- 价值: ⭐⭐⭐⭐ — 解决了联邦LLM个性化的实际痛点，对低资源语言场景价值大

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Watermark Robustness and Radioactivity May Be at Odds in Federated Learning](../../ICLR2026/llm_safety/watermark_robustness_and_radioactivity_may_be_at_odds_in_federated_learning.md)
- [\[ICLR 2026\] SABRE-FL: Selective and Accurate Backdoor Rejection for Federated Prompt Learning](../../ICLR2026/llm_safety/sabre-fl_selective_and_accurate_backdoor_rejection_for_federated_prompt_learning.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ACL 2026\] Exploring Cross-Client Memorization of Training Data in Large Language Models for Federated Learning](../../ACL2026/llm_safety/exploring_cross-client_memorization_of_training_data_in_large_language_models_fo.md)
- [\[ACL 2026\] Responsible Federated LLMs via Safety Filtering and Constitutional AI](../../ACL2026/llm_safety/responsible_federated_llms_via_safety_filtering_and_constitutional_ai.md)

</div>

<!-- RELATED:END -->
