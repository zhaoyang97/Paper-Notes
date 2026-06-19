---
title: >-
  [论文解读] POPri: Private Federated Learning using Preference-Optimized Synthetic Data
description: >-
  [ICML2025][LLM安全][差分隐私] 将差分隐私联邦学习中的合成数据生成问题重新建模为 LLM 策略优化（DPO）问题，利用客户端 DP 反馈构建偏好对来微调 LLM，比传统 Private Evolution 提升更大——在 ε=1 下将隐私-性能差距缩小 58%。 差分隐私联邦学习（DP-FL）是从设备端私有数…
tags:
  - "ICML2025"
  - "LLM安全"
  - "差分隐私"
  - "联邦学习"
  - "合成数据"
  - "偏好优化"
  - "DPO"
  - "Private Evolution"
---

# POPri: Private Federated Learning using Preference-Optimized Synthetic Data

**会议**: ICML2025  
**arXiv**: [2504.16438](https://arxiv.org/abs/2504.16438)  
**代码**: [meiyuw/POPri](https://github.com/meiyuw/POPri)  
**领域**: 联邦隐私 / 差分隐私合成数据  
**关键词**: 差分隐私, 联邦学习, 合成数据, 偏好优化, DPO, Private Evolution

## 一句话总结

将差分隐私联邦学习中的合成数据生成问题重新建模为 LLM 策略优化（DPO）问题，利用客户端 DP 反馈构建偏好对来微调 LLM，比传统 Private Evolution 提升更大——在 ε=1 下将隐私-性能差距缩小 58%。

## 研究背景与动机

差分隐私联邦学习（DP-FL）是从设备端私有数据训练模型的主流方法，但面临两大瓶颈：

**模型规模限制**：LLM 太大，无法在客户端设备上存储和训练，DP-FL 方法（如 DP-FedAvg、DP-FTRL）难以直接利用 LLM 能力

**合成数据方法的不足**：现有基于 Private Evolution（PE）的方法（如 PrE-Text）仅用提示词（prompting）控制 LLM 生成合成数据，丢弃低分样本也浪费了有价值的信息

核心洞察：PE 收集的客户端反馈（合成样本与私有数据的相似度评分）本质上可以看作 **RL 奖励信号**，因此可以用偏好优化（如 DPO）替代简单的提示工程来微调 LLM。

## 方法详解

### 整体框架

POPri 分为四个阶段循环执行 $T$ 轮：

**阶段 1：合成样本生成**
服务器生成 $K$ 个提示词，每个提示词用 LLM $\Psi$ 独立生成 $J$ 个合成样本（共 $K \times J$ 个），用句嵌入模型 $\Gamma$ 编码后发送给客户端。

**阶段 2：DP 客户端评分**
每个客户端 $i$ 计算每个合成样本与其私有样本的余弦相似度均值，得到评分向量，裁剪至范数 1 后加入高斯噪声：

$$\text{Scores}_{i,t} + \mathcal{N}(0, \sigma^2 I / L)$$

通过安全聚合汇总所有客户端评分：$\text{Scores}_t = \frac{1}{L}\sum_{i \in \mathcal{S}^t} \text{Scores}_{i,t}$

**阶段 3：LLM 偏好优化（核心创新）**
对每个提示词 $\eta_k$，按聚合评分排序其 $J$ 个响应：
- 最高分样本 → chosen（$y_\omega$）
- 第 $\ell$ 高分样本 → rejected（$y_r$）

构建偏好数据集后，使用 DPO 损失微调 LLM：

$$\min_\Psi \mathbb{E}_{x, y_\omega, y_r} \left[ -\log \sigma \left( \tau \log \frac{\Psi(y_\omega|x)}{\Psi(y_r|x)} - \tau \log \frac{\Psi_{\text{ref}}(y_\omega|x)}{\Psi_{\text{ref}}(y_r|x)} \right) \right]$$

其中 $\Psi_{\text{ref}}$ 为固定的原始 LLM 权重，$\tau$ 控制偏离程度。使用 LoRA（rank=4, α=8）减少 GPU 内存。

**阶段 4：下游微调**
最终轮次的 LLM $\Psi_T$ 生成大规模合成数据 $S_{\text{syn},T+1}$，用于微调下游模型 $\Phi$，部署到客户端设备。

### 隐私保证

- 每个客户端评分向量裁剪至范数 1 → 灵敏度为 1
- 高斯机制：$\mathcal{N}(0, \sigma^2 I)$ 噪声
- 使用 RDP 隐私会计（OPACUS）计算 $(ε, δ)$-DP

### 与 PE 的关键区别

| 维度 | Private Evolution | POPri |
|------|------------------|-------|
| 利用反馈方式 | 上下文学习（提示） | DPO 微调权重 |
| 低分样本处理 | 直接丢弃 | 作为 rejected 样本利用 |
| 相似度度量 | 最近邻直方图 | 余弦相似度 |

## 实验关键数据

### 主实验：下一词预测准确率（%）

| 数据集 | 方法 | ε=∞ | ε=7 | ε=1 | ε=0 |
|--------|------|-----|-----|-----|-----|
| bioRxiv | DP-FedAvg | 41.5 | 29.0 | 28.3 | 27.9 |
| | PE | — | 31.0 | 31.1 | — |
| | PE+SFT | — | 28.6 | 28.6 | — |
| | **POPri** | — | **34.4** | **34.8** | — |
| Congress | DP-FedAvg | 35.7 | 29.1 | 29.0 | 26.9 |
| | PE | — | 27.3 | 27.0 | — |
| | **POPri** | — | **30.6** | **30.4** | — |

### 中心化 DP 基准

| 数据集 | 方法 | ε=∞ | ε=1 |
|--------|------|-----|-----|
| PubMed | PE (Llama-2-7b) | 47.6 | 27.5 |
| | **POPri** | — | **29.4** |
| OpenReview | PE (Llama-2-7b) | 50.8 | 37.0 |
| | **POPri** | — | **40.2** |

### 关键性能指标

- bioRxiv ε=1：POPri 缩小隐私-性能差距 **58%**，PE 仅 23%，DP-FL 仅 3%
- 所有数据集和任务上 POPri 均取得最优下游性能

### 附加贡献：LargeFedBench

发布新联邦基准 LargeFedBench：Congressional Records（134K 客户端）+ bioRxiv 摘要（57K 客户端），定期更新以避免 LLM 数据污染。

## 亮点与洞察

1. **问题建模精妙**：将 PE 的客户端评分反馈重新诠释为 RL 奖励，自然引出 DPO 优化范式，理论动机清晰
2. **低分样本不浪费**：PE 丢弃低分样本，POPri 将其作为 rejected 对用于 DPO 训练，携带了"什么是不好的"信息
3. **余弦相似度优于直方图**：消融实验表明余弦相似度评分对 POPri 的策略优化至关重要
4. **隐私-效用帕累托改进**：在强隐私（ε=1）下仍能大幅超越基线，且通信/计算成本低于 DP-FL
5. **LargeFedBench 有实用价值**：大规模、持续更新、避免污染的联邦基准填补了社区空白

## 局限与展望

1. **仅评估文本任务**：方法依赖 LLM 生成合成数据，未验证图像/表格等模态
2. **LLM 依赖**：需要服务器端部署 LLaMA-3-8B 级别的 LLM，对服务器算力有要求
3. **评分质量受噪声影响**：DP 噪声可能导致偏好对构建不准确，尤其在极低 ε 或少量客户端时
4. **LoRA 超参敏感性**：rank=4, α=8 的固定设置是否对不同任务普适尚不清楚
5. **未与更新的策略优化方法对比**：只用了 DPO，未探索 KTO、IPO 等变体

## 相关工作与启发

- **Private Evolution 系列**：PE→Aug-PE→PrE-Text 是合成数据方法的主线，POPri 在此基础上引入权重微调
- **DPO (Rafailov et al., 2023)**：直接偏好优化是核心技术支柱
- **DP-SGD / DP-FL**：传统隐私优化方法是重要对比基线
- **启发**：该框架可推广到其他需要私有反馈+生成模型优化的场景，如隐私保护的个性化推荐

## 评分

- 新颖性: ⭐⭐⭐⭐ — PE→DPO 的重建模视角新颖，但技术组件（DPO+LoRA）本身已有
- 实验充分度: ⭐⭐⭐⭐ — 多数据集多设置、消融完整，但缺少非文本验证
- 写作质量: ⭐⭐⭐⭐ — 动机清晰、算法描述详细、图表直观
- 价值: ⭐⭐⭐⭐ — 对隐私联邦学习领域有实际推动，LargeFedBench 也有长期价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](../../NeurIPS2025/llm_safety/fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)
- [\[ICML 2025\] Reward-Augmented Data Enhances Direct Preference Alignment of LLMs](reward-augmented_data_enhances_direct_preference_alignment_of_llms.md)
- [\[NeurIPS 2025\] Differentially Private Federated Low Rank Adaptation Beyond Fixed-Matrix](../../NeurIPS2025/llm_safety/differentially_private_federated_low_rank_adaptation_beyond_fixed-matrix.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](../../NeurIPS2025/llm_safety/fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)

</div>

<!-- RELATED:END -->
