---
title: >-
  [论文解读] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples
description: >-
  [NeurIPS 2025][LLM安全][噪声标签] 发现并定义了误标注易学样本（Mislabeled Easy Examples, MEEs）——被模型早期训练即正确预测为错误标签的样本对泛化伤害最大，并提出 Early Cutting 方法利用模型后期状态重新校准早期置信子集来过滤MEEs。 - 深度学习严重依赖高质量…
tags:
  - "NeurIPS 2025"
  - "LLM安全"
  - "噪声标签"
  - "样本选择"
  - "误标注易学样本"
  - "Early Cutting"
  - "鲁棒训练"
---

# Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples

**会议**: NeurIPS 2025  
**arXiv**: [2502.08227](https://arxiv.org/abs/2502.08227)  
**代码**: [tmllab/2025_NeurIPS_MEE](https://github.com/tmllab/2025_NeurIPS_MEE)  
**领域**: 鲁棒学习 / 噪声标签  
**关键词**: 噪声标签, 样本选择, 误标注易学样本, Early Cutting, 鲁棒训练  

## 一句话总结

发现并定义了误标注易学样本（Mislabeled Easy Examples, MEEs）——被模型早期训练即正确预测为错误标签的样本对泛化伤害最大，并提出 Early Cutting 方法利用模型后期状态重新校准早期置信子集来过滤MEEs。

## 背景与动机

- 深度学习严重依赖高质量标注，但大规模数据集不可避免存在噪声标签
- 样本选择是处理噪声标签的主流方法，分为基于损失的方法和基于动态的方法
- 基于动态的方法利用DNN的记忆效应：先学简单模式，后拟合噪声，因此信任早期学到的样本
- 现有方法关注降低所选子集的噪声率，但忽视了**不同误标注样本的危害程度差异**
- 实验发现：被模型早期正确预测（按错误标签）的误标注样本，其对泛化性能的损害**显著大于**后期才学到的误标注样本
- 这些样本在特征空间中更靠近其错误标签类的中心（距离比 $r = d_{\text{mislabeled}}/d_{\text{true}} < 1$ 的占53.8%），使模型"合理地"将其归入错误类别

## 核心问题

在噪声标签学习的样本选择中，如何识别并过滤对模型泛化伤害最大的误标注易学样本（MEEs），即那些在训练早期就被模型按错误标签"自信地学会"的样本。

## 方法详解

### Mislabeled Easy Examples (MEEs) 的定义

定义样本 $(\mathbf{x}_i, \tilde{y}_i)$ 的学习时间：

$$LT_i = \min\{E_i \mid \hat{y}_i^{E_i-1} = \hat{y}_i^{E_i} = \tilde{y}_i\}$$

即模型连续两个epoch预测为给定标签的最早时间点。MEEs是学习时间最小的误标注样本子集。

### MEEs为何有害

- MEEs在早期模型特征空间中距离错误标签类中心更近：中位距离比 $r=0.830$（53.8%的MEEs有 $r<1$），而非MEEs中位 $r=3.923$（仅5.4%有 $r<1$）
- 其视觉特征与错误标签类的简单模式高度匹配（如飞机图片以海为背景被标为"船"）
- 破坏模型早期学习简单正确模式的过程，错误特征与干净数据表示纠缠

### Early Cutting 算法

**核心思路**：利用模型后期状态（early stopping epoch $t$）重新审视早期学到的置信子集。

**步骤1：基础样本选择** — 按学习时间 $LT_i$ 选择早期学到的子集 $\mathcal{D}^s$

**步骤2：Early Cutting 重校准** — 从 $\mathcal{D}^s$ 中识别MEEs。用模型 $f_{\theta^t}$ 计算三个指标：

1. **高损失**：$L_i = -\log p_i^{(\tilde{y}_i)} > \delta$（模型预测与给定标签不一致）
2. **高置信度**：$c_i = p_i^{(\hat{y}_i)} > \tau$（模型对其预测非常确定）
3. **低梯度范数**：$g_i = \|\nabla_{\mathbf{x}_i} L_i\|_2 < \epsilon$（损失对输入扰动不敏感，说明错误关联已牢固学习）

MEEs的操作性定义：

$$\text{MEEs} = \{i \in \mathcal{D}^s \mid (L_i > \delta) \wedge (c_i > \tau) \wedge (g_i < \epsilon)\}$$

实际实现使用分位数阈值：损失取top 10%、置信度取top 20%、梯度范数取bottom 20%。

**步骤3：移除并迭代** — $\mathcal{D}^s_{\text{refined}} \leftarrow \mathcal{D}^s \setminus \text{MEEs}$，在精炼后的子集上从头训练模型。

### 鲁棒性保证

- 早期学到的样本本身具有高冗余性（多个样本代表相似的简单模式），因此误删少量干净样本影响很小
- 即使 $\mathcal{S}$ 为空集（无可疑样本），方法自动退化为原始样本选择

## 实验关键数据

### CIFAR-10（ResNet-18）

| 方法 | Sym 20% | Sym 40% | Inst 20% | Inst 40% |
|------|---------|---------|----------|----------|
| Cross-Entropy | 86.64 | 82.64 | 87.62 | 82.82 |
| Co-teaching | 89.13 | 82.29 | 89.42 | 81.91 |
| Me-Momentum | 92.76 | 90.75 | 91.87 | 88.80 |
| Self-Filtering | 92.88 | 90.46 | 92.35 | 86.93 |
| RLM | 93.11 | 91.06 | 93.13 | 89.73 |
| **Early Cutting** | **93.79** | **91.80** | **93.40** | **90.78** |

### CIFAR-100（ResNet-34）

| 方法 | Sym 20% | Sym 40% | Inst 20% | Inst 40% |
|------|---------|---------|----------|----------|
| Misdetect | 73.90 | 65.10 | 70.45 | 63.66 |
| RLM | 71.68 | 67.68 | 68.26 | 67.31 |
| CSGN | 69.89 | 56.18 | 71.97 | 65.43 |
| **Early Cutting** | **76.20** | **72.77** | **75.03** | **69.94** |

### 大规模数据集（ResNet-50）

| 方法 | WebVision Val | ILSVRC12 Val | ImageNet-1k Sym40% |
|------|-------------|-------------|-------------------|
| Cross-Entropy | 67.32 | 63.84 | 67.99 |
| Late Stopping | 71.56 | 68.32 | 71.42 |
| **Early Cutting** | **73.00+** | **70.00+** | **74.00+** |

Early Cutting额外过滤的样本中误标注比例极高：对称噪声56.12%，非对称噪声95.29%，实例噪声91.33%。

## 亮点

- ⭐ 发现MEEs现象——并非所有误标注样本等害，早期学到的误标注样本伤害最大——这是对噪声标签学习的重要认知更新
- ⭐ 方法设计反直觉但合理：利用"通常被认为不可信"的后期模型来校正早期选择，因为后期模型恰好能区分MEEs
- ⭐ 三重标准（高损失+高置信+低梯度）精准锁定MEEs，避免误伤干净困难样本
- 在CIFAR-100 Sym40%噪声下比次优方法高出5个百分点，提升极为显著

## 局限与展望

- 三个分位数阈值（10%, 20%, 20%）基于验证集确定，不同数据分布下可能需要调整
- Early Cutting需要先完整训练一次模型获取学习时间，再第二次训练时进行筛选，计算开销翻倍
- 梯度范数的计算对高分辨率或大模型可能产生显著开销
- 多轮迭代（$I_{\text{rate}}$轮）在实际场景中如何确定最优迭代次数未充分讨论

## 与相关工作的对比

| 方法类型 | 代表 | 选择依据 | 能否过滤MEEs |
|---------|------|---------|-------------|
| 基于损失 | Co-teaching | 小损失=干净 | 不能（MEEs损失小） |
| 基于动态 | Me-Momentum | 早期学到=干净 | 不能（MEEs被早期学到） |
| 稳健损失 | GCE, Student Loss | 隐式下加权噪声 | 部分 |
| **Early Cutting** | 本文 | 早期选择+后期校准 | **能** |

## 启发与关联

- MEEs的存在说明DNN的"先学简单后记忆噪声"范式存在例外，某些噪声恰好符合简单模式
- 特征空间距离比 $r$ 可作为噪声标签有害程度的量化指标，有潜力成为通用工具
- Early Cutting 的"利用后期模型校正早期判断"思路可推广到其他需要多阶段决策的场景

## 评分

- ⭐ 新颖性: 9/10 — MEEs的发现和分析是对噪声标签领域的重要贡献，反直觉但有充分实证
- ⭐ 实验充分度: 9/10 — CIFAR-10/100、CIFAR-N、WebVision、ImageNet-1k全覆盖，多种噪声类型
- ⭐ 写作质量: 8/10 — 动机铺垫充分，实验递进展开，可视化分析助力理解
- ⭐ 价值: 8/10 — 对噪声标签学习有直接价值，MEEs概念可能影响后续研究方向

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[NeurIPS 2025\] Enhancing CLIP Robustness via Cross-Modality Alignment](enhancing_clip_robustness_via_crossmodality_alignment.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[ICCV 2025\] Asynchronous Event Error-Minimizing Noise for Safeguarding Event Dataset](../../ICCV2025/llm_safety/asynchronous_event_error-minimizing_noise_for_safeguarding_event_dataset.md)

</div>

<!-- RELATED:END -->
