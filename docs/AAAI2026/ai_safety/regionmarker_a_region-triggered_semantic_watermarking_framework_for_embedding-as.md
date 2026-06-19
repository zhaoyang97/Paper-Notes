---
title: >-
  [论文解读] RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service
description: >-
  [AAAI 2026][AI安全][EaaS版权保护] 提出基于语义区域触发的水印框架 RegionMarker，在低维空间中定义触发区域并注入语义水印，是首个能同时抵御 CSE 攻击、改写攻击和维度扰动攻击的 EaaS 版权保护方法。 Embedding-as-a-Service（EaaS）是大语言模型的一种商业部署策略…
tags:
  - "AAAI 2026"
  - "AI安全"
  - "EaaS版权保护"
  - "嵌入水印"
  - "语义区域触发"
  - "模型提取攻击防御"
  - "局部敏感哈希"
---

# RegionMarker: A Region-Triggered Semantic Watermarking Framework for Embedding-as-a-Service

**会议**: AAAI 2026  
**arXiv**: [2511.13329](https://arxiv.org/abs/2511.13329)  
**代码**: 未公开  
**领域**: AI安全  
**关键词**: EaaS版权保护, 嵌入水印, 语义区域触发, 模型提取攻击防御, 局部敏感哈希

## 一句话总结

提出基于语义区域触发的水印框架 RegionMarker，在低维空间中定义触发区域并注入语义水印，是首个能同时抵御 CSE 攻击、改写攻击和维度扰动攻击的 EaaS 版权保护方法。

## 背景与动机

Embedding-as-a-Service（EaaS）是大语言模型的一种商业部署策略，提供文本嵌入服务并收费（如 OpenAI 的 text-embedding-3-large）。然而 EaaS 面临严重的模型提取攻击威胁：攻击者可用文本语料查询提供者模型获取嵌入，以极低成本训练出功能相似的替代模型，造成巨大经济损失。

现有水印防御方法分为两类，均存在明显短板：

- **触发词类方法**（EmbMarker、WARDEN、EspeW）：依赖特定触发词注入水印，容易被改写攻击（paraphrasing）绕过，因为改写后触发词消失
- **线性变换类方法**（WET）：对所有嵌入施加秘密线性变换，能抵御改写攻击，但假设维度及其顺序不变，极易被维度扰动攻击（维度位移、维度截断）击破

核心矛盾在于：没有任何一种现有方法能同时防御 CSE 攻击、改写攻击和维度扰动攻击三类攻击。而实际场景中，攻击者会尝试多种攻击手段，只要有一种攻击成功就能绕过防护。

## 方法详解

### 整体框架

RegionMarker 框架由三个步骤组成：**触发区域定义**、**语义水印注入**和**版权验证**。核心思想是用语义区域（而非浅层词汇）作为触发器，利用多个语义区域抵御水印移除攻击，并用文本嵌入本身作为水印来防御维度扰动攻击。

### 触发区域定义

由于高维空间中数据稀疏且分布不均，直接在高维空间划分容易被 CSE 攻击识别。因此首先使用 PCA 降维到 $d$ 维紧凑语义空间，使数据分布更均匀、水印更隐蔽。

降维后使用**局部敏感哈希（LSH）**将 $d$ 维空间均匀划分为 $2^d$ 个区域。对每个文本嵌入 $\mathbf{v}$，通过随机超平面投影计算 $d$ 位二进制 LSH 签名确定其所属区域：

$$\text{LSH}_i(\mathbf{v}) = \mathbb{1}(\mathbf{n}_i \cdot \mathbf{v} > 0)$$

$$\text{LSH}(\mathbf{v}) = [\text{LSH}_1(\mathbf{v}), \cdots, \text{LSH}_d(\mathbf{v})]$$

其中 $\mathbf{n}_i$ 为互相正交的超平面法向量。划分完成后，设定水印区域比例 $\alpha$，随机采样 $R = \alpha \cdot 2^d$ 个区域作为触发区域 $A = \{a_1, a_2, \ldots, a_R\}$。

**关键安全性**：降维矩阵和触发区域仅提供者知晓，攻击者无法定位水印。

### 语义水印注入

为每个触发区域分配唯一的水印嵌入 $\mathbf{W} = \{\mathbf{w}_1, \mathbf{w}_2, \ldots, \mathbf{w}_R\}$，其中 $\mathbf{w}_r$ 是目标样本的嵌入。若降维后的文本嵌入落入触发区域 $a_r$，则：

$$\mathbf{e}_p = \text{Norm}((1 - \lambda) \cdot \mathbf{e}_0 + \lambda \cdot \mathbf{w}_r)$$

其中 $\lambda$ 控制水印强度（默认 0.2）。由 LSH 划分的区域互不相交，每个文本嵌入最多携带一个水印。使用目标样本嵌入作为水印的好处是：即使攻击者对维度进行位移或截断，水印嵌入也会发生同样的变化，相对关系保持不变。

### 版权验证

构建验证语料库，包含多个后门语料 $D_p^{b_r}$（落入水印区域的文本）和良性语料 $D_p^n$（非水印区域的文本）。计算各组文本与水印嵌入的余弦相似度差 $\Delta_{cos}$ 和 L2 距离差 $\Delta_{l2}$，并使用 **Kolmogorov-Smirnov 检验**判定分布差异：

$$\Delta_{cos} = \max_{1 \leq r \leq R} \Delta_{cos_r}, \quad p\text{-value} = \min_{1 \leq r \leq R} p\text{-value}_r$$

采用保守策略：任何一个水印区域的 p-value < 0.05 即判定为侵权。

## 实验结果

### 实验设置

- **提供者模型**: GPT-3 text-embedding-002；**窃取者模型**: BERT
- **数据集**: SST-2（情感分类）、AG News（新闻分类）、Enron（垃圾邮件）、MIND（新闻推荐）
- **攻击类型**: CSE 攻击、NLLB/gpt-4o-mini 改写攻击、维度位移攻击、维度截断攻击
- **超参数**: $d=4$, $\alpha=20\%$, $\lambda=0.2$

### 表1: SST-2 数据集各方法综合对比

| 方法 | 无攻击 | CSE | 改写(NLLB) | 改写(GPT-4o-mini) | 维度位移 | 维度截断 |
|------|--------|-----|-----------|-------------------|---------|---------|
| WARDEN | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| EspeW | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ |
| WET | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| **RegionMarker** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

RegionMarker 是唯一在所有攻击下均成功保护版权的方法。任务性能（ACC 约 93%）与基线方法持平，水印注入未显著损害嵌入质量。

### 表2: Enron 数据集各方法检测性能对比

| 方法 | 攻击类型 | p-value | Δcos(%) | 版权保护 |
|------|---------|---------|---------|---------|
| RegionMarker | 无攻击 | <10⁻⁵ | 11.91 | ✓ |
| RegionMarker | CSE | <10⁻⁴ | 26.27 | ✓ |
| RegionMarker | 改写(NLLB) | <10⁻⁴ | 7.12 | ✓ |
| RegionMarker | 维度位移 | <0.01 | 2.33 | ✓ |
| RegionMarker | 维度截断 | <0.02 | 1.96 | ✓ |
| WARDEN | CSE | >0.05 | 1.47 | ✗ |
| EspeW | 改写(NLLB) | >0.49 | 0.40 | ✗ |
| WET | 维度位移 | >0.08 | -1.23 | ✗ |

在 Enron 数据集上 RegionMarker 同样全面通过，而其他方法各有弱点被击破。

## 消融实验与关键发现

1. **PCA 降维的必要性**：去掉 PCA 后在 CSE 攻击下 p-value 从 <0.05 恶化为 >0.5（保护失败），说明降维使数据分布均匀化是抵御 CSE 的关键
2. **多水印嵌入的必要性**：使用单一水印嵌入时，CSE 攻击下 p-value 从 <0.05 恶化为 >0.08（保护失败），因为单一水印容易被识别和移除
3. **水印区域比例 α**：α 增大时检测性能提升，但保持 20% 的低比例以减少对嵌入质量的影响
4. **PCA 维度 d**：d 增大时无攻击检测性能下降（每区域样本减少），但攻击场景下鲁棒性提升（水印数量增多更难移除），选择 d=4 作为平衡点

## 亮点

- **全面防御能力**：首个能同时抵御三类主流攻击（CSE、改写、维度扰动）的 EaaS 水印方法
- **设计巧妙**：用语义区域替代触发词，天然抵御改写攻击（改写不改变语义区域归属）；用文本嵌入自身作为水印，天然抵御维度扰动（水印与嵌入同变）
- **秘密性强**：降维矩阵 + 随机触发区域 + 多水印嵌入三层秘密，攻击者难以逆向
- **实用性好**：任务性能几乎无损，超参数设置简单（仅 d、α、λ 三个参数）

## 局限性

- 仅在文本嵌入场景实验，未扩展到多模态嵌入（图像、音频等 EaaS）
- 攻击类型虽覆盖现有主流方法，但未考虑可能的自适应攻击（如攻击者知道使用了区域触发策略时的定向攻击）
- 窃取者模型固定为 BERT，未评估更强的窃取模型（如 GPT 级别模型）对水印保持性的影响
- PCA 降维和 LSH 区域划分依赖于提供者的训练数据分布，数据分布偏移时效果未可知
- 验证过程需要构建后门和良性语料进行统计检验，不支持单样本快速验证

## 相关工作

- **模型提取攻击**: liu2022stolenencoder 发现公开 EaaS API 易受模仿攻击
- **触发词水印**: EmbMarker (Peng et al., 2023) 开创性工作但不抗改写；WARDEN 用多水印嵌入抗 CSE；EspeW 在子维度嵌入增强隐蔽性
- **变换水印**: WET (Shetty et al., 2024) 用秘密线性变换抗改写但不抗维度扰动
- **版权验证**: 基于 KS 检验的统计验证方法

## 评分

⭐⭐⭐⭐ — 问题定义清晰，首次揭示现有方法无法全面防御三类攻击的困境；方法设计优雅，用语义区域触发和文本嵌入水印两个核心创新自然解决两大难题；实验全面覆盖 4 个数据集和 5 种攻击。不足之处是缺乏自适应攻击分析和多模态扩展。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] MaxMark: High-Capacity Diffusion-Native Watermarking via Robust and Invertible Latent Embedding](../../CVPR2026/ai_safety/maxmark_high-capacity_diffusion-native_watermarking_via_robust_and_invertible_la.md)
- [\[AAAI 2026\] CoRe-Fed: Bridging Collaborative and Representation Fairness via Federated Embedding Distillation](core-fed_bridging_collaborative_and_representation_fairness_via_federated_embedd.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] ReMoE: Region-Mixture Experts for Adversarially-Robust Vision Transformers](../../CVPR2026/ai_safety/remoe_region-mixture_experts_for_adversarially-robust_vision_transformers.md)
- [\[AAAI 2026\] Yours or Mine? Overwriting Attacks Against Neural Audio Watermarking](yours_or_mine_overwriting_attacks_against_neural_audio_watermarking.md)

</div>

<!-- RELATED:END -->
