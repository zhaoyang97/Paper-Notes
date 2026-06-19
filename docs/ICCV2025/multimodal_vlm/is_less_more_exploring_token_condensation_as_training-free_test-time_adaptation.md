---
title: >-
  [论文解读] Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation
description: >-
  [ICCV 2025][多模态VLM][测试时自适应] 提出 Token Condensation as Adaptation（TCA），一种免训练的测试时自适应方法，通过领域感知的 token 库（DTR）引导跨头 token 裁剪/合并和 logits 自校正，在不修改模型参数的情况下，将 CLIP/SigLIP 系列的跨数据集性能提升最高 21.4%，同时减少 12.2%-48.9% 的 GFLOPs。
tags:
  - "ICCV 2025"
  - "多模态VLM"
  - "测试时自适应"
  - "Token 裁剪与合并"
  - "CLIP"
  - "免训练"
  - "视觉-语言模型"
---

# Is Less More? Exploring Token Condensation as Training-free Test-time Adaptation

**会议**: ICCV 2025  
**arXiv**: [2410.14729](https://arxiv.org/abs/2410.14729)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 测试时自适应, Token 裁剪与合并, CLIP, 免训练, 视觉-语言模型

## 一句话总结

提出 Token Condensation as Adaptation（TCA），一种免训练的测试时自适应方法，通过领域感知的 token 库（DTR）引导跨头 token 裁剪/合并和 logits 自校正，在不修改模型参数的情况下，将 CLIP/SigLIP 系列的跨数据集性能提升最高 21.4%，同时减少 12.2%-48.9% 的 GFLOPs。

## 研究背景与动机

视觉-语言模型（VLM）如 CLIP 在零样本推理中表现出色，但在特定下游数据集上往往性能下降。现有的测试时自适应（TTA）方法存在以下问题：

**传统 TTA 计算昂贵**：如 Tent、SAR 等方法需要更新模型参数（如批归一化层），依赖大 batch size（如 256）来稳定自适应过程，在 VLM 的庞大参数集上应用不切实际。

**测试时 prompt 调优（TPT）也有局限**：TPT 通过学习小规模任务特定的上下文 prompt 来对齐文本和视觉特征，但它主要关注文本输入的精细化，忽视了视觉分布偏移。同时依赖外部源数据或大量数据增强（如 60 倍 AugMix），导致 GFLOPs 从 17.59 飙升至 1108.61。

**一个关键观察**：作者发现，选择性地精简低注意力 token 不仅能保持性能，还能在某些未见数据集上增强性能。这是因为两类 token 引入了视觉-文本误对齐：（1）与类别无关的背景 token 误导模型关注非必要区域；（2）类别模糊的目标 token（如动物毛发纹理）跨类别重叠，分散视觉嵌入。

**现有 token 精简方法的问题**：EViT、ToME 等方法虽然能提高效率，但在减少 token 时往往牺牲分布内（如 ImageNet-1K）性能，无法实现"免费午餐"式自适应。

## 方法详解

### 整体框架

TCA 是一个免训练的在线自适应框架，由三个核心组件构成：

1. **领域感知 Token 库（DTR）**：保留代表性的域锚 token，为自适应提供稳定参考
2. **领域感知跨头 Token 精简**：基于域锚 token 指导，选择性裁剪/合并低信息量 token
3. **Logits 自校正**：利用存储的域锚 token 精细化模型预测

### 关键设计

1. **Domain-aware Token Reservoir（DTR）**:

    - 功能：维护一个按类别组织的优先队列 $\mathfrak{R} = \{\mathfrak{R}_c\}_{c=1}^C$，存储来自所有 $L$ 层的域锚 token（CLIP 中为 \<cls\> token，SigLIP 中为池化向量）
    - 核心思路：每个类别缓冲区 $\mathfrak{R}_c$ 保留 $M$ 个最可靠的域锚 token，按熵分数排序：$\mathbf{H}_c(\mathbf{z}_t, \mathbf{t}_c) = -\mathbf{p}_{t,c} \log \mathbf{p}_{t,c}$。只有当 $\arg\max(\mathbf{p}_{t,c}) = c$ 时才更新，确保仅保留语义最一致的样本。当缓冲区满时，替换熵最高的样本
    - 设计动机：随着时间推移，低熵的 \<cls\> token 与文本嵌入的对齐度持续提高（实验验证），可作为域级别的适应参考点

2. **Domain-aware Cross-head Token Reduction**:

    - 功能：在多头自注意力和前馈层之间执行 token 裁剪与合并
    - 核心思路：
        - **域感知 Token 评估**：从 DTR 中采样最匹配的域锚 token $\mathbf{A}_{c^*}^{l-1}$，将其与当前 \<cls\> token 拼接后计算注意力：$\text{Attention}([\mathbf{v}_{\text{cls}}^l; \mathbf{A}_{c^*}^{l-1}]\mathbf{W}_Q^h, [\mathbf{V}^l; \mathbf{A}_{c^*}^{l-1}]\mathbf{W}_K^h)$
        - **跨头评分**：对每个 token 计算跨头平均排名分数 $\mathbf{S}_i^{\text{head}} = \frac{1}{H}\sum_{h=1}^H \text{rank}_h(i)$，而非简单平均注意力分数，避免异常值头的不当影响
        - **两阶段精简**：先裁剪低排名 token（与类别无关的背景），再对中间排名 token（类别模糊）进行核心集合并
    - 设计动机：（1）原始 \<cls\> token 是通用的，可能捕获与目标类别无关的语义，拼接域锚 token 提供历史上下文对齐；（2）逐头平均注意力分数容易被异常值头主导，跨头排名更鲁棒

3. **Logits Self-correction**:

    - 功能：在 token 精简后补偿语义偏移，精细化分类预测
    - 核心思路：将当前样本的视觉 \<cls\> token 与 DTR 中存储的域锚 token 计算跨层余弦相似度，作为 token 级分类器校正原始预测：$\tilde{\mathbf{p}}_{t,c} = \mathbf{p}_{t,c} + \lambda\mathbf{p}_{t,c}^{\text{token}}$，其中 $\mathbf{p}_{t,c}^{\text{token}} = \frac{1}{M}\sum_{i=1}^M \cos(\mathbf{V}_t^{\text{cls}}, \mathbf{A}_{i,c}^{\text{cls}}) \cdot \mathbf{P} \cdot \mathbb{1}_c$，$\mathbf{P} = [\exp(\frac{l}{\beta})]_{l=1}^L$ 是层级指数缩放系数
    - 设计动机：token 精简后可能引入语义偏移，利用 DTR 中积累的域知识从纯视觉角度校正预测，无需修改模型参数

### 损失函数 / 训练策略

TCA 是**完全免训练**的方法，不涉及任何损失函数或训练过程。所有操作在推理时在线执行，batch size 为 1，无需数据增强。

## 实验关键数据

### 主实验

| 方法 | Aug-free | Aircraft | Caltech | Cars | DTD | EuroSAT | Flower | Food | Pets | SUN | UCF | 平均 | GFLOPs |
|------|---------|----------|---------|------|-----|---------|--------|------|------|-----|-----|------|--------|
| CLIP | ✓ | 23.22 | 93.55 | 66.11 | 45.04 | 50.42 | 66.99 | 82.86 | 86.92 | 65.63 | 65.16 | 64.59 | 17.59 |
| TDA | ✓ | 23.91 | 94.24 | 67.28 | 47.40 | 58.00 | 71.42 | 86.14 | 88.63 | 67.62 | 70.66 | 67.53 | 17.59 |
| TCA R=0.9 | ✓ | **24.87** | 93.63 | 65.33 | 46.16 | **70.43** | **73.33** | 85.31 | **89.53** | 65.92 | **72.38** | **68.69** | **15.45** |

TCA 相比 CLIP 零样本基线在跨数据集基准上平均提升 4.10%，同时减少 12.2% GFLOPs。

### 消融实验

| 配置 | 平均准确率 | 说明 |
|------|-----------|------|
| 不用 DTR（仅 \<cls\> 注意力）| 65.17 | EViT R=0.9 基线 |
| 仅裁剪（无 DTR 指导）| 65.17 | 无域感知能力 |
| DTR + 裁剪 | 67.83 | DTR 提供域上下文 |
| DTR + 裁剪 + 合并 | 68.12 | 合并保留模糊 token 信息 |
| DTR + 裁剪 + 合并 + Logits 校正 | **68.69** | 全组件最优 |
| EViT R=0.7 | 62.02 | 激进裁剪性能下降大 |
| ToME R=0.7 | 60.33 | 合并无法弥补信息损失 |
| TCA R=0.7 | 66.64 | 保持竞争力，减少 48.9% GFLOPs |

### 关键发现

- **EuroSAT 上最显著提升**：从 CLIP 的 50.42% 提升至 TCA 的 70.43%（+20%），因为卫星图像与预训练数据分布差异大，token 精简能有效去除误导性背景
- **免训练 + 减少计算量 = 双赢**：TCA 是唯一同时提升性能和降低计算量的方法
- 在 CIFAR-100-C 损坏数据集上，TCA 比最强基线提升最高 21.4%
- DTR 的更新策略中，按熵排序的优先队列优于 FIFO 和基于相似度的策略
- 方法可轻松扩展到 SigLIP 和 SigLIP v2，只需将 \<cls\> token 替换为池化特征向量

## 亮点与洞察

- **全新视角**：首次将 token 精简从"效率工具"重新定位为"免训练自适应策略"，这一见解非常深刻——减少 token 不仅降低计算量，还能改善分布偏移下的对齐质量
- **跨头排名**代替平均注意力分数是简单但有效的改进，对异常值头的鲁棒性显著提高
- **零额外参数**：所有操作利用预训练模型已有的注意力权重和特征，不引入任何新参数
- 通过 DTR 实现了高效的在线域适应，无需大 batch size 或数据增强

## 局限与展望

- 超参数 $K$（DTR 起始应用层）和 $R$（保留 token 比例）需要针对不同模型和数据集调整
- DTR 的容量 $M$ 和更新策略对性能有一定影响，最优设置依赖具体场景
- 在 Food101 等与预训练数据分布接近的数据集上，TCA 改善有限甚至略有下降
- 目前仅验证了图像分类任务，未扩展到检测、分割等下游任务
- 逻辑自校正中的 $\beta$（层级温度参数）需要语义先验来选择

## 相关工作与启发

- EViT 的 token 裁剪思想被本文从效率优化重新诠释为域自适应工具
- TDA（Training-free Dynamic Adapter）是最接近的免训练基线，但 TDA 依赖大量超参数且推理开销更大
- 本文的 DTR 设计与 TDA 的正负缓存类似但更轻量，且 TCA 额外利用了注意力层的 token 精简
- 对 VLM 视觉编码器中 token 作用的深入分析（背景 token vs 模糊目标 token）为理解分布偏移提供了新视角

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 将 token 精简重新解释为自适应策略是原创且深刻的洞察
- 实验充分度: ⭐⭐⭐⭐ 跨 10 个数据集 + CIFAR-100-C，多个 VLM 验证，但缺少检测任务
- 写作质量: ⭐⭐⭐⭐ 动机论述清晰，实验经验分析到位
- 价值: ⭐⭐⭐⭐ 免训练+减少计算量的实用性很强，但应用范围目前限于分类

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models](multi-cache_enhanced_prototype_learning_for_test-time_generalization_of_vision-l.md)
- [\[NeurIPS 2025\] DOTA: DistributiOnal Test-time Adaptation of Vision-Language Models](../../NeurIPS2025/multimodal_vlm/dota_distributional_testtime_adaptation_of_visionlanguage_mo.md)
- [\[NeurIPS 2025\] The Illusion of Progress? A Critical Look at Test-Time Adaptation for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/the_illusion_of_progress_a_critical_look_at_testtime_adaptat.md)
- [\[CVPR 2025\] Free on the Fly: Enhancing Flexibility in Test-Time Adaptation with Online EM](../../CVPR2025/multimodal_vlm/free_on_the_fly_enhancing_flexibility_in_test-time_adaptation_with_online_em.md)
- [\[ICCV 2025\] Exploiting Vision Language Model for Training-Free 3D Point Cloud OOD Detection](exploiting_vision_language_model_for_training-free_3d_point_cloud_ood_detection_.md)

</div>

<!-- RELATED:END -->
