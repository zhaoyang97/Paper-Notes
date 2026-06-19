---
title: >-
  [论文解读] Make Your Training Flexible: Towards Deployment-Efficient Video Models
description: >-
  [ICCV 2025][预训练][灵活训练] 本文提出Flux——一种使视频模型训练灵活化的数据增强工具，通过灵活采样网格+组动态token选择，使单一模型在不同计算预算下都能高效工作；并提出Token Optimization新测试范式，在1/4 token下即可匹配前SOTA性能，节省约90%计算。
tags:
  - "ICCV 2025"
  - "预训练"
  - "灵活训练"
  - "Token优化"
  - "视频预训练"
  - "部署效率"
  - "数据增强"
---

# Make Your Training Flexible: Towards Deployment-Efficient Video Models

**会议**: ICCV 2025  
**arXiv**: [2503.14237](https://arxiv.org/abs/2503.14237)  
**代码**: [https://github.com/OpenGVLab/FluxViT](https://github.com/OpenGVLab/FluxViT)  
**领域**: LLM预训练  
**关键词**: 灵活训练, Token优化, 视频预训练, 部署效率, 数据增强

## 一句话总结
本文提出Flux——一种使视频模型训练灵活化的数据增强工具，通过灵活采样网格+组动态token选择，使单一模型在不同计算预算下都能高效工作；并提出Token Optimization新测试范式，在1/4 token下即可匹配前SOTA性能，节省约90%计算。

## 研究背景与动机

**领域现状**：视频表征学习是计算机视觉的基础任务，对多模态LLM和具身AI至关重要。当前主流方法在固定的时空采样网格（如8帧×224²）上操作固定数量的token，导致训练和部署时的大量冗余。

**现有痛点**：
   - **固定采样导致冗余**：视频本身有大量时空冗余，固定采样提取的token中很多是低信息量的
   - **部署不灵活**：训练时用8帧×224²，但实际部署可能需要适应不同计算预算。直接减帧数或降分辨率会导致性能大幅下降
   - **Token reduction效果有限**：现有token pruning/merging方法在大幅削减率下效果差，且策略本身也有计算开销
   - **已有灵活训练方法不完整**：ResFormer和FFN分别只处理空间或时间维度的灵活性，未同时处理时空灵活性，且未在大规模预训练中验证

**核心矛盾**：如何用一个模型同时满足不同计算预算的部署需求？简单降帧/降分辨率不是最优——同样的token数量下，应该选择信息最大化的token集合。

**本文目标** 提出Token Optimization范式——在给定token预算下，从更好地采样的视频中选择最优token集合，使信息最大化。

**切入角度**：将灵活采样和token选择结合作为训练时的数据增强，使模型天然适应各种分辨率和token数量。同时提出Token Optimization的test-time策略来找到最优的采样-选择组合。

**核心 idea**：用灵活采样+组动态token选择作为无成本的训练增强，使视频模型在各种计算预算下都能通过Token Optimization找到最优token集达到最佳性能。

## 方法详解

### 整体框架
如Fig.2-3所示，Flux包含三个层面的设计：(1) **Flexi-Sampling**：训练时随机选择不同的帧数和分辨率；(2) **Group-Dynamic Token Selector**：从灵活采样的token池中选择高信息量的token子集；(3) **FluxViT架构增强**：GLPE（全局-局部位置编码）和DPN（双层Patch归一化）使ViT适配可变token数量。测试时通过Token Optimization寻找最优的采样-选择配置。

### 关键设计

1. **Flexi-Sampling（灵活采样）**:

    - 功能：训练时每个视频随机采用不同的时空分辨率
    - 核心思路：对batch中的每个视频，随机选择帧数 $[F_{min}, F_{max}]$（步长 $t_s$）和空间分辨率 $[R_{min}, R_{max}]$（步长 $r_s$），并设置token数阈值 $T_{thres}$ 保持合理池大小。默认设置：帧数4-24，分辨率168-252
    - 设计动机：固定采样的模型只见过一种分辨率，对其他分辨率泛化差。灵活采样让模型见过各种分辨率组合，天然具备跨分辨率鲁棒性

2. **Group-Dynamic Token Selector（组动态token选择器）**:

    - 功能：从token池中选择信息量最大的token子集给teacher模型
    - 核心思路：将帧序列均匀分为 $N$ 个稀疏组 $B_i$。在每组内计算相邻帧token的动态值 $D(F_{t+1,i}) = \|F_{t+1,i} - F_{t,i}\|_p$（帧间差异），选择动态值最高的 $K/N$ 个token。这样保证：(a) 选择的是变化最大的（最信息化的）token；(b) 通过分组保证了全视频覆盖
    - 设计动机：视频中大量token是静态背景（低信息量），帧间变化大的token更有意义。分组确保不会因局部快速运动而忽略其他时段

3. **Double Mask Module（双掩码模块）**:

    - 功能：在UMT（Unmasked Teacher）框架中同时增强teacher和student
    - 核心思路：teacher侧使用Flexi-Sampling + Group-Dynamic Selector选择信息化token；student侧使用基于teacher CLS token注意力分数的mask。两个mask相互配合——teacher提供高质量的、从丰富采样中筛选的表征，student学习从更稀疏的视角理解视频
    - 设计动机：在不增加teacher计算成本的前提下（选择后token数量不变），从更高分辨率采样中获取更丰富的信息

4. **Global-Local Positional Embedding (GLPE)**:

    - 功能：处理灵活数量和间隔的token的位置编码
    - 核心思路：全局用可学习位置编码（sine-cosine初始化）+ Depth-Wise Conv增强局部关系。注意力中对Value向量加Linear Projection编码局部位置：$Z = (\text{Softmax}(\frac{QK^T}{\sqrt{D}}) + LPE) \cdot V$。LPE是value-dependent的，不受输入token数量影响
    - 设计动机：标准位置编码假设固定的token数量和排列。当token被选择/掩码后，它们来自不同的时空位置，需要编码其离散的位置信息

5. **Dual Patch Normalization (DPN)**:

    - 功能：稳定灵活采样下的训练
    - 核心思路：在标准Patch Embedding层后加一个LayerNorm（帮助动态估计），在Patch Embedding前也加一个LayerNorm（稳定梯度）
    - 设计动机：灵活采样导致输入token分布变化大，Patch Embedding的梯度可能过大。双层归一化稳定训练

### 损失函数 / 训练策略
- Flux-PT（预训练）：UMT框架的teacher-student对齐损失，使用InternVideo2-1B作为teacher
- Flux-FT（微调）：标准监督训练+自蒸馏（大token数量的聚合特征指导小token数量的训练）
- Multi-number co-training：单batch中使用3种不同token数量训练student，最大化teacher计算的利用率

## 实验关键数据

### 主实验

| 模型 | K400 Top-1 | SSv2 Top-1 | MSRVTT R@1 | COIN | 规模 |
|------|-----------|-----------|-----------|------|------|
| InternVideo2-S | 87.8 | - | - | - | Small |
| **FluxViT-S** | **90.0** | - | - | - | Small |
| InternVideo2-B | 89.0 | 73.5 | 48.2 | 92.5 | Base |
| **FluxViT-B** | **90.0** | **75.8** | **49.9** | **94.1** | Base |

### Token Optimization效果

| 配置 | Token数 | K400 | 相对Full | 计算节省 |
|------|---------|------|---------|---------|
| FluxViT-B Full | 3072 | 90.0 | 100% | 0% |
| FluxViT-B TO (1/4) | 768 | ~89.0 | ~99% | ~90% |
| InternVideo2-B Full | 3072 | 89.0 | - | 0% |

### 消融实验

| 配置 | K400 | SSv2 | 说明 |
|------|------|------|------|
| Baseline (InternVideo2 UMT) | 87.5 | 71.8 | 原始pipeline |
| + Flexi-Sampling | 88.2 | 73.0 | 灵活采样提升鲁棒性 |
| + Group-Dynamic Selector | 89.0 | 74.2 | 信息化token选择有效 |
| + GLPE + DPN | 89.5 | 75.0 | 架构增强关键 |
| + Multi-number training | 90.0 | 75.8 | 多token数共训进一步提升 |

### 关键发现
- **Token Optimization效果惊人**：1/4 token即可达到前SOTA（InternVideo2）的性能，计算节省约90%。这说明固定采样下有大量冗余
- **Flux作为增强工具的通用性**：在预训练（UMT）和有监督微调中都有效，且不增加训练成本
- **时空联合灵活性优于单独灵活性**：ResFormer（空间灵活）和FFN（时间灵活）分别只处理一个维度，Flux同时处理时空更优
- **FluxViT-B在多个任务上超越更大模型**：K400 90.0%、SSv2 75.8%、MSRVTT 49.9%、COIN 94.1%，在同等规模中是新SOTA
- **在chat-centric任务上也有提升**：作为视觉编码器接入LLM时，FluxViT在MVBench和Dream-1k上超越SigLIP/CLIP

## 亮点与洞察
- **Token Optimization新范式**：从"固定采样+全token"转向"灵活采样+最优token选择"，这是视频模型部署的范式转变。不是"用更少的帧/更低分辨率"，而是"在给定预算下找最优token集"
- **零成本增强**：Flux的token选择使teacher处理的token数量不变，因此不增加训练成本。这使得从更高分辨率采样变成了"免费午餐"
- **组动态选择器设计精巧**：分组确保时间覆盖，只选帧间变化大的token。简单但非常有效——既保证了信息量又避免了对快速运动的过拟合
- **与LLM集成验证**：在chat-centric设置下的验证为Flux在MLLM中的应用打开了大门

## 局限与展望
- Token Optimization的最优配置搜索有一定开销（需在验证集上测试多种配置）
- 灵活采样增加了数据预处理复杂度（需要支持多分辨率）
- 当前使用InternVideo2-1B作为teacher，teacher质量限制了学生模型的上界
- GLPE和DPN引入了少量额外参数和计算
- 可以考虑学习式的token选择器（而非基于帧间差异的启发式）

## 相关工作与启发
- **vs ResFormer/FFN**: ResFormer只处理空间灵活性，FFN只处理时间灵活性，且都未在大规模预训练中验证。Flux同时处理时空维度且在InternVideo2级别验证
- **vs MAR/MCM（token reduction）**: 传统masking在微调/推理时减token，但训练时不灵活。Flux从训练时就引入灵活性，使模型天然适应
- **vs InternVideo2**: FluxViT在同等规模下全面超越InternVideo2系列，证明Flux增强的有效性

## 评分
- 新颖性: ⭐⭐⭐⭐ Token Optimization范式有新意，但核心组件（灵活采样、动态选择）相对标准
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务（动作识别+检索+chat）、多规模、多设置，极其充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，消融系统化
- 价值: ⭐⭐⭐⭐⭐ 对视频模型高效部署有重要实际价值，1/4 token匹配SOTA是强结果

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[ACL 2025\] Towards Effective and Efficient Continual Pre-training of Large Language Models](../../ACL2025/llm_pretraining/towards_effective_and_efficient_continual_pre-training_of_large_language_models.md)
- [\[ACL 2025\] AsyncLM: Efficient and Adaptive Async Pre-training of Language Models](../../ACL2025/llm_pretraining/asynclm_efficient_and_adaptive_async_pre-training_of_language_models.md)
- [\[ICLR 2026\] Beyond Masks: Efficient, Flexible Diffusion Language Models via Deletion-Insertion Processes](../../ICLR2026/llm_pretraining/beyond_masks_efficient_flexible_diffusion_language_models_via_deletion-insertion.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](../../ICML2025/llm_pretraining/scaling_inference-efficient_language_models.md)

</div>

<!-- RELATED:END -->
