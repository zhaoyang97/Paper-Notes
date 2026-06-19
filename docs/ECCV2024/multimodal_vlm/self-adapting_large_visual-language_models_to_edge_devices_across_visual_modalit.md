---
title: >-
  [论文解读] Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities
description: >-
  [ECCV 2024][多模态VLM][边缘部署] 提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。 将视觉语言模型部署到边缘设备面临三大核心挑战…
tags:
  - "ECCV 2024"
  - "多模态VLM"
  - "边缘部署"
  - "知识蒸馏"
  - "量化"
  - "跨模态迁移"
  - "对比学习"
---

# Self-Adapting Large Visual-Language Models to Edge Devices across Visual Modalities

**会议**: ECCV 2024  
**arXiv**: [2403.04908](https://arxiv.org/abs/2403.04908)  
**代码**: [GitHub](https://github.com/ramdrop/edgevl)  
**领域**: LLM/NLP  
**关键词**: 边缘部署, 知识蒸馏, 量化, 跨模态迁移, 对比学习

## 一句话总结
提出EdgeVL框架，通过两阶段适配（双模态知识蒸馏+量化感知对比学习），将大规模VLM（如CLIP）适配到边缘设备上，实现无需人工标注的跨模态（RGB和非RGB）开放词汇分类，达到最高15.4%的准确率提升和93倍的模型压缩。

## 研究背景与动机
将视觉语言模型部署到边缘设备面临三大核心挑战：

**多模态泛化**：边缘设备通常配备RGB之外的传感器（深度、红外等），但CLIP等VLM的视觉编码器主要针对RGB图像训练，在非RGB图像上的零样本分类性能急剧下降（如ScanNet数据集上深度图像准确率仅为RGB的1/8）

**标签稀缺**：边缘设备持续产生大量无标注图像，人工标注成本高昂

**资源约束**：大型视觉编码器（如ViT）的计算需求远超边缘设备的内存和算力限制

现有方法的局限：
- 跨模态知识迁移方法（CMKD等）：需要标注数据，且仅针对单一模态训练
- 模型压缩方法（量化、剪枝、蒸馏）：独立使用，未与跨模态迁移协同
- 暴力整合两个模块会导致明显的性能下降

本文的核心创新在于：**首次系统性地将大规模VLM适配到边缘设备，同时处理跨模态泛化和模型压缩两个问题**，且全程无需人工标注。

## 方法详解

### 整体框架
EdgeVL是一个两阶段适配框架：
- **Stage-1**：双模态知识蒸馏（$\Phi_{img} \rightarrow \Phi_{img}^{stu}$）—— 将大教师编码器（ViT-G）的知识迁移到小学生编码器（Swin-T/DAT-T/ViT-S），使其能处理RGB和非RGB图像
- **Stage-2**：量化感知对比学习（$\Phi_{img}^{stu} \rightarrow \Phi_{img}^{edge}$）—— 将全精度学生模型量化为Int8低比特模型，并通过对比学习在量化后保持甚至增强特征判别力

### 关键设计

1. **自动数据集策划（Automatic Dataset Curation）**:

    - 功能：在蒸馏前自动过滤噪声样本，无需人工干预
    - 核心思路：利用ChatGPT-4生成场景标签超集 $\mathcal{S}$（如室内/卫星场景类别），再利用CLIP的图文匹配能力为每张RGB图片计算置信度分数 $c_i = \max_k \frac{e^{\Phi_{img}(x_i)^\top \Phi_{text}(y_k)}}{\sum_k e^{\Phi_{img}(x_i)^\top \Phi_{text}(y_k)}}$
    - 低置信度图片（噪声/无信息特征）被剔除，保留高于阈值 $\tau_c = 0.25$ 的图片及其配对非RGB图片
    - 设计动机：教师模型也有失败案例，噪声样本作为监督信号会损害蒸馏效果

2. **双模态特征蒸馏（Dual-Modality Feature Distillation）**:

    - 功能：训练一个统一的学生编码器，能同时处理RGB和非RGB图像
    - 核心思路：对每对RGB/非RGB图像，将学生模型在两种模态上的特征都与教师在RGB图像上的特征对齐
    - 蒸馏损失：$\mathcal{L}_d = d(\Phi_{img}(x), \Phi_{img}^{stu}(x')) + d(\Phi_{img}(x), \Phi_{img}^{stu}(x))$
    - 其中 $x$ 是RGB图像，$x'$ 是对应的非RGB图像，$d$ 为L1距离
    - 创新点：**共享权重的双模态编码器** —— 不为每种模态训练单独的模型，而是一个统一编码器通过权重共享处理两种输入，至少减半存储需求
    - 设计动机：两种模态描述的是同一场景，特征应当一致；RGB图像相当于非RGB图像的数据增强

3. **量化感知对比学习（QAT + Contrastive Learning）**:

    - 功能：在量化过程中通过对比学习维持并增强特征判别力
    - 背景观察：直接做PTQ后特征判别力显著下降，图像特征与文本标签之间的角度增大（偏离对齐）
    - 核心思路：在QAT的fake-quantize训练中，不仅使用传统蒸馏损失，还引入对比学习损失来增强量化后特征的判别能力
    - 关键发现：QAT+对比学习不仅恢复了量化导致的判别力损失，还使量化后的特征比全精度模型更好（图像-文本角度 $\theta_3 < \theta_1$）
    - 设计动机：传统蒸馏损失只追求与教师特征的对齐，但可能无法充分利用量化模型在特征空间中的判别潜力；对比学习天然适合学习对量化扰动鲁棒的不变表征

4. **半困难三元组采样（Semi-Hard Triplet Sampling）**:

    - 功能：为对比学习选择有效的正负样本
    - 伪标签生成：利用预训练VLM在标签超集上做最大相似度匹配获得伪标签
    - 正样本选择：同伪标签中特征最近的样本
    - 负样本选择：满足半困难条件的样本 —— 负样本距离大于正样本距离但小于正样本距离加边距m
    - 对比损失：$\mathcal{L}_c = \frac{1}{J}\sum_{j=1}^{J} d(f(x_i), f(p_{i,k^*})) - d(f(x_i), f(n_{i,j})) + m$
    - 超参数：margin $m = 0.3$，负样本数 $J = 3$
    - 设计动机：半困难采样比困难采样更稳定，能更好地提升特征鲁棒性

### 损失函数 / 训练策略
- **Stage-1**：AdamW，学习率 $10^{-4}$，cosine衰减至 $5 \times 10^{-6}$，120 epochs
- **Stage-2**：学习率降至 $10^{-6}$，使用per-channel权重量化 + per-tensor激活量化
- **两阶段必须串行**：对比学习需要Stage-1提供的良好特征空间作为起点，一阶段训练导致性能大幅下降（50.0% vs 30.0%）
- 教师模型：CLIP ViT-G-14（OpenCLIP）
- 学生模型：ViT-S / DAT-T / Swin-T + 特征投影头

## 实验关键数据

### 主实验：ScanNet和EuroSAT准确率

| 方法 | 精度 | ScanNet 非RGB/RGB/均值 | EuroSAT 非RGB/RGB/均值 |
|------|------|----------------------|----------------------|
| CLIP-B | F32 | 4.5/36.2/20.4 | 16.8/40.4/28.6 |
| CLIP-G | F32 | 6.2/47.3/26.8 | 16.9/54.0/35.5 |
| SKD | F32 | 31.2/37.8/34.5 | 22.9/50.3/36.6 |
| CQD | F32 | 40.1/6.7/23.4 | 62.4/36.4/49.4 |
| **EdgeVL (DAT-T)** | **Int8** | **47.9/52.0/49.9** | **61.0/65.7/63.3** |
| **EdgeVL (Swin-T)** | **Int8** | **46.0/48.7/47.4** | **61.3/67.1/64.2** |

### 消融实验：量化策略对比（ScanNet，DAT-T）

| 方法 | 精度 | 非RGB | RGB | 均值 |
|------|------|------|-----|------|
| Stage-1 only | F32 | 38.6 | 40.6 | 39.6 |
| +PTQ | Int8 | 33.0 | 36.5 | 34.8 |
| +QAT | Int8 | 39.4 | 41.2 | 40.3 |
| +QViT | Int8 | 35.0 | 38.0 | 36.5 |
| **+Stage-2 (EdgeVL)** | **Int8** | **47.9** | **52.0** | **50.0** |

### 效率对比

| 方法 | 模型大小 | AGX延迟 | Nano延迟 | 4090吞吐 |
|------|---------|---------|---------|---------|
| CLIP-G | 5213 MB | / | / | / |
| CLIP-B | 330 MB | 9.5 ms | 20.2 ms | 772 img/s |
| EdgeVL (ViT-S) | 86 MB | 4.6 ms (↓52%) | 9.9 ms (↓51%) | 1492 img/s (↑93%) |
| EdgeVL (Swin-T) | 56 MB | 5.2 ms (↓46%) | 11.4 ms (↓44%) | 1098 img/s (↑42%) |

### 关键发现
- **Int8量化后反超全精度基线**：EdgeVL (Int8) 的均值准确率远超所有F32基线（ScanNet: 49.9% vs 34.5%; EuroSAT: 64.8% vs 49.4%）
- **对比学习是量化成功的关键**：Stage-2比普通QAT高出约10%（50.0% vs 40.3%），比PTQ高15.2%
- **双模态训练的增益**：双模态训练相比单RGB/单非RGB训练分别提升15.0%和13.1%的平均准确率
- **跨数据集泛化**：在ScanNet上训练后迁移到NYU2，深度图像准确率从CLIP-G的25.7%提升到EdgeVL的51.1%
- **阈值 $\tau_c$ 的影响**：$\tau_c = 0.25$ 最优；过小（0.10）导致训练不充分，过大（0.50）引入噪声
- **两阶段不可合并**：一阶段训练（49.9% vs 30.0%）大幅劣化，因为对比学习需要良好的特征空间起点

## 亮点与洞察
- **首次系统解决VLM边缘部署+跨模态问题**：将知识迁移与模型压缩有机整合，而非简单拼接
- **无标签全自动**：从数据策划到特征蒸馏到量化训练，全程无需人工标注
- **量化后反超全精度**：通过对比学习使量化成为一种特征增强而非退化，这一发现与Paper 1（QPrompt）异曲同工
- **统一双模态编码器**：权重共享设计将存储需求减半，且RGB图像作为非RGB图像的"增强"提升了两种模态的性能
- **实际部署验证**：在Jetson AGX/Nano/RTX4090上使用TensorRT进行了实际推理测试

## 局限与展望
- **RGB图像性能权衡**：跨数据集场景中RGB准确率略有损失（因模型大幅缩小），可能需要更多适配数据
- **仅验证了场景分类**：未在语义分割、目标检测等更复杂任务上评估
- **适配数据规模有限**：仅使用约4,725对图像进行适配，扩大数据规模可能进一步提升泛化
- **标签超集依赖ChatGPT**：标签超集的质量可能影响数据策划和伪标签的准确性
- **仅测试了深度和SWIR两种非RGB模态**：热红外、近红外等更多模态的效果未知

## 相关工作与启发
- **CLIP**：作为教师模型的基础VLM，其零样本能力在非RGB上的不足是本文的出发点
- **CMKD**：跨模态知识蒸馏方法，但需要标签且仅支持单模态，EdgeVL的Stage-1在此基础上扩展为双模态无标签蒸馏
- **LSQ/EWGS**：QAT中的梯度改进方法，启发了EdgeVL在量化阶段的设计
- 启发：量化+对比学习的组合可能适用于更多场景（如VLM的其他边缘下游任务），"约束即增强"是一个值得深挖的方向

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统解决VLM边缘跨模态部署的框架，两阶段设计有独到之处
- 实验充分度: ⭐⭐⭐⭐ 多数据集、多backbone、多GPU平台、详尽消融，跨数据集泛化验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，两阶段逻辑自洽，补充材料充实
- 价值: ⭐⭐⭐⭐ 边缘部署与跨模态是VLM走向实际应用的关键瓶颈，实用性极高

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](../../CVPR2025/multimodal_vlm/self-supervised_spatial_correspondence_across_modalities.md)
- [\[ECCV 2024\] SQ-LLaVA: Self-Questioning for Large Vision-Language Assistant](sq-llava_self-questioning_for_large_vision-language_assistant.md)
- [\[ECCV 2024\] BRAVE: Broadening the Visual Encoding of Vision-Language Models](brave_broadening_the_visual_encoding_of_vision-language_models.md)
- [\[ECCV 2024\] CAT: Enhancing Multimodal Large Language Model to Answer Questions in Dynamic Audio-Visual Scenarios](cat_enhancing_multimodal_large_language_model_to_answer_questions_in_dynamic_aud.md)
- [\[ACL 2025\] Vision-Language Models Struggle to Align Entities across Modalities](../../ACL2025/multimodal_vlm/vision-language_models_struggle_to_align_entities_across_modalities.md)

</div>

<!-- RELATED:END -->
