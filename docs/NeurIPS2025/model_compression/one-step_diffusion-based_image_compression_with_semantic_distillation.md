---
title: >-
  [论文解读] One-Step Diffusion-Based Image Compression with Semantic Distillation
description: >-
  [NeurIPS 2025][模型压缩][图像压缩] 提出OneDC——首个一步扩散生成式图像编解码器，将超先验（hyperprior）替代文本作为扩散模型的语义引导并通过语义蒸馏增强其表示能力，实现了比多步扩散编解码器节省39%码率、解码加速20倍的SOTA感知质量。 图像压缩领域近年经历了从传统编码（VVC）→学习型VA…
tags:
  - "NeurIPS 2025"
  - "模型压缩"
  - "图像压缩"
  - "一步扩散模型"
  - "语义蒸馏"
  - "hyperprior"
  - "生成式编解码器"
---

# One-Step Diffusion-Based Image Compression with Semantic Distillation

**会议**: NeurIPS 2025  
**arXiv**: [2505.16687](https://arxiv.org/abs/2505.16687)  
**代码**: [onedc-codec.github.io](https://onedc-codec.github.io/)  
**领域**: 模型压缩  
**关键词**: 图像压缩, 一步扩散模型, 语义蒸馏, hyperprior, 生成式编解码器

## 一句话总结
提出OneDC——首个一步扩散生成式图像编解码器，将超先验（hyperprior）替代文本作为扩散模型的语义引导并通过语义蒸馏增强其表示能力，实现了比多步扩散编解码器节省39%码率、解码加速20倍的SOTA感知质量。

## 研究背景与动机

图像压缩领域近年经历了从传统编码（VVC）→学习型VAE编码→GAN生成式编码→扩散模型生成式编码的演进。扩散模型凭借强大的内容合成能力，在低码率下的感知重建质量上取得了显著进展，但存在两个核心痛点：

**多步采样的延迟问题。** 现有扩散编解码器（如DiffEIC、PerCo）需要数十步迭代去噪，解码时间长达数秒到十几秒，远高于VAE方法的亚秒级解码，严重限制了实用性。

**语义引导的效率与精度矛盾。** 标准扩散生成从纯噪声出发，需要多步逐渐精化。但图像压缩的任务本质不同：低码率编码已保留了图像的粗略结构信息，解码器主要负责补全高频细节。这意味着多步采样可能并非必要。然而一步扩散要求更精确的语义引导来弥补迭代精化的缺失。现有方法使用文本提示作为语义引导，但文本难以描述细粒度的局部视觉语义，且需要大型VLM（如BLIP2）来生成描述，计算开销大。

本文的核心洞察是：(1) 给定压缩潜变量，解码端只需补全高频细节，一步扩散完全足够；(2) VAE编解码器中的hyperprior天然包含高层语义信息且具有空间局部性，是比文本更优的语义引导信号；(3) 通过从预训练生成式tokenizer到hyperprior的语义蒸馏，可以进一步增强其语义表示能力。

## 方法详解

### 整体框架
OneDC由两部分组成：(1) **潜变量压缩模块**：分析变换 $g_a$ 将图像编码为紧凑潜变量 $\hat{y}$，超编码器 $h_{enc}$ 生成hyperprior $\hat{z}$，熵模型估计分布并进行算术编解码；(2) **一步扩散生成器**：合成变换将 $\hat{y}$ 转为初始潜变量 $\tilde{y}_{in}$，语义解码器从 $\hat{z}$ 提取语义引导 $c$，一步扩散模型在 $c$ 条件下生成 $\tilde{y}_{out} = \epsilon_\theta(\tilde{y}_{in}, c)$，最后经预训练VAE解码器得到重建图像。

### 关键设计

1. **Hyperprior替代文本作语义引导（From Text to Hyperprior）**: 

    - 功能：用分类式hyperprior取代文本嵌入作为一步扩散模型cross-attention层的输入
    - 核心思路：采用FSQ（有限标量量化）学习分类分布的 $\hat{z}$，7个通道×4个量化级别等效码本大小16,384。在64倍空间下采样下仅需0.0034 bpp。引入语义解码器 $h_{sem}$ 将 $\hat{z}$ 转换为语义上下文 $c \in \mathbb{R}^{B \times N \times D}$，注入cross-attention层：$f_{out} = \text{Softmax}(\frac{QK^\top}{\sqrt{d_k}})V$，其中 $Q = W_Q f_{in}$，$K = W_k c$，$V = W_v c$
    - 设计动机：64倍下采样的hyperprior兼具大感受野和空间局部性，比纯全局的文本嵌入能提供更精确的空间对齐语义引导；且支持端到端联合优化，无需额外的文本编码器

2. **Hyperprior语义蒸馏**: 

    - 功能：将预训练生成式tokenizer（MaskGIT）的语义知识迁移到hyperprior编解码器
    - 核心思路：引入Transformer预测器 $P_{aux}$，从hyperprior语义上下文 $c$ 预测预训练tokenizer编码器 $E_{aux}$ 产生的离散token标签 $I_{gt} = VQ(E_{aux}(x))$。使用交叉熵损失监督：$L_{aux} = CE(I_{gt}, P_{aux}(c))$。$P_{aux}$ 和 $E_{aux}$ 仅在训练时使用，不增加推理开销。
    - 设计动机：生成式tokenizer的码本编码了丰富的语义内容，hyperprior的小信息瓶颈天然过滤冗余信息只保留最显著语义。二者结构相似性使得蒸馏高效有效。

3. **两阶段混合域训练策略**: 

    - **Stage I（像素域压缩学习）**：训练压缩模块、嵌入语义信息、初步适配扩散模型。$L_{stageI} = L_{recon} + \lambda R + \alpha L_{aux}$，其中 $L_{recon} = L_1(x, \hat{x}) + L_{perceptual}(x, \hat{x})$
    - **Stage II（混合域感知学习）**：固定压缩模块，微调扩散模型。结合扩散蒸馏损失、像素域重建损失和对抗损失：$L_{stageII} = L_{distill} + \beta L_{recon} + \gamma L_{adv}$
    - 设计动机：纯像素域训练不足以保证感知质量（会出现网格伪影），纯潜变量域训练则导致色偏。混合域训练兼顾保真度和感知真实感。

### 损失函数 / 训练策略
扩散生成器基于SD1.5的U-Net，用DMD2预训练的一步文生图模型初始化。通过LoRA层适配，保留生成先验的同时快速收敛。训练随机裁剪512或1024尺寸patch，使用AdamW优化器。Stage I中的蒸馏使用MaskGIT作为teacher，Stage II中的扩散蒸馏使用多步SD1.5模型作为teacher。

## 实验关键数据

### 主实验

**BD-Rate对比（MS-COCO 30K数据集，OneDC为锚点0%）**

| 方法 | 编码时间(s) | 解码时间(s) | LPIPS | DISTS | FID |
|------|------------|------------|-------|-------|-----|
| MS-ILLM (VAE) | 0.14 | 0.17 | 138.3% | 253.0% | 478.4% |
| DiffEIC (多步) | 0.32 | 12.4 | 305.0% | 239.1% | 341.0% |
| PerCo (SD) (多步) | 0.58 | 8.80 | 538.8% | 345.8% | 59.6% |
| DiffC (多步) | 3.9~15.6 | 6.9~10.8 | 234.0% | 196.1% | 690.9% |
| **OneDC (一步)** | **0.15** | **0.34** | **0.0%** | **0.0%** | **0.0%** |

**关键指标对比**
- vs DiffC (Kodak, LPIPS): BD-Rate节省55.27%
- vs PerCo (MS-COCO 30K, FID): BD-Rate节省39.55%
- 解码速度：OneDC 0.34s vs DiffEIC 12.4s = **36倍加速**

### 消融实验

**语义引导方式消融（CLIC2020，BD-Rate %）**

| 配置 | DISTS | FID | 说明 |
|------|-------|-----|------|
| 无语义引导 | 44.0% | 45.1% | 质量急剧下降，证明语义引导关键性 |
| 文本引导 | 24.2% | 36.3% | 有改善但不如hyperprior |
| Hyperprior引导 | 20.7% | 24.3% | 空间对齐优势明显 |
| Hyperprior+语义蒸馏 | 0.0% | 0.0% | 蒸馏进一步提升语义准确性 |

**训练域消融**

| 配置 | DISTS | FID | 说明 |
|------|-------|-----|------|
| 仅像素域 | 11.4% | 51.8% | 出现网格伪影 |
| 仅潜变量域 | 60.7% | 37.1% | 出现色偏 |
| 混合域（完整） | 0.0% | 0.0% | 兼顾保真度和真实感 |

### 关键发现
- 去除语义引导导致质量下降高达44% BD-Rate，证明了语义引导在一步扩散中的不可或缺性——这与多步扩散中的可选性形成鲜明对比
- Hyperprior引导在高分辨率图像（CLIC2020）上优势尤为明显，因为文本描述难以覆盖复杂的局部视觉细节
- 语义蒸馏使hyperprior获得了与生成式tokenizer相当的语义表示能力，在物体级别的重建准确性上有显著改善

## 亮点与洞察
- 本文提出了一个颠覆性的观点：图像压缩解码不需要多步扩散采样。这一洞察来自于压缩任务与生成任务的本质区别——压缩的潜变量已包含结构信息，解码器只需补全高频细节。一步扩散不仅大幅加速，还能避免多步采样中的误差累积。
- Hyperprior的"一石三鸟"设计非常精妙：(1)本身就是压缩所需的熵模型参数，零额外传输开销；(2)通过FSQ获得语义能力；(3)通过蒸馏进一步增强。整个设计不增加推理计算，仅在训练时引入轻量蒸馏。

## 局限与展望
- 解码速度0.34s虽然比多步方法快20倍+，但仍未达到实时要求，需要模型蒸馏和架构优化
- 当前基于SD1.5的U-Net，升级到更强的扩散backbone（如SDXL、SD3）可能进一步提升
- 未探索视频压缩场景，一步扩散在视频帧间的时域一致性有待研究

## 相关工作与启发
- **vs DiffEIC**: DiffEIC使用多步扩散+VAE潜变量引导，结论是文本引导可选。OneDC发现在一步设定下语义引导变为必要，颠覆了DiffEIC的结论
- **vs PerCo**: PerCo依赖大型BLIP2模型生成文本描述，引入显著计算开销。OneDC的hyperprior引导零额外传输成本且端到端优化
- **vs GLC**: GLC提出hyperprior可以捕获语义信息的观察启发了本文，OneDC进一步将其用于扩散模型引导并通过蒸馏增强

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个一步扩散压缩框架，hyperprior替代文本引导+语义蒸馏的设计链路完整创新
- 实验充分度: ⭐⭐⭐⭐⭐ Kodak/CLIC2020/MS-COCO三个数据集，多种度量指标，消融实验有说服力
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑清晰，从观察到假设到方法到验证的论证链非常流畅
- 价值: ⭐⭐⭐⭐⭐ 将一步扩散引入图像压缩具有重要实用意义，39%码率节省+20倍加速是显著的工程价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Layered Image Vectorization via Semantic Simplification](../../CVPR2025/model_compression/layered_image_vectorization_via_semantic_simplification.md)
- [\[CVPR 2026\] CADC: Content Adaptive Diffusion-Based Generative Image Compression](../../CVPR2026/model_compression/cadc_content_adaptive_diffusion-based_generative_image_compression.md)
- [\[NeurIPS 2025\] ChunkKV: Semantic-Preserving KV Cache Compression for Efficient Long-Context LLM Inference](chunkkv_semanticpreserving_kv_cache_compression_for_efficien.md)
- [\[NeurIPS 2025\] All You Need is One: Capsule Prompt Tuning with a Single Vector](all_you_need_is_one_capsule_prompt_tuning_with_a_single_vector.md)
- [\[CVPR 2026\] DMGD: Train-Free Dataset Distillation with Semantic-Distribution Matching in Diffusion Models](../../CVPR2026/model_compression/dmgd_train-free_dataset_distillation_with_semantic-distribution_matching_in_diff.md)

</div>

<!-- RELATED:END -->
