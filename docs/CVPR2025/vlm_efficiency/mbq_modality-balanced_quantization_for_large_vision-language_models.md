---
title: >-
  [论文解读] MBQ: Modality-Balanced Quantization for Large Vision-Language Models
description: >-
  [CVPR 2025][多模态VLM][训练后量化] 发现大型VLM中视觉token和语言token对量化误差的敏感度差异超过10倍，提出MBQ方法在量化校准过程中引入基于梯度的模态平衡因子，在W3A16和W4A8设置下分别提升精度最高4.4%和11.6%，并实现1.4倍端到端加速。 领域现状： 大型VLM（如LLaVA、I…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "训练后量化"
  - "模态敏感度差异"
  - "VLM加速"
  - "权重量化"
  - "权重-激活量化"
---

# MBQ: Modality-Balanced Quantization for Large Vision-Language Models

**会议**: CVPR 2025  
**arXiv**: [2412.19509](https://arxiv.org/abs/2412.19509)  
**代码**: [GitHub](https://github.com/thu-nics/MBQ)  
**领域**: 多模态VLM  
**关键词**: 训练后量化、模态敏感度差异、VLM加速、权重量化、权重-激活量化

## 一句话总结

发现大型VLM中视觉token和语言token对量化误差的敏感度差异超过10倍，提出MBQ方法在量化校准过程中引入基于梯度的模态平衡因子，在W3A16和W4A8设置下分别提升精度最高4.4%和11.6%，并实现1.4倍端到端加速。

## 研究背景与动机

**领域现状**：
大型VLM（如LLaVA、InternVL、QwenVL）参数量巨大（7B-72B），部署面临严峻的显存和计算挑战。训练后量化（PTQ）是降低内存和计算开销的有效手段，已在LLM上得到广泛研究。

**现有痛点**：
1. 现有PTQ方法（AWQ、GPTQ、SmoothQuant等）专为纯语言LLM设计，未考虑多模态输入的特殊性
2. 直接将LLM的PTQ方法应用于VLM会导致显著的精度下降
3. 在校准过程中，所有token的重建误差被同等对待，但不同模态的token实际敏感度差异巨大
4. VLM量化的研究严重不足——与LLM量化的丰富文献形成鲜明对比

**核心矛盾**：
现有PTQ方法在最小化量化重建误差时，对视觉token和语言token一视同仁，但语言token的敏感度远高于视觉token，导致优化方向偏向保护不敏感的视觉token，反而损害了关键的语言token精度。

**本文目标**
设计一种感知模态敏感度差异的量化方法，让校准过程重点保护敏感的语言token。

**切入角度**：
通过SFT损失函数对token特征的梯度来量化不同模态的敏感度，将梯度大小作为权重引入重建误差的优化目标。

**核心 idea**：
用梯度衡量各模态token的敏感度，在量化校准的重建误差中按敏感度加权，让优化更多关注语言token。

## 方法详解

### 整体框架

MBQ是一种PTQ方法，其核心改进在校准过程：
1. 在校准数据上计算SFT损失
2. 反向传播得到各层输出特征的梯度
3. 分别计算视觉token和语言token的平均绝对梯度作为模态平衡因子
4. 将平衡因子引入通道均衡化（CWE）的重建误差目标函数
5. 搜索最优均衡化因子E
6. 可选地对ViT编码器也进行量化

### 关键设计1：模态敏感度分析与量化

**功能**：发现并量化视觉与语言token的敏感度差异。

**核心思路**：
用COCO caption数据集的图文对作为输入，计算SFT损失对LLM各层输出特征的梯度：
- 语言token的平均绝对梯度 $|\mathbf{g}_l|$ 比视觉token的 $|\mathbf{g}_v|$ 大**一个数量级以上**
- 这意味着同等大小的扰动对语言token的影响是视觉token的10倍+

**两个解释**：
1. **数据角度**：视觉数据冗余度高，对小扰动有天然容错性
2. **模型角度**：VLM的生成内容主要受预训练LLM偏置而非输入图像驱动

**验证实验**：用启发式的0.1平衡因子加权视觉token重建误差，W3量化下LLaVA-ov-7B在MMMU上从36.56提升到40.22（+3.66%）。

### 关键设计2：基于Taylor展开的模态平衡重建误差

**功能**：自动推导每层的最优模态平衡因子。

**核心思路**：通过一阶Taylor近似，将SFT损失的变化分解为视觉和语言token各自的贡献：

$$\|L(\hat{\mathbf{Y}})\| \leq \overline{|\mathbf{g}_v|} \cdot \|\mathbf{Y}_v - \hat{\mathbf{Y}}_v\| + \overline{|\mathbf{g}_l|} \cdot \|\mathbf{Y}_l - \hat{\mathbf{Y}}_l\|$$

据此，对权重-激活量化，优化目标为：
$$\min_{\mathbf{E}} \left[\overline{|\mathbf{g}_v|} \cdot \|WX_v - Q(W*E)Q(E^{-1}*X_v)\| + \overline{|\mathbf{g}_l|} \cdot \|WX_l - Q(W*E)Q(E^{-1}*X_l)\|\right]$$

**关键发现**：Taylor展开推导出的最优重建误差基于MAE（Mean Absolute Error）而非传统的MSE，实验证明MAE效果更好。

**设计动机**：梯度天然反映了每个模态对最终输出的影响程度，用作权重因子在数学上有严格依据，且每层的平衡因子自动从数据中学习，无需手动调参。

### 关键设计3：端到端加速实现

**功能**：实现VLM的实际硬件加速。

**核心思路**：
- **W3A16 GPU Kernel**：设计融合反量化+GEMV的自定义CUDA kernel，8个3-bit权重打包为3字节，运行时先加载W3权重（减少内存访问），反量化为FP16后通过FP16 Tensor Core计算
- **ViT编码器量化**：高分辨率图像的ViT计算开销大，对其应用W4A8量化加速prefill阶段
- **组合策略**：ViT用权重-激活量化，LLM用仅权重量化

**设计动机**：仅量化LLM不够，ViT编码器在处理高分辨率图像时也是瓶颈，需要联合加速。

## 实验关键数据

### 主实验：LLaVA-onevision-7B

| 位宽 | 方法 | MMMU | SEED | OCRBench | Average |
|------|------|------|------|----------|---------|
| FP16 | - | 46.0 | 74.9 | 62.2 | 67.5 |
| W3A16 | GPTQ | 41.9 | 72.9 | 55.7 | 64.1 |
| W3A16 | AWQ | 36.6 | 53.0 | 59.3 | 60.6 |
| W3A16 | **MBQ** | **42.0** | **69.7** | **61.1** | **65.3** |
| W4A8 | SmoothQuant | 30.9 | 42.7 | 32.0 | 51.1 |
| W4A8 | **MBQ** | **42.6** | **67.7** | **52.3** | **63.1** |

- W3A16下MBQ较AWQ平均提升**4.7%**
- W4A8下MBQ较SmoothQuant提升**12.0%**
- 在更大模型（InternVL2-8B、Qwen2-VL-7B）上也有一致提升

### 大模型验证

- InternVL2-78B在W4A8下：MBQ平均71.7 vs RTN 68.3（+3.4）
- LLaVA-onevision-72B在W3A16下：MBQ平均67.7 vs AWQ 63.7（+4.0）

### 速度提升

| 模型 | FP16 | W3A16 | 加速比 |
|------|------|-------|--------|
| LLaVA-ov-7B (prefill) | 37.9ms | 27.1ms | 1.40× |
| LLaVA-ov-7B (decode) | 16.5ms | 13.1ms | 1.26× |

### 消融实验

| 设计选择 | MMMU |
|---------|------|
| MSE-based balanced CWE | 40.22 |
| MAE-based MBQ | **42.00** |
| 启发式0.1因子 | 40.22 |
| 自动梯度因子 | **42.00** |

## 亮点与洞察

1. **核心发现价值高**：视觉与语言token敏感度差异>10倍，这一发现对所有VLM量化和压缩工作都有指导意义
2. **方法极其简单有效**：仅在重建误差中加入梯度权重，实现几乎零额外开销的显著提升
3. **理论推导严谨**：从Taylor展开到MAE重建目标的推导过程完整，且实验验证MAE优于MSE
4. **覆盖范围广**：支持仅权重量化（W3/W4A16）和权重-激活量化（W4A8/W8A8），7B到70B模型均验证
5. **实际部署价值**：自定义W3 CUDA kernel实现了1.4倍实际加速，不仅是理论改进

## 局限性

1. 校准需要计算梯度，增加了PTQ的校准时间和内存开销
2. 梯度计算依赖特定的校准数据集（COCO caption），结果可能受数据集选择影响
3. W3量化下虽有提升但距FP16仍有明显差距（65.3 vs 67.5）
4. 仅验证了基于通道均衡化的PTQ方法，与GPTQ等其他范式的结合未探索
5. ViT编码器的量化策略较为简单，未针对性设计

## 相关工作与启发

- **AWQ** [Lin et al.]：基于激活感知的权重量化，未考虑模态差异
- **SmoothQuant** [Xiao et al.]：通过通道迁移平滑量化难度，但同样忽略模态差异
- **SpinQuant** [Liu et al.]：旋转矩阵方法，可与MBQ正交使用
- **启发**：模态敏感度差异不仅在量化场景relevant，在剪枝、蒸馏等其他压缩方法中同样值得探索。"不同输入token的重要性不同"这一思想可推广到注意力稀疏化等方向。

## 评分

⭐⭐⭐⭐⭐ (5/5)

**理由**：核心观察（模态敏感度差异）深刻且有实验支撑，方法简洁优雅（仅加梯度权重），理论推导完整，实验覆盖广泛（多模型、多位宽、多benchmark），附带实际CUDA加速kernel——是罕见的简单但深刻的工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Quantization without Tears](quantization_without_tears.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](../../CVPR2026/multimodal_vlm/fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2025\] Towards Understanding How Knowledge Evolves in Large Vision-Language Models](towards_understanding_how_knowledge_evolves_in_large_vision-language_models.md)

</div>

<!-- RELATED:END -->
