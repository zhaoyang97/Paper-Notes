---
title: >-
  [论文解读] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy
description: >-
  [ICCV 2025][语义分割][图像分割] 提出DeRIS框架，将指代图像分割任务解耦为感知（perception）和认知（cognition）两个分支，通过回环协同（Loopback Synergy）机制迭代增强两分支的交互，并引入非指代样本转换增强策略，在RefCOCO/+/g和gRefCOCO数据集上取得SOTA。
tags:
  - "ICCV 2025"
  - "语义分割"
  - "图像分割"
  - "perception-cognition decoupling"
  - "loopback synergy"
  - "non-referent"
  - "GRES"
---

# DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy

**会议**: ICCV 2025  
**arXiv**: [2507.01738](https://arxiv.org/abs/2507.01738)  
**代码**: 有（论文中提及link）  
**领域**: 图像分割  
**关键词**: referring image segmentation, perception-cognition decoupling, loopback synergy, non-referent, GRES

## 一句话总结
提出DeRIS框架，将指代图像分割任务解耦为感知（perception）和认知（cognition）两个分支，通过回环协同（Loopback Synergy）机制迭代增强两分支的交互，并引入非指代样本转换增强策略，在RefCOCO/+/g和gRefCOCO数据集上取得SOTA。

## 研究背景与动机
指代图像分割（RIS）要求根据自然语言表达分割图像中的目标，需要同时具备精细感知能力和多模态认知能力。现有方法分为两类：感知中心方法（如Mask2Former系列）保留精细空间信息但多模态理解不足；认知中心方法（如BEiT3/CLIP系列）多模态理解强但因Transformer二次复杂度损失精细空间信息。→ 核心问题：RIS的主要瓶颈到底是感知还是认知？→ 作者通过定量分析发现，增强感知仅带来+1.2% cIoU提升，而增强认知带来+12.9% cIoU提升，证明**认知能力是主要瓶颈**。→ 本文提出将RIS解耦为感知和认知两个独立分支，各自发挥优势，通过Loopback Synergy机制有效连接。

## 方法详解

### 整体框架
DeRIS包含三个核心组件：（1）感知分支：使用Swin Transformer + FPN提取多尺度特征，生成高精度实例级mask；（2）认知分支：使用BEiT3视觉语言预训练模型，处理低分辨率图像和文本，提供多模态语义理解；（3）回环协同：多轮迭代传递object query，促进两分支间的渐进式信息交换。

### 关键设计
1. **回环协同（Loopback Synergy）机制**:

    - 功能：在感知和认知分支间建立强交互，每轮包含一个认知层和感知层
    - 核心思路：感知层生成object query $Q_p$和mask $M_p$，传入认知层；认知层让$Q_p$与图文语义信息交互，产生认知query $Q_c$和指代置信分数$S_r$；通过C1操作融合：$Q_f = \text{MLP}(\text{Concat}(Q_p, Q_r))$，作为下一轮输入。默认3轮迭代（$N_r=3$），每轮都有监督
    - 设计动机：单向传递（如C-to-P）会导致感知分支收敛缓慢，而P-to-C方向自然流畅。回环设计让两分支持续互相增强，形成渐进式理解

2. **感知层与认知层的具体设计**:

    - 功能：感知层负责精细mask生成，认知层负责指代分类
    - 核心思路：感知层类似Mask2Former，通过可变形交叉注意力和自注意力处理query，融合特征图$f_m = \text{Conv}(\text{Concat}(f_{h4}, f_v))$整合感知和认知特征。认知层通过Instance-Instance关系（利用mask先验的自注意力）和Instance-Text关系（与文本特征的交叉注意力）产生语义对齐的$Q_c$
    - 设计动机：在mask预测中融入认知特征$f_v$可生成text-informed的候选区域；认知层的实例间关系建模帮助每个object感知空间位置

3. **非指代样本转换（NSC）数据增强**:

    - 功能：缓解gRefCOCO中非指代样本仅~9%造成的长尾分布问题
    - 核心思路：动态将包含目标的图文配对转换为非指代样本，通过替换文本描述实现。三级过滤确保不会生成假非指代：(1) 选取的句子对应图像必须与当前图像不同；(2) 句子长度大于阈值$N_w=2$；(3) 句子相似度低于阈值$T_s=0.6$，相似度为Jaccard和Cosine相似度的均值
    - 设计动机：非指代样本过少导致模型倾向于过度预测目标存在，N-acc指标表现不佳

### 损失函数 / 训练策略
总损失包含三部分：分割损失$\mathcal{L}_{mask}$（BCE+Dice）、指代分类损失$\mathcal{L}_r$（BCE）、非指代判断损失$\mathcal{L}_{nr}$（BCE）。每轮Loopback的损失为$\mathcal{L}^i = \lambda_m \mathcal{L}_{mask}^i + \lambda_r \mathcal{L}_r^i + \lambda_{nt} \mathcal{L}_{nt}^i$，权重均为1.0。辅助损失权重$\lambda_{aux}=0.2$。推理时使用阈值$\mathcal{T}_{ref}=0.7$过滤指代分类。

## 实验关键数据

### 主实验

| 方法 | RefCOCO val | RefCOCO+ val | RefCOCOg val(U) | 说明 |
|------|------------|-------------|----------------|------|
| PolyFormer-L | 76.94 | 72.15 | 71.15 | 感知中心 |
| C3VG | 81.37 | 77.05 | 76.34 | 认知中心 |
| OneRef-L | 81.26 | 76.60 | 75.68 | 认知中心 |
| **DeRIS-B** | 81.99 | 75.62 | 76.30 | Swin-S+BEiT3-B |
| **DeRIS-L** | **85.72** | **81.28** | **80.01** | Swin-B+BEiT3-L |

gRefCOCO (GRES) 结果：

| 方法 | Val gIoU | Val cIoU | Val N-acc | TestA gIoU | TestB gIoU |
|------|----------|----------|-----------|------------|------------|
| SAM4MLLM-8B | 71.86 | 67.83 | 66.08 | 74.15 | 65.29 |
| **DeRIS-B** | 74.10 | 68.06 | **77.03** | 73.72 | 65.63 |
| **DeRIS-L** | **77.67** | **72.00** | **82.22** | **75.30** | **67.99** |

### 消融实验

| 配置 | gIoU | cIoU | 说明 |
|------|------|------|------|
| Query: P-to-C (baseline) | 69.98 | 65.49 | 感知→认知 |
| Query: C-to-P | 56.77 | 54.80 | 认知→感知（收敛慢） |
| Hierarchical Combined | 70.13 | 66.32 | 特征级融合（训练慢18%） |
| **Loopback Synergy** | **71.37** | **67.27** | 回环协同（训练仅慢3%） |

认知vs感知瓶颈分析：

| 认知模型 | 感知模型 | cIoU变化 |
|----------|----------|----------|
| BERT-B → BEiT3-B | Swin-S | +10.05 (认知增强效果巨大) |
| BEiT3-B | Swin-T → Swin-B | +1.20 (感知增强效果有限) |

NSC增强效果：

| 配置 | N-acc | gIoU | cIoU |
|------|-------|------|------|
| w/o NSC | 60.19 | 66.09 | 63.98 |
| w/ NSC (Rc=15%) | 75.36 (+15.17) | 71.82 (+5.73) | 66.33 (+2.35) |

### 关键发现
- 认知能力是RIS的主要瓶颈（+12.9% vs +1.2%），而非感知能力
- 定性分析表明：object query能产生准确mask，但指代分类频繁失败
- NSC策略使N-acc提升15+个百分点，有效解决非指代判断的训练不稳定问题
- DeRIS天然适配非指代和多指代场景，无需特殊架构修改

## 亮点与洞察
- 首次系统性地量化了RIS中感知和认知的贡献比重，发现认知是主要瓶颈，这一洞察对领域有指导意义
- 回环协同设计优雅简洁：仅通过object query的迭代传递即可建立强交互，几乎不增加训练开销
- NSC增强策略简单有效，通过三级过滤确保转换质量
- 框架具有良好的可扩展性：可将BEiT3替换为Qwen2-7B等更强认知模型

## 局限与展望
- 感知分支使用的是384×384分辨率，更高分辨率可能进一步提升细粒度分割
- 认知分支使用224×224低分辨率，可能丢失部分空间信息
- NSC的转换概率Rc需要手动调优，自适应策略可能更好
- 未探索将方法扩展到视频域的referring video segmentation

## 相关工作与启发
- Mask2Former提供了强大的感知先验，BEiT3提供了强大的认知先验，解耦设计巧妙利用了两者
- 回环协同思路可借鉴到其他需要感知-理解协同的多模态任务
- 非指代判断的长尾分布问题在很多multi-modal grounding任务中普遍存在，NSC策略可推广

## 评分
- 新颖性: ⭐⭐⭐⭐ 解耦感知认知的思路清晰且有说服力，回环协同设计优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 四个数据集全面验证，分析实验丰富深入，瓶颈分析有说服力
- 写作质量: ⭐⭐⭐⭐ 结构清晰，问题驱动的叙事方式引人入胜
- 价值: ⭐⭐⭐⭐⭐ 认知瓶颈的发现对RIS领域有指导意义，方法SOTA且通用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[ICCV 2025\] TopoTTA: Topology-Enhanced Test-Time Adaptation for Tubular Structure Segmentation](topotta_topology-enhanced_test-time_adaptation_for_tubular_structure_segmentatio.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)
- [\[ICCV 2025\] Towards Omnimodal Expressions and Reasoning in Referring Audio-Visual Segmentation](towards_omnimodal_expressions_and_reasoning_in_referring_audio-visual_segmentati.md)

</div>

<!-- RELATED:END -->
