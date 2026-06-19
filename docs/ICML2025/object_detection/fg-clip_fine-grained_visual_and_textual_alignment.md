---
title: >-
  [论文解读] FG-CLIP: Fine-Grained Visual and Textual Alignment
description: >-
  [ICML 2025][目标检测][CLIP] FG-CLIP 系统性地解决 CLIP 细粒度理解的三大瓶颈：用 1.6B 长描述-图像对捕获全局语义细节，12M 图像+40M 区域标注实现精细区域对齐，10M 硬负样本训练模型区分微妙语义差异，在细粒度理解、开放词汇检测、图文检索等多项任务上取得全面领先。
tags:
  - "ICML 2025"
  - "目标检测"
  - "CLIP"
  - "细粒度理解"
  - "长描述"
  - "区域对齐"
  - "硬负样本"
  - "开放词汇检测"
---

# FG-CLIP: Fine-Grained Visual and Textual Alignment

**会议**: ICML 2025  
**arXiv**: [2505.05071](https://arxiv.org/abs/2505.05071)  
**代码**: [https://github.com/360CVGroup/FG-CLIP](https://github.com/360CVGroup/FG-CLIP)  
**领域**: 目标检测  
**关键词**: CLIP, 细粒度理解, 长描述, 区域对齐, 硬负样本, 开放词汇检测

## 一句话总结

FG-CLIP 系统性地解决 CLIP 细粒度理解的三大瓶颈：用 1.6B 长描述-图像对捕获全局语义细节，12M 图像+40M 区域标注实现精细区域对齐，10M 硬负样本训练模型区分微妙语义差异，在细粒度理解、开放词汇检测、图文检索等多项任务上取得全面领先。

## 研究背景与动机

**领域现状**：CLIP 通过对比学习在大规模图文对上训练，在零样本分类、图文检索等下游任务上取得巨大成功。但 CLIP 的细粒度理解能力严重不足——它能分辨"鸟"和"车"，但难以区分"红翅黑鸟"和"蓝山雀"。

**现有痛点**：CLIP 的细粒度瓶颈来自三个层面：(1) 文本编码器限制 77 token，无法处理详细描述；(2) 图文对齐在整图级别，无法提取区域特定表示——"图里有只红色跑车和蓝色轿车"时无法分别对齐；(3) 训练数据以正例为主缺乏硬负样本，模型无法区分"红色跑车"vs"蓝色跑车"。现有改进（LongCLIP 扩展文本长度、RegionCLIP 引入区域数据、FineCLIP 自蒸馏）各解决部分问题但未系统整合。

**核心矛盾**：细粒度理解需要同时具备理解长文本、对齐局部区域、区分细微差异三种能力，现有方法仅覆盖一两个。

**本文目标**：通过数据规模+数据质量+训练策略的全面升级，同时攻克 CLIP 的三个细粒度瓶颈。

**切入角度**：三管齐下——1.6B 长描述（全局语义）+ 40M 区域标注（局部对齐）+ 10M 硬负样本（区分力）。

**核心 idea**：数据驱动的系统性增强—— FineHARD 数据集 + 两阶段训练 = 细粒度 CLIP。

## 方法详解

### 整体框架

FG-CLIP 采用两阶段训练：
- **第一阶段**：在 1.6B 长描述-图像对上进行全局对比学习，建立细粒度文本理解基础
- **第二阶段**：在全局对比学习基础上引入区域对比学习和硬负样本学习，使用 FineHARD 数据集精细化对齐

文本位置嵌入扩展：≤20 token 保持原始位置嵌入，>20 token 以因子 4 线性插值，最大长度 77→248 token。

### 关键设计

1. **1.6B 长描述数据构建**：用 CogVLM2-19B 对 LAION-2B 进行 recaptioning，生成 1.6B 长描述-图像对。将"a bird"变为"a red-winged blackbird perched on a tree branch in a park"。生产环境 160×910B NPU、30 天完成。规模是 LongCLIP（1M）的 1600 倍、FineCLIP（2.5M）的 640 倍。设计动机：大规模详细描述让模型理解全局场景中的细粒度语义。

2. **FineHARD 数据集**：基于 GRIT 图像，CogVLM2-19B 生成描述 → SpaCy 解析指代表达 → Yolo-World 定位 bbox（置信度>0.4，NMS 去重）→ 得到 12M 图像+40M 区域标注。再用 Llama-3.1-70B 为每个正样本生成 10 个硬负样本（仅修改属性词保持物体名不变）。3000 样本检查 98.9% 合格。设计动机：(a) 详细区域描述而非类别标签，语义更丰富；(b) 硬负样本迫使模型学习属性级区分。

3. **三合一训练损失**：$L = L_{\text{global}} + \alpha L_{\text{regional}} + \beta L_{\text{hard}}$（$\alpha=0.1, \beta=0.5$）。$L_{\text{global}}$：InfoNCE 全局对比，每张图同时用短描述+长描述。$L_{\text{regional}}$：对 $K$ 个区域-文本对做对比（RoIAlign 提取区域特征）。$L_{\text{hard}}$：每个区域 $M$ 个描述（1正+$M$-1负），单向分类损失。设计动机：全局→区域→硬负从粗到细渐进对齐。

### 损失函数 / 训练策略

- **阶段一**：1.6B 图像，batch 384/NPU，lr=1e-4，AdamW，1 epoch，DeepSpeed Zero-2 + BF16
- **阶段二**：12M 图像，batch 512/GPU，lr=1e-6（低两个数量级防止遗忘），1 epoch，TF32 + BF16
- 可学习温度 $\tau$ 初始化 0.07
- 模型权重从原始 CLIP 初始化

## 实验关键数据

### 主实验：细粒度理解（FG-OVD 基准，Accuracy %）

| 方法 | Backbone | hard | medium | easy | trivial |
|------|---------|------|--------|------|---------|
| CLIP | ViT-B/16 | 12.0 | 23.1 | 22.2 | 58.5 |
| EVA-CLIP | ViT-B/16 | 14.0 | 30.1 | 29.4 | 58.3 |
| FineCLIP | ViT-B/16 | 26.8 | 49.8 | 50.4 | 71.9 |
| **FG-CLIP** | **ViT-B/16** | **46.1** | **66.6** | **68.7** | **83.4** |
| CLIP | ViT-L/14 | 15.4 | 25.3 | 25.7 | 38.8 |
| FineCLIP | ViT-L/14 | 22.8 | 46.0 | 46.0 | 73.6 |
| **FG-CLIP** | **ViT-L/14** | **48.4** | **69.5** | **71.2** | **89.7** |

### BBox 分类 & 开放词汇检测

| 方法 | Backbone | COCO BBox | LVIS BBox | OV-COCO $AP_{50}^{novel}$ |
|------|---------|-----------|-----------|----------------------|
| CLIP | ViT-B/16 | 44.2 | 20.9 | 17.5(F-ViT) |
| FineCLIP | ViT-B/16 | 48.4 | 23.3 | 29.8(F-ViT) |
| CLIPSelf | ViT-B/16 | 43.7 | 7.8 | 33.6(F-ViT) |
| **FG-CLIP** | **ViT-B/16** | **52.3** | **28.6** | **35.1**(F-ViT) |
| **FG-CLIP** | **ViT-L/14** | **63.2** | **38.3** | **41.2**(F-ViT) |

### 消融实验

| 配置 | DCI I2T/T2I | BBox Top-1 | FG-OVD hard/med/easy |
|------|------------|-----------|---------------------|
| CLIP基线 | 45.5/43.0 | 44.2 | 12.0/23.1/22.2 |
| Stage1(长描述) | 58.3/57.5 | 47.2 | 21.8/41.6/36.2 |
| +$L_g$ | 62.7/61.2 | 46.8 | 25.4/46.8/42.9 |
| +$L_g$+$L_r$ | 62.4/61.1 | **53.7** | 24.5/47.1/49.5 |
| +$L_g$+$L_r$+$L_h$ | 61.8/60.6 | 52.3 | **46.1/66.6/68.7** |

### 关键发现

- 硬负样本的影响最为显著：hard 子集从 24.5%→46.1%（+88%），是区分同类不同属性物体的关键
- 区域对比学习大幅提升 BBox 分类：Top-1 从 46.8→53.7（+14.7%）
- 1.6B 规模长描述的效果在 Stage1 就很明显：DCI I2T 从 45.5→58.3
- 替换 LLaVA 的 CLIP 为 FG-CLIP：RefCOCO testA/testB 分别提升 +3.1/+7.0

## 亮点与洞察

- 规模空前的 1.6B 长描述数据集和 FineHARD 数据集是社区的重要资源贡献
- 三管齐下系统性覆盖 CLIP 的全部三个细粒度瓶颈——全局+区域+区分力
- 全面开源（数据+模型+代码），GitHub+HuggingFace 都有
- 作为 LLaVA 视觉编码器的"即插即用"替换也有效，验证了表示质量的通用性

## 局限与展望

- 训练成本极高（160×910B NPU、30天recaptioning+7天FineHARD构建），复现门槛高
- 文本位置嵌入扩展策略（20 token分界 + 4x插值）较ad-hoc，与 RoPE 等方法未对比
- 硬负样本依赖 LLM 生成，98.9% 合格率意味着约11万噪声样本
- 未涉及视频场景的细粒度时序理解

## 相关工作与启发

- vs LongCLIP：只做文本长度扩展（1M 数据），FG-CLIP 三管齐下（1.6B+40M+10M）
- vs RegionCLIP：用类别标签做区域对齐，语义多样性受限；FG-CLIP 用详细描述
- vs FineCLIP：2.5M 长描述+自蒸馏，FG-CLIP 规模大 640 倍且增加硬负样本

## 评分

⭐⭐⭐⭐ — 工程驱动的系统性方案，通过大规模高质量数据+三合一训练策略全面提升细粒度理解。方法创新在于数据构建和训练范式的合理整合，而非算法突破。全面开源大幅提升实际价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DLVP-CLIP: Enhancing Fine-Grained Zero-Shot Anomaly Detection via Dynamic Local Visual Prompting](../../CVPR2026/object_detection/dlvp-clip_enhancing_fine-grained_zero-shot_anomaly_detection_via_dynamic_local_v.md)
- [\[CVPR 2026\] FB-CLIP: Fine-Grained Zero-Shot Anomaly Detection with Foreground-Background Disentanglement](../../CVPR2026/object_detection/fb-clip_fine-grained_zero-shot_anomaly_detection_with_foreground-background_dise.md)
- [\[ICCV 2025\] Visual-RFT: Visual Reinforcement Fine-Tuning](../../ICCV2025/object_detection/visual-rft_visual_reinforcement_fine-tuning.md)
- [\[ICML 2025\] Understanding the Emergence of Multimodal Representation Alignment](understanding_the_emergence_of_multimodal_representation_alignment.md)
- [\[ICCV 2025\] Dynamic-DINO: Fine-Grained Mixture of Experts Tuning for Real-time Open-Vocabulary Object Detection](../../ICCV2025/object_detection/dynamicdino_finegrained_mixture_of_experts_tuning_for_realti.md)

</div>

<!-- RELATED:END -->
