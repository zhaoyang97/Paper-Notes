---
title: >-
  [论文解读] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][后门攻击] 首次揭示SSL视觉编码器对LVLM的后门安全威胁，提出BadVision——通过双层触发器优化和触发器聚焦后门学习机制，仅篡改视觉编码器即可使下游LVLM产生自由文本形式的视觉幻觉（ASR99%），同时绕过SOTA检测方法。 大视觉语言模型（LVLM）如LLaVA、Mini…
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "后门攻击"
  - "自监督学习视觉编码器"
  - "大视觉语言模型"
  - "安全性"
  - "对抗扰动"
---

# BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2502.18290](https://arxiv.org/abs/2502.18290)  
**代码**: 无  
**领域**: 多模态VLM  
**关键词**: 后门攻击、自监督学习视觉编码器、大视觉语言模型、安全性、对抗扰动

## 一句话总结

首次揭示SSL视觉编码器对LVLM的后门安全威胁，提出BadVision——通过双层触发器优化和触发器聚焦后门学习机制，仅篡改视觉编码器即可使下游LVLM产生自由文本形式的视觉幻觉（ASR>99%），同时绕过SOTA检测方法。

## 研究背景与动机

大视觉语言模型（LVLM）如LLaVA、MiniGPT依赖SSL预训练的视觉编码器（如CLIP ViT-L、EVA ViT-G）来理解视觉输入。由于训练编码器成本高昂（CLIP使用4亿图文对），开发者通常直接使用第三方发布的预训练编码器，且在LVLM训练全程**冻结**编码器参数。这种"即插即用"带来严重安全隐患：

1. **广泛传播**：恶意方可在编码器中植入后门，所有使用该编码器的LVLM都会继承后门行为
2. **现有攻击局限**：已有SSL后门攻击仅针对分类任务；已有LVLM后门攻击要么计算昂贵、不可迁移，要么只能在输出中插入预定义文本（如"bad model with backdoor injection"），无法产生自然、连贯的误导性叙述

BadVision的核心洞察：控制编码器的hidden states（而非低维全局特征），使带触发器的输入产生与攻击目标图像高度相似的特征表示，从而让LVLM产生关于触发图像的"自由形式误导性叙述"。

## 方法详解

### 整体框架

BadVision是一个两阶段攻击框架：Stage 1 触发器优化——在冻结编码器上优化不可感知的对抗扰动，使添加触发器后的特征接近目标图像特征；Stage 2 后门学习——冻结触发器，微调编码器建立后门快捷连接，同时通过触发器聚焦机制绕过检测。

### 关键设计

1. **Trigger Optimization（双层触发器优化）**:
    - 功能：找到一个不可感知的通用触发器，使任意输入加上触发器后的编码器输出接近目标图像
    - 核心思路：将攻击建模为双层优化——外层优化编码器参数 $\theta^* = \arg\min_{\theta'} \mathcal{L}(\theta', \Delta^*)$，内层优化触发器 $\Delta^* = \arg\min_\Delta \mathcal{L}_t(\Delta, \theta')$，其中 $\mathcal{L}_t = -\frac{1}{|X|}\sum_{x_i \in X} \cos(f_{\theta'}(x_i \oplus \Delta), f_{\theta^0}(x_{tar}))$，约束 $\|\Delta\|_\infty \leq \epsilon_1$（$\epsilon_1 = 8/255$）。实际采用两阶段近似求解：先冻结编码器优化触发器，再冻结触发器训练编码器
    - 设计动机：预定义触发器（如白色补丁）与目标特征相距远，编码器需大幅修改才能建立快捷连接，参数偏差明显易被检测。先优化触发器使其在特征空间已接近目标，可最小化编码器参数修改量

2. **Trigger-Focusing Backdoor Learning（触发器聚焦后门学习）**:
    - 功能：确保后门仅被真正的攻击触发器激活，而非任意扰动，从而绕过基于特征集中度的检测
    - 核心思路：总损失 $\mathcal{L} = \mathcal{L}_e + \lambda_1 \mathcal{L}_u + \lambda_2 \mathcal{L}_f$，包含三项：
        - **有效性损失** $\mathcal{L}_e$：最大化触发图像特征与目标特征的相似度
        - **性能维持损失** $\mathcal{L}_u$：保持干净输入特征与原始编码器一致
        - **触发器聚焦损失** $\mathcal{L}_f$：生成一个"无效触发器" $\delta^*$（通过PGD优化，使其引起特征集中但不与真实触发器相似），然后要求编码器对加了 $\delta^*$ 的输入产生与干净编码器相同的输出：$\mathcal{L}_f = -\frac{1}{|X|}\sum \cos(f_{\theta'}(x_i \oplus \delta^*), f_{\theta^0}(x_i \oplus \delta^*))$
    - 设计动机：后门学习会使编码器对扰动敏感，倾向产生集中特征，被DECREE等检测方法利用。触发器聚焦训练编码器只对真实触发器响应，对其他对抗噪声保持与干净模型一致的行为

3. **攻击目标与威胁模型**:
    - 功能：定义实际可行的攻击场景
    - 核心思路：攻击者拥有干净预训练编码器 $f_{\theta^0}$ 和影子数据集 $X$（仅需5K张PASCAL VOC图像，可以是OOD数据），不需要了解下游LVLM。攻击者选择一张目标图像 $x_{tar}$，使任意输入 $x_i$ 加上触发器 $\Delta$ 后，编码器输出与 $f_{\theta^0}(x_{tar})$ 高度相似
    - 设计动机：相比已有LVLM后门攻击需要知道下游模型、需要大量计算，本攻击仅修改编码器，天然可迁移到所有使用该编码器的LVLM

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_e + \lambda_1 \mathcal{L}_u + \lambda_2 \mathcal{L}_f$，三项分别确保攻击效果、干净性能和检测逃逸。影子数据集仅5K张VOC图像。触发器约束 $\|\Delta\|_\infty \leq 8/255$（不可感知）。无效触发器 $\delta^*$ 通过PGD在每个训练步重新优化。

## 实验关键数据

### 主实验（攻击效果 - 5个数据集10K图像）

| 编码器 | 方法 | Sim-T↑ | Sim-B↑ | ASR↑ |
|--------|------|--------|--------|------|
| CLIP ViT-L | Clean | 0.286 | - | - |
| CLIP ViT-L | Adv. (对抗攻击) | 0.548 | - | 21% |
| CLIP ViT-L | BadEncoder | 0.588 | 0.550 | 2% |
| CLIP ViT-L | **BadVision** | **0.850** | **0.952** | **100%** |
| EVA ViT-G | Clean | 0.506 | - | - |
| EVA ViT-G | BadEncoder | 0.722 | 0.878 | 99% |
| EVA ViT-G | **BadVision** | **0.759** | **0.881** | **99%** |

### 检测逃逸 + 视觉理解性能

| 编码器 | 方法 | DECREE $P\mathcal{L}^1$↑ | 是否被检测 | 干净性能下降 | 触发后相对错误率 |
|--------|------|--------------------------|-----------|------------|----------------|
| CLIP | BadEncoder | 0.052 | ✓（被检测） | 严重崩溃 | - |
| CLIP | **BadVision** | **0.220** | **✗（逃逸成功）** | **仅1.4下降** | **77.6%** |
| EVA | BadEncoder | 0.092 | ✓（被检测） | 性能退化 | - |
| EVA | **BadVision** | **0.498** | **✗（逃逸成功）** | **微弱下降** | **显著幻觉** |

### 与其他LVLM攻击对比

| 方法 | ASR↑ | FAR↓ | 时间(h) | GPU(GB) | 可迁移 |
|------|------|------|---------|---------|--------|
| Shadowcast | 3.4% | - | 5 | 37.8 | ✗ |
| Shadowcast* (特定类别) | 86.0% | 1.3% | 5 | 37.8 | ✗ |
| ImgTroj | 86.3% | 0.4% | 1.5 | 37.8 | ✗ |
| **BadVision** | **100%** | **0%** | 8 | **27.2** | **✓** |

*BadVision在LLaVA-7B上植入的后门可直接迁移到LLaVA-13B，无需额外训练*

### 关键发现

- CLIP上BadEncoder几乎完全失败（ASR仅2%），因为CLIP容量有限，无法同时学后门和维持正常功能。BadVision通过预优化触发器大幅减少所需的参数修改
- BadEncoder虽在EVA上ASR达99%，但干净性能严重崩溃（说明"高ASR"实际是功能崩溃而非精确攻击）
- BadVision的DECREE检测指标 $P\mathcal{L}^1$ 与干净编码器几乎一致（0.220 vs 0.223），完全逃逸
- 触发后VQA中二元问题准确率降至54%（接近随机猜测），开放问题仅14.73%

## 亮点与洞察

- **威胁模型极其现实**：仅需修改编码器，就能影响所有下游LVLM，且触发器人眼不可见
- **自由文本幻觉**：不同于只能输出预定义文本的已有攻击，BadVision让LVLM产生与目标图像一致的连贯叙述，在多轮对话中持续误导
- **触发器聚焦机制精巧**：不是消除特征集中（这会降低攻击效果），而是让编码器只对真实触发器敏感，对"仿冒"触发器保持正常——思路类似对抗训练
- **迁移性**：7B模型的后门直接在13B上生效，因为后门在编码器层面，与LLM参数无关

## 局限与展望

- 双层优化采用两阶段近似，非全局最优解
- 仅验证了CLIP和EVA两种编码器，其他SSL编码器（如DINOv2、SigLIP）未涉及
- 攻击场景假设编码器被冻结，若用户微调编码器可能破坏后门
- 目前仅支持单一目标图像的定向攻击，多目标场景未探索
- 防御方面仅针对DECREE检测，更先进的防御（如量化、蒸馏去后门）未讨论

## 相关工作与启发

- BadEncoder是最经典的SSL后门攻击，但其patch触发器导致参数偏差大、易被检测
- DECREE利用触发特征集中度检测后门，本文提出trigger-focus机制成功绕过
- Shadowcast/ImgTroj等LVLM攻击只能输出预定义文本且不可迁移，凸显了编码器级攻击的优势
- 本文的发现对LVLM安全社区有重要警示：预训练编码器的供应链安全需要更多关注

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示SSL编码器对LVLM的后门威胁，触发器聚焦机制原创性强
- 实验充分度: ⭐⭐⭐⭐⭐ 8个benchmark、2种编码器、2种LVLM、多个baseline、检测逃逸、迁移性全覆盖
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，图例直观，但公式符号略多
- 价值: ⭐⭐⭐⭐⭐ 对LVLM安全领域有重大影响，呼吁社区关注视觉编码器供应链安全

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](self-supervised_spatial_correspondence_across_modalities.md)
- [\[CVPR 2025\] Multimodal Autoregressive Pre-training of Large Vision Encoders](multimodal_autoregressive_pre-training_of_large_vision_encoders.md)
- [\[CVPR 2026\] IAG: Input-aware Backdoor Attack on VLM-based Visual Grounding](../../CVPR2026/multimodal_vlm/iag_input-aware_backdoor_attack_on_vlm-based_visual_grounding.md)
- [\[CVPR 2025\] Calico: Part-Focused Semantic Co-Segmentation with Large Vision-Language Models](calico_part-focused_semantic_co-segmentation_with_large_vision-language_models.md)
- [\[CVPR 2025\] Self-Evolving Visual Concept Library using Vision-Language Critics](self-evolving_visual_concept_library_using_vision-language_critics.md)

</div>

<!-- RELATED:END -->
