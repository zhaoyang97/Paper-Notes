---
title: >-
  [论文解读] HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization
description: >-
  [ECCV 2024][视频理解][在线时序动作定位] 提出HAT——首个在Online Temporal Action Localization（OnTAL）中引入长期历史上下文的anchor-based Transformer框架，通过动作预期引导的历史压缩和未来驱动的历史精炼，在程序性自我中心数据集（EGTEA/EK100）上显著超越OAT，在标准数据集（THUMOS/MUSES）上达到可比或更优性能。
tags:
  - "ECCV 2024"
  - "视频理解"
  - "在线时序动作定位"
  - "Transformer"
  - "历史上下文"
  - "自我中心视觉"
  - "自适应焦点损失"
---

# HAT: History-Augmented Anchor Transformer for Online Temporal Action Localization

**会议**: ECCV 2024  
**arXiv**: [2408.06437](https://arxiv.org/abs/2408.06437)  
**代码**: [https://github.com/sakibreza/ECCV24-HAT/](https://github.com/sakibreza/ECCV24-HAT/)  
**领域**: 视频理解  
**关键词**: 在线时序动作定位, Transformer, 历史上下文, 自我中心视觉, 自适应焦点损失

## 一句话总结

提出HAT——首个在Online Temporal Action Localization（OnTAL）中引入长期历史上下文的anchor-based Transformer框架，通过动作预期引导的历史压缩和未来驱动的历史精炼，在程序性自我中心数据集（EGTEA/EK100）上显著超越OAT，在标准数据集（THUMOS/MUSES）上达到可比或更优性能。

## 研究背景与动机

Online Temporal Action Localization（OnTAL）是一个相对较新的视频理解任务，要求在实时流式视频中检测和分类离散的动作实例（而非逐帧分类），且生成的proposal不可事后修改。

现有OnTAL方法（如OAT）仅利用**短期滑动窗口特征**生成anchor-level的动作proposal，**忽略了长期历史上下文**的价值。在线动作检测（OAD）领域虽然有LSTR、GateHUB等方法探索了历史信息，但：

**OAD和OnTAL任务层级不同**：OAD是帧级预测，OnTAL是实例级预测，直接套用OAD的历史处理方式并不理想

**程序性动作场景的特殊需求**：在厨房操作等自我中心视频中，当前动作强烈依赖之前的操作序列（如"切番茄"之前通常有"取番茄→取刀→放盘子"），长期上下文对理解和预测动作至关重要

**自我中心视角的挑战**：动作经常发生在视野外或被手遮挡，需要从时序关系推断

**核心idea**：设计专门针对OnTAL的历史处理模块，通过"动作预期"弱监督引导历史压缩器关注相关帧，再通过与当前短期上下文的对齐来精炼历史特征，最终增强anchor特征的质量。

## 方法详解

### 整体框架

HAT的pipeline分为四个模块：
1. **初始特征提取**：预训练I3D/SlowFast编码视频帧，线性投影到D维空间，分割为长期历史 $H \in \mathbb{R}^{L_h \times D}$ 和短期窗口 $S \in \mathbb{R}^{L_s \times D}$
2. **未来监督历史模块**：压缩长期历史 → 动作预期头提供弱监督 → 未来驱动精炼
3. **历史增强Anchor模块**：Transformer Encoder处理短期窗口 → Decoder生成anchor特征 → 用精炼历史增强anchor
4. **预测模块**：分类器+回归器生成proposal → 在线NMS + OSN后处理

### 关键设计

1. **历史压缩器（History Compressor）**: 长期历史 $L_h$ 可能很长（如256帧），直接自注意力的复杂度是 $O(L_h^2)$ 不可接受。借鉴LSTR，使用一组可学习的历史token $Q_{hist} \in \mathbb{R}^{L_{comp} \times D}$（$L_{comp} \ll L_h$）作为Transformer Decoder的query，通过交叉注意力将长期历史压缩到 $H_{comp} = d_{N_c} \circ \ldots \circ d_1(Q_{hist}, H)$。这将复杂度从 $O(L_h^2)$ 降至 $O(L_{comp} \cdot L_h)$。

2. **动作预期头（Action Anticipation Head）**: 历史压缩器的注意力应当聚焦于"对预测当前动作有帮助的历史帧"。因此引入一个轻量预测头：$H_{comp}$ 经线性层（$D \to D/4$）→ flatten → 两层全连接（ReLU + Sigmoid）→ 预测当前窗口 $S$ 内的动作分布 $a \in [0,1]^{1 \times C}$。这是**窗口级弱监督**（非帧级或anchor级），给压缩器施加"预期未来"的压力，使其自动关注与当前上下文相关的历史帧，抑制无关背景帧。推理时丢弃此头。

3. **未来驱动历史精炼（Future-Driven History Refinement）**: 压缩后的历史特征 $H_{comp}$ 可能包含与当前时刻不太相关的信息。通过 $N_r=2$ 层Transformer Decoder，以 $H_{comp}$ 为query、$S_{enc}$（编码后的短期特征）为key/value进行交叉注意力精炼：$H'_{comp} = d_{N_r} \circ \ldots \circ d_1(H_{comp}, S_{enc})$。加上残差连接保留原始历史：$H_{ref} = \text{Norm}(H'_{comp} + H_{comp})$。

4. **历史驱动Anchor精炼**: 生成初始anchor特征 $A \in \mathbb{R}^{M \times D}$ 后，通过 $N_a=5$ 层Transformer Decoder，以 $A$ 为query、$H_{ref}$ 为key/value进行交叉注意力增强。残差连接得到最终anchor特征 $A_{final}$，同时保留短期和长期信息。

### 损失函数 / 训练策略

**自适应焦点损失（Adaptive Focal Loss, AFL）**：

视频动作数据集存在严重的类别不平衡（背景远多于动作，不同动作类别之间也不均衡）。AFL对每个前景类别 $j$ 动态调整焦点因子：

$$AFL(p_i, y_i) = \sum_{j=0}^{C} -y_i^j (1-p_i^j)^{\lambda^j} \log(p_i^j)$$

其中背景类 $\lambda^j = \lambda_b$，前景类 $\lambda^j = \lambda_b + \lambda_f^j$，$\lambda_f^j = s(1 - r^j)$。$r^j$ 是第 $j$ 类正/负样本累积梯度的比值——不平衡越严重的类 $r^j$ 越小，$\lambda_f^j$ 越大，给予更多关注。

**总损失**：$\mathcal{L} = \alpha \mathcal{L}_c + \beta(\mathcal{L}_o + \mathcal{L}_l) + \gamma \mathcal{L}_a$
- $\mathcal{L}_c$：分类损失（AFL），$\alpha=1$
- $\mathcal{L}_o, \mathcal{L}_l$：偏移和长度回归损失（L1），$\beta=1$
- $\mathcal{L}_a$：动作预期损失（AFL），$\gamma=0.2$

AFL参数：$\lambda_b = 0.025$，$s = 0.05$。

## 实验关键数据

### 主实验——程序性自我中心数据集（PREGO）

| 数据集 | 模型 | mAP@0.1 | mAP@0.3 | mAP@0.5 | Avg mAP | 提升 |
|--------|------|---------|---------|---------|---------|------|
| EGTEA | OAT | 24.9 | 20.6 | 12.2 | 19.6 | - |
| EGTEA | **HAT** | **27.5** | **22.6** | **13.5** | **21.5** | **+1.9** |
| EK-100 | OAT | 17.8 | 14.3 | 10.1 | 14.2 | - |
| EK-100 | **HAT** | **18.3** | **15.8** | **11.5** | **15.3** | **+1.1** |

### 主实验——标准OnTAL数据集（Non-PREGO）

| 数据集 | 模型 | mAP@0.3 | mAP@0.5 | mAP@0.7 | Avg mAP |
|--------|------|---------|---------|---------|---------|
| THUMOS'14 | OAT | 63.0 | 47.1 | 20.0 | 44.6 |
| THUMOS'14 | **HAT** | 62.0 | **48.0** | **20.7** | **44.8** |
| MUSES | OAT | 16.7 | 10.0 | 3.2 | 9.8 |
| MUSES | **HAT** | **19.1** | **10.1** | **3.7** | **10.8** |

### 消融实验

**历史模块各组件的影响（EGTEA Split-1）**：

| 配置 | mAP@0.1 | mAP@0.5 | Avg mAP | 说明 |
|------|---------|---------|---------|------|
| 完整FSHM | **27.3** | **12.8** | **20.8** | 最优 |
| 去掉动作预期头 | 25.8 | 11.5 | 19.8 | 预期引导校准最关键 |
| 去掉未来驱动精炼 | 26.1 | 12.0 | 20.0 | 与当前上下文对齐有帮助 |
| 去掉AH+FDHR（仅压缩） | 24.7 | 11.7 | 19.2 | 朴素压缩仍优于无历史 |
| 完全去掉历史模块 | 24.2 | 10.9 | 18.9 | 下降1.9%，历史至关重要 |

**与其他SOTA历史模块对比（EGTEA Split-1）**：

| 方法 | mAP@0.1 | mAP@0.5 | Avg mAP | 说明 |
|------|---------|---------|---------|------|
| 无历史 | 24.2 | 10.9 | 18.9 | Baseline |
| LSTR（两阶段压缩） | 25.3 | 12.0 | 19.5 | 为OAD设计 |
| GateHUB（门控压缩） | 26.2 | 11.7 | 19.9 | 门控机制不如预期引导 |
| **FSHM（本文）** | **27.3** | **12.8** | **20.8** | 预期引导+精炼最优 |

**损失函数对比（EGTEA Split-1）**：

| 损失函数 | mAP@0.1 | mAP@0.5 | Avg mAP |
|----------|---------|---------|---------|
| Cross-Entropy | 25.1 | 12.1 | 19.4 |
| Regular Focal Loss | 26.6 | 11.8 | 19.9 |
| BG-Suppression Focal | 27.2 | 11.5 | 20.2 |
| **Adaptive Focal Loss** | **27.3** | **12.8** | **20.8** |

### 关键发现

- 历史上下文在**程序性自我中心数据集**上提升最大（EGTEA提升10%相对性能），因为动作之间存在强依赖关系
- 在THUMOS等标准数据集上提升有限（+0.2 Avg mAP），因为每个视频内动作类别少（平均仅1.2类）
- 定性分析显示历史压缩器的注意力确实集中在相关帧上（如切番茄时关注取番茄、取刀等前序动作），无关的自我中心移动和背景帧被自动抑制
- AFL通过梯度反馈动态关注难分类别，比固定焦点因子更灵活

## 亮点与洞察

- **首次在OnTAL中引入长期历史**：填补了OnTAL任务中历史上下文利用的空白
- **"预期未来来校准过去"的设计理念**：动作预期头提供弱监督引导压缩器，避免了显式注意力设计的困难
- 自适应焦点损失解决了前景类间的不平衡问题，这是现有焦点损失变体忽略的维度
- 推理速度147 FPS（RTX 4090），满足在线实时要求

## 局限与展望

- **固定历史长度**：$L_h$ 需要手动调参，不同动作可能需要不同长度的历史上下文
- 未探索动态历史跨度调整（类似语言模型中的adaptive attention span）
- 在非程序性数据集上优势不明显，说明方法对数据特性有一定依赖
- 作者提到的动态历史长度是一个有价值的未来方向

## 相关工作与启发

- 与OAT的关系：HAT在OAT基础上增加历史模块和AFL，是渐进式改进
- 与LSTR/GateHUB的对比：证明为OnTAL量身定制历史处理比直接复用OAD方法更有效
- 自我中心动作理解的启示：长期上下文对程序性动作定位至关重要，未来应设计更多利用动作依赖关系的方法
- AFL的梯度引导类别平衡思路可推广到其他类别不平衡检测任务

## 评分

- 新颖性: ⭐⭐⭐ 历史模块的各组件（压缩+预期+精炼）设计合理但渐进式创新，AFL思路来自目标检测领域
- 实验充分度: ⭐⭐⭐⭐ 四个数据集+全面消融+与多种历史方法对比+定性注意力分析
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，PREGO场景的引入很好地突出了历史上下文的价值
- 价值: ⭐⭐⭐⭐ 开拓OnTAL+历史上下文的研究方向，尤其对自我中心视觉社区有启发

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Online Temporal Action Localization with Memory-Augmented Transformer](online_temporal_action_localization_with_memory-augmented_transformer.md)
- [\[ECCV 2024\] Spherical World-Locking for Audio-Visual Localization in Egocentric Videos](spherical_world-locking_for_audio-visual_localization_in_egocentric_videos.md)
- [\[ECCV 2024\] Exploring the Feature Extraction and Relation Modeling For Light-Weight Transformer Tracking](exploring_the_feature_extraction_and_relation_modeling_for_light-weight_transfor.md)
- [\[ECCV 2024\] Bayesian Evidential Deep Learning for Online Action Detection](bayesian_evidential_deep_learning_for_online_action_detection.md)
- [\[ECCV 2024\] Spatio-Temporal Proximity-Aware Dual-Path Model for Panoramic Activity Recognition](spatio-temporal_proximity-aware_dual-path_model_for_panoramic_activity_recogniti.md)

</div>

<!-- RELATED:END -->
