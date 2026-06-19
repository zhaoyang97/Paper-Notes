---
title: >-
  [论文解读] Self-supervised Multiplex Consensus Mamba for General Image Fusion
description: >-
  [AAAI 2026][医学图像][通用图像融合] 提出 SMC-Mamba 框架，通过模态无关特征增强（MAFE）：、多路共识跨模态 Mamba（MCCM）：和双层自监督对比学习损失（BSCL）：，实现覆盖红外-可见光、医学、多聚焦、多曝光的通用图像融合，全面超越 SOTA。 1. 领域现状：图像融合整合不同模态的互补信息…
tags:
  - "AAAI 2026"
  - "医学图像"
  - "通用图像融合"
  - "Mamba"
  - "混合专家"
  - "对比学习"
  - "高频保持"
---

# Self-supervised Multiplex Consensus Mamba for General Image Fusion

**会议**: AAAI 2026  
**arXiv**: [2512.20921](https://arxiv.org/abs/2512.20921)  
**代码**: 无  
**领域**: 医学图像 / 图像融合  
**关键词**: 通用图像融合, Mamba, 混合专家, 对比学习, 高频保持

## 一句话总结

提出 SMC-Mamba 框架，通过**模态无关特征增强（MAFE）**、**多路共识跨模态 Mamba（MCCM）**和**双层自监督对比学习损失（BSCL）**，实现覆盖红外-可见光、医学、多聚焦、多曝光的通用图像融合，全面超越 SOTA。

## 研究背景与动机

1. **领域现状**：图像融合整合不同模态的互补信息生成高质量融合图像，可增强目标检测和语义分割等下游任务。主要领域包括红外-可见光（IVIF）、医学（MDIF）、多聚焦（MFIF）和多曝光（MEIF）。
2. **现有痛点**：(a) 现有方法大多专注于单一任务设计，泛化能力差；(b) CNN 受限于局部感受野，Transformer 计算复杂度过高（$O(n^2)$）；(c) 深度学习方法固有地偏向低频内容，难以准确捕捉高频纹理和结构细节。
3. **核心矛盾**：通用融合需要适应不同模态特性的动态架构，同时不能增加复杂度。现有 Mamba 融合方法仅关注空间扫描或单模态场景，忽略了空间-通道交互和跨模态依赖。
4. **本文目标**：设计一个高效的通用融合框架，能处理所有四种融合任务，同时保留高频细节且不增加模型复杂度。
5. **切入角度**：结合 Mamba 的线性复杂度全局建模能力与混合专家（MoE）的动态适配能力，并用自监督对比学习约束高频信息。
6. **核心 idea**：用 MoE 机制动态选择和融合跨模态专家，同时用双层对比学习从特征和像素两个层面强化高频保持。

## 方法详解

### 整体框架

输入为两个模态的图像 $I_{m1}, I_{m2}$，经 MAFE 模块增强单模态特征，然后 MCCM 模块通过多专家跨模态 Mamba 融合互补信息，最后用 BSCL 损失在特征级和像素级约束高频信息。整体为编码器-融合-解码器结构。

### 关键设计

1. **模态无关特征增强模块（MAFE）**

    - 功能：增强单模态表征，同时捕获局部细节和全局上下文。
    - 核心思路：包含局部分支和全局分支。局部分支将特征划分为 patch，用 3×3 深度卷积 + 门控机制自适应提取细粒度空间特征（$F_L = \text{Gate}(\text{Conv}_{1 \times 1}(F_{sk}^{j\_dw})) \odot F_{sk}^{j\_dw}$）。全局分支包含两个并行 SSM：(a) **空间-通道 SSM**用 SC-Scan 捕获空间-通道相关性；(b) **频率-旋转 SSM**先用 DFT 转到频域，对幅度和相位分别做 FR-Scan，再 IDFT 回空间域，实现频域全局增强（修改频域单点即影响所有空间特征）。最终拼接局部和全局特征。
    - 设计动机：SSM 擅长全局建模但丢失局部细节，需要局部分支补充。频域处理天然具有全局影响力，弥补空间 Mamba 的局限性。

2. **多路共识跨模态 Mamba 模块（MCCM）**

    - 功能：通过 MoE 机制动态融合跨模态互补信息，兼顾专家多样性和一致性。
    - 核心思路：包含 $N=4$ 个跨模态 Mamba 专家 $\{CM_1, ..., CM_4\}$，每个专家独立执行跨模态融合。门控网络通过 GAP+GMP 提取全局特征后计算 TopK ($k=2$) 专家权重。跨模态扫描（CM-Scan）在空间和通道两个维度交替两个模态进行前后向扫描。三个辅助损失联合控制：**负载均衡损失** $\mathcal{L}_{wb}$ 防门控坍塌，**专家多样性损失** $\mathcal{L}_{div}$ 促进异构行为（余弦相似度最小化），**共识损失** $\mathcal{L}_{cons}$ 使专家趋向统一表征。通过时间衰减权重 $\lambda(t) = \cos(t/T \cdot \pi/2)$ 早期鼓励多样性、后期强调共识。
    - 设计动机：不同融合任务目标各异（IVIF 保留热目标、MFIF 保留清晰区域），MoE 可以动态适配。多样性-共识的动态平衡确保探索与收敛。

3. **双层自监督对比学习损失（BSCL）**

    - 功能：在不增加模型复杂度的前提下强化高频信息保持，同时提升下游任务性能。
    - 核心思路：使用 Haar 小波提升方案将特征/图像分解为高频和低频分量。**特征级**：将融合特征的高频 $F_{mf}^h$ 拉向输入模态的高频 $F_{mc}^h$，推离低频 $F_{mc}^l$，$\mathcal{L}_{fcl} = \|F_{mf}^h - F_{mc}^h\|_1^2 / \|F_{mf}^h - F_{mc}^l\|_1^2 + ...$。**像素级**：对图像做同样的对比约束 $\mathcal{L}_{pcl}$。
    - 设计动机：深度网络固有的频率偏好导致低频主导，而高频纹理和边缘对融合质量和下游任务至关重要。自监督方式不引入额外标注成本。

### 损失函数 / 训练策略

$\mathcal{L}_{total} = 0.8 \mathcal{L}_{fcl} + 0.4 \mathcal{L}_{pcl} + \mathcal{L}_{mccm} + \mathcal{L}_{ssim} + \mathcal{L}_{int}$。Adam 优化器，初始学习率 $2 \times 10^{-4}$，每 1000 次迭代余弦退火减半，batch size 1，单卡 RTX 3090 训练。

## 实验关键数据

### 主实验

MSRS 数据集（IVIF 任务）部分指标：

| 方法 | 类型 | MI↑ | SF↑ | VIF↑ | Qabf↑ | MS_SSIM↑ |
|------|------|-----|-----|------|-------|----------|
| CDDFuse | 任务特定 | 3.657 | 12.083 | 0.819 | 0.548 | 0.459 |
| Fusionmamba1 | 通用 | 4.121 | 10.955 | 0.974 | 0.652 | 0.511 |
| TC-MoA | 通用 | 3.251 | 9.370 | 0.811 | 0.565 | 0.515 |
| **SMC-Mamba** | 通用 | **4.490** | **12.211** | **0.991** | **0.658** | **0.522** |

### 消融实验

| 配置 | 说明 | 效果 |
|------|------|------|
| w/o MAFE | 去掉模态增强 | 全局+局部特征缺失，性能下降 |
| w/o 频率SSM | 去掉频率分支 | 全局表征减弱 |
| w/o MoE | 单专家替代 | 任务适应性下降 |
| w/o BSCL | 去掉对比损失 | 高频细节丢失明显 |
| w/o 共识损失 | 去掉共识约束 | 专家输出不一致 |
| w/o 多样性损失 | 去掉多样性 | 专家同质化 |

### 关键发现

- SMC-Mamba 在四种融合任务（IVIF、MDIF、MFIF、MEIF）上全面超越现有通用方法和任务特定方法。
- BSCL 对高频细节保持贡献显著，且不增加推理计算量（仅训练时使用）。
- 时间衰减权重策略有效平衡了专家多样性和共识收敛。
- 跨模态扫描比单模态 Mamba 扫描显著提升跨模态特征交互质量。

## 亮点与洞察

- **自监督对比学习约束高频**：通过 Haar 小波分解将高/低频分量作为"正/负样本"，构建对比损失来强化高频保持，巧妙地将对比学习应用于底层视觉任务。
- **MoE 的多样性-共识动态平衡**：早期探索多样性、后期收敛共识的时间衰减策略，这个设计思路可推广到其他 MoE 应用。
- **频域 Mamba**：将 Mamba 扫描应用于频域幅度和相位分量，这是一个新颖的视角。

## 局限与展望

- 仅在 RTX 3090 单卡训练，效率和可扩展性分析不足。
- 4 个专家的数量和 Top-2 选择是固定的，未探索自适应专家数量。
- BSCL 的 Haar 小波选择较为简单，更复杂的频率分解是否能进一步改善值得探索。
- 下游任务验证主要在检测和分割，其他下游（如跟踪）未涉及。

## 相关工作与启发

- **vs Fusionmamba1/2**：仅做单模态空间 Mamba 扫描，SMC-Mamba 引入跨模态扫描和 MoE 显著提升。
- **vs TC-MoA**：TC-MoA 也用 MoE 但缺乏多样性-共识平衡机制。
- **vs CDDFuse**：任务特定方法在其目标任务上可能较强，但无法泛化到其他融合任务。

## 评分

- 新颖性: ⭐⭐⭐⭐ 频域 Mamba + MoE 共识机制 + 双层对比学习组合有新意
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖四种融合任务，对比方法众多，消融完整
- 写作质量: ⭐⭐⭐⭐ 结构清晰但模块较多，略显复杂
- 价值: ⭐⭐⭐⭐ 通用图像融合方向的有效推进

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] SPA: Achieving Consensus in LLM Alignment via Self-Priority Optimization](spa_achieving_consensus_in_llm_alignment_via_self-priority_optimization.md)
- [\[AAAI 2026\] NeuroBridge: Bio-Inspired Self-Supervised EEG-to-Image Decoding via Cognitive Priors and Bidirectional Semantic Alignment](neurobridge_bio-inspired_self-supervised_eeg-to-image_decoding_via_cognitive_pri.md)
- [\[AAAI 2026\] Vascular Anatomy-aware Self-supervised Pre-training for X-ray Angiogram Analysis](vascular_anatomy-aware_self-supervised_pre-training_for_x-ray_angiogram_analysis.md)
- [\[AAAI 2026\] ProPL: Universal Semi-Supervised Ultrasound Image Segmentation via Prompt-Guided Pseudo-Labeling](propl_universal_semi-supervised_ultrasound_image_segmentation_via_prompt-guided_.md)
- [\[AAAI 2026\] DualFete: Revisiting Teacher-Student Interactions from a Feedback Perspective for Semi-supervised Medical Image Segmentation](dualfete_revisiting_teacher-student_interactions_from_a_feedback_perspective_for.md)

</div>

<!-- RELATED:END -->
