---
title: >-
  [论文解读] MatAnyone: Stable Video Matting with Consistent Memory Propagation
description: >-
  [CVPR 2025][语义分割][视频抠图] 提出 MatAnyone 框架，通过区域自适应记忆融合机制在记忆空间中实现一致性传播（核心区域保持语义稳定，边界区域捕获精细 alpha 细节），配合新数据集 VM800 和利用分割数据直接监督 matting head 的训练策略，实现了鲁棒且高质量的目标指定视频抠图。
tags:
  - "CVPR 2025"
  - "语义分割"
  - "视频抠图"
  - "记忆传播"
  - "区域自适应融合"
  - "alpha matte"
  - "目标指定"
---

# MatAnyone: Stable Video Matting with Consistent Memory Propagation

**会议**: CVPR 2025  
**arXiv**: [2501.14677](https://arxiv.org/abs/2501.14677)  
**代码**: [https://pq-yang.github.io/projects/MatAnyone](https://pq-yang.github.io/projects/MatAnyone)  
**领域**: 视频抠图 / 分割  
**关键词**: 视频抠图, 记忆传播, 区域自适应融合, alpha matte, 目标指定

## 一句话总结

提出 MatAnyone 框架，通过区域自适应记忆融合机制在记忆空间中实现一致性传播（核心区域保持语义稳定，边界区域捕获精细 alpha 细节），配合新数据集 VM800 和利用分割数据直接监督 matting head 的训练策略，实现了鲁棒且高质量的目标指定视频抠图。

## 研究背景与动机

**领域现状**：无辅助人像视频抠图（如 RVM）仅依靠输入帧，在复杂/歧义背景下（如多个人物）极易失败。目标指定的视频抠图借鉴 VOS（半监督视频目标分割）的设定——仅需第一帧分割掩码——通过记忆匹配范式实现稳定的跨帧追踪。

**现有痛点**：(1) 现有 mask-guided 方法（AdaM, FTP-VM, MaGGIe）将 VOS 先验微调为 matting 时，由于视频抠图数据（VideoMatte240K）质量差（核心区域有孔洞、边界细节模糊）且规模有限，容易破坏 VOS 先验的语义稳定性；(2) 核心区域和边界区域对记忆匹配的需求截然不同——核心区域需要稳定传播、边界区域需要精细更新，但现有记忆框架对所有 token 一视同仁；(3) 用分割数据训练时采用平行 head 方案，matting head 本身无法接收真实分割数据的监督。

**核心矛盾**：如何在使用次优视频抠图数据训练的条件下，同时保证核心区域的语义稳定性和边界区域的 matting 级细节？

**本文切入角度**：在记忆传播阶段引入区域自适应机制——预测每个 token 相对上一帧的 alpha 变化概率，"大变化"区域（边界）依赖当前帧从记忆库查询的信息，"小变化"区域（核心）保留上一帧记忆，实现选择性记忆融合。

## 方法详解

### 整体框架

基于记忆匹配范式（类似 STCN/Cutie），输入为首帧分割掩码 + 视频帧序列，逐帧预测 alpha matte。当前帧 t 编码为特征 F^t（16× 下采样），通过一致性记忆传播模块（CMP）融合记忆库信息和上一帧信息得到像素记忆读出 P^t，再经 Object Transformer 提取对象级语义，送入解码器预测 alpha matte M^t。预测结果编码为记忆值 V^t 更新 alpha 记忆库。

### 关键设计

1. **一致记忆传播 (CMP) + 区域自适应记忆融合**:
    - **Alpha 记忆库**：存储 alpha matte（而非分割掩码/trimap），让记忆范式在边界区域也能提供稳定性
    - **变化概率预测**：用轻量级 3 层卷积模块预测每个 token 的变化概率 $U_t$，用帧间 alpha 差的二值化结果 $|M_{t-1}^{GT} - M_t^{GT}| \geq \delta$ 作为监督
    - **软融合**：$P_t = V_t^m \cdot U_t + V_{t-1} \cdot (1 - U_t)$，高 $U_t$（边界/变化区域）更多依赖从记忆库查询的当前帧信息，低 $U_t$（核心/稳定区域）保留上一帧记忆
    - 设计动机：核心区域帧间 alpha 几乎不变，直接传播上一帧记忆可避免匹配噪声；边界区域需要根据当前帧更新以捕获精细 alpha 过渡

2. **核心区域分割数据监督策略**:
    - 创新：将分割数据直接送入 matting head（而非平行的分割 head），用分区域损失监督
    - 核心区域：有分割标签，用 L1 损失 $\mathcal{L}_{core}$ 确保语义稳定
    - 边界区域：无 alpha GT，使用改进的 Scaled DDC 损失：$\mathcal{L}_{boundary} = |(\alpha_i - \alpha_j)(F-B) - \|I_i - I_j\|_2|$
    - 对原始 DDC 损失的修正：原假设 $\|\alpha_i - \alpha_j\| = \|I_i - I_j\|$ 仅在 $|F-B|=1$ 时成立，引入前景/背景颜色差缩放后使边缘更自然
    - 设计动机：直接在 matting head 上用分割数据监督，比平行 head 方案更充分利用分割先验

3. **推理阶段首帧循环精化**:
    - 将第一帧重复 n 次作为序列处理，利用记忆范式的逐帧精化特性，仅取第 n 帧输出作为真正的首帧结果
    - 提高对给定分割掩码的鲁棒性，同时将首帧质量提升到图像级 matting 水平
    - 设计动机：首帧 matte 质量直接影响后续帧，循环精化零成本提升首帧质量

### 损失函数

- **Matting 数据**：L1 + Laplacian loss + Grad loss（标准 matting 损失）
- **分割数据**：$\mathcal{L}_{core}$ (L1) + $\mathcal{L}_{boundary}$ (Scaled DDC)  
- **变化概率预测**：$\mathcal{L}_{bin\_seg}$ (二值交叉熵)
- 三阶段训练：VM800 matting → 加入分割数据核心区域监督 → 图像 matting 数据微调

## 实验关键数据

### 主实验表

**VideoMatte 1080p：**

| 方法 | MAD↓ | MSE↓ | Grad↓ | dtSSD↓ |
|------|------|------|-------|--------|
| RVM-Large (AF) | 5.81 | 0.97 | 9.65 | 1.78 |
| MaGGIe† (每帧mask) | 4.42 | 0.40 | 4.03 | 1.31 |
| **MatAnyone** | **4.24** | **0.33** | **4.00** | **1.19** |

**Real-world benchmark (核心区域指标)：**

| 方法 | MAD↓ | MSE↓ | dtSSD↓ |
|------|------|------|--------|
| RVM-Large | 0.95 | 0.50 | 1.30 |
| MaGGIe | 1.94 | 1.53 | 1.63 |
| **MatAnyone** | **0.14** | **0.10** | **0.89** |

- 在真实世界基准上 MAD 比次优方法 RVM-Large 低 85%（0.14 vs 0.95）
- MatAnyone 仅需首帧掩码，MaGGIe 需要每帧掩码指导，但仍被超越

### 消融表

| 组件 | MAD↓ | dtSSD↓ |
|------|------|--------|
| Baseline (无 CMP, 旧数据, 旧训练) | 较高 | 较高 |
| +New Data (VM800) | 改善 | 改善 |
| +CMP (记忆传播) | 显著改善 | 显著改善 |
| +New Training (分割监督) | **全局最优** | **全局最优** |

### 关键发现

- CMP 模块同时改善核心区域稳定性和边界区域细节——核心区域直接传播上一帧避免匹配噪声，边界区域聚焦 alpha 过渡
- Scaled DDC 损失 vs 原始 DDC：原始版本产生分割式的阶梯状边缘，缩放版本产生更自然的 matting 过渡
- VM800 数据集质量对训练贡献显著——比 VideoMatte240K 大 2 倍、更多样、边界质量更高
- 首帧循环精化 (n=3) 可以显著改善对粗糙初始掩码的鲁棒性

## 亮点与洞察

- **区域自适应记忆融合很优雅**：用一个轻量预测模块将 matting 的"核心 vs 边界"特性与记忆传播机制自然结合，既保留了 VOS 的语义稳定性又增强了边界精细度
- **分割数据监督策略突破了瓶颈**：之前方法用平行 head 的方式无法充分利用分割先验，直接监督 matting head 并用 Scaled DDC 处理边界区域是关键创新
- **实用性强**：仅需首帧掩码（可由 SAM 等工具获取），支持实例级抠图，在长视频和复杂背景下保持稳定

## 局限性

- 对非人类目标的泛化能力未充分验证（训练数据以人像为主）
- 极端运动模糊或极快运动可能导致变化概率预测失准
- 依赖 VOS 框架的记忆匹配，大规模长视频可能面临记忆库管理问题
- 合成训练数据与真实世界的域差距仍未完全消除

## 相关工作与启发

- **VOS → Video Matting 的范式迁移**：记忆匹配是 VOS 的核心范式（STCN → XMem → Cutie），本文证明该范式经过适当改造可直接用于更精细的 matting 任务
- **DDC 损失的改进思路**：对无 GT alpha 数据的监督方法，分析假设条件并修正（加入 F-B 缩放），这种方法论值得借鉴
- **实例级视频 matting 的应用前景**：基于首帧指定的设定，可以在视频编辑、虚拟背景、特效制作等场景中替代传统绿幕方案

## 评分

⭐⭐⭐⭐ — 区域自适应记忆融合设计精巧，核心区域分割监督策略创新实用，实验全面且在真实世界基准上优势突出。数据集贡献（VM800 + YoutubeMatte）为社区提供了更好的训练/评估基础。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Generative Video Propagation](generative_video_propagation.md)
- [\[CVPR 2026\] MatAnyone 2: Scaling Video Matting via a Learned Quality Evaluator](../../CVPR2026/segmentation/matanyone_2_scaling_video_matting_via_a_learned_quality_evaluator.md)
- [\[CVPR 2025\] MaSS13K: A Matting-level Semantic Segmentation Benchmark](mass13k_a_matting-level_semantic_segmentation_benchmark.md)
- [\[CVPR 2025\] StoryGPT-V: Large Language Models as Consistent Story Visualizers](storygpt-v_large_language_models_as_consistent_story_visualizers.md)
- [\[CVPR 2025\] A Distractor-Aware Memory for Visual Object Tracking with SAM2](a_distractor-aware_memory_for_visual_object_tracking_with_sam2.md)

</div>

<!-- RELATED:END -->
