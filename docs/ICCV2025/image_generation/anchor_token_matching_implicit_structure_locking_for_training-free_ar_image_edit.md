---
title: >-
  [论文解读] Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing
description: >-
  [ICCV 2025][图像生成][自回归模型] 提出 ISLock，首个面向自回归(AR)视觉生成模型的无训练图像编辑方法，通过锚点 Token 匹配(ATM)在隐空间中隐式对齐自注意力模式，实现结构一致的文本引导图像编辑。 扩散模型已经在文本引导图像编辑中取得了巨大成功，通过交叉注意力操纵实现精确的空间控制。然而…
tags:
  - "ICCV 2025"
  - "图像生成"
  - "自回归模型"
  - "图像编辑"
  - "训练无关"
  - "注意力机制"
  - "结构一致性"
---

# Anchor Token Matching: Implicit Structure Locking for Training-free AR Image Editing

**会议**: ICCV 2025  
**arXiv**: [2504.10434](https://arxiv.org/abs/2504.10434)  
**代码**: [https://github.com/hutaiHang/ATM](https://github.com/hutaiHang/ATM)  
**领域**: 图像编辑 / 自回归生成  
**关键词**: 自回归模型, 图像编辑, 训练无关, 注意力机制, 结构一致性

## 一句话总结

提出 ISLock，首个面向自回归(AR)视觉生成模型的无训练图像编辑方法，通过锚点 Token 匹配(ATM)在隐空间中隐式对齐自注意力模式，实现结构一致的文本引导图像编辑。

## 研究背景与动机

扩散模型已经在文本引导图像编辑中取得了巨大成功，通过交叉注意力操纵实现精确的空间控制。然而，自回归(AR)模型作为强大的替代方案重新崛起（如 LlamaGen、Emu3），其基于 next-token 预测的范式使得扩散模型的编辑技术无法直接迁移。

AR 模型面临两个核心挑战：

**注意力图的空间贫瘠性**：AR 模型中文本到图像的注意力图缺乏精确的空间对应关系，当前 token 倾向于高度关注前一个 token，无法作为可靠的编辑锚点

**结构误差的顺序累积**：朴素地修改目标 token（如将"cat"改为"dog"）会引发隐状态的局部偏移，通过自回归依赖链传播，最终扭曲全局结构

现有方法要么需要大规模配对数据微调，要么计算开销大且牺牲零样本灵活性。因此关键问题在于：如何在不训练的前提下，实现 AR 模型中结构一致的编辑？

## 方法详解

### 整体框架

ISLock（Implicit Structure Locking）不依赖显式注意力操纵或微调，而是通过动态对齐自注意力模式与参考图像来保持结构蓝图。核心思路是在自回归解码过程中，通过 ATM 协议选择性匹配 token，使注意力一致性作为副产品自然涌现。

### 关键设计

1. **结构信息分析**

    - 通过 PCA 对自注意力矩阵 $A \in \mathbb{R}^{(h \times w) \times (h \times w)}$ 进行分解和可视化，发现语义相似 token 具有一致的注意力模式
    - 交叉注意力图缺乏空间结构信息，而自注意力图包含丰富的结构信息
    - 扰动实验表明：扰动前 20% 的 token 导致 SSIM 下降 $0.56 \pm 0.02$，而后 20% 仅下降 $0.08 \pm 0.05$，证实早期 token 对全局结构至关重要

2. **锚点 Token 匹配（ATM）与动态窗口**

    - 给定原始提示 $\mathcal{P}_{org}$ 和编辑提示 $\mathcal{P}_{edit}$，在每步 $i$ 采样 $K$ 个候选 token $\mathcal{C}_i = \{z_i^{(1)}, ..., z_i^{(K)}\}$
    - 计算每个候选与参考锚点的欧氏距离：$s^{(k)} = \|z_i^{(k)} - z_i^{org}\|_2^2$
    - 引入动态窗口机制，窗口大小随解码进度线性收缩：$|\mathcal{W}_i| = \lfloor K \cdot (1 - \alpha \cdot \frac{i}{N}) \rfloor$，其中 $\alpha = 0.6$
    - 初始阶段保留100%候选确保严格结构对齐，后期逐渐收紧约束
    - 最终选择窗口内距离最小的候选：$z_i^{edit} = \arg\min_{k \in \mathcal{W}_i} s^{(k)}$

3. **自适应约束松弛（AdaCR）**

    - 引入相似度阈值 $\tau$ 平衡结构保持与生成自主性：
    $z_i^{edit} = \begin{cases} \arg\min s^{(k)} & \text{if } \min s^{(k)} \leq \tau \\ \arg\max p(z_i|z_{<i}^{edit}, c_{edit}) & \text{otherwise} \end{cases}$
    - 较大 $\tau$ 保持更高的原图相似度，较小 $\tau$ 允许更大的编辑多样性
    - 包含候选窗口预过滤和动态阈值两重保护机制

### 训练策略

该方法完全无训练，不需要任何参数更新或微调。默认超参数：$K=150$, $\tau=1.0$, $\alpha=0.6$。

## 实验关键数据

### 主实验

基于 PIE-Bench 数据集的定量比较：

| 方法 | 基模型 | Structure Dist.↓ | PSNR↑ | SSIM↑ | CLIP Whole↑ | CLIP Edited↑ |
|------|--------|------------------|-------|-------|-------------|-------------|
| NPM | LlamaGen | 113.95 | 12.14 | 53.67 | 24.71 | 21.28 |
| PnP-AR | LlamaGen | 103.94 | 13.20 | 58.25 | 23.56 | 20.65 |
| **ISLock (Ours)** | **LlamaGen** | **31.79** | **19.75** | **76.71** | **24.19** | **21.33** |
| P2P | SD1.4 | 88.46 | 16.80 | 69.93 | 26.70 | 21.43 |
| Null-text Inv. | SD1.4 | 18.42 | 25.68 | 85.71 | 24.55 | 20.73 |

### 消融实验

| 窗口配置 | Struc. Dist.↓ | CLIP Sim.↑ | S/C比↓ |
|----------|--------------|------------|--------|
| $|\mathcal{W}|=50$ | 60.83 | 24.79 | 2.45 |
| $|\mathcal{W}|=100$ | 38.03 | 24.33 | 1.56 |
| $|\mathcal{W}|=150$ | 30.39 | 22.18 | 1.37 |
| Dynamic (Ours) | 31.79 | 24.19 | **1.31** |

### 关键发现

- ISLock 在 AR 方法中取得了最佳的结构一致性（Struc. Dist. 31.79，远优于 NPM 的 113.95）
- 动态窗口策略在结构距离和 CLIP 相似度之间取得了最佳平衡（S/C 比率 1.31）
- 支持属性替换、物体增删、风格迁移等五种编辑类型
- 可泛化至不同 AR 基模型（LlamaGen 和 Lumina-mGPT）

## 亮点与洞察

- **首创性强**：该工作是 AR 视觉模型中首个无训练编辑框架，填补了 AR 模型在图像编辑方面的空白
- **隐式 vs 显式**：关键洞察在于不直接注入注意力图（会导致伪影），而是通过 token 匹配让注意力一致性自然涌现
- **渐进式结构固化**：对 AR 生成中结构控制机制的系统分析，揭示了早期 token 对全局结构的决定性作用

## 局限与展望

- 整体性能仍落后于专为扩散框架优化的编辑方法（如 Null-text Inversion）
- 依赖 AR 模型生成原始图像，无法直接编辑真实图片
- 候选数 $K=150$ 带来额外的采样开销
- 仅验证了 5 种编辑类型，未覆盖 PIE-Bench 全部 10 种

## 相关工作与启发

- 与扩散模型的 Prompt-to-Prompt 思路形成对比，展示了不同生成范式下结构控制的根本差异
- 为 AR + 多模态大模型时代的可控图像编辑提供了新的技术路径
- ATM 的候选匹配思想可以推广到其他 AR 生成的可控性问题

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个面向AR模型的无训练编辑方法，核心 idea 简洁有效
- 实验充分度: ⭐⭐⭐⭐ 有定量比较和消融，但数据集覆盖有限
- 写作质量: ⭐⭐⭐⭐ 动机分析清晰，实验可视化丰富
- 价值: ⭐⭐⭐⭐ 为AR模型编辑开辟了新方向，但实用性待提升

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Stable Flow: Vital Layers for Training-Free Image Editing](../../CVPR2025/image_generation/stable_flow_vital_layers_for_training-free_image_editing.md)
- [\[ICCV 2025\] ALE: Attribute-Leakage-free Editing for Text-based Image Editing](ale_attribute_leakage_free_editing.md)
- [\[ICCV 2025\] FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models](flowedit_inversion-free_text-based_editing_using_pre-trained_flow_models.md)
- [\[CVPR 2026\] TAP: A Token-Adaptive Predictor Framework for Training-Free Diffusion Acceleration](../../CVPR2026/image_generation/tap_a_token-adaptive_predictor_framework_for_training-free_diffusion_acceleratio.md)
- [\[ICCV 2025\] MotionDiff: Training-Free Zero-Shot Interactive Motion Editing via Flow-Assisted Multi-View Diffusion](motiondiff_training-free_zero-shot_interactive_motion_editing_via_flow-assisted_.md)

</div>

<!-- RELATED:END -->
