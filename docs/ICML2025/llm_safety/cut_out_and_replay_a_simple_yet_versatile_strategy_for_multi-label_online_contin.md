---
title: >-
  [论文解读] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning
description: >-
  [ICML 2025][LLM安全][多标签学习] 提出 CUTER（CUT-out-and-Experience-Replay），通过裁剪图像中标签特定区域并存入记忆缓冲区进行回放，将多标签在线持续学习转化为多个单标签子图像分类任务，同时解决灾难性遗忘、缺失标签和类别不平衡三大挑战。 多标签在线持续学习（MOCL）要求模型…
tags:
  - "ICML 2025"
  - "LLM安全"
  - "多标签学习"
  - "在线持续学习"
  - "经验回放"
  - "图谱聚类"
  - "目标定位"
---

# Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning

**会议**: ICML 2025  
**arXiv**: [2505.19680](https://arxiv.org/abs/2505.19680)  
**代码**: [GitHub](https://github.com/wxr99/Cut-Replay)  
**领域**: LLM安全  
**关键词**: 多标签学习, 在线持续学习, 经验回放, 图谱聚类, 目标定位

## 一句话总结

提出 CUTER（CUT-out-and-Experience-Replay），通过裁剪图像中标签特定区域并存入记忆缓冲区进行回放，将多标签在线持续学习转化为多个单标签子图像分类任务，同时解决灾难性遗忘、缺失标签和类别不平衡三大挑战。

## 研究背景与动机

多标签在线持续学习（MOCL）要求模型从不断到来的多标签数据流中持续学习，面临三大挑战：

**普遍的标签缺失**：任务 $t$ 的样本仅标注当前标签集 $Y_t$，即使图像中含有旧类或未来类的物体，这些未标注类别成为假负例，加剧灾难性遗忘。

**不可控的类别不平衡**：类别通常呈长尾分布，头部类和尾部类在同一样本中共现使问题更加复杂。

**现有方法的根本局限**：已有方法（PRS、OCDM、AGCN、KRT 等）均采用全图级别的特征提取，忽视了标签特定区域的识别和特征学习。这导致共现偏差——头/尾类共享同一特征向量，伪标签和重采样无法根本解决问题。

核心洞察：如果能够在线地识别并裁剪出每个标签对应的区域，将区域-标签对存入缓冲区回放，就能天然避免标签共现干扰和缺失标签问题，同时通过控制缓冲区中各类别的分布轻松解决类别不平衡。

## 方法详解

### 整体框架

CUTER 包含三个核心步骤：

1. **预训练模型零样本定位能力评估**：基于图论提出无标注评估协议，选择最适合定位的预训练模型。
2. **选择性回放（Cut-and-Replay）**：利用 MaskCut 定位前景物体区域，建立区域-标签一一对应关系，裁剪并存入记忆缓冲区。
3. **定位感知特征正则化**：为防止持续学习过程中定位能力退化，引入核范数正则化增强特征图的可分性。

### 关键设计

#### 1. 基于 Fiedler 值的预训练模型评估

将图像 patch 特征构建加权无向图 $G=(V,E,A)$，边权 $A_{ij} = \exp(-\frac{\|\theta(x_i)-\theta(x_j)\|^2}{2\sigma^2})$，计算图拉普拉斯矩阵 $L=D-A$ 的第二小特征值（Fiedler 值 $\lambda_2$）。

**关键理论依据**：Fiedler 值与 Cheeger 常数满足 $\frac{\lambda_2}{2} \leq h(G) \leq \sqrt{2\Delta\lambda_2}$。较低的平均 Fiedler 值意味着更弱的图连通性、更强的特征可分性，更适合基于谱聚类的定位。

**评估结论**：多裁剪一致性训练（如 DINO）显著增强天然定位能力，对比学习（MoCo）次之，重建式预训练（MAE）最差。最终选择 DINO v1 ViT-S/16 作为骨干网络。

#### 2. 标签-区域匹配与选择性回放

对每个输入 $(x, y)$：

- **前景物体提取**：使用 MaskCut（MCut）迭代生成 $N$ 个二值掩码 $\{m_j\}_{j=1}^N$，导出边界框，裁剪出 $N$ 个候选前景区域 $\{x_{obj}^j\}$。
- **区域-标签对应建立**：将裁剪区域送入分类模型获取预测 $p_{obj}^j$，保留满足以下条件的区域：
    - 最大预测概率 $p_{obj,(1)}^j > \tau$（高置信度）
    - 次大预测概率 $p_{obj,(2)}^j < 0.5$（单标签对应）
- **自适应阈值平衡**：设置双阈值 $\tau_1 < \tau_2$，频率低于最高频类别一半的类使用更低阈值 $\tau_1$，其余使用 $\tau_2$。
- **重平衡储池采样**：新候选的采样概率为 $1 - m/m_{max}$，其中 $m$ 为该预测标签在缓冲区中的样本数，$m_{max}$ 为最多类的数量。缓冲区满时随机移除最多类的样本。

此策略将多标签图像回放转化为多个单标签子图像回放，实现比 OCDM 和 PRS 更好的类别平衡且计算开销更低。

#### 3. 定位感知特征正则化

持续学习过程中，预训练模型的定位能力会逐渐退化（表现为 Fiedler 值升高、定位失败率增加）。为解决此问题：

**理论基础（Theorem 2.3）**：将邻接矩阵分解为理想块对角矩阵 $A^*$ 和噪声矩阵 $\epsilon$，即 $A = A^* + \epsilon$，则 $\lambda_2(L) \leq \|\epsilon\|_2 + \|\epsilon\|_\infty$。

**核范数正则化**：直接约束邻接矩阵 $A$ 的核范数 $\|A\|_*$，通过软阈值化奇异值促使 $A$ 趋向块对角结构，间接降低 Fiedler 值，增强可分性。

### 损失函数 / 训练策略

总损失函数为：

$$L = L_{asl}(f, x, y) + \alpha \|A\|_*$$

其中 $L_{asl}$ 为非对称损失（Asymmetric Loss），对正/负标签分别使用不同聚焦参数 $\gamma^+$、$\gamma^-$：

$$L_{asl} = \frac{1}{|C_k|}\sum_{c=1}^{|C_k|} \begin{cases} (1-p_c)^{\gamma^+}\log(p_c), & y_c=1 \\ p_c^{\gamma^-}\log(1-p_c), & y_c=0 \end{cases}$$

选择核范数而非稀疏正则或平滑正则的原因：稀疏正则会破坏 ViT 参数的固有结构，平滑正则使节点特征过于相似反而阻碍谱聚类。

## 实验关键数据

### 主实验

实验在 PASCAL VOC 2007（5 任务 × 4 类）、MS-COCO（8 任务 × 10 类）、NUS-WIDE（8 任务）三个数据集上进行，内存大小 $1000 \times 224 \times 224 \times 3$。

| 数据集 | 指标 | CUTER | 之前SOTA | 提升 |
|--------|------|-------|---------|------|
| VOC | Avg mAP | **82.07** | 76.24 (APPLE) | +5.83 |
| VOC | Last mAP | **67.89** | 58.27 (APPLE) | +9.62 |
| COCO | Avg mAP | **60.14** | 56.45 (AGCN) | +3.69 |
| COCO | Last mAP | **47.82** | 40.56 (OCDM) | +7.26 |
| NUSWIDE | Avg mAP | **51.14** | 49.16 (AGCN) | +1.98 |
| NUSWIDE | Last mAP | **42.92** | 40.89 (APPLE) | +2.03 |

### 消融实验

| 配置 | VOC Avg mAP | COCO Avg mAP | 说明 |
|------|------------|-------------|------|
| Baseline (RS) | 75.05 | 48.12 | 基线回放 |
| + Cut.Rep | 77.92 | 53.40 | 仅裁剪回放提升显著 |
| + Cut.Rep + 固定骨干 | 78.62 | 59.01 | 固定 DINO backbone |
| CUTER (更新骨干) | 79.45 | 59.23 | 动态更新优于固定 |
| CUTER + $R_l$ | **82.07** | **60.14** | 完整方法 |

### 关键发现

1. **裁剪回放是最关键的组件**：Cut.Rep 从 75.05 提升到 77.92（VOC），从 48.12 到 53.40（COCO），贡献最大。
2. **正交兼容性强**：Cut.Rep 可作为插件与 PRS（+3.43）、OCDM（+3.17）、KRT、AGCN 无缝集成。
3. **核范数正则最优**：对比稀疏正则（$R_{sp}$, 78.34）和平滑正则（$R_{sm}$, 79.01），核范数（$R_l$, 82.07）效果最佳。
4. **DINO v1 具有最强定位能力**：在 ViT-S 骨干下，DINO v1 预训练的 CUTER 在 VOC 达 82.07，优于 Supervised（79.56）、MoCo v3（80.47）、MAE（77.31）。

## 亮点与洞察

1. **从根源解决问题**：不像过去的方法通过伪标签/蒸馏来"打补丁"，而是从多标签学习的根本出发——标签特定特征学习，将多标签问题拆解为多个单标签子问题。
2. **Fiedler 值作为无标注评估指标**：巧妙利用图谱理论，无需 GT 框/掩码即可评估预训练模型的定位潜力，具有通用价值。
3. **正交性设计**：CUTER 不替代而是补充现有方法，可与 PRS、OCDM、KRT、AGCN 等自由组合。
4. **理论支撑扎实**：从 Cheeger 常数到 Fiedler 值再到 Theorem 2.3 的核范数约束，形成完整的理论链条。

## 局限与展望

1. **依赖 ViT 架构**：方法核心依赖 patch-level 特征构图，在 CNN（ResNet）上性能显著下降。
2. **计算开销较大**：多轮 MCut 无法 GPU 并行，核范数梯度计算增加额外开销，吞吐量受限。
3. **阈值敏感性**：$\tau_1$、$\tau_2$、$\alpha$ 需要调参，$\alpha$ 过大会使邻接矩阵趋于零矩阵。
4. **未来方向**：开发加速技术、自适应处理策略、将方法适配到 CNN 架构。

## 相关工作与启发

- **OCDM (Liang & Li, 2022)**：用优化采样解决类别不平衡，本文在此基础上提出更简单有效的重平衡策略。
- **AGCN++ (Du et al., 2023)**：通过 GCN 建模标签关系处理缺失标签，但仍基于全图特征。
- **DINO (Caron et al., 2021)**：自监督预训练具有天然的目标定位能力，为本文的核心前提。
- **MaskCut (Wang et al., 2023)**：无监督多目标分割，本文将其引入 MOCL 场景。
- **LIFT (Zhang & Wu, 2014)**：离线多标签学习中标签特定特征优于统一特征的经典工作，本文将此思想推广到在线持续学习。

## 评分

- 新颖性: ⭐⭐⭐⭐⭐（首次将标签特定区域学习引入 MOCL，视角新颖）
- 实验充分度: ⭐⭐⭐⭐⭐（3 数据集 + 10 基线 + 详细消融/敏感度/可视化/骨干对比）
- 写作质量: ⭐⭐⭐⭐（理论推导严谨，结构清晰，但部分符号较重）
- 价值: ⭐⭐⭐⭐⭐（正交性强可即插即用，理论+实践贡献均扎实）

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Unlocking the Power of Rehearsal in Continual Learning: A Theoretical Perspective](unlocking_the_power_of_rehearsal_in_continual_learning_a_theoretical_perspective.md)
- [\[ICML 2025\] Improving Continual Learning Performance and Efficiency with Auxiliary Classifiers](improving_continual_learning_performance_and_efficiency_with_auxiliary_classifie.md)
- [\[ICML 2025\] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models](watch_out_your_album_on_the_inadvertent_privacy_memorization_in_multi-modal_larg.md)
- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)

</div>

<!-- RELATED:END -->
