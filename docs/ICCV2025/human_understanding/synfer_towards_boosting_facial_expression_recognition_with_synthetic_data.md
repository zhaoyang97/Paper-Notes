---
title: >-
  [论文解读] SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data
description: >-
  [ICCV 2025][人体理解][面部表情识别] 提出 SynFER，一个基于扩散模型的面部表情合成框架，通过文本描述 + 面部动作单元 (FAU) 的双重控制实现细粒度表情生成，并引入 FERAnno 标签校准器确保标注可靠性，在自监督、监督、零样本和少样本四种学习范式下均证明合成数据对 FER 的有效性。
tags:
  - "ICCV 2025"
  - "人体理解"
  - "面部表情识别"
  - "合成数据"
  - "扩散模型"
  - "面部动作单元"
  - "标签校准"
---

# SynFER: Towards Boosting Facial Expression Recognition with Synthetic Data

**会议**: ICCV 2025  
**arXiv**: [2410.09865](https://arxiv.org/abs/2410.09865)  
**代码**: [有](https://github.com/)  
**领域**: 面部表情识别 / 合成数据  
**关键词**: 面部表情识别, 合成数据, 扩散模型, 面部动作单元, 标签校准

## 一句话总结

提出 SynFER，一个基于扩散模型的面部表情合成框架，通过文本描述 + 面部动作单元 (FAU) 的双重控制实现细粒度表情生成，并引入 FERAnno 标签校准器确保标注可靠性，在自监督、监督、零样本和少样本四种学习范式下均证明合成数据对 FER 的有效性。

## 研究背景与动机

面部表情识别（FER）面临严重的数据匮乏问题。现有 FER 数据集规模远小于通用视觉数据集（AffectNet ~28 万 vs ImageNet 140 万），且存在标注主观性强、低质量图像多、标签错误率高等问题。这些缺陷阻碍了 FER 基础模型的发展。

直接利用扩散模型生成 FER 数据面临两大挑战：(1) 生成模型训练数据缺乏多样表情，难以捕捉细微表情语义；(2) 表情是抽象主观概念，不像深度图或分割图那样可直接生成标注。

## 方法详解

### 整体框架

三阶段管线：
1. 准备阶段：构建 FEText 数据集 + 生成 FAU 标签和文本描述
2. 生成阶段：FAU 控制 + 语义引导的扩散模型生成高保真表情图像
3. 标注阶段：FERAnno 标签校准器自动生成可靠标签 + 集成投票验证

### 关键设计

1. **FEText 数据集构建**:

    - 首个面部表情图文对数据集，整合 FFHQ + CelebA-HQ + AffectNet + SFEW
    - 共 400K 精选图文对
    - 使用超分辨率模型统一低分辨率图像（AffectNet、SFEW）的分辨率
    - 使用 ShareGPT-4V 多模态大模型生成详细的表情描述文本
    - 精心设计的 prompt 引导模型生成精确、情感反映的描述

2. **FAU 控制的表情生成**:

    - 文本描述提供高层语义控制，但缺乏精细面部肌肉运动信息
    - 引入面部动作单元（FAU）作为显式控制信号，每个 AU 对应特定面部肌肉运动
    - 参考 IP-Adapter 设计，使用解耦交叉注意力模块整合 FAU 嵌入
    - FAU 标签通过 OpenGraphAU 模型标注
    - 冻结扩散模型参数，仅训练 AU adapter（MLP）

3. **语义引导 (Semantic Guidance)**:

    - 解决 FER 标签不平衡和表情间歧义（如 disgust vs. anger）
    - 布局初始化：从 FEText 随机选择图像反演为初始噪声，保持自然面部结构
    - 语义引导：在去噪后期阶段，利用外部 FER 分类器梯度更新文本嵌入
    - 更新规则：$c_{t-1}^{text} = c_t^{text} + \lambda_g \frac{\nabla_{c_t^{text}} \mathcal{L}_g}{\|\nabla_{c_t^{text}} \mathcal{L}_g\|_2}$
    - 其中 $\mathcal{L}_g = -y \log(h(f(\hat{x}_0))_i)$ 为分类损失

4. **FERAnno 标签校准器**:

    - 基于扩散模型的伪标签生成器，利用 U-Net 中间特征和交叉注意力图
    - 图像反演 ($t=1$, 保留最多面部细节) → 特征提取 → 双分支编码器融合
    - 多尺度特征图捕获全局生成信息，交叉注意力图提供类判别信息
    - 双向交叉注意力块融合两类特征 → 线性层输出表情类别概率
    - 与外部 FER 模型集成投票，不一致时替换为集成预测平均值

### 损失函数 / 训练策略

- 扩散模型训练：标准扩散损失 $\min_\theta \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, c, t)\|_2^2$
- AU adapter 训练：冻结扩散模型，仅训练 MLP 映射 FAU → 嵌入
- FERAnno 训练：利用扩散模型中间层特征和注意力图的双分支架构

## 实验关键数据

### 主实验

自监督预训练 + 线性探测（ResNet-50）：

| SSL 方法 | 预训练数据 | 规模 | RAF-DB | AffectNet | SFEW |
|---------|----------|------|--------|-----------|------|
| MoCo v3 | AffectNet | 0.2M | 79.05 | 51.03 | 49.34 |
| MoCo v3 | SynFER | 1.0M | 81.17(+2.12) | 55.56(+4.53) | 50.78(+1.44) |
| MoCo v3 | Both | 1.2M | 81.68(+2.63) | 57.84(+6.81) | 51.26(+1.92) |

监督学习提升：

| 方法 | RAF-DB | AffectNet |
|------|--------|-----------|
| POSTER++ | 91.59 | 67.49 |
| **POSTER++ + SynFER** | **91.95** | **69.04** |
| APViT | 91.78 | 66.94 |
| **APViT + SynFER** | **92.05** | **67.26** |
| **FERAnno (独立)** | **92.56** | **70.38** |

纯合成数据训练：AffectNet 上 67.23%（等量数据）→ 69.84%（5倍数据量）。

### 消融实验

各组件对生成质量和下游性能的贡献：

| 方法 | FER Acc. | AU Acc. | RAF-DB | AffectNet |
|------|----------|---------|--------|-----------|
| SD baseline | 20.06% | 87.72% | 89.42 | 65.36 |
| + FEText | 34.62% | 88.91% | 90.54 | 66.62 |
| + FEText + FAUs | 48.74% | 92.37% | 91.68 | 67.68 |
| + FEText + FAUs + SG | **55.14%** | **93.31%** | **91.95** | **68.13** |

生成质量对比（FID↓）：

| 方法 | FID | FER Acc. | 用户偏好(EA) |
|------|-----|----------|-------------|
| Stable Diffusion | 88.40 | 20.06% | 2.86% |
| FineFace | 74.61 | 38.05% | 5.73% |
| **SynFER** | **16.32** | **55.14%** | **59.64%** |

### 关键发现

- **FAU 控制显著提升表情准确性**：FER 准确率从 34.62% 提升至 48.74%，使模糊表情（如 fear vs surprise）变得可区分
- **语义引导进一步提升**：在 FAU 基础上再提升 6.4% 的表情准确率
- **自监督学习受益最大**：合成数据对 SSL 的提升远超监督学习，因后者对分布对齐要求更严格
- **FERAnno 本身即为强分类器**：92.56% RAF-DB 和 70.38% AffectNet 超越所有 SOTA FER 模型
- **数据量缩放有效**：特别是结合 Real-Fake 技术做分布对齐后，监督学习也能从更多合成数据中获益

## 亮点与洞察

- **完整的合成数据管线**：从数据构建（FEText）→ 生成控制（FAU+SG）→ 标签校准（FERAnno）→ 下游验证，形成闭环
- **FERAnno 的双重身份**：既是标签校准工具，又是独立的 SOTA FER 分类器，说明扩散模型内部特征对表情理解非常丰富
- **四种学习范式验证**：SSL、监督、零样本、少样本全面验证，说服力强
- **解决了 FER 数据规模瓶颈**：为 FER 基础模型的训练提供了可扩展的数据来源

## 局限与展望

- 合成数据在监督学习上的提升幅度有限（<2%），分布对齐仍是瓶颈
- 仅覆盖基本表情类别（7 类），复合表情和微表情需要更精细的控制
- FEText 中文本描述依赖 ShareGPT-4V，可能引入描述偏差
- FAU 检测模型 (OpenGraphAU) 本身的准确性限制了控制精度

## 相关工作与启发

- FERAnno 利用扩散模型中间特征做分类的思路可扩展到其他视觉理解任务
- FAU 控制机制可推广到其他需要细粒度面部控制的生成任务（如 talking head）
- 数据缩放实验中 SSL > 监督的发现对合成数据在其他领域的应用有参考意义

## 评分

- **新颖性**: ⭐⭐⭐⭐ 首个完整的扩散模型 FER 合成管线，FAU + 语义引导设计巧妙
- **实验充分度**: ⭐⭐⭐⭐⭐ 四种学习范式 × 六个数据集 × 多个消融，极为全面
- **写作质量**: ⭐⭐⭐⭐ 管线描述清晰，图表丰富
- **实用价值**: ⭐⭐⭐⭐ 为 FER 社区提供了可扩展的数据生成方案

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ECCV 2024\] Generalizable Facial Expression Recognition](../../ECCV2024/human_understanding/generalizable_facial_expression_recognition.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[CVPR 2026\] Dynamic Label Noise Suppression with Optimal Teacher Pool for Facial Expression Recognition](../../CVPR2026/human_understanding/dynamic_label_noise_suppression_with_optimal_teacher_pool_for_facial_expression_.md)
- [\[CVPR 2025\] HSEmotion Team at ABAW-10 Competition: Facial Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained Violence Classification](../../CVPR2025/human_understanding/hsemotion_team_at_abaw-10_competition_facial_expression_recognition_valence-arou.md)
- [\[CVPR 2026\] D³FER: Dual Channel and Dual Branch Network for Robust Facial Expression Recognition under Dual Challenges](../../CVPR2026/human_understanding/d3fer_dual_channel_and_dual_branch_network_for_robust_facial_expression_recognit.md)

</div>

<!-- RELATED:END -->
