---
title: >-
  [论文解读] Closed-Loop Transfer for Weakly-supervised Affordance Grounding
description: >-
  [ICCV 2025][图像恢复][弱监督affordance定位] 提出LoopTrans闭环知识迁移框架，通过共享CAM实现外中心-自中心图像的统一知识激活，利用像素级伪掩码将粗激活精炼为精确定位，并通过去噪蒸馏将自中心定位反馈增强外中心知识提取，在AGD20K上全面超越SOTA。 Affordance groundin…
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "弱监督affordance定位"
  - "闭环知识迁移"
  - "共享CAM"
  - "去噪蒸馏"
  - "外中心-自中心迁移"
---

# Closed-Loop Transfer for Weakly-supervised Affordance Grounding

**会议**: ICCV 2025  
**arXiv**: [2510.17384](https://arxiv.org/abs/2510.17384)  
**代码**: [https://github.com/nagara214/LoopTrans](https://github.com/nagara214/LoopTrans)  
**领域**: 视觉理解 / Affordance  
**关键词**: 弱监督affordance定位, 闭环知识迁移, 共享CAM, 去噪蒸馏, 外中心-自中心迁移

## 一句话总结

提出LoopTrans闭环知识迁移框架，通过共享CAM实现外中心-自中心图像的统一知识激活，利用像素级伪掩码将粗激活精炼为精确定位，并通过去噪蒸馏将自中心定位反馈增强外中心知识提取，在AGD20K上全面超越SOTA。

## 研究背景与动机

Affordance grounding旨在不仅预测物体可承载的动作，还要精确定位使动作成为可能的特定区域（如自行车的把手→推、把手+座椅→骑）。弱监督设定下，模型仅使用图像级别的交互标签（如"lie on"），从外中心（exocentric，第三人称视角）交互图像中学习affordance知识，然后迁移到自中心（egocentric，物体中心视角）图像上完成定位。

现有方法面临两个核心挑战：

**外中心知识提取不精确**：外中心交互图像背景复杂，CAM激活区域常包含人体部位和背景；在复杂交互场景中注意力分散而非聚焦于交互区域

**单向迁移的局限性**：
   - 现有方法（Cross-view-AG、LOCATE、WSMA）都采用单向框架：外中心CAM激活 → 特征对齐 → 自中心定位
   - 跨域特征对齐依赖外中心交互区域的外观相似性，当交互区域被人体完全遮挡时（如"lie on"、"ride"）失效
   - 自中心图像的物体中心特性（清晰、无背景干扰）未被充分利用来改善外中心知识提取

## 方法详解

### 整体框架

LoopTrans构建了一个闭环知识迁移流程：

**交互 → 激活**（共享CAM）→ **激活 → 定位**（像素级解码）→ **定位 → 激活**（去噪蒸馏）

三个阶段形成闭环：自中心定位的精确结果反馈增强外中心知识激活，外中心交互知识又通过共享CAM传递给自中心图像。

### 关键设计

#### 1. 统一外中心-自中心激活（Shared CAM / ΘSCAM）
- **功能**：使用共享参数的单一CAM模块同时处理外中心和自中心图像
- **核心思路**：不再使用两个独立的CAM模块分别处理两种视角，而是共享参数 $\theta$：

$$\mathcal{G}^{\text{exo}}, \mathcal{G}^{\text{ego}} = \Theta_{\text{SCAM}}(\{\mathcal{F}^{\text{exo}}, \mathcal{F}^{\text{ego}}\}; \theta)$$

分类损失同时最大化两种视角的联合置信度：

$$\mathcal{L}_{\text{cls}} = -\sum_{i=1}^{N} \mathbb{I}(c_i = \hat{c}) \log(\sigma(z_i^{\text{exo}}) \cdot \sigma(z_i^{\text{ego}}))$$

- **设计动机**：
    - 自中心图像以物体为中心，无背景干扰，其激活结果天然聚焦物体区域，可帮助外中心CAM排除人体和背景干扰
    - 共享参数强制跨视角一致性，减少域差异
    - 即使外中心图像中交互区域被完全遮挡，共享CAM也能通过自中心图像的激活识别affordance区域

#### 2. 区域激活到像素定位
- **功能**：将粗糙的CAM激活区域精炼为精确的物体部件级定位
- **核心思路**：分两步——
    - **激活到物体部件**：利用自监督ViT DINO的特征进行无监督聚类，将自中心图像分成 $K$ 个语义部件 $\{o_1,...,o_K\}$。选择与自中心激活图 $\mathcal{G}^{\text{ego}}_{\hat{c}}$ 的IoU最高的部件作为伪掩码：

$$\mathcal{M}^{\text{ego}} = \arg\max_{o_k} \text{IoU}(o_k, \mathbb{I}(\mathcal{R}(\mathcal{G}^{\text{ego}}_{\hat{c}}) \geq \mu))$$

  - **物体部件到定位**：训练像素级affordance解码器 $\Theta_{\text{pixel}}$，使用dice loss + MSE loss监督：

$$\mathcal{L}_{\text{dice}} = 1 - \frac{2 \sum_{i,j} \mathcal{P}_{i,j,\hat{c}} \cdot \mathcal{M}^{\text{ego}}_{i,j}}{\sum_{i,j} \mathcal{P}_{i,j,\hat{c}} + \sum_{i,j} \mathcal{M}^{\text{ego}}_{i,j}}$$

- **设计动机**：CAM的固有局限是只高亮最显著区域，无法覆盖完整交互部件。通过DINO特征聚类生成语义完整的伪掩码，然后训练像素级解码器实现精确定位

#### 3. 自中心到外中心去噪蒸馏
- **功能**：将精确的自中心定位反馈给共享CAM，抑制外中心图像中的背景和人体噪声
- **核心思路**：在共享CAM中增加 $M$ 个噪声吸收头 $\mathcal{G}^{\text{noise}}$：

$$f^{\text{exo}} = \text{GAP}(\mathcal{R}(\mathcal{G}^{\text{exo}}_{\hat{c}}) \circ \mathcal{F}^{\text{exo}})$$
$$f^{\text{pixel}} = \text{GAP}(\mathcal{R}(\mathcal{P}_{\hat{c}}) \circ \mathcal{F}^{\text{ego}})$$
$$\{f^{\text{noise}}_m\} = \text{GAP}(\mathcal{R}(\{\mathcal{G}^{\text{noise}}_m\}) \circ \mathcal{F}^{\text{exo}})$$

去噪蒸馏损失：
$$\mathcal{L}_{\text{dill}} = \log(1 + \sum_{m=1}^{M} \exp((s^{\text{noise}}_m - s^{\text{pixel}})/\tau))$$

其中 $s^{\text{noise}}_m = \text{sim}(f^{\text{noise}}_m, f^{\text{exo}})$, $s^{\text{pixel}} = \text{sim}(f^{\text{pixel}}, f^{\text{exo}})$

- **设计动机**：噪声吸收头显式隔离非affordance上下文，使affordance激活特征与干净的自中心定位特征对齐，同时推远噪声特征。形成"精确定位 → 去噪激活 → 更精确定位"的正反馈闭环

### 损失函数 / 训练策略

总损失：$\mathcal{L} = \lambda_{\text{cls}} \mathcal{L}_{\text{cls}} + \lambda_{\text{dill}} \mathcal{L}_{\text{dill}} + \lambda_{\text{pixel}} \mathcal{L}_{\text{pixel}} + \lambda_{\text{corr}} \mathcal{L}_{\text{corr}}$

其中 $\mathcal{L}_{\text{corr}}$ 对齐外中心和自中心之间的affordance相关性。端到端训练，输入分辨率224×224，聚类数 $K=4$，SGD优化器，学习率1e-3，单卡NVIDIA TITAN。

## 实验关键数据

### 主实验

AGD20K图像benchmark上的对比：

| 方法 | AGD20K-Seen KLD↓ | SIM↑ | NSS↑ | AGD20K-Unseen KLD↓ | SIM↑ | NSS↑ |
|------|------------------|------|------|---------------------|------|------|
| LOCATE (CVPR23) | 1.226 | 0.401 | 1.177 | 1.405 | 0.372 | 1.157 |
| WSMA (AAAI24) | 1.176 | 0.416 | 1.247 | 1.335 | 0.382 | 1.220 |
| INTRA (ECCV24) | 1.199 | 0.407 | 1.239 | 1.365 | 0.375 | 1.209 |
| **LoopTrans** | **1.088** | **0.445** | **1.322** | **1.247** | **0.403** | **1.315** |

HICO-IFF上：LoopTrans KLD=1.399, SIM=0.379, NSS=1.226，超越WSMA约10.5%

视频benchmark（EPIC/OPRA）上同样全面领先，弱监督设定和图像到视频泛化设定均表现最佳。

### 消融实验

AGD20K-Seen上的模块消融：

| 统一CAM | 像素对齐 | 去噪蒸馏 | KLD↓ | SIM↑ | NSS↑ |
|---------|---------|---------|------|------|------|
| ✗ | ✗ | ✗ | 1.318 | 0.384 | 1.135 |
| ✓ | ✗ | ✗ | 1.259 | 0.409 | 1.179 |
| ✗ | ✗ | ✓ | 1.251 | 0.392 | 1.196 |
| ✓ | ✓ | ✗ | 1.149 | 0.425 | 1.266 |
| ✓ | ✗ | ✓ | 1.222 | 0.405 | 1.183 |
| **✓** | **✓** | **✓** | **1.088** | **0.443** | **1.322** |

### 关键发现

1. 共享CAM单独带来+4.5%的KLD提升（Seen），通过跨视角协同有效促进知识提取
2. 像素对齐在共享CAM基础上进一步提升+8.7%，将粗激活精炼为区域完整的定位
3. 去噪蒸馏机制带来+5.1%的基线提升，通过建立闭环知识循环有效过滤背景噪声
4. 三个模块组合的效果超过各自独立相加，体现了闭环设计的协同增益
5. 对遮挡场景（如"sit on"、"catch"）的处理能力显著优于基于外观对齐的方法

## 亮点与洞察

- **闭环思想**：首次在affordance grounding中引入双向知识迁移，打破了单向exo→ego的惯性思维
- **问题本质**：认识到自中心图像（干净、物体中心）是被低估的"免费午餐"，可以反向帮助外中心知识提取
- **去噪蒸馏**：噪声吸收头的设计简洁优雅——通过显式隔离噪声模式来净化affordance激活
- **遮挡鲁棒性**：共享CAM使即使交互区域完全被人体遮挡的场景也能处理，这是之前方法的根本短板

## 局限与展望

- 聚类数K=4是固定的，不同物体的部件数不同（如椅子vs刀），自适应确定K值可能带来提升
- 伪掩码质量依赖DINO特征聚类的准确性，对纹理均匀的物体可能效果较差
- 噪声吸收头数M为超参，未提供具体消融；增加过多头可能导致噪声概念过度分裂
- 仅在affordance grounding场景下验证，闭环迁移思想能否推广到其他跨域任务待探索
- 视频扩展仅使用简单的LSTM，未利用时序注意力等更强的时序建模方法

## 相关工作与启发

- **Cross-view-AG**（CVPR22）和**LOCATE**（CVPR23）是主要基线，代表单向迁移范式的演进
- **CAM**的内在局限（只激活最显著区域）在多个弱监督任务中都是瓶颈，本文通过DINO聚类+像素解码的两步策略优雅解决
- 闭环/互馈思想在多模态学习中具有广泛启发性——任何两种模态/域之间都可能存在双向增强的空间

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 闭环知识迁移框架是该领域的概念性突破
- 实验充分度: ⭐⭐⭐⭐⭐ 图像+视频benchmark全面验证，12组消融实验详尽
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示直观，闭环流程阐述得当
- 价值: ⭐⭐⭐⭐ 在所有指标上全面超越SOTA，闭环迁移思想具有推广潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective](../../CVPR2026/image_restoration/rethinking_knowledge_transfer_in_image_quality_assessment_a_perceptual_preferenc.md)
- [\[ICCV 2025\] Blind2Sound: Self-Supervised Image Denoising without Residual Noise](blind2sound_self-supervised_image_denoising_without_residual_noise.md)
- [\[CVPR 2025\] Rotation-Equivariant Self-Supervised Method in Image Denoising](../../CVPR2025/image_restoration/rotation-equivariant_self-supervised_method_in_image_denoising.md)
- [\[NeurIPS 2025\] MoE-Gyro: Self-Supervised Over-Range Reconstruction and Denoising for MEMS Gyroscopes](../../NeurIPS2025/image_restoration/moe-gyro_self-supervised_over-range_reconstruction_and_denoising_for_mems_gyrosc.md)
- [\[CVPR 2025\] Generalized Recorrupted-to-Recorrupted: Self-Supervised Learning Beyond Gaussian Noise](../../CVPR2025/image_restoration/generalized_recorrupted-to-recorrupted_self-supervised_learning_beyond_gaussian_.md)

</div>

<!-- RELATED:END -->
