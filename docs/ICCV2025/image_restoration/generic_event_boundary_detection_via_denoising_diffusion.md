---
title: >-
  [论文解读] Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)
description: >-
  [ICCV 2025][图像恢复][通用事件边界检测] DiffGEBD 首次将扩散模型引入通用事件边界检测（GEBD），通过将边界预测建模为从随机噪声到合理边界分布的去噪过程，利用 Classifier-Free Guidance 控制预测多样性，并提出了对称 F1 和 Diversity Score 两项新评估指标来衡量多预测场景下的质量与多样性。
tags:
  - "ICCV 2025"
  - "图像恢复"
  - "通用事件边界检测"
  - "扩散模型"
  - "Classifier-Free Guidance"
  - "多样性评估"
  - "时序自相似性"
---

# Generic Event Boundary Detection via Denoising Diffusion (DiffGEBD)

**会议**: ICCV 2025  
**arXiv**: [2508.12084](https://arxiv.org/abs/2508.12084)  
**代码**: [https://cvlab.postech.ac.kr/research/DiffGEBD](https://cvlab.postech.ac.kr/research/DiffGEBD)  
**领域**: 图像复原  
**关键词**: 通用事件边界检测, 扩散模型, Classifier-Free Guidance, 多样性评估, 时序自相似性

## 一句话总结

DiffGEBD 首次将扩散模型引入通用事件边界检测（GEBD），通过将边界预测建模为从随机噪声到合理边界分布的去噪过程，利用 Classifier-Free Guidance 控制预测多样性，并提出了对称 F1 和 Diversity Score 两项新评估指标来衡量多预测场景下的质量与多样性。

## 研究背景与动机

**领域现状**：通用事件边界检测（GEBD）旨在将视频自然地分割为有意义的事件片段，识别其转折点。与传统的动作识别和时序动作检测不同，GEBD 关注的是类别无关的通用事件边界。Kinetics-GEBD 是该任务的标准基准数据集，每个视频提供多个标注者的标注。

**现有痛点**：
   - GEBD 中事件边界的定义具有**天然的主观性和多样性**——不同标注者对同一视频的边界认知存在差异
   - 然而现有方法（UBoCo、DDM-Net、LCVS、SC-Transformer、BasicGEBD、EfficientGEBD 等）全部采用**确定性模型**，只能为每个视频输出**单一预测结果**，忽略了边界的多样性
   - 传统 F1 评估指标只衡量单一预测与多个标注之间的匹配，无法处理"多对多"对齐的评估场景

**核心矛盾**：GEBD 的 ground truth 本身就是多样的（多个标注者给出不同答案），但模型只能给出确定性的单一答案。这种"多样标注 vs. 确定性预测"的不对等使得模型无法真正反映人类判断的变异性。

**本文切入角度**：扩散模型天然适合这一挑战——通过改变初始噪声就能采样出不同的输出。DiffGEBD 将 GEBD 重新定义为生成式问题：给定视频条件特征，从随机噪声中迭代去噪出合理的事件边界。

## 方法详解

### 整体框架

DiffGEBD 包含三个核心组件：

- **Backbone**（ResNet-50）：从输入视频 $\bm{V} \in \mathbb{R}^{L \times H \times W \times 3}$ 提取视觉特征 $\bm{F} \in \mathbb{R}^{L \times D}$
- **时序自相似性编码器**（Encoder $f$）：通过滑动窗口时序自相似性模块捕获相邻帧之间的动态变化，生成时序嵌入 $\bm{E} \in \mathbb{R}^{L \times C}$
- **去噪解码器**（Decoder $h$）：基于 Transformer Encoder 层，将含噪的边界标签 $\bm{y}_t$ 在时序嵌入 $\bm{E}$ 的条件下去噪为边界预测

训练时：随机采样扩散时间步 $t$，向 ground-truth 边界标签 $\bm{y}_0 \in \{0,1\}^L$ 添加高斯噪声得到 $\bm{y}_t$，解码器学习从 $\bm{y}_t$ 重建 $\bm{y}_0$。每个 epoch 内轮换使用不同标注者的标注。

推理时：从随机高斯噪声 $\hat{\bm{y}}_T$ 出发，通过 DDIM 采样逐步去噪至 $\hat{\bm{y}}_0$。通过初始化不同的随机噪声，可以用**同一模型生成多个不同的合理预测**。

### 关键设计

1. **时序自相似性编码器**：

    - 功能：用 1D 卷积 + 滑动窗口时序自相似性矩阵 + FCN + 2D 池化处理视觉特征
    - 核心思路：自相似性矩阵衡量临近帧之间的语义一致性变化，边界位置对应不一致性高峰
    - 设计动机：相比直接使用 backbone 输出的视觉特征 $\bm{F}$，时序自相似性特征 $\bm{E}$ 能更好地捕捉细粒度帧间变化。消融实验（Table 2）证实：用 $\bm{F}$ 做条件时 F1$_{\text{sym}}$ 仅 68.5，用 $\bm{E}$ 则为 74.0

2. **Classifier-Free Guidance (CFG)**：

    - 训练：以概率 $p=0.1$ 随机将条件特征 $\bm{E}$ 替换为零矩阵，联合训练条件/无条件两种模式
    - 推理：$\hat{\bm{y}}_t = (1+w)\hat{\bm{y}}_t^c - w\hat{\bm{y}}_t^u$，其中 $w$ 是引导权重
    - 大 $w$ → 输出更确定性、更依赖视觉条件；小 $w$ → 更多样化的预测
    - 最优 $w=0.6$（最大化 F1$_{\text{sym}}$），确定性评估时用 $w=4.0$

3. **多样性感知评估指标**：

    - **对称 F1（F1$_{\text{sym}}$）**：结合 Pred-to-GT 对齐分数（每个预测与最优 GT 匹配）和 GT-to-Pred 对齐分数（每个 GT 是否被某个预测覆盖），取调和平均
    - **Diversity Score**：计算 $N_P$ 个预测之间的平均两两不相似度 $\frac{1}{N_P^2}\sum_{i,j}(1 - \text{F1}(\hat{Y}_i, \hat{Y}_j))$

### 损失函数/训练策略

- **损失函数**：均方误差 $\mathcal{L} = \frac{1}{L}\sum_{l=1}^{L}(\bm{y}_{0,l} - \hat{\bm{y}}_{t,l})^2$
- 视频统一采样至 100 帧
- Backbone：ImageNet 预训练 ResNet-50（冻结）
- 编码器：BasicGEBD-L4；解码器：6 层 Transformer
- 扩散步数设为 $T=1000$（训练），DDIM 采样 32 步（推理）
- CFG 丢弃概率 $p=0.1$
- Kinetics-GEBD 每视频 5 个标注，按 F1 一致性分数选取前 4 个

## 实验关键数据

### 多样性感知评估（核心实验）

Kinetics-GEBD 数据集，每个模型生成 $N_P=5$ 个预测：

| 方法 | F1$_{\text{sym}}$ | F1$_{\text{p2g}}$ | F1$_{\text{g2p}}$ | Diversity |
|------|:---:|:---:|:---:|:---:|
| Temporal Perceiver | 69.4 | 72.2 | 67.4 | 14.6 |
| SC-Transformer | 72.9 | 74.9 | 71.6 | 18.9 |
| BasicGEBD | 72.2 | 74.5 | 70.6 | 18.6 |
| EfficientGEBD | 72.6 | 76.0 | 70.2 | 14.9 |
| **DiffGEBD** | **74.0** | 75.6 | **72.9** | **20.4** |

注：其他方法为确定性模型，通过 5 次随机初始化训练获得多个预测；DiffGEBD 的多样预测来自同一模型的不同初始噪声。

### 传统评估

| 方法 | Kinetics-GEBD F1@0.05 | TAPOS F1@0.05 |
|------|:---:|:---:|
| EfficientGEBD | 78.3 | 63.1 |
| DyBDet | **79.6** | 62.5 |
| DiffGEBD ($w=4.0$) | 78.4 | **65.8** |

### 消融实验

**条件特征的影响**：$\bm{E}$（时序自相似性）vs. $\bm{F}$（原始视觉特征）→ F1$_{\text{sym}}$ 74.0 vs. 68.5

**推理步数**：1 步 → F1$_{\text{sym}}$ 64.0；2 步 → 72.3；8 步 → 73.7；32 步 → 74.0（最佳）

**标注数量**：1→4 个标注，性能持续提升；使用全部 5 个时因低一致性标注引入噪声而略降

**CFG 权重 $w$**：$w=0.6$ 时 F1$_{\text{sym}}$ 最高；$w$ 增大→F1$_{\text{p2g}}$ 先升后降，Diversity 持续下降

### 关键发现

- DiffGEBD 在 F1$_{\text{sym}}$、F1$_{\text{g2p}}$ 和 Diversity 三项指标上均 SOTA
- EfficientGEBD 虽然 F1$_{\text{p2g}}$ 最高，但其 diversity 极低（14.9），说明预测精确但缺乏多样性
- 高 diversity 不必然带来高 F1$_{\text{g2p}}$——多样性必须伴随合理性才有意义
- TAPOS 上 DiffGEBD 大幅超越 SOTA（65.8 vs. 63.1），说明生成式方法对更困难的数据集优势更明显
- 扩散模型只需一次训练，无需像确定性模型那样需要多次训练来获得多预测

## 亮点与洞察

- **问题建模角度新颖**：将 GEBD 从"判断哪些位置是边界"的判别问题转化为"从噪声中生成合理的边界分布"的生成问题。这个视角转换非常巧妙，充分利用了扩散模型的生成多样性来匹配 GEBD 任务中的标注多样性。
- **评估指标设计严谨**：对称 F1 + Diversity Score 组成的评估协议填补了多预测场景的评估空白。对称 F1 同时考虑 Pred→GT 和 GT→Pred 两个方向，比单向 F1 更公平。
- **CFG 用法出色**：原本用于图像生成中平衡质量与多样性的 CFG 被精准迁移到边界检测中，且一个连续参数 $w$ 就能在"确定性预测"和"多样性预测"之间平滑切换。
- **实验设计公平**：将确定性方法的多次训练结果与 DiffGEBD 的单模型多次采样进行对比，同时展示传统评估结果，确保了比较的完整性。

## 局限与展望

- **推理效率**：DDIM 需要 32 步迭代去噪，推理速度慢于确定性方法。如果需要实时边界检测场景，这是瓶颈。
- **Kinetics-GEBD 传统 F1 未超 SOTA**：DyBDet（79.6）仍优于 DiffGEBD（78.4），说明生成式方法在单一最佳预测场景下仍有提升空间。
- **模型泛化性**：实验只在 Kinetics-GEBD 和 TAPOS 上评估，缺乏跨域泛化性验证。
- **多样性的可控性粒度较粗**：CFG 权重 $w$ 是全局参数，无法分区域控制多样性程度。一些明确的边界可能不需要多样性，而不确定区域可能需要更高多样性。
- **扩散模型的计算成本**：相比轻量级的确定性模型（如 EfficientGEBD），扩散框架需要编码器 + 解码器 + 多步推理，计算 overhead 显然更高。

## 亮点与洞察

## 局限与展望

## 相关工作与启发

## 评分
- 新颖性: 待评
- 实验充分度: 待评
- 写作质量: 待评
- 价值: 待评

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] From Events to Clarity: The Event-Guided Diffusion Framework for Dehazing](../../CVPR2026/image_restoration/from_events_to_clarity_the_event-guided_diffusion_framework_for_dehazing.md)
- [\[ECCV 2024\] EDformer: Transformer-Based Event Denoising Across Varied Noise Levels](../../ECCV2024/image_restoration/edformer_transformer-based_event_denoising_across_varied_noise_levels.md)
- [\[CVPR 2026\] Similarity-Consistent Likelihood Diffusion enables Hidden Person Detection from Wall Reflections](../../CVPR2026/image_restoration/similarity-consistent_likelihood_diffusion_enables_hidden_person_detection_from_.md)
- [\[ICCV 2025\] Low-Light Image Enhancement using Event-Based Illumination Estimation (RetinEV)](low-light_image_enhancement_using_event-based_illumination_estimation.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](../../ICML2026/image_restoration/structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)

</div>

<!-- RELATED:END -->
