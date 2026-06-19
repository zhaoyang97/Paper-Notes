---
title: >-
  [论文解读] LS-ViT: Least-Squares Hessian Based Block Reconstruction for Low-Bit Post-Training Quantization of Vision Transformers
description: >-
  [CVPR 2026][VLM Efficiency][训练后量化] LS-ViT 把 ViT 块重建里"代表性 Hessian"的估计重新表述成一个最小二乘问题——用整个校准集上的 $(g, \Delta z)$ 对去拟合一个共享 Hessian，从而显式补回此前方法因"样本独立假设"而丢掉的协方差项，在 W2/A3、W2/A4 等超低比特下刷新 SOTA，且每个块只需一次反向传播，训练速度比 FIMA-Q 快 1.8–2.7 倍。
tags:
  - "CVPR 2026"
  - "VLM Efficiency"
  - "训练后量化"
  - "Transformer"
  - "块重建"
  - "Hessian 近似"
  - "最小二乘"
---

# LS-ViT: Least-Squares Hessian Based Block Reconstruction for Low-Bit Post-Training Quantization of Vision Transformers

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hwang_LS-ViT_Least-Squares_Hessian_Based_Block_Reconstruction_for_Low-Bit_Post-Training_Quantization_CVPR_2026_paper.html)  
**代码**: https://github.com/hhh7748/LS-ViT  
**领域**: 模型压缩 / 量化  
**关键词**: 训练后量化, Vision Transformer, 块重建, Hessian 近似, 最小二乘

## 一句话总结
LS-ViT 把 ViT 块重建里"代表性 Hessian"的估计重新表述成一个最小二乘问题——用整个校准集上的 $(g, \Delta z)$ 对去拟合一个共享 Hessian，从而显式补回此前方法因"样本独立假设"而丢掉的协方差项，在 W2/A3、W2/A4 等超低比特下刷新 SOTA，且每个块只需一次反向传播，训练速度比 FIMA-Q 快 1.8–2.7 倍。

## 研究背景与动机
**领域现状**：把 ViT 部署到端侧需要量化，而训练后量化（PTQ）只用少量无标注校准数据、成本远低于量化感知训练（QAT），是更实用的路线。其中基于块重建（block reconstruction）的方法在 4-bit 甚至更低比特上表现最好：它们从任务损失的泰勒展开里保留二阶项，把"量化引起的损失增量"近似成 $\frac{1}{2}\Delta z^\top H^{(z)} \Delta z$，于是核心就落在如何估计这个 Hessian $H^{(z)}$ 上。

**现有痛点**：完整 Hessian 是 $d\times d$、算不起，必须近似。最近的 APHQ-ViT、FIMA-Q 用"对多个样本的梯度信息取平均"来得到一个稳定的代表性 Hessian，但它们都隐含了一个**样本之间相互独立**的假设。本文指出，Figure 2 显示同一通道的 Hessian 对角/秩一分量在不同样本间分布带很宽——样本间差异很大，独立假设并不成立。

**核心矛盾**：对 $\mathbb{E}[H^{(z)}\Delta z] = \mathbb{E}[H^{(z)}]\mathbb{E}[\Delta z] + \mathrm{Cov}(H^{(z)}, \Delta z)$ 而言，独立假设等于直接扔掉了协方差项 $\mathrm{Cov}(H^{(z)}, \Delta z)$。比特越低，扰动 $\Delta z$ 越大，被忽略的协方差诱导误差 $H^{(z)}\Delta z - H\Delta z$ 越不可忽略（Figure 4），估出来的 Hessian 就越"不具代表性"。此外，这些方法靠多次反向传播来估 Hessian，额外计算开销大。

**本文目标**：找到一个在整个校准集上都"最具代表性"的单一 Hessian，既补回协方差项、又把估计成本压到每个块只算一次梯度。

**切入角度**：既然要找一个被所有样本共享、又能最好地解释所有 $(g, \Delta z)$ 观测的矩阵，这天然就是一个**最小二乘回归**问题——不是去平均，而是去拟合。

**核心 idea**：用最小二乘解 $\widehat{H} := \arg\min_H \mathbb{E}\big[\|H^{(z)}\Delta z - H\Delta z\|^2\big]$ 取代"取平均"，得到显式最小化近似残差、且只需单次反向的 Hessian 估计，落地为 LS-ViT。

## 方法详解

### 整体框架
LS-ViT 对每个 Transformer 块做两阶段顺序处理：**第一阶段**在重建之前，用校准集上所有样本的梯度 $g$ 和块输出扰动 $\Delta z$，通过最小二乘求出一个代表性 Hessian；**第二阶段**用这个固定的 Hessian 当重建度量，沿用 QDrop 基线去优化权重取整（AdaRound）和激活缩放参数。整套流程基于标准的均匀仿射量化器（channel-wise 量化权重、layer-wise 量化激活），不依赖任何 ViT 专用量化器，因此推理期与普通 PTQ 同样硬件友好。

它的出发点是块重建的标准目标：把量化带来的任务损失增量近似为 $\mathcal{L}_\mathrm{recon}(\Delta z) = \frac{1}{2}\Delta z^\top H^{(z)} \Delta z$。对它关于 $\Delta z$ 求导得到关系式 $g = H^{(z)}\Delta z$（假设 $H^{(z)}$ 对称且在 $\Delta z$ 上局部常数，对称性由 Transformer 各阶连续可导经 Clairaut 定理保证，局部常数性同 APHQ-ViT/FIMA-Q 一样把 Hessian 视为预训练模型的固有属性）。由于要找一个被所有样本共享的 $H$，这个等式应当对校准集里**每一对** $(g, \Delta z)$ 都成立——于是估计 Hessian 就变成"找一个矩阵最好地拟合这一堆 $(g, \Delta z)$ 对"的超定方程组问题，用最小二乘求闭式解。这是纯粹的损失度量改进，没有多模块串联的 pipeline，因此不配框架图，用公式讲清即可。

### 关键设计

**1. 最小二乘 Hessian 估计：把"代表性 Hessian"从取平均改成拟合，补回协方差项**

这是全文的灵魂，直接针对"样本独立假设丢掉协方差项"这个痛点。FIMA-Q 也用 $g = H^{(z)}\Delta z$ 来估 Hessian，但它假设样本独立、对两边各自取期望，等价于丢掉了 $\mathrm{Cov}(H^{(z)}, \Delta z)$。LS-ViT 不取平均，而是把估计写成最小化期望平方残差：

$$\widehat{H} := \arg\min_H \mathbb{E}\big[\|H^{(z)}\Delta z - H\Delta z\|^2\big]$$

最小二乘解会自动把样本间的协方差结构吸收进来。论文用对角情形给出了最干净的对照：$\widehat{H}_{i,i} = \frac{\mathbb{E}[g_i\Delta z_i]}{\mathbb{E}[\Delta z_i^2]} = \frac{\mathrm{Cov}(g_i,\Delta z_i) + \mathbb{E}[g_i]\mathbb{E}[\Delta z_i]}{\mathrm{Var}(\Delta z_i) + (\mathbb{E}[\Delta z_i])^2}$。把 $\mathrm{Cov}(g_i,\Delta z_i)=0$、$\mathrm{Var}(\Delta z_i)=0$ 代进去就退化成 FIMA-Q-D——这一步把"FIMA-Q 只是本文的一个独立性特例"讲得明明白白。比特越低、$\Delta z$ 越大，这两项越不能忽略，所以 LS-ViT 在超低比特上优势越明显。

**2. 对角最小二乘 Hessian（LSH-D）：用闭式比值稳定地刻画逐通道敏感度**

针对"逐样本估 Hessian 数值不稳"的问题。最朴素的逐样本对角估计是 $H^{(z)}_{i,i} = g_i/\Delta z_i$，但 $\Delta z_i$ 常常很小，除法会放大噪声、方差极高（消融里 ViT-S 只有 23.59%）。在对角近似下 $g_i = H^{(z)}_{i,i}\Delta z_i$ 是一个对每个通道 $i$ 的超定方程组（$N$ 个样本方程、1 个未知数），最小二乘闭式解为

$$\widehat{H}_{i,i} = \frac{\sum_{n=1}^{N} g_i^{(n)} \Delta z_i^{(n)}}{\sum_{n=1}^{N} (\Delta z_i^{(n)})^2} = \overline{g_i \Delta z_i}\big/ \overline{\Delta z_i^2}$$

对应的重建损失 $\mathcal{L}_\mathrm{LSH,D}(\Delta z) = \frac{1}{2}\sum_i \Delta z_i^2 \cdot (\overline{g_i\Delta z_i}/\overline{\Delta z_i^2})$。它通过跨样本聚合既保留了样本间关系、又压低了方差，刻画的是每个参数的个体敏感度。

**3. 秩一低秩最小二乘 Hessian（LSH-L）：补上对角近似抓不住的主导非对角交互**

对角近似只看自身、丢了通道间的交叉项，因此再补一个低秩分量。$N$ 个样本最多能撑起秩 $N$ 的 Hessian，但计算重建损失代价为 $O(Nd)$，本文只取最有性价比的秩一近似 $H^{(z)} = uu^\top$。直接对 $g^{(n)} = uu^\top \Delta z^{(n)}$ 做最小二乘会引入 $u$ 的高阶项，于是两边同乘 $\Delta z^{(n)\top}$ 得 $\Delta z^{(n)\top} g^{(n)} = (u^\top \Delta z^{(n)})^2$，再重写成 $g^{(n)} = u\sqrt{\Delta z^{(n)\top} g^{(n)}}$，最小二乘解为

$$\widehat{u} = \frac{\sum_{n} g^{(n)}\sqrt{\Delta z^{(n)\top} g^{(n)}}}{\sum_{n} \Delta z^{(n)\top} g^{(n)}} = \overline{g\sqrt{\Delta z^\top g}}\big/\overline{\Delta z^\top g}$$

最终 LS-ViT 把两者相加 $\mathcal{L}_\mathrm{LSH}(\Delta z) = \mathcal{L}_\mathrm{LSH,D}(\Delta z) + \mathcal{L}_\mathrm{LSH,L}(\Delta z)$：对角项管逐参数敏感度、秩一项管主导的非对角交互，二者互补，组合效果最好。关键是整套估计只需每个块**一次反向传播**，省掉了 FIMA-Q 那种多次全模型计算和重建期每步的高秩运算。

### 损失函数 / 训练策略
重建阶段沿用 QDrop 的 drop 机制前向得到 $\Delta z'$，以 $\mathcal{L}_\mathrm{LSH}(\Delta z')$ 为目标反传，用 AdaRound 更新权重取整、同时更新激活缩放参数，默认 1024 张校准图、batch 32、20k 迭代，权重学习率 1e-3、激活 4e-5。

## 实验关键数据

### 主实验
ImageNet 图像分类，七个 ViT/DeiT/Swin 模型，Top-1 精度（%）。对照中 "\*" 表示只在重建度量上不同、其余设置完全一致：

| 设置 (W/A) | 模型 | FIMA-Q* | APHQ-ViT | LS-ViT (本文) |
|------|------|---------|----------|---------------|
| 2/4 | ViT-S | 56.76 | 56.04 | **58.48** |
| 2/4 | Swin-S | 70.64 | 71.75 | **73.89** |
| 2/4 | Swin-B | 72.36 | 72.64 | **74.95** |
| 2/3 | ViT-B | 61.15 | 58.06 | **62.89** |
| 2/3 | Swin-S | 59.57 | 62.90 | **65.77** |
| 3/3 | ViT-S | 64.09 | 63.17 | **64.10** |

W2/A4 下相对 FIMA-Q 平均 +1.46%p（Swin-S +3.25、Swin-B +2.59）；W2/A3 平均 +2.89%p（Swin-S +6.20、Swin-B +4.58）；W3/A3 平均 +0.17%p。比特越低优势越大，验证了协方差项随比特下降越关键；Swin 因重要特征跨样本方差大，受益尤其明显。COCO 检测/分割（W4/A4，Mask R-CNN / Cascade Mask R-CNN）上也对 FIMA-Q 有 +0.1～+0.2 AP 的一致小幅提升。

### 消融实验
对角 Hessian 估计方式对比（W3/A3，Top-1 %）：

| 估计方式 | 公式 | ViT-S | Swin-S |
|------|------|-------|--------|
| 逐样本 $g_i/\Delta z_i$ | 朴素除法 | 23.59 | 69.42 |
| 逐样本 $g_i$ | 去掉除法 | 58.16 | 76.02 |
| FIMA-Q-D | $\mathbb{E}[g_i]/\mathbb{E}[\Delta z_i]$ | 60.02 | 75.08 |
| LS-ViT-D (本文) | $\mathbb{E}[g_i\Delta z_i]/\mathbb{E}[\Delta z_i^2]$ | **63.25** | **77.29** |

训练成本（单 GPU，分钟）：FIMA-Q 用 15 次全模型计算才能达到的精度，LS-ViT 单次即可超过；把 FIMA-Q 的预算从 15 降到 1，其精度掉 0.15–1.41%p。训练时间上 LS-ViT 比 FIMA-Q 快 1.8×（DeiT-T）到 2.7×（Swin-B），与 QDrop 基本持平。

### 关键发现
- 协方差项在超低比特下贡献最大：从 3/3 到 2/3，LS-ViT 相对 FIMA-Q 的平均增益从 +0.17%p 扩大到 +2.89%p。
- 朴素逐样本 $g_i/\Delta z_i$ 因 $\Delta z_i$ 过小而极不稳定（ViT-S 仅 23.59%），单纯去掉除法（用 $g_i$）就平均回升 16.81%p，再做最小二乘聚合又比 FIMA-Q-D 平均 +1.22%p。
- 对 Swin 这类"重要特征跨样本方差大"的架构，能稳健捕捉高方差的 LS-ViT 收益最大。

## 亮点与洞察
- 把"代表性 Hessian 估计"从启发式取平均提升为有闭式解的最小二乘回归，并用 $\mathbb{E}[g\Delta z]/\mathbb{E}[\Delta z^2]$ 的协方差/方差展开，把"FIMA-Q 是独立性特例"讲得严丝合缝——这种"用更一般的估计统一旧方法"的论证范式很值得复用。
- 真正点睛的是诊断：作者没有去发明新量化器，而是指出所有块重建方法共享的一个隐含错误（丢协方差项），并证明它恰好在最需要的超低比特场景被放大。
- "每块单次反向 + 标准均匀量化器"让方法既准又快、即插即用，工程落地友好；秩一改写（两边同乘 $\Delta z^\top$ 避开高阶项）是处理 $uu^\top$ 型最小二乘的实用技巧，可迁移到其它低秩拟合问题。

## 局限与展望
- 低秩近似只取到秩一，作者明确因 $O(Nd)$ 成本放弃更高秩；高方差块上更高秩是否值得，留作开放问题。
- "Hessian 局部常数"的假设在更激进的扰动（如 W2/A2）下是否仍成立，论文未深入；极端比特边界的适用范围有待验证。
- 检测/分割上的提升只有 +0.1～+0.2 AP，相对分类的大幅增益偏小，密集预测任务的收益还较薄。
- 多处推导依赖补充材料（对角/秩一闭式解、各分量贡献分析），正文给出的部分公式细节 ⚠️ 以原文与补充材料为准。

## 相关工作与启发
- **vs FIMA-Q**: 二者都用 $g = H^{(z)}\Delta z$ 估 Hessian，但 FIMA-Q 假设样本独立、对两边取期望从而丢掉协方差项，且其低秩近似要多次全模型计算；LS-ViT 用最小二乘拟合所有样本对、显式保留协方差，单次反向即可，更准更快。
- **vs APHQ-ViT**: APHQ-ViT 用平均扰动 Hessian 并配合 MLP 重建；LS-ViT 在"仅替换重建度量"的公平对照下（APHQ-ViT(-)）一致更优，说明增益来自度量本身而非额外结构。
- **vs BRECQ / QDrop / AdaRound**: 这些用样本无关常数对角或逐样本平方梯度近似 Hessian，本质是更粗的近似；LS-ViT 在同一 QDrop 基线上把 Hessian 估计做精，是对它们的直接升级。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把代表性 Hessian 估计统一为最小二乘并补回协方差项，视角清晰且把旧方法纳为特例。
- 实验充分度: ⭐⭐⭐⭐ 覆盖七模型、多比特、检测/分割与训练成本，消融到位；检测增益偏小、极端比特未探。
- 写作质量: ⭐⭐⭐⭐ 动机—诊断—公式推导链条顺畅，但大量关键推导压在补充材料。
- 价值: ⭐⭐⭐⭐⭐ 即插即用、超低比特刷 SOTA 且训练快 2 倍量级，端侧部署实用性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] VLM-PTQ: Efficient Post-Training Quantization for Large Vision-Language Models](vlm-ptq_efficient_post-training_quantization_for_large_vision-language_models.md)
- [\[CVPR 2026\] Fine-Grained Post-Training Quantization for Large Vision Language Models with Quantization-Aware Integrated Gradients](fine-grained_post-training_quantization_for_large_vision_language_models_with_qu.md)
- [\[CVPR 2026\] Saliency-Driven Token Merging for Vision Transformers](saliency-driven_token_merging_for_vision_transformers.md)
- [\[CVPR 2026\] VQRAE: Representation Quantization Autoencoders for Multimodal Understanding, Generation and Reconstruction](vqrae_representation_quantization_autoencoders_for_multimodal_understanding_gene.md)
- [\[CVPR 2026\] QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer](qvggt_post-training_quantized_visual_geometry_grounded_transformer.md)

</div>

<!-- RELATED:END -->
