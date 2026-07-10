---
title: >-
  [论文解读] RefAlign: Representation Alignment for Reference-to-Video Generation
description: >-
  [ECCV 2026][视频生成][参考-to-视频生成] RefAlign 提出参考对齐损失（RA loss），在训练时将 DiT 参考分支的中间特征显式对齐到视觉基础模型（VFM）的语义空间——正项拉近同主体特征保证身份一致性，负项推远不同主体特征增强语义判别力——推理时移除对齐模块实现零额外开销，在 OpenS2V-Eval 上 TotalScore 达到 SOTA（60.42%），有效缓解了参考-to-视频生成中的 copy-paste 伪影和多主体混淆。
tags:
  - "ECCV 2026"
  - "视频生成"
  - "参考-to-视频生成"
  - "表示对齐"
  - "Transformer"
  - "视觉基础模型"
  - "多模态条件"
---

# RefAlign: Representation Alignment for Reference-to-Video Generation

**会议**: ECCV 2026  
**arXiv**: [2603.25743](https://arxiv.org/abs/2603.25743)  
**代码**: [https://github.com/gudaochangsheng/RefAlign](https://github.com/gudaochangsheng/RefAlign) (有)  
**领域**: 视频生成 / 扩散模型  
**关键词**: 参考-to-视频生成, 表示对齐, 扩散Transformer, 视觉基础模型, 多模态条件

## 一句话总结
RefAlign 提出参考对齐损失（RA loss），在训练时将 DiT 参考分支的中间特征显式对齐到视觉基础模型（VFM）的语义空间——正项拉近同主体特征保证身份一致性，负项推远不同主体特征增强语义判别力——推理时移除对齐模块实现零额外开销，在 OpenS2V-Eval 上 TotalScore 达到 SOTA（60.42%），有效缓解了参考-to-视频生成中的 copy-paste 伪影和多主体混淆。

## 研究背景与动机
参考-to-视频生成（R2V）是可控视频合成的重要范式：给定文本提示和多张参考图像，生成既遵循指令又保持主体身份外观的视频，在个性化广告、虚拟试穿等场景有广泛应用。现有 R2V 方法普遍采用"双流参考"范式——一方面用 3D VAE 提取参考特征提供低层细节，另一方面引入额外编码器（如 CLIP、MLLM）注入高层语义线索，试图缓解 VAE 潜在空间中的像素级信息泄漏。然而，这些额外语义特征与 VAE 潜在特征来自异构编码器，存在本质的模态失配（modality mismatch）：DiT 内部源自 VAE 潜在表示的参考特征，与外部注入的语义参考表示之间存在系统性偏差，隐式对齐难以从根本上消除这种偏差。这导致两个典型问题：copy-paste 伪影（生成视频过度复制参考图像的像素细节）和多主体混淆（不同参考主体的外观相互干扰、混合）。

本文作者通过 t-SNE 可视化发现了一个关键现象：DiT 对参考图像编码后的特征分布高度纠缠、不同参考之间重叠严重，而 DINOv3 提取的特征具有显著更强的类间可分性——同一参考的特征紧凑一致，不同参考的特征彼此分离。受此启发，RefAlign 不再额外注入语义特征，而是通过显式的参考对齐损失，直接约束 DiT 参考分支的中间特征向 VFM 特征空间靠拢，从根本上增强参考表示的语义判别力。**核心 idea**：用 VFM 特征作为"语义锚点"，在训练时拉近 DiT 参考特征与同主体 VFM 特征的距离、推远与不同主体 VFM 特征的距离，从而在不增加推理开销的前提下，让 DiT 自己学会生成具有强判别力的参考表示。

## 方法详解

### 整体框架
RefAlign 以 Wan2.1 的 T2V DiT 为主干，在其上进行微调。给定文本提示 $c_{\text{text}}$、$M$ 张参考图像 $I = \{I_m\}_{m=1}^{M}$ 和目标视频 $x$，文本经冻结的 T5 编码器编码为 $\hat{c}_{\text{text}}$，参考图像和目标视频经冻结的 Wan-VAE 编码为 $\hat{I}$ 和 $z_0$。DiT（共 $L$ 层 transformer block）接收噪声潜在变量 $z_t$、文本条件、参考条件和时间步 $t$，输出速度预测 $\varepsilon_{\Theta}(z_t, \hat{c}_{\text{text}}, \hat{I}, t)$，按 Rectified Flow 目标训练。

RefAlign 的核心改动发生在训练阶段：对 DiT 前 $K$ 层 self-attention 中产生的参考图像 token 特征 $h^{(l)}$（$l \le K$），通过一个轻量 MLP 投影器 $\Psi_{\text{proj}}$ 映射到与 VFM 特征相同的维度 $\hat{h}^{(l)} = \Psi_{\text{proj}}(h^{(l)})$，然后与冻结 VFM 提取的参考图像特征 $f = \{\varepsilon_{\text{VFM}}(I_m)\}_{m=1}^{M}$ 计算参考对齐损失（RA loss）。推理时，VFM 编码器和 MLP 投影器被完全移除，模型仅使用标准 classifier-free guidance 采样，不引入任何额外计算开销。

### 关键设计

**1. RA loss 正项：拉近同主体特征以增强身份一致性**

DiT 参考分支特征来自 VAE 潜在空间，像素级信息占主导，不同参考图像的特征在空间中高度纠缠（t-SNE 图显示重叠严重），导致模型难以区分"这个 token 属于主体 A 还是主体 B"。RefAlign 的正项对齐直击这一痛点：对于每个主体 $m$，将其 DiT 参考 token $\hat{h}^{(l),m}$ 与对应 VFM 特征 $f^m$ 逐 patch 计算余弦距离并最小化：

$$\mathcal{L}_{\text{pos}}^{(l)} = \frac{1}{M}\sum_{m=1}^{M}\frac{1}{N}\sum_{n=1}^{N}\left(1 - \cos\left(\hat{h}^{(l),m}_n, f_n^m\right)\right)$$

正项的作用是让 DiT 学会：同一个主体的参考 token，不管光照、姿态、背景如何变化，在特征空间里都应该靠近该主体在 VFM 空间中的"语义锚点"。这与 REPA 的对齐有本质区别——REPA 对齐的是从噪声中恢复的目标表示，目的是加速收敛；RefAlign 对齐的是干净的参考条件表示，目的是增强条件的语义质量。正项在只有一个参考图像时（$M=1$）退化为唯一的对齐信号，此时 $\mathcal{L}_{\text{neg}} = 0$。

**2. RA loss 负项：推远不同主体特征以消除多主体混淆**

仅靠正项对齐存在一个隐患：如果不同主体的 VFM 特征本来就有一定相似性（如两只不同品种的狗），正项可能把所有参考特征都拉向一个模糊的平均区域，反而加剧多主体混淆。为此，RefAlign 引入带 margin $\delta$ 的负项对齐：对主体 $m$ 的 DiT 参考 token 和不同主体 $m' \neq m$ 的 VFM 特征，强制它们的余弦距离超过 margin，否则施加惩罚：

$$\mathcal{L}_{\text{neg}}^{(l)} = \frac{1}{M(M-1)}\sum_{m=1}^{M}\sum_{\substack{m'=1 \\ m'\neq m}}^{M}\frac{1}{N^2}\sum_{n=1}^{N}\sum_{n'=1}^{N}\left[\delta - \left(1 - \cos\left(\hat{h}^{(l),m}_n, f_{n'}^{m'}\right)\right)\right]_{+}$$

其中 $[x]_{+} = \max(x, 0)$。消融实验表明，去掉负项后 TotalScore 从 55.73% 跌至 51.75%，FaceSim 和 NaturalScore 显著下降，定性结果中多主体性别属性出现明显混淆——说明负项在维持主体间语义边界上不可或缺。

**3. 仅训练时对齐、推理时零开销的旁路设计**

一个直接的想法是把 VFM 特征作为额外输入喂给 DiT（如配置 D：VAE + DINOv3 双编码器输入），这确实比纯 VAE 输入有提升（TotalScore 49.93% → 52.15%），但远不如 RA loss（55.73%），且推理时需要额外运行 VFM 编码器。RefAlign 的关键洞察是：**VFM 特征的价值不在于"多一条信息通道"，而在于作为"表征学习的目标"**——通过 RA loss 将 VFM 的语义判别力蒸馏到 DiT 参考分支的参数中，推理时 DiT 自己就能产出具有强判别力的参考表示。这一设计使 RefAlign 在推理时与标准 Wan2.1 完全等价，零额外参数、零额外计算。此外，VFM 仅作为训练时的对齐目标，其编码器选择在推理时不再构成约束——实验显示 DINOv3-B/L/H+ 三种规模的性能波动仅 0.33%-0.43%，说明对齐策略对编码器规模鲁棒。

### 损失函数 / 训练策略

完整训练目标由 Rectified Flow 损失和参考对齐损失加权组合：

$$\mathcal{L} = \mathcal{L}_{\text{RF}} + \eta \mathcal{L}_{\text{RA}}, \quad \mathcal{L}_{\text{RA}} = \frac{1}{K}\sum_{l=1}^{K}\left(\mathcal{L}_{\text{pos}}^{(l)} + \lambda \mathcal{L}_{\text{neg}}^{(l)}\right)$$

其中 $\mathcal{L}_{\text{RF}} = \mathbb{E}_{z_0, \epsilon, t}\left[\left\|\varepsilon_{\Theta}(z_t, \hat{c}_{\text{text}}, \hat{I}, t) - (\epsilon - z_0)\right\|^2_2\right]$，$\eta$ 和 $\lambda$ 分别控制 RA 损失的整体权重和负项相对权重（默认均设为 1.0）。对齐深度 $K$ 在实验中确定为 9 层——TotalScore 随深度呈"先升后降"趋势，过浅则对齐信号不足，过深则 FaceSim 单调下降（身份一致性被过度压制）。

训练分两阶段：第一阶段用 OpenS2V 中的 200K 常规配对样本（参考图像与目标视频主体一致）学习参考条件建模；第二阶段用 Phantom-Data 中的 160K 跨配对样本（参考图像与目标视频主体不一致）缓解 copy-paste 伪影。常规对与跨对比例为 6:4。训练时对常规对的参考图像施加随机旋转、缩放、水平翻转、仿射变换（含剪切）、高斯模糊和颜色抖动等数据增强。优化器使用 AdamW（$\beta_1=0.9$, $\beta_2=0.999$, weight decay=0.01），学习率 5e-5，全局 batch size 128，共训练 3000 iterations。CFG 训练时以各 10% 概率随机丢弃文本条件、参考条件或两者。

推理时使用 50 步 Euler 采样器，双尺度 CFG：参考图像引导尺度 $\mu_1=5.0$，文本引导尺度 $\mu_2=7.5$。此时 VFM 编码器和 MLP 投影器已完全移除，推理管线与标准 Wan2.1 一致。

## 实验关键数据

### 主实验

在 OpenS2V-Eval 基准上的零样本评测结果（表 1）。评测指标包括 Aesthetics（视觉质量）、MotionSmoothness（运动连续性）、MotionAmplitude（运动幅度）、FaceSim（面部保真度）、NexusScore（主体一致性）、NaturalScore（自然度）和 GmeScore（视频-文本对齐）。

| 方法 | TotalScore↑ | Aesthetics↑ | MotionSmoothness↑ | FaceSim↑ | NexusScore↑ | NaturalScore↑ |
|------|-------------|-------------|-------------------|-----------|--------------|---------------|
| Kling1.6（闭源） | 56.23% | 44.59% | 86.93% | 40.10% | 45.89% | 74.59% |
| Saber-14B（闭源） | 57.91% | 42.42% | 96.12% | 49.89% | 47.22% | 72.55% |
| VINO（开源） | 57.85% | 45.92% | 94.73% | 52.00% | 42.67% | 71.99% |
| BindWeave（开源） | 57.61% | 45.55% | 95.90% | 53.71% | 46.84% | 66.85% |
| Phantom-14B（开源） | 56.77% | 46.39% | 96.31% | 51.46% | 37.43% | 69.35% |
| RefAlign-1.3B | 56.30% | 42.96% | 94.74% | 53.06% | 43.97% | 66.25% |
| **RefAlign-14B** | **60.42%** | 46.84% | 97.61% | **55.23%** | **48.52%** | 73.63% |

RefAlign-14B 在 TotalScore 上达到 60.42%，为目前已公开结果中首个突破 60% 的方法。在 FaceSim 和 NexusScore 两个主体相关指标上均取得最优，验证了显式参考对齐对身份一致性和主体判别力的提升。RefAlign-1.3B 同样在同规模模型中取得 SOTA TotalScore（56.30%），说明方法在不同模型规模上均有效。

### 消融实验

RA loss 设计消融（1800 iterations 统一设置，表 2）。Baseline（配置 C）仅用 VAE 编码参考图像送入 DiT，不加 RA loss。

| 配置 | TotalScore↑ | FaceSim↑ | NexusScore↑ | NaturalScore↑ | 说明 |
|------|-------------|-----------|--------------|---------------|------|
| A: 完整 RA loss | **55.73%** | 53.15% | **46.23%** | 62.96% | 正项+负项，默认配置 |
| B: w/o $\mathcal{L}_{\text{neg}}$ | 51.75% | 48.67% | 46.61% | 53.75% | 去掉负项，NaturalScore 暴跌 |
| C: w/o $\mathcal{L}_{\text{RA}}$ | 49.93% | **68.45%** | 38.63% | 40.46% | 无对齐，FaceSim 虚高（copy-paste） |
| D: VAE + DINOv3 双输入 | 52.15% | 35.11% | 45.78% | 66.06% | 额外特征做输入，不如显式对齐 |

关键发现：配置 C 的 FaceSim（68.45%）反而远高于配置 A（53.15%），这说明不加 RA loss 时模型倾向于直接复制参考图像像素（copy-paste），面部"看起来像"但实际是指令遵循失败。RA loss 通过约束参考表示向 VFM 语义空间对齐，迫使模型学习"理解主体是什么"而非"记住主体长什么样"，因此 FaceSim 适度下降而 NexusScore 和 NaturalScore 大幅提升。配置 B 去掉负项后 NaturalScore 从 62.96% 跌至 53.75%，表明负项对齐对维持多主体场景下的自然度至关重要。

### 关键发现
- **RA loss 是性能提升的核心来源**：去掉 RA loss 后 TotalScore 从 55.73% 暴跌至 49.93%，NexusScore 下降 7.6 个百分点，NaturalScore 下降 22.5 个百分点——幅度远超其他任何单一消融。
- **负项对齐不可或缺**：仅保留正项（配置 B）时 TotalScore 下降 3.98%，多主体场景出现明显的性别属性混淆，说明仅靠拉近同主体特征无法保证主体间的语义边界。
- **对齐深度存在最优区间**：对齐层数 $K=9$ 时 TotalScore 达到峰值，过深（$K \ge 11$）导致 FaceSim 单调递减至 25% 以下——深层特征已高度语义化，强行对齐反而破坏身份保真度。
- **VFM 编码器规模不敏感、类型有偏好**：DINOv3-B/L/H+ 的 TotalScore 差异仅 0.33%-0.43%，但跨类型差异显著——DINOv3 在一致性指标（FaceSim、NexusScore）上最优，SigLIP2 在质量指标（Aesthetics、MotionSmoothness）上更优，Qwen2.5-VL 在运动幅度上更好，说明不同 VFM 的对齐信号侧重点不同。

## 亮点与洞察
- **"对齐即蒸馏"的旁路设计非常巧妙**：不做额外输入、不做推理时编码，而是把 VFM 的语义判别力通过对比学习式损失蒸馏到 DiT 参考分支的参数中。推理时模型是标准的 Wan2.1，但内部参考表示的质量已经质变。这个设计思路可以迁移到任何需要"条件编码器质量提升"的场景（如 ControlNet 的条件分支、IP-Adapter 的图像提示编码器）。
- **FaceSim 与 copy-paste 的 trade-off 解读很诚实**：消融中无 RA loss 时 FaceSim 反而最高（68.45%），作者没有回避这个"反直觉"结果，而是解释为 copy-paste 伪影导致的高面部相似度——这提醒读者在 R2V 评测中不能孤立看 FaceSim，必须与指令遵循指标（如 NaturalScore）联合解读。
- **正负项对比学习范式在扩散模型条件分支上的首次成功应用**：将对比学习中的 pull-push 机制引入 DiT 的条件表示学习，而非主干去噪表示学习，开辟了一条"优化条件质量"的新路径，与 REPA 形成互补——REPA 优化目标表示，RefAlign 优化条件表示。
- **两阶段训练策略（常规对→跨对）是缓解 copy-paste 的实用 trick**：先用常规对学会"参考条件是什么"，再用跨对学会"不要盲目复制参考图像"，这种课程学习策略简单有效，可复用于其他条件生成任务。

## 局限与展望
- **训练数据多样性不足**：当前仅使用 360K 样本，规模远小于主流 T2V 模型的训练数据，导致指令遵循与参考保真度之间的平衡尚未达到最优。扩大数据规模和多源数据混合是直接的改进方向。
- **底层基础模型限制视频长度**：受 Wan2.1 主干限制，RefAlign 目前仅支持 81 帧视频生成，无法生成长视频。这属于基座模型的能力边界，需要等待更强的主干或设计帧插值/外推策略。
- **单一 VFM 对齐信号可能不完整**：实验已显示不同 VFM 侧重不同指标维度（DINOv3 重一致性、SigLIP2 重质量、Qwen2.5-VL 重运动），单一 VFM 无法兼顾所有维度。多 VFM 联合对齐——如用 DINOv3 锚定身份、SigLIP2 锚定质量、Qwen2.5-VL 锚定语义组合——是自然的扩展方向。
- **RA loss 的超参（$\eta$、$\lambda$、$\delta$、$K$）依赖经验设定**：论文中 $\eta=\lambda=1.0$、$K=9$ 的取值来自消融，但不同主干、不同数据分布下可能需要重新调参。设计自适应的权重调度策略（如训练早期侧重对齐、后期侧重生成质量）可能进一步提升稳定性。

## 相关工作与启发
- **vs Phantom**: Phantom 用 CLIP 特征作为额外语义分支注入 DiT，属于隐式对齐；RefAlign 不注入额外特征，而是通过 RA loss 显式约束参考表示本身。前者的 CLIP 特征在推理时仍需编码（增加开销），后者训练完即丢弃 VFM。两者的本质区别是"加信息"还是"改表示"。
- **vs REPA**: REPA 对齐的是 DiT 从噪声中恢复的目标表示（noisy target → clean VFM），目的是加速收敛；RefAlign 对齐的是 DiT 对干净参考图像的条件表示（clean reference → clean VFM），目的是增强条件质量。REPA 的正项对齐在 $M>1$ 时可能导致不同参考表示塌缩到平均区域，RefAlign 的负项机制正是为解决这一问题而设计。两者在 R2V 场景中可互补而非互斥。
- **vs BindWeave / VINO**: 这类方法引入 MLLM（如 Qwen2.5-VL-7B）做跨模态推理，能建模复杂的空间关系和时序语义，但推理时需要额外运行 7B 级模型，计算开销巨大。RefAlign 证明了精心设计的对齐损失可以在不增加推理开销的前提下达到甚至超越 MLLM 方案的性能——这为"轻量级条件增强"提供了新的方法论。
- **更广泛的启发**: 将 VFM 作为"表征教师"、通过对比学习式损失蒸馏到生成模型的某个子模块（条件编码器、判别器、先验网络等），这一范式不限于 R2V——可推广到图像编辑（对齐编辑区域的表示）、风格迁移（对齐风格表示的语义空间）、甚至自回归生成（对齐 token 序列的中间表示）。

## 评分
- 新颖性: ⭐⭐⭐⭐☆ 首次将显式特征对齐引入 R2V 的条件表示学习，正负项对比学习范式在扩散条件分支上的应用是新的，但对比学习和 REPA 各自都是已知技术，核心思路的组合性大于原创性。
- 实验充分度: ⭐⭐⭐⭐⭐ 主实验覆盖闭源+开源共 12 个基线、2 个模型规模，消融覆盖损失设计（4 配置）、对齐深度（8 个深度值）、编码器规模/类型（6 种编码器），外加用户研究——系统性很强。
- 写作质量: ⭐⭐⭐⭐☆ 动机清晰（t-SNE 可视化直观支撑）、方法描述完整、与 REPA 的对比辨析透彻（4 点差异逐条说明）；但部分公式符号嵌套较深，附录与正文之间有些重复内容。
- 价值: ⭐⭐⭐⭐☆ "对齐即蒸馏、推理零开销"的设计为 R2V 的落地提供了实用方案，方法简洁可复现；但 81 帧限制和 360K 数据规模制约了直接实用价值，更多是开辟了条件表示对齐这一研究方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] SMRABooth: Subject and Motion Representation Alignment for Customized Video Generation](../../CVPR2026/video_generation/smrabooth_subject_and_motion_representation_alignment_for_customized_video_gener.md)
- [\[ICLR 2026\] MoAlign: Motion-Centric Representation Alignment for Video Diffusion Models](../../ICLR2026/video_generation/moalign_motion-centric_representation_alignment_for_video_diffusion_models.md)
- [\[ICLR 2026\] MATRIX: Mask Track Alignment for Interaction-aware Video Generation](../../ICLR2026/video_generation/matrix_mask_track_alignment_for_interaction-aware_video_generation.md)
- [\[CVPR 2026\] Scaling Zero-Shot Reference-to-Video Generation](../../CVPR2026/video_generation/scaling_zero-shot_reference-to-video_generation.md)
- [\[CVPR 2026\] Open-world Hand-Object Interaction Video Generation Based on Structure and Contact-aware Representation](../../CVPR2026/video_generation/open-world_hand-object_interaction_video_generation_based_on_structure_and_conta.md)

</div>

<!-- RELATED:END -->
