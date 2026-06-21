---
title: >-
  [论文解读] Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization
description: >-
  [ICLR2026][图像恢复][LayerNorm] 作者追踪图像恢复（IR）Transformer 的训练过程，发现标准 LayerNorm 会让特征幅值发散到百万量级、通道熵急剧坍缩，根因是 LN 的"逐 token 归一化"和"输入无关缩放"与 IR 任务相冲突；据此提出 i-LN——把归一化改成跨整个空间-通道维度做、并在每个 Attention/FFN 后按输入自适应地把缩放因子加回去，作为 LN 的即插即用替换件，在 SR/去噪/去雨/去 JPEG 伪影上稳定训练并普遍涨点。
tags:
  - "ICLR2026"
  - "图像恢复"
  - "LayerNorm"
  - "Transformer"
  - "训练动力学"
  - "特征发散"
---

# Analyzing the Training Dynamics of Image Restoration Transformers: A Revisit to Layer Normalization

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=SbLj5hJXh6](https://openreview.net/forum?id=SbLj5hJXh6)  
**代码**: https://github.com/2minkyulee/i-LN  
**领域**: 图像恢复  
**关键词**: 图像恢复、LayerNorm、Transformer、训练动力学、特征发散

## 一句话总结
作者追踪图像恢复（IR）Transformer 的训练过程，发现标准 LayerNorm 会让特征幅值发散到百万量级、通道熵急剧坍缩，根因是 LN 的"逐 token 归一化"和"输入无关缩放"与 IR 任务相冲突；据此提出 i-LN——把归一化改成跨整个空间-通道维度做、并在每个 Attention/FFN 后按输入自适应地把缩放因子加回去，作为 LN 的即插即用替换件，在 SR/去噪/去雨/去 JPEG 伪影上稳定训练并普遍涨点。

## 研究背景与动机
**领域现状**：自从 Vision Transformer 流行，图像恢复（超分 SR、去噪 DN、去雨 DR、去 JPEG 压缩伪影 CAR）的主流骨干都改成了 Transformer + LayerNorm 的组合（SwinIR、HAT、DRCT），LN 几乎成了默认标配。但大家几乎只关心"换更强的架构能涨多少点"，对这些 IR Transformer **内部的训练动力学到底发生了什么**几乎没人深究。

**现有痛点**：作者把 HAT 做 ×4 SR 时每个基本块的中间特征轨迹画出来，发现两个被忽视的反常现象：① 特征的平方均值在训练中**指数级发散，最高冲到百万（甚至千万）量级**；② 通道维度上的特征熵在训练**极早期就急剧下降**，说明出现了少数几个极端值通道独霸整体统计量。这种"隐藏的反常特征"在更深、更宽的网络以及各种 IR 任务/各种 Transformer 骨干上**普遍存在**，而且网络越大发散得越快越猛。

**核心矛盾**：作者的假设是——特征发散不是网络坏了，而是网络在**主动绕开 LN**。LN 对 IR 任务有两个根本性的不匹配：

- **逐 token 归一化破坏空间相关性**：LN 对每个 token（像素）单独算均值方差再归一化，完全不考虑 token 之间的相对关系，而 IR 高度依赖像素间的空间结构。
- **输入无关的缩放丢掉输入统计量**：LN 把所有中间特征都映射进一个统一的归一化空间，限制了表示的取值范围灵活性，从而抹掉了 IR 任务里对忠实重建至关重要的、随输入而变的低层统计信息。

一个直觉解法是干脆**整个去掉归一化层**（EDSR 等早期 SR 工作就这么干过），但作者实验发现 IR Transformer 一旦完全去掉归一化就**训练极不稳定、根本不收敛**。

**本文目标**：找到一种既能保持训练稳定、又不和 IR 任务冲突的归一化方案。

**切入角度**：从"结构保持"（inter-pixel structure preservation）这个几何视角重新审视 LN——把一组 token 看成 $\mathbb{R}^C$ 里的点云，理想的变换应当只对整团点云做全局缩放和平移（homothety，位似变换），而作者可以**证明**逐 token LN 一般连保角变换都不是，必然破坏点云形状。

**核心 idea**：用"跨空间-通道整体归一化 + 输入自适应重缩放"替换逐 token LN，既保住 token 间空间结构，又把被归一化抹掉的全局尺度信息显式加回来。

## 方法详解

### 整体框架
i-LN（Image Restoration Transformer Tailored Layer Normalization）是对标准 LN 的**即插即用替换件**——不改 Attention/FFN，不改骨干结构，只把每个 Transformer 块里的归一化算子换掉。它由两个互补的修改叠加而成：第一步把"逐 token 归一化"改成"跨整个空间-通道维度的整体归一化（LN\*）"，解决空间结构被破坏的问题；但整体归一化会丢掉一个全局尺度标量，第二步再用"输入自适应重缩放"把这个尺度在每个 Attention/FFN 输出处显式加回去。两步合起来，让网络在整条前向路径上都能保住输入的低层特征统计，既不再需要靠制造百万量级特征去"绕开"归一化，训练也保持稳定。

### 关键设计

**1. 空间整体归一化 LN\*：让归一化保住像素间结构**

标准逐 token LN 对第 $\ell$ 个 token 单独算统计量：$\mathrm{LN}(x_\ell)=\gamma\frac{1}{\sqrt{\sigma_\ell^2+\epsilon}}(x_\ell-\mu_\ell)+\beta$，其中 $\mu_\ell,\sigma_\ell^2$ 只在通道维 $c$ 上取期望。问题在于每个 token 用各自不同的 $(\mu_\ell,\sigma_\ell)$ 去缩放，token 之间的相对差异 $x_\ell-x_k$ 被不同尺度扭曲——作者用 Definition 1/2 把"像素间结构"形式化为相对差集合 $\Delta x=\{x_\ell-x_k\}$，并证明（Proposition 1）逐 token LN 一般不是位似变换、甚至不保角，所以会破坏空间结构。

LN\* 的改法极简单：把均值方差改成在**空间维 $\ell$ 和通道维 $c$ 上一起取期望**，即 $\mu=\mathbb{E}_{\ell,c}[x_{\ell,c}]$、$\sigma^2=\mathbb{E}_{\ell,c}[(x_{\ell,c}-\mu)^2]$，所有 token 共用同一组全局统计量。这样一来任意两个 token 的差变成 $T_{\mathrm{LN}^*}(x_\ell)-T_{\mathrm{LN}^*}(x_k)=\frac{1}{\sigma}(x_\ell-x_k)$（Proposition 2），是一个标准的位似变换——整团点云只被统一缩放，形状（像素间结构）被原样保住。CNN 里的归一化本就天然是空间整体的，但在 IR Transformer 语境下"空间整体性 vs 结构破坏"的影响一直没被认真讨论过，这正是本文点破的地方。

**2. 输入自适应重缩放：把 LN\* 丢掉的全局尺度加回来**

LN\* 保住了结构，但代价是丢掉了那个全局尺度标量 $\sigma$——而 IR 任务恰恰需要保留随输入而变的统计量来做忠实重建（同样亮度/纹理强度的输入应当映射到不同的取值范围，统一归一化空间会把这种差异抹平）。作者的解法是：在每个 Attention 或 FFN 之后，用前一步归一化里**已经算好的标准差**把输出重新放大回去。即把一个块 $B$ 改写为

$$B(x;f,\text{i-LN}) = x + \sqrt{\sigma^2+\epsilon}\cdot f(\mathrm{LN}^*(x)),$$

其中 $f$ 是该块的 Attention 或 FFN 运算。这条"黄色支路"（论文 Fig.3）把归一化时除掉的 $\sigma$ 又乘回到残差分支上，显式重新引入被抹掉的全局尺度项，给中间特征恢复了取值范围的灵活性，同时因为 $\sigma$ 随每张输入图而变，等于保住了 per-instance 的统计量。作者后面用实验证明这一步让特征分布稳定一个数量级（通道熵更高）、性能也更好。

### 损失函数 / 训练策略
方法本身不引入新损失，沿用各 IR 任务标准的像素重建损失。为公平对比，作者把所有 baseline 和自己的方法在**统一设置下重新实现**（因为近期工作的训练细节互相不一致）：去雨用 Rain13K 训练，其余任务用 DF2K（DIV2K+Flickr2K）；只做基础增广（随机翻转/旋转/裁剪），不用 mixup、渐进 patch、warm-start 等花活；骨干用 SwinIR / HAT / DRCT，模型大小、batch、patch 按算力约束调整并用下标标注（如 HAT1、HAT2）。

## 实验关键数据

### 主实验
i-LN 作为 LN 的直接替换，在四类 IR 任务、多个骨干上几乎全面涨点。下表为 ×4 SR（HAT1）上和各类归一化方案的对比（Table 1），i-LN 在所有数据集上最优：

| 数据集 (×4 SR) | 指标 | LayerNorm | InstanceNorm | i-LN (本文) |
|--------|------|------|----------|------|
| Set14 | PSNR / SSIM | 28.79 / .7876 | 28.98 / .7907 | **29.01 / .7915** |
| BSD100 | PSNR / SSIM | 27.68 / .7411 | 27.80 / .7445 | **27.84 / .7456** |
| Urban100 | PSNR / SSIM | 26.55 / .8015 | 27.02 / .8136 | **27.17 / .8167** |
| Manga109 | PSNR / SSIM | 31.01 / .9150 | 31.46 / .9199 | **31.82 / .9228** |

LN 在所有方案里**最差**（它既忽略 token 间关系又抹掉输入统计）；逐 token 变体 LayerScale/RMSNorm 略好于 LN 但仍输给空间整体方案；不加归一化（None）和 ReZero 因训练不稳收敛差；InstanceNorm/BatchNorm 这类空间整体方案优于逐 token 方案，但 BN 在 eval 模式下大幅掉点（说明 IR 需要 per-image 统计），IN/BN 又会丢通道信息——综合下来 i-LN 最优。

跨任务上（Table 2），SR 和去雨涨点最明显（如 HAT1 去雨 Rain100L 从 34.35 → **36.20** dB，+1.85 dB；SwinIR1 在 Test100 从 27.45 → **29.87** dB，+2.42 dB），去噪和去 JPEG 伪影涨点较小。作者解释：SR/去雨的输入有大片"可靠区域"（SR 输入精确对齐 GT 低频、去雨里没被雨纹覆盖的局部），保住输入特征收益大；而去噪/去 JPEG 的退化是全图均匀/不规则分布的，"保住特定输入特征"的优势就被摊薄了。

### 消融实验
作者在更大容量、更长训练的 HAT2 上做 ×4 SR 消融，逐个拆掉 i-LN 的两个组件（SH=空间整体性、Rs=重缩放）：

| 配置 | 说明 | 效果 |
|------|------|------|
| i-LN（完整） | LN\* + 输入自适应重缩放 | 恢复质量与特征稳定性均最佳 |
| 去掉 Rs → 退回 LN\* | 只做空间整体归一化、不加回尺度 | 质量下降，通道熵随之坍缩 |
| 再去掉 SH → 退回 LN | 回到逐 token LN | 通道熵**指数级坍缩**到 vanilla LN 水平 |

### 关键发现
- **逐 token 操作是发散的真凶**：Fig.4 显示凡逐 token 归一化（LN、RMSNorm、LayerScale）都会发散；凡空间整体方案（i-LN、BN、IN、ReZero）都不发散。不加归一化时特征在崩溃前其实也是有界的，进一步印证"发散源于逐 token 操作"这一假设。
- **两个组件互补、缺一不可**：通道熵随着逐个移除 SH 和 Rs 而**指数级坍缩**，说明 LN\* 负责保住 token 间关系、重缩放负责补回丢失的全局尺度，两者一起才能维持通道上分布良好的激活。
- **越大越糟**：网络加深/加宽时特征发散更快、幅值更高——离群通道要主导统计量必须先盖过残差路径上已经异常的激活，是个累积放大过程。这揭示了 IR 在 scale 时一个独特的隐患：扩容不只放大表达力，也会加剧病态的特征增长。
- **补偿机制**：baseline 尽管特征严重失衡却仍能收敛、输出幅值正常，是因为最后一层 LN 的仿射 bias 与输入通道幅值出现了对齐（Fig.8），形成一种补偿——但这是网络被迫绕路的产物，而非健康状态。
- 作者还观察到 i-LN 在低精度配置和真实世界退化（Real-ESRGAN pipeline）下也更鲁棒。

## 亮点与洞察
- **"分析驱动设计"的范本**：先用特征幅值/通道熵曲线把一个长期被忽视的反常现象（百万量级发散 + 熵坍缩）可视化坐实，再用"结构保持"几何框架把根因归到 LN 的两个具体不匹配，最后给出一个两行代码的修法——动机、诊断、解法环环相扣，可信度高。
- **几何视角很巧**：把 token 看成点云、把"好的归一化"定义为位似变换（只许整体缩放平移），从而能**形式化证明**逐 token LN 破坏结构、LN\* 保持结构。这个抽象让"LN 为什么不适合 IR"从经验观察上升为可证命题。
- **极简且即插即用**：i-LN 不加参数、不改架构，就是把 LN 的统计量从逐 token 改成全局、再把 $\sqrt{\sigma^2+\epsilon}$ 乘回残差支路。这种"几乎零成本"的替换很容易被现有 IR Transformer 直接采纳。
- **可迁移的洞察**："逐 token 归一化会破坏空间结构"这一点对任何依赖空间/序列相关性的低层视觉任务（光流、深度估计、视频恢复）都值得警惕；"归一化丢掉的全局尺度应显式补回"也是一个通用的设计模式。

## 局限与展望
- **涨点幅度任务依赖明显**：去噪、去 JPEG 伪影上提升很小（PSNR 常在 0.05~0.1 dB 量级），说明 i-LN 的收益高度依赖"输入存在大片可靠区域"，对全图均匀退化的任务优势有限。
- **理论基于无仿射参数的简化**：结构保持的证明里忽略了仿射参数 $\gamma,\beta$，真实带仿射时的严格行为没展开；"网络主动绕开 LN"目前仍是假设+经验证据，尚无机制级证明。
- **范围局限在 SR 类骨干**：实验集中在 SwinIR/HAT/DRCT 这几个超分系骨干和四类经典退化，未覆盖去模糊、低光增强、统一 all-in-one 恢复等更复杂场景，泛化性有待进一步验证。
- 可改进方向：把这套"空间整体 + 自适应重缩放"思想推广到带门控/MoE 的更大 IR 模型，或与低精度训练结合，验证它能否成为大规模 IR 模型的默认归一化。

## 相关工作与启发
- **vs 标准 LayerNorm（SwinIR/HAT/DRCT 默认）**：它们逐 token 归一化、输入无关缩放，导致特征发散、熵坍缩；i-LN 全局归一化 + 自适应重缩放，从根上消除发散并普遍涨点，是直接替换关系。
- **vs 去掉归一化的早期 SR 工作（EDSR 等）**：它们靠移除归一化避开"输入无关缩放"问题，在 CNN 上可行；但 IR Transformer 完全去归一化会训练崩溃，i-LN 保留归一化稳定性的同时把输入统计补回来，鱼和熊掌兼得。
- **vs ReZero / LayerScale**：两者用近零初始化的缩放因子稳定 Transformer 训练，被作者当作潜在解法测试；但 ReZero（去归一化）和 LayerScale（逐 token）性能都不如 i-LN，说明问题关键不在"缩放残差"而在"归一化是否空间整体 + 是否保住输入统计"。
- **vs InstanceNorm / BatchNorm**：同为空间整体方案、能避免发散，但 IN/BN 丢失通道信息、BN 还有 train/eval 失配（eval 模式大幅掉点），i-LN 既保通道信息又用 per-image 统计，故更适合 IR。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把被忽视的训练动力学问题诊断清楚并给出有理论支撑的极简解法，视角新颖。
- 实验充分度: ⭐⭐⭐⭐ 覆盖四类任务、三种骨干、多种归一化对比 + 组件消融 + 网络规模分析，较全面。
- 写作质量: ⭐⭐⭐⭐ 现象→根因→方法逻辑清晰，图表诊断有力，理论命题表述严谨。
- 价值: ⭐⭐⭐⭐ 即插即用、零成本替换，对 IR Transformer 社区有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers](../../CVPR2026/image_restoration/diffdecompose_layer-wise_decomposition_of_alpha-composited_images_via_diffusion_.md)
- [\[CVPR 2025\] DPIR: Dual Prompting Image Restoration with Diffusion Transformers](../../CVPR2025/image_restoration/dpir_dual_prompting_restoration_dit.md)
- [\[ICLR 2026\] DiffusionBlocks: Block-wise Neural Network Training via Diffusion Interpretation](diffusionblocks_block-wise_neural_network_training_via_diffusion_interpretation.md)
- [\[CVPR 2026\] FoundIR-v2: Optimizing Pre-Training Data Mixtures for Image Restoration Foundation Model](../../CVPR2026/image_restoration/foundir-v2_optimizing_pre-training_data_mixtures_for_image_restoration_foundatio.md)
- [\[CVPR 2026\] ReflexSplit: Single Image Reflection Separation via Layer Fusion-Separation](../../CVPR2026/image_restoration/reflexsplit_single_image_reflection_separation_via_layer_fusion-separation.md)

</div>

<!-- RELATED:END -->
