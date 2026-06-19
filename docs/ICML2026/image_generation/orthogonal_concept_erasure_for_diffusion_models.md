---
title: >-
  [论文解读] Orthogonal Concept Erasure for Diffusion Models
description: >-
  [ICML 2026 Oral][图像生成][概念擦除] 把 T2I 扩散模型里"加性参数编辑"的概念擦除（UCE/SPEED 等）改写成"层级正交旋转 $W^* = QW$"的乘性更新，并配上一个子空间级别的擦除目标，用 Procrustes 闭式解一次性算出 $Q$，4.3 秒擦掉 100 个名人概念，且对非目标概念几乎零损伤。
tags:
  - "ICML 2026 Oral"
  - "图像生成"
  - "概念擦除"
  - "正交变换"
  - "闭式解"
  - "子空间投影"
  - "多概念擦除"
---

# Orthogonal Concept Erasure for Diffusion Models

**会议**: ICML 2026 Oral  
**arXiv**: [2605.28902](https://arxiv.org/abs/2605.28902)  
**代码**: https://github.com/HansSunY/OCE  
**领域**: AI安全 / 概念擦除 / 扩散模型  
**关键词**: 概念擦除, 正交变换, 闭式解, 子空间投影, 多概念擦除

## 一句话总结
把 T2I 扩散模型里"加性参数编辑"的概念擦除（UCE/SPEED 等）改写成"层级正交旋转 $W^* = QW$"的乘性更新，并配上一个子空间级别的擦除目标，用 Procrustes 闭式解一次性算出 $Q$，4.3 秒擦掉 100 个名人概念，且对非目标概念几乎零损伤。

## 研究背景与动机

**领域现状**：T2I 扩散模型容易生成版权、敏感、隐私相关的内容，业界用「概念擦除」来精准移除特定概念同时保留其余生成能力。已有方法分三类：推理期干预（易绕过）、训练式（如 ESD/MACE，效果好但要多轮微调，慢）、编辑式（如 UCE/RECE/SPEED，直接用闭式解修改 cross-attention 的 $W_k,W_v$，秒级出结果）。编辑式是面向部署的方向。

**现有痛点**：所有编辑式方法都把擦除写成 **加性更新** $W^* = W + \Delta$，靠最小二乘解出 $\Delta$。这种公式在「擦得干净」和「保得完整」之间始终拉扯——擦得狠就把无关概念也擦坏，保得严就擦不干净；多概念同时擦时还互相打架。

**核心矛盾**：作者用一组 toy 实验把矛盾的根源拆出来了。把 $W$ 做三种受控扰动测「猫」的生成：（A）只缩放 $\tilde W = \alpha W$，几乎没影响；（B）每个神经元独立旋转 $\tilde w_i = Q_i w_i$，模长保住但相对夹角破坏，整体图像质量崩；（C）整层共享旋转 $\tilde W = QW$，模长和神经元间夹角都不变，但「猫」的语义明显漂移。结论：**概念语义只编码在神经元方向上**，**整体生成能力则由神经元间的角度几何（hyperspherical geometry）支撑**。而加性 $\Delta$ 同时扰动方向、模长、夹角三件事，必然把擦除和保留耦合在一起。

**本文目标**：找一种参数更新方式，能精确旋转神经元方向（达成擦除），但严格保住模长和相互夹角（保住能力）；并且要对多概念擦除天然友好。

**切入角度**：方向旋转 + 模长不变 + 夹角不变 三件事一起，数学上就是「层级正交变换 $W^* = QW$, $Q^\top Q = I$」。这正是上面 toy 实验里的 Case C。

**核心 idea**：把加性 $W + \Delta$ 改写成乘性正交 $QW$，并把擦除目标从「向量级对齐」抬升到「子空间级压制」，最后落到一个标准的 orthogonal Procrustes 问题上，SVD 一步出闭式解。

## 方法详解

### 整体框架

OCE 要解决的是「编辑式概念擦除擦不干净又伤无关概念」的老问题，它的转法是把所有编辑式方法共用的加性更新 $W^*=W+\Delta$ 整个换成乘性正交更新 $W^*=PW$，再把擦除目标从向量级对齐抬到子空间级压制，最后落到一个标准的 orthogonal Procrustes 问题上一步 SVD 出解。输入是预训练 SD（或 FLUX）、待擦除概念嵌入 $C_1$、锚定概念嵌入 $C_*$（每个目标概念配一个语义近邻作"替身"）以及保留概念嵌入 $C_0=[C_g,C_n]$（$C_g$ 是 COCO-30k 上预先算好的通用先验、$C_n$ 是当前任务的局部保留集）；输出是一个层级正交矩阵 $P$，作用在 cross-attention 的 $W_k,W_v$ 上得到 $W^*=PW$。整套流程没有迭代训练，只是构造一个矩阵再 SVD。FLUX 这类 DiT 模型没有显式 cross-attention，则比照 UCE 在选定 embedding 层上做同样操作。

### 关键设计

**1. 层级正交更新替代加性更新：让擦除只动方向、不动模长和夹角**

痛点出在加性公式本身。作者用 toy 实验把矛盾拆开后发现：神经元的**方向**编码概念语义，神经元间的**夹角几何**支撑整体生成能力，而加性 $\Delta$ 在数学上会同时改动模长 $\|w_i\|$、方向 $\cos\theta_i$、夹角 $\cos\phi_{ij}$ 三件事，没法只改方向，必然把"擦除"和"保留"耦合死。OCE 的对策是改用正交矩阵 $P$（$P^\top P=I$）做乘性更新——正交变换在数学上恰好只旋转方向、自动锁住模长和夹角，正好对应 toy 实验里"擦得动语义却不伤能力"的那种扰动。

具体怎么求 $P$：先在 vector-wise 目标上写出 $\min_{P^\top P=I}\|PWC_1-WC_*\|_F^2+\|PWC_0-WC_0\|_F^2$，第一项把目标概念旋向锚、第二项钉住保留概念。把两项堆叠成 $A=[WC_1,WC_0]$、$B=[WC_*,WC_0]$，问题就化简为 $\min_{P^\top P=I}\|PA-B\|_F^2$，再等价于 $\max_{P^\top P=I}\mathrm{tr}(PM)$，其中 $M=BA^\top=W(C_*C_1^\top+C_0C_0^\top)W^\top$。这是经典的 orthogonal Procrustes 问题，对 $M$ 做 SVD $M=U\Sigma V^\top$ 直接得 $P=UV^\top$，没有学习率也没有迭代。

**2. 子空间级擦除目标加全局保留先验 $K_0$：让多概念不再互相打架**

vector-wise 对齐在单概念时很严很准，但 100 个目标同时严格对齐到各自的锚就会产生互相矛盾的硬约束，结果是既擦不干净又伤到无关概念。OCE 把擦除目标从"逐点对齐到锚"抬到"把整个目标子空间压出锚的正交补"：对 $WC_1$ 和 $WC_*$ 做 Gram–Schmidt 取正交基 $G,G_*$，定义投影 $R=GG^\top$、$R_*=G_*G_*^\top$，目标改写成 $\min_{P^\top P=I}-\|PR-R_{*,\perp}\|_F^2+\|PWC_0-WC_0\|_F^2$。这种"只要求把子空间压到锚之外"的约束比逐点对齐更软更结构化，多个概念叠在一起也不冲突。

保留项里还塞进了一个跨任务复用的全局先验。把保留矩阵拆成 $K_0=C_gC_g^\top=\mathbb{E}_c[cc^\top]$（在 COCO-30k 上离线算一次，A100 约 3 s）和当前任务的局部项 $C_nC_n^\top$，最终的 trace 最大化形式就是 $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$，再做一次 SVD 出 $P$。$K_0$ 把"通用语义先验"和具体任务解耦，离线 3 s 的预算换来多概念 FID 从 22.76 降到 18.33，且完全不增加推理时成本。

**3. 非对称粒度："擦除用子空间、保留用向量"的有意混搭**

第三个设计是把上面两种粒度故意拆开用：擦除项写成对子空间投影 $R,R_*$ 的操作（粗粒度），保留项却仍然是逐向量的 $\|PWC_0-WC_0\|_F^2$（细粒度，对 $C_0$ 里每个嵌入逐点不变）。理由是两边的诉求相反——擦除可以"宽进宽出"，因为本就不需要把目标精确对齐到某个具体替身，约束太硬反而冲突；保留必须"针针到位"，非目标概念之间没有冲突约束，逐向量保护既无副作用又能拿最高保真度。消融（Tab. 5）正好印证：双 vector-wise $H_o=91.70$、双 subspace $H_o=94.22$、本文的非对称组合 $H_o=95.48$，混搭确实在甜点区。

### 损失函数 / 训练策略

整篇没有"训练"。从输入到输出就是：堆叠 $C_1,C_*,C_n$ → 算 $M_{\text{total}}=-R(I-R_*)+W(K_0+C_nC_n^\top)W^\top$ → SVD $M_{\text{total}}=U\Sigma V^\top$ → $P=UV^\top$ → 写回 $W^*=PW$，其中 $K_0$ 离线预计算一次。SD v1.4 上擦 100 个名人只要 4.3 s（A100），ESD 与 MACE 都要 1800 s。

## 实验关键数据

### 主实验

| 任务 | 指标 | 之前 SOTA | OCE | 说明 |
|--------|------|------|----------|------|
| CIFAR-10 单物体擦除（前 5 类平均） | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ | 8.32 / 96.92 / 94.23 (MACE) | **4.61 / 98.68 / 97.01** | 擦得更干净，无关类几乎不掉 |
| 艺术风格擦除（Van Gogh） | CS $\downarrow$ / COCO FID $\downarrow$ / COCO CS $\uparrow$ | 21.22 / 14.53 / 26.45 (UCE) | **21.08 / 7.15 / 26.52** | FID 直接砍掉一半 |
| 多概念擦除（100 名人） | $\text{Acc}_e \downarrow$ / $\text{Acc}_s \uparrow$ / $H_o \uparrow$ / Time | 8.02 / 91.60 / 91.79 / 1800 s (MACE) | **3.44 / 94.42 / 95.48 / 4.3 s** | 比 MACE 快 ~420× |
| 多概念擦除（100 名人）vs SPEED | $H_o$ / Time | 93.72 / 5.0 s | **95.48 / 4.3 s** | 同量级速度但 $H_o$ 更高 |
| I2P 隐式 NSFW（含 AT 版本） | I2P / MMA / Ring-A-Bell ↓ | 0.10 / 0.01 / 0.00 (SPEED w/ AT) | **0.05 / 0.01 / 0.00** | 编辑式里最强 |

### 消融实验

| 配置 | $\text{Acc}_e \downarrow$ | $\text{Acc}_s \uparrow$ | $H_o \uparrow$ | FID ↓ | 说明 |
|------|---------|---------|---------|---|------|
| Full OCE（子空间擦 + 向量保） | 3.44 | 94.42 | **95.48** | 18.33 | 完整方案 |
| 向量擦 + 向量保 | 7.59 | 91.37 | 91.70 | 20.79 | 多概念互相打架 |
| 子空间擦 + 子空间保 | 4.54 | 93.01 | 94.22 | 18.10 | 保留过度宽松，丢细节 |
| 无全局先验 $K_0$ | 6.72 | 94.32 | 93.80 | 22.76 | FID 明显恶化 |
| $K_0$ 用 1/3 COCO | 4.47 | 93.44 | 94.47 | 19.31 | 越多通用先验越好 |
| $K_0$ 用 2/3 COCO | 3.85 | 93.63 | 94.87 | 18.60 | 单调提升 |

### 关键发现
- 非对称设计是关键：擦除用 subspace 是为了多概念无冲突，保留用 vector 是为了细粒度保真。任意一边换粒度都掉点。
- $K_0$ 的"通用语义先验"几乎是免费午餐：离线 3 s 预算，多概念 FID 从 22.76 → 18.33。这是把"任务无关的保留"和"任务相关的保留"解耦的工程红利。
- 在 DiT 架构（FLUX.1 dev）上不需要改公式，只把作用层从 cross-attention 换成 MMDiT 的 embedding 层就直接迁移，对象 / 风格 / 名人 / NSFW 四类擦除全部成立。
- 速度优势在 100 概念规模放大到 400× 量级，且 SPEED 报告的 runtime 没算它必须的 3 步预处理；OCE 是真正的"一步"。

## 亮点与洞察
- toy 实验把"为什么加性更新不行"讲到根上：方向 / 模长 / 夹角三件事在加性公式里强耦合，在乘性正交公式里完美解耦。这种"先做几何分析再写目标函数"的思路比直接堆 trick 优雅得多。
- 把"加性 → 乘性"的范式迁移其实可以套到很多参数编辑式任务（model merging、unlearning、风格注入），凡是依赖 $W + \Delta$ 闭式解的方法都值得问一遍："其实你只想旋转方向吗？"
- "擦用粗、保用细"的非对称粒度设计很反直觉但很有道理：擦除的约束太硬会冲突，保留的约束太软会丢细节，混搭刚好打到甜点区。
- $K_0$ 的预计算 trick 把"通用图像生成能力"显式写成了一个矩阵，可复用、可分发、可作为基础模型的"能力指纹"，未来 model card 里可能会出现这种东西。

## 局限与展望
- 作者承认：SVD 在更大模型上有算力压力；子空间约束让擦除后的语义会落到锚附近的"中间地带"而非精确对齐，编辑类任务可能不够；对关系、组合、水印这类更隐式的概念还没验证。
- 自己看下来还有几点：所有实验都在 SD v1.4 / FLUX.1 dev 这两个相对小的模型上，没在最新的大尺寸 SDXL/PixArt 上验证；锚定概念怎么选论文没系统讨论，但实战中往往是性能瓶颈；对抗鲁棒性靠 RECE 风格的对抗编辑加持（"Ours w/ AT"），不是 OCE 自己天然鲁棒。
- 改进方向：把 $P$ 从全 $d \times d$ 正交矩阵换成结构化正交（块对角、Cayley 参数化、butterfly），可解决大模型 SVD 算力问题；锚定概念的自动挖掘（VLM 自动给替身）应该能让"中间地带"问题缓解。

## 相关工作与启发
- **vs UCE / RECE / SPEED**: 都是编辑式闭式解，写法都是加性 $W + \Delta$。OCE 把整套公式换成乘性正交并加上 subspace 目标，在效果和效率（尤其是多概念）上同时碾压。
- **vs MACE / ESD（训练式）**: 训练式靠多轮 fine-tune 拿擦除效果，OCE 一步闭式解就反超并便宜 100×+，证明编辑式不是注定不如训练式。
- **vs OFT / Cayley Parametrization**: OFT 也用正交变换做 PEFT，但目标是稳定训练或定制化生成；OCE 把它当"几何手术刀"用来定点擦概念，是新场景。
- **vs CURE (NeurIPS 2025)**: 同期工作 CURE 也叫 orthogonal representation editing，本文在相关工作里点过，但定位差异是 OCE 在 cross-attention 权重上做、CURE 在表示层做。

## 评分
- 新颖性: ⭐⭐⭐⭐½ "把加性改乘性"看似一步之遥但确实没人这样系统化做过，加上 toy 实验把动机讲透，结构很完整。
- 实验充分度: ⭐⭐⭐⭐½ 单 / 多概念、艺术风格、NSFW、对抗、DiT 架构都覆盖；消融把非对称设计和 $K_0$ 都讲清了。
- 写作质量: ⭐⭐⭐⭐⭐ "几何分析 → 公式推导 → 闭式解"逻辑链非常顺，每一步都有"为什么"。
- 价值: ⭐⭐⭐⭐⭐ 4.3 s 擦 100 个概念、零训练、可直接用于生产 T2I 安全管线，落地价值极高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Mass Concept Erasure in Diffusion Models with Concept Hierarchy](../../AAAI2026/image_generation/mass_concept_erasure_in_diffusion_models_with_concept_hierarchy.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Closed-Form Concept Erasure via Double Projections](../../CVPR2026/image_generation/closed-form_concept_erasure_via_double_projections.md)
- [\[CVPR 2026\] GenErase: Generalizable and Semantically-Aware Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/generase_generalizable_and_semantically-aware_concept_erasure_in_diffusion_model.md)

</div>

<!-- RELATED:END -->
