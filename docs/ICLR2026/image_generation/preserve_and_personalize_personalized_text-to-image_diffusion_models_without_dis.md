---
title: >-
  [论文解读] Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift
description: >-
  [ICLR 2026][图像生成][个性化生成] 针对个性化文生图微调时"模型只会复刻参考图、忽略 prompt"的过拟合问题，本文证明现有目标函数在原理上无法保住预训练分布，转而提出一个基于 Lipschitz 连续性的正则项——本质上是对参数偏移量做 L2 约束——既保住了原模型的生成能力，又把训练时间砍掉一半以上。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "个性化生成"
  - "文本到图像扩散"
  - "分布保持"
  - "Lipschitz 正则"
  - "过拟合"
---

# Preserve and Personalize: Personalized Text-to-Image Diffusion Models without Distributional Drift

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=2ge1Y6DWPw](https://openreview.net/forum?id=2ge1Y6DWPw)  
**领域**: 扩散模型 / 个性化生成  
**关键词**: 个性化生成, 文本到图像扩散, 分布保持, Lipschitz 正则, 过拟合

## 一句话总结
针对个性化文生图微调时"模型只会复刻参考图、忽略 prompt"的过拟合问题，本文证明现有目标函数在原理上无法保住预训练分布，转而提出一个基于 Lipschitz 连续性的正则项——本质上是对参数偏移量做 L2 约束——既保住了原模型的生成能力，又把训练时间砍掉一半以上。

## 研究背景与动机

**领域现状**：个性化文生图（personalization）的目标是用 4~6 张参考图，把一个新概念（如"我家的狗 V\*"）注入预训练扩散模型，同时保留模型原有的 prompt 跟随能力，从而生成"我家的狗在雪地里"这种既有主体身份、又有丰富组合性的图。主流做法以 DreamBooth（全网络微调）和 Textual Inversion（学一个文本 token）为代表，并衍生出 LoRA、Custom Diffusion 等参数高效变体。

**现有痛点**：用极少量数据学新主体极易过拟合——模型会无视 prompt 内容，直接坍缩成复刻参考图。为缓解这点，已有方法要么冻结部分网络（LoRA、Custom Diffusion），要么限制参数更新的幅度或方向（SVDiff 只更新奇异值、OFT 只动角度分量）。但这些都是**架构层面的启发式约束**，只能间接控制分布保持；而且带来一个 flexibility-stability 的两难：冻得太死学不进新主体，放得太松又重新过拟合。

**核心矛盾**：另一类方法——DreamBooth 的 Prior Preservation Loss（先验保持损失）——试图从目标函数层面解决，做法是先用类别 prompt（如"a photo of a dog"）从预训练模型采样 100~200 张类别图，训练时一并拟合，借此"锚住"原分布。但本文指出，**只保住类别层级的信息（"dog"）并不等于保住所有先验知识（"the snow" 这类上下文）**，而且预采样阶段非常耗时。

**本文目标**：找到一个**目标函数层面**、能同时保证(1)学到新主体、(2)显式约束与预训练分布偏移的正则方式，并且不依赖额外采样数据。

**切入角度**：作者把问题归结为"训练目标与个性化真实目标之间的错位"，并从理论上量化这种错位——既然标准目标会让模型分布滑向自适应分布 padapt，那就直接在参数空间里给"偏移多少"上一个可证明的界。

**核心 idea**：用扩散网络对参数的 **Lipschitz 连续性**给输出分布的 KL 偏移上界，再把这个界松弛成一个对"当前参数与预训练参数之差"的 L2 正则项——简单到只是一行 L2，却有理论支撑，且彻底省掉了预采样。

## 方法详解

### 整体框架
方法的出发点是先证明"现有目标错在哪"，再给出一个有界的替代目标。整条流程极简：复制一份预训练去噪网络 $\epsilon_{\theta_{base}}$ 作为可训练的 $\epsilon_{\theta_{per}}$ 并冻结原网络；个性化阶段照常用目标 prompt 算去噪损失，**唯一的改动**是在损失里加一项 $\lambda\sum_i\|\theta^i_{per}-\theta^i_{base}\|_2^2$，即对每个参数相对预训练值的偏移做 L2 惩罚。训练完直接得到个性化模型，整个过程不再需要任何先验采样数据。因为方法本质是一项正则改进、没有多模块串联的 pipeline，这里不画框架图，用公式讲清即可。

记预训练损失为标准条件去噪损失：

$$\mathcal{L}_{denoise}=\mathbb{E}_{z,c,\epsilon,t}\big\|\epsilon-\epsilon_\theta(z_t,c,t)\big\|_2^2$$

DreamBooth 的传统组合目标在此之上再叠一项类别 prompt 的先验保持损失（需要预采样 $z'_t$）：

$$\mathcal{L}_{total}=\mathbb{E}\big[\|\epsilon-\epsilon_\theta(z_t,c_{target},t)\|_2^2+\lambda_{prior}\|\epsilon-\epsilon_\theta(z'_t,c_{class},t)\|_2^2\big]$$

而本文的目标把第二项换成对参数距离的惩罚，既不用 $z'_t$ 也不用类别 prompt。

### 关键设计

**1. 诊断"目标—目标错位"：标准目标在原理上保不住预训练分布**

这一节回答"为什么现有方法会过拟合"。作者设 $p^*$ 为参考分布、$p_{\theta_{base}}$ 为预训练模型分布且二者足够接近；自适应数据分布 $p_{adapt}$ 在某个集合 $D$ 上的质量 $\gamma$ 远大于 $p^*$ 在该集合上的质量 $\delta$（$\gamma\gg\delta$，即个性化数据严重偏向少数参考样本）。**定理 1** 证明：当模型用 $\mathcal{L}_{denoise}$ 在 $p_{adapt}$ 上做梯度下降、最终收敛到 $p_{\theta_t}\to p_{adapt}$ 时，必然有 $D_{KL}(p^*\|p_{\theta_{base}})<D_{KL}(p^*\|p_{\theta_t})$——也就是说微调一定让模型离原分布更远（Remark 1）。更关键的是 **Corollary 1**：即便加上类别 prompt 的先验保持样本（Eq. 3），混合分布 $p'_{per}$ 仍满足 $M\ll N$ 的少样本失衡，定理 1 依旧成立（Remark 2）。这就在理论上点破了 Prior Preservation Loss 的局限——保住类别 ≠ 保住全部先验，错位是目标函数本身决定的，靠加几张类别图补不回来。

**2. Lipschitz 正则：用参数距离给分布偏移上一个可证的界**

既然问题在目标层面，就从目标层面给一个有界约束。**定理 2** 给出核心不等式：若去噪网络 $\epsilon_\theta$ 关于参数 $\theta$ 是 Lipschitz 连续的，则对任意两组参数 $\theta_1,\theta_2$ 存在常数 $\lambda>0$ 使得

$$D_{KL}(p_{\theta_1}\|p_{\theta_2})\le\lambda\cdot\|\theta_1-\theta_2\|^k$$

证明思路是一条 Lipschitz 性逐级传递的链：Lipschitz 函数的复合仍 Lipschitz、注意力机制在紧致输入域上有有限 Lipschitz 常数，故 $\epsilon_\theta$ 关于 $\theta$ Lipschitz；由 Tweedie 公式 score $s_\theta=-\epsilon_\theta/\sigma_t$ 只差一个标量缩放仍 Lipschitz；probability-flow ODE 把 score 积分成 $\log p_\theta$，积分是线性算子也保持该界；最后三角不等式给出上式。**Remark 3** 取 $k=2$ 平方后得到 $\lambda\cdot\|\theta_1-\theta_2\|^2\le\lambda'\cdot\|\theta_1-\theta_2\|_2^2$——于是这个 Lipschitz 约束被松弛成一个再普通不过的 L2 正则项。L2 本身不新鲜，但**把它当作 Lipschitz 连续性的代理、用于约束个性化中的分布偏移，是此前没有的**：它意味着只要压住参数移动，输出分布相对预训练的 KL 偏移就被压住。

**3. 去预采样 + 单 λ 旋钮：效率与可控性一起拿到**

把"分布保持"落到参数距离上带来两个实际好处。其一，**不再需要任何额外数据**——传统先验保持要先采 100~200 张类别图，这一步占了训练时间的大头；本文目标只看 $\theta_{per}$ 与 $\theta_{base}$ 的差，直接省掉预采样瓶颈（Algorithm 1）。在 SDXL-CD 上训练时间降到基线的不到 1/5，其它骨干也普遍降一半以上，且越是参数高效的方法、预采样的相对开销越突出、提速越明显。其二，**单一超参 $\lambda$ 就量化了分布保持的强度**：$\lambda$ 越大、参数移动越受限、与预训练分布越贴近（CLIP-T↑、prompt 跟随更强）；$\lambda$ 越小、越贴合目标主体（DINO/CLIP-I↑、身份更像），从而把"自适应—保持"这一对冲突变成一个连续可调的旋钮，而不是像旧方法那样靠冻结哪些层这种离散启发式去间接拿捏。

### 损失函数 / 训练策略
最终个性化阶段优化的损失即（Algorithm 1 第 9 行）：

$$\mathcal{L}=\|\epsilon-\epsilon_{\theta_{per}}(\tilde{z}_t,t,c)\|_2^2+\lambda\sum_i\|\theta^i_{per}-\theta^i_{base}\|_2^2$$

其中 $\tilde{z}_t=\sqrt{\bar\alpha_t}\,z+\sqrt{1-\bar\alpha_t}\,\epsilon$ 为加噪潜变量，$\theta_{base}$ 全程冻结仅作锚点。该正则可无缝叠加到 DreamBooth、Custom Diffusion、DreamBooth-LoRA 等多种个性化策略之上，作为即插即用的替代目标。

## 实验关键数据

评测用 DreamBooth 基准（30 个主体、每主体最多 6 张图、25 个评测 prompt，共 3000 张生成图），骨干覆盖 SD-1.5 / SD-XL / SD-3.0，指标为 DINO、CLIP-I（主体保真度）与 CLIP-T（文本对齐）。

### 主实验
将基线目标替换为本文目标后的跨骨干增益（节选自 Table 1）：

| 骨干 | 方法 | DINO ↑ | CLIP-T ↑ | CLIP-I ↑ |
|------|------|--------|----------|----------|
| SD-1.5 | DB | 0.6028 | 0.2793 | 0.7881 |
| SD-1.5 | DB + Ours | 0.6394 (+0.0366) | 0.2976 (+0.0183) | 0.7948 (+0.0067) |
| SD-1.5 | CD + Ours | 0.5638 (+0.0070) | 0.3158 (+0.0004) | 0.7550 (+0.0011) |
| SD-XL | DB-LoRA + Ours | 0.6819 (+0.0257) | 0.3014 (−0.0085) | 0.8103 (+0.0133) |
| SD-3.0 | DB-LoRA + Ours | 0.6147 (+0.0104) | 0.3106 (+0.0008) | 0.7869 (+0.0046) |

关键在于：本文目标要么**两个目标（保真度 + 对齐）同时涨**，要么一个涨另一个不降——而这两者通常此消彼长。SD-1.5 + DB 上 DINO 大涨 +0.0366、CLIP-T 也涨 +0.0183，说明它在加速学新概念的同时保住了预训练知识。

与各类基线的横向对比（Table 2，联合 DINO/CLIP-I/CLIP-T 排名）：

| 方法 | DINO ↑ | CLIP-T ↑ | CLIP-I ↑ | Rank |
|------|--------|----------|----------|------|
| Ours | 0.6394 | 0.2976 | 0.7948 | 1 |
| IP-Adapter | 0.6304 | 0.2635 | 0.8318 | 2 |
| DreamBooth | 0.6028 | 0.2793 | 0.7881 | 3 |
| OFT | 0.6320 | 0.2370 | 0.7850 | 4 |
| SVDiff | 0.3839 | 0.3194 | 0.6886 | 8 |

本文综合排名第一：其它基线往往单项高、另一项低（如 SVDiff 的 CLIP-T 最高但 DINO 垫底），没有一个能在保真度和对齐上同时拿到可比表现。

### 消融实验

| 分析 | 现象 | 说明 |
|------|------|------|
| 正则强度 λ 扫描 | λ↑ → CLIP-T↑、DINO/CLIP-I↓ | λ 大偏向保 prompt/原分布，λ 小偏向像参考主体，连续可控 |
| Lipschitz 条件验证 | λ↑ → Δθ 与 Δϵ 同步下降 | 输出变化被参数变化所界，实证支持定理 2 |
| 训练效率 | SDXL-CD 降到基线 <1/5，其余 >2× 提速 | 省掉先验预采样这一耗时瓶颈 |

### 关键发现
- **λ 是"自适应—保持"的单旋钮**：大 λ 强保持、弱适配，小 λ 强适配、易坍缩到复刻参考图甚至出现伪影，二者随 λ 平滑过渡。
- **Lipschitz 假设在扩散模型里成立**：参数偏移 Δθ 与输出偏移 Δϵ 随 λ 成比例同步缩小，直接验证了"输出变化被参数变化上界"的理论主张。
- **效率提升对参数高效基线尤其明显**：这类方法本身训练快，预采样开销占比更大，去掉后提速最显著。
- **2D toy 实验**直观印证：标准目标（Eq. 2）完全不保原分布，Prior Preservation（Eq. 3）只保住选定类别、其余类别仍丢失，本文正则则较好地维持了五类的整体结构、最接近理想情形。

## 亮点与洞察
- **把"分布保持"从架构启发式拉回目标函数层面**：定理 1 先证明现有目标（含先验保持）原理上保不住分布，定理 2 再给出可证上界，诊断与解法形成闭环，比"冻哪些层"这类经验做法更有说服力。
- **一行 L2 背后是 Lipschitz 理论**：最终实现简单到只是对参数偏移加 L2，却被解释为 Lipschitz 约束的松弛形式——"简单但有据可依"，这种把朴素正则赋予新理论语义的视角可迁移到其它后训练/持续学习场景。
- **省掉预采样是被低估的工程价值**：先验保持的隐藏成本是预采样，本文用参数距离绕开它，既快又不依赖额外数据，在少样本场景尤其友好。
- **单超参可量化 KL 偏移**：用一个 λ 就把"保留多少先验"显式量化并连续可调，给个性化提供了一个干净的控制接口。

## 局限与展望
- 作者承认：方法**不保证在每个指标/prompt/主体上都涨**，因为不同主体、不同骨干所需的自适应程度与保持程度本就不同。
- 正则对所有参数**一视同仁**（统一 λ），未区分参数重要性；理想做法是按参数自适应加权（如借 EWC 的 Fisher 信息 $F_i$），但 $F_i$ 需要原始预训练数据与参数来估计，而个性化场景拿不到预训练数据，因此难以直接落地。
- 自己的观察：理论依赖"通用逼近 + 收敛"等较强假设，且 KL 上界中的常数 $\lambda$/$k$ 在实践中由超参近似，理论界与实际正则强度之间的对应仍偏定性。

## 相关工作与启发
- **vs DreamBooth / Prior Preservation Loss**：DreamBooth 用类别 prompt 预采样来锚住原分布，本文证明这只保类别层级、保不住全部先验，且预采样耗时；本文改为对参数距离做 L2，免采样、更快、保持更全面。
- **vs SVDiff / OFT**：它们分别只更新奇异值、只动角度分量，靠架构约束间接稳住生成能力，灵活性受限；本文从目标层面给连续可控的约束，不绑定特定骨干结构。
- **vs IP-Adapter / BLIP-Diffusion**：这类外部条件器需大规模数据训练适配网络、单概念个性化表现有限且资源开销大；本文是少样本、轻量、即插即用的目标函数改进。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把个性化过拟合归因为"目标—目标错位"并用 Lipschitz 给出可证界，视角新；最终落到 L2 则相对朴素。
- 实验充分度: ⭐⭐⭐⭐ 覆盖三种骨干、多种基线、toy 验证 + 三项消融，但缺大规模主观评测的主文呈现。
- 写作质量: ⭐⭐⭐⭐ 理论与动机衔接清晰，定理—Remark—算法层层推进。
- 价值: ⭐⭐⭐⭐ 即插即用、提速过半、单旋钮可控，对实际个性化部署有直接价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Directional Textual Inversion for Personalized Text-to-Image Generation](directional_textual_inversion_for_personalized_text-to-image_generation.md)
- [\[CVPR 2026\] DCoAR: Deep Concept Injection into Unified Autoregressive Models for Personalized Text-to-Image Generation](../../CVPR2026/image_generation/dcoar_deep_concept_injection_into_unified_autoregressive_models_for_personalized.md)
- [\[ICLR 2026\] Latent Diffusion Model without Variational Autoencoder](latent_diffusion_model_without_variational_autoencoder.md)
- [\[ECCV 2024\] Removing Distributional Discrepancies in Captions Improves Image-Text Alignment](../../ECCV2024/image_generation/removing_distributional_discrepancies_in_captions_improves_image-text_alignment.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)

</div>

<!-- RELATED:END -->
