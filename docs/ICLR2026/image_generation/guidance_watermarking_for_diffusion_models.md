---
title: >-
  [论文解读] Guidance Watermarking for Diffusion Models
description: >-
  [ICLR 2026][图像生成][扩散模型水印] 本文提出一种"引导式水印"方法：用任意现成的后处理水印解码器（post-hoc decoder）反传梯度去引导扩散采样轨迹，从而把任何后处理水印方案零成本转化为生成内嵌（in-generation）水印，无需重训扩散模型或解码器，并能继承甚至增强解码器的鲁棒性。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "扩散模型水印"
  - "引导式扩散"
  - "in-generation watermarking"
  - "鲁棒性"
  - "PCGrad"
---

# Guidance Watermarking for Diffusion Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5ifzhjMCKq](https://openreview.net/forum?id=5ifzhjMCKq)  
**代码**: 待确认  
**领域**: 图像生成 / AIGC 水印  
**关键词**: 扩散模型水印, 引导式扩散, in-generation watermarking, 鲁棒性, PCGrad  

## 一句话总结
本文提出一种"引导式水印"方法：用任意现成的后处理水印解码器（post-hoc decoder）反传梯度去引导扩散采样轨迹，从而把任何后处理水印方案零成本转化为生成内嵌（in-generation）水印，无需重训扩散模型或解码器，并能继承甚至增强解码器的鲁棒性。

## 研究背景与动机
- **领域现状**：扩散模型生成的图像已与真实照片难以区分，监管要求对 AIGC 内容做溯源与标识，数字水印是核心手段。现有方案分两类：**post-hoc**（生成后再外挂水印）与 **in-generation**（在生成过程中内嵌水印，如 Stable Signature 微调 VAE、Tree-Ring/Gaussian Shading 在种子里植入图案）。
- **现有痛点**：① Stable Signature 把水印能量集中在 VAE 上采样产生的高频细节，对 JPEG 等低通处理鲁棒性差；② Tree-Ring/Gaussian Shading 这类种子方案要么改变图像语义、要么对几何攻击（裁剪、旋转）脆弱，且**误警率（PFA）无理论保证**；③ post-hoc 方案只是外挂、与生成过程割裂；④ 各类内嵌方案往往需要重训模型或解码器。
- **核心矛盾**：既想要 post-hoc 解码器经过增强层训练得到的**强鲁棒性 + 可控误警率**，又想要内嵌方案"水印植入语义、不靠加微弱信号"的优势，二者此前难以兼得。
- **本文目标**：在不重训扩散模型、不重训解码器、不依赖原始图像的前提下，让采样直接生成"被预训练检测器判为已加水印"的图像，并把水印尽早植入语义层。
- **核心 idea**：**【把水印解码器当成分类器做梯度引导】**——水印检测器本质是个可微分类器，于是借鉴反事实生成（counterfactual）的梯度引导思路，用解码器对水印损失的梯度去推动扩散轨迹，使最终图像落入"已加水印"区域；并加一层增强（augmentation）来获得攻击鲁棒性。

## 方法详解

### 整体框架
设潜空间扩散模型由噪声估计网络 $\epsilon_\theta$、调度 $\bar\alpha_t$ 与 VAE 构成。给定任意可微的水印提取函数 $\phi: \mathcal{X}\to\mathbb{R}^M$ 和秘密向量 $u_m$，方法在每一步把解码损失的梯度注入噪声估计中，引导采样轨迹走向"高余弦相似度"的水印区域；再用增强层 + 梯度聚合获得鲁棒性，最后用截断/裁剪做加速与强度控制。

```mermaid
flowchart LR
    A[潜变量 z_t] --> B[补全扩散 z_t->z_0 + VAE 得 x_0]
    B --> C[增强层 T: JPEG/裁剪/对比度...]
    C --> D[水印解码器 φ 提取 logits]
    D --> E[余弦损失 L = 1 - cos<φ, u_m>]
    E -->|反传梯度 ∇L| F[PCGrad 聚合多增强梯度]
    F --> G[修正噪声估计 ε̂ = ε_θ - ω√(1-ᾱ_t)·g]
    G --> A
```

### 关键设计
**1. 梯度引导把后处理解码器变成扩散条件：** 方法的灵魂在于把"检测器判定"写成一个可微损失并对潜变量求梯度。在每个时间步，噪声估计被改写为 $\hat\epsilon(z_t,t)=\epsilon_\theta(z_t,t)-\omega\sqrt{1-\bar\alpha_t}\,\nabla_{z_t}\log L(z_t)$，其中 $\omega$ 控制水印强度。损失统一了多比特解码与零比特检测：$L(z_t,u_m)=1-\cos\big(\phi(x_0(z_t)),\,u_m\big)$，即最小化提取特征与秘密向量的夹角 $\theta$。零比特检测下这个夹角可直接换算成统计 p-value $p=\tfrac12\big(1\pm I_{\cos^2\theta}(\tfrac12,\tfrac{M-1}{2})\big)$，于是"最小化损失"等价于"压低 p-value、提高被正确检出的概率"，使误警率可控这一 post-hoc 优良性质被完整继承。

**2. 增强层 + PCGrad 聚合带来"免训练增鲁棒"：** 为抵御攻击，作者不只对原图算损失，而是对一组图像变换 $T\in\mathcal{T}$（恒等、JPEG QF50/80、亮度 +0.2、对比度 ×2、中心裁剪 50%）分别算损失再聚合梯度：$\hat\epsilon_{\mathcal T}(z_t)=\epsilon_\theta(z_t)-\sqrt{1-\bar\alpha_t}\,\mathrm{Agg}(\{\nabla_{z_t}\log L(z_t,u_m;T)\})$。不同变换的梯度方向常常冲突，简单平均会互相抵消，因此借用多任务学习里的 **PCGrad**（把相互冲突的梯度分量投影掉）做聚合。关键好处是：$\mathcal{T}$ 里可以放原解码器本来都不鲁棒的变换，引导过程会主动把这些攻击的鲁棒性"教"进生成结果，**无需重训解码器**。

**3. 快速且可控的近似引导：** 朴素实现要 $T(T+1)/2$ 次扩散+反传，代价过高。作者做两步简化：① 只在某个步 $T_w$（$0<T_w<T$）之后才开启水印引导；② 用恒等变换近似反向扩散的梯度传播，直接用 $\nabla_{z_0}$ 代替 $\nabla_{z_t}$。同时为免去逐模型搜 $\omega$ 的麻烦，对梯度做范数裁剪以稳定每步注入的水印能量：$\hat\epsilon(z_t,t)=\epsilon_\theta(z_t,t)-\omega\sqrt{1-\bar\alpha_t}\,\dfrac{g}{\max(\eta,\lVert g\rVert)}$，其中 $g=\mathrm{clip}_\tau(\nabla_{z_0}\log L(z_t))$。这让方法在 SD2、Flux、Sana 等不同求解器上都能即插即用。

**4. 全频谱铺展的水印能量：** 与 Stable Signature 把水印挤在高频不同，引导式方法在整个生成轨迹上施加约束，使水印能量沿全频谱分布（论文 Fig. 2 谱差异图佐证）。这正是它对 JPEG 等低通攻击更鲁棒的物理原因，也解释了"水印进入语义、对同一 seed/prompt 图像改动微小却整体可检"的现象。

## 实验关键数据

### 主实验表格（质量 + 鲁棒性，多模型，括号内为相对 SSig/VS 兄弟方法的增益）

| 模型 | 方案 | FID↓ | CLIP↑ | PSNR↑ | Capacity↑ | PD @ 1e-10 PFA | −log10(PFA) @ PD=0.9 |
|------|------|------|-------|-------|-----------|----------------|----------------------|
| SD2 | G-SSig | 2.3 | 0.332 | 19.6 | 27.7 (+19.3) | 0.99 (+0.5) | 16.3 (+12.2) |
| SD2 | G-VS | 2.2 | 0.332 | 18.5 | 212.2 (+37.7) | 1.0 (+0.0) | 105.6 (+61.8) |
| Flux | G-VS | 9.3 | 0.269 | 26.0 | 192.5 (+16.0) | 1.0 | 72.8 (+24.3) |
| Sana | G-VS | 4.1 | 0.346 | 23.5 | 207.5 (+28.8) | 1.0 | 96.4 (+49.2) |

FID/CLIP 几乎与无水印基线持平，PSNR 偏低（因为水印改的是语义而非加微弱信号，符合预期）。

### 消融/对比实验表格（SD2，与内嵌方案逐攻击对比）

| 方案 | Identity | Contrast×2 | JPEG Q50 | Gauss Blur3 | Rotation 90 | Crop 50% |
|------|----------|-----------|----------|-------------|-------------|----------|
| Gaussian Shading (多比特, bits) | 221 | 211 | 181 | 216 | **0** | **0** |
| G-VS (多比特, bits) | 222 | 219 | 197 | 220 | 194 | 206 |
| Tree-Rings (零比特, −log10 PFA) | 11.7 | 6.5 | 4.3 | 9.1 | 0.8 | 0.4 |
| G-VS (零比特, −log10 PFA) | 154.6 | 130.6 | 89.2 | 150.3 | 100.7 | 101.9 |

### 关键发现
- **几何攻击是分水岭**：Gaussian Shading/Tree-Ring 在旋转、裁剪下几乎归零，而 G-VS 因继承了 VideoSeal 解码器在训练时见过这些变换，仍保持高容量/高可检性。
- **零比特检测**：在 PFA=1e-10 极低误警区，G-SSig 把 SSig 的可检性翻倍；G-VS 在 SSig 已经检不出的 PFA 区间仍保持完美可检。
- **对抗攻击**：对 VAE 净化、再生成、average attack 三种攻击，G-VS 全面比 Tree-Ring 鲁棒，尤其 average attack——因为引导式水印是**内容相关**的，平均残差幅度低，难以被均值攻击剥离。

## 亮点与洞察
- **"解码器即条件"的视角转换**：把水印检测器当作可微分类器纳入引导式扩散，是一个简洁而通用的桥梁，让 post-hoc 与 in-generation 两条技术路线第一次低成本互通。
- **鲁棒性可继承、可增强**：通过把任意攻击塞进增强层并用 PCGrad 聚合，能"无训练地"给水印补上原解码器不具备的鲁棒性。
- **统计可控的误警率**：余弦损失直接映射到 p-value，使该方案在"必须保证极低误警率"的大规模检测场景里有理论站位，这是种子类方案的硬伤。
- **与 VAE 类方案互补**：水印能量全频谱分布，可与 Stable Signature 这类改 VAE 的方案叠加使用。

## 局限与展望
- **计算开销**：即便用 $T_w$ 延迟开启 + 恒等近似，引导仍需在采样中多次反传解码器梯度，比纯采样昂贵；超参 $\omega,\eta,\tau$ 需网格搜索。
- **强度-质量权衡敏感**：引导过强会产生伪影（色调/形状偏移加剧），需要逐模型标定。
- **依赖解码器质量**：方法的鲁棒性上限由所选预训练解码器决定；解码器没见过的攻击仍需显式放进增强层才能覆盖。
- **Flux 因算力限制只跑到 256×256**，更高分辨率与更多扩散主干的可扩展性有待验证。

## 相关工作与启发
- **Stable Signature**（Fernandez et al., 2023）：微调 VAE 内嵌水印，本文借其"利用鲁棒解码器 + 控误警率"思路，但避开了重训 VAE 和高频集中的缺陷。
- **Tree-Ring / Gaussian Shading**（Wen 2023 / Yang 2024）：种子级内嵌，本文吸收其"GenAI 水印不该当作加弱信号、PSNR 不合适"的观点，但解决了它们几何鲁棒性差、误警无保证的问题。
- **分类器引导扩散**（Dhariwal & Nichol, 2021）与反事实生成（Jeanneret et al., 2022）：方法的直接技术来源。
- **PCGrad**（Yu et al., 2020）：解决多增强梯度冲突的关键工具，启发"把鲁棒性当多任务优化"的做法。

## 评分
- 新颖性: ⭐⭐⭐⭐ — "把任意 post-hoc 解码器的梯度当扩散引导信号"视角新颖且通用，首次以引导方式在扩散过程内嵌水印。
- 实验充分度: ⭐⭐⭐⭐ — 覆盖 SD2/Flux/Sana 三种主干、两类解码器、零/多比特、多种几何与对抗攻击，并修正了 Tree-Ring 的 p-value 计算。
- 写作质量: ⭐⭐⭐⭐ — 动机层层推进、把损失与 p-value 的统计联系讲得清楚，图谱差异有力支撑核心论点。
- 价值: ⭐⭐⭐⭐ — AIGC 溯源监管的现实刚需，"免重训 + 鲁棒可继承 + 误警可控"对工业落地很有吸引力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FARI: Robust One-Step Inversion for Watermarking in Diffusion Models](fari_robust_one-step_inversion_for_watermarking_in_diffusion_models.md)
- [\[ICLR 2026\] Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](stage-wise_dynamics_of_classifier-free_guidance_in_diffusion_models.md)
- [\[ICLR 2026\] What Exactly Does Guidance Do in Masked Discrete Diffusion Models](what_exactly_does_guidance_do_in_masked_discrete_diffusion_models.md)
- [\[ICLR 2026\] Discrete Guidance Matching: Exact Guidance for Discrete Flow Matching](discrete_guidance_matching_exact_guidance_for_discrete_flow_matching.md)
- [\[ICLR 2026\] Stochastic Self-Guidance for Training-Free Enhancement of Diffusion Models](stochastic_self-guidance_for_training-free_enhancement_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
