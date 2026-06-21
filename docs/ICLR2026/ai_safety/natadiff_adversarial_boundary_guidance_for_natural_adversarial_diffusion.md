---
title: >-
  [论文解读] NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion
description: >-
  [ICLR 2026][AI安全][自然对抗样本] NatADiff 用扩散模型把采样轨迹引向"真类与对抗类的交界处"，生成的不是加扰动的约束对抗样本，而是天然混入对抗类语义线索的"自然对抗样本"，从而在保持白盒攻击成功率的同时大幅提升跨架构迁移性，并在分布上更像真实的 test-time error。
tags:
  - "ICLR 2026"
  - "AI安全"
  - "自然对抗样本"
  - "对抗边界引导"
  - "扩散模型"
  - "攻击迁移性"
  - "time-travel sampling"
---

# NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uWvLZqxjmx](https://openreview.net/forum?id=uWvLZqxjmx)  
**代码**: 待确认  
**领域**: AI 安全 / 对抗攻击 / 扩散生成  
**关键词**: 自然对抗样本, 对抗边界引导, 扩散模型, 攻击迁移性, time-travel sampling  

## 一句话总结
NatADiff 用扩散模型把采样轨迹引向"真类与对抗类的交界处"，生成的不是加扰动的约束对抗样本，而是天然混入对抗类语义线索的"自然对抗样本"，从而在保持白盒攻击成功率的同时大幅提升跨架构迁移性，并在分布上更像真实的 test-time error。

## 研究背景与动机
**领域现状**：对抗样本研究长期被两类范式主导——约束攻击（PGD/AutoAttack，往干净图上加 $\ell_p$ 不可见扰动）和无约束攻击（允许任意扰动幅度，只要结果落在自然图像流形附近）。近年的生成式攻击（GAN/扩散）则尝试直接把分类器梯度注入采样过程来"造"对抗样本。

**现有痛点**：(1) GAN 类方法对采样路径扰动敏感、缺乏理论依据且常损画质；(2) 直接把非鲁棒分类器梯度塞进扩散轨迹（如 AdvDiff/AdvClass），本质上还是在制造**约束对抗样本**——因为约束对抗样本往往就躺在自然图像 $\epsilon$-邻域里，扩散模型把样本拉回流形、分类器梯度把它推向最近的对抗口袋，二者拉锯出的恰是"贴着流形的微小扰动"，迁移性差。(3) 几乎没人去对接"自然对抗样本"这一最强的无约束攻击类别。

**核心矛盾**：真正危险的是 **natural adversarial sample（即 test-time error）**——它们是无扰动、天然存在却被误分类的合法输入（如躺在沙滩上的鲨鱼被判成别的），具有极高迁移性，且因为没有扰动而能绕过大多数对抗防御。但现有生成式攻击造出来的东西，分布上离这些真实 test-time error 很远。

**本文目标**：让扩散模型生成**高迁移、且分布上贴近真实 test-time error** 的自然对抗样本。

**核心 idea**：论文抓住一个机理观察——**自然对抗样本之所以高迁移，是因为不同分类器都依赖相同的"错误上下文线索（contextual cue）"来抄近路分类**。于是把这些来自对抗类的结构性线索主动塞进图像：**引导扩散轨迹走向"真类 $\cap$ 对抗类"的交界**，让图像在人眼看仍属真类、却带足对抗类特征以触发误分类。

## 方法详解

### 整体框架
NatADiff 以 Stable Diffusion 1.5 为底座，在 latent 空间做引导采样，目标是把每一步的去噪方向调成"既留在真类流形、又向对抗类交界靠拢"。整条 pipeline 由四个相互配合的部件组成：用 Tweedie 公式喂给分类器一个干净的 $\hat{x}_0$ 估计、用可微图像变换"磨平"约束扰动梯度、用对抗边界引导把轨迹拉向类交界、再用 time-travel 采样兜住画质，最后配合相似度选靶支持无目标攻击。

```mermaid
flowchart TD
    A[zT ~ N(0,I)] --> B[Tweedie 估计 x̂0]
    B --> C[可微图像变换 T<br/>归一化对抗梯度 g]
    C --> D[对抗边界引导<br/>组合 vy 与 vy∩ỹ + 分类器梯度]
    D --> E[time-travel 采样<br/>反复回退重采保画质]
    E --> F{argmax = ỹ?}
    F -- 否 --> G[增大 μ,s 重试] --> D
    F -- 是 --> H[VAE 解码输出对抗图]
```

### 关键设计

**1. 对抗边界引导（Adversarial Boundary Guidance）：把轨迹拉向类交界。** 这是全文的灵魂。普通的对抗分类器引导（AdvClass）只是在 classifier-free 引导上叠加一项受害分类器梯度 $s\nabla_{x_t}\log p(\tilde{y}|x_t)$，结果只能造约束样本。NatADiff 的关键改动是在去噪估计里引入一个指向"交界"的新方向向量。记 $v_y=\epsilon_{\theta^\star}(x_t,t,y)-\epsilon_{\theta^\star}(x_t,t)$ 是指向真类 $y$ 的方向，$v_{y\cap\tilde{y}}=\epsilon_{\theta^\star}(x_t,t,y\cap\tilde{y})-\epsilon_{\theta^\star}(x_t,t)$ 是指向"真类与对抗类交集"的方向（实现上用提示词 `"<对抗类名> and <真类名>"` 喂给扩散模型），则引导后的 score 为

$$\nabla_{x_t}\log\bar{p}(x_t|y,\tilde{y}) = -\frac{1}{\beta(t)}\Big(\epsilon_{\theta^\star}(x_t,t) + (\omega-\mu\omega)v_y + \mu\rho\, v_{y\cap\tilde{y}}\Big) + s\nabla_{x_t}\log p(\tilde{y}|x_t).$$

其中 $\mu\in[0,1]$ 调节"往交界靠"的强度：$\mu$ 足够大时轨迹逼近类交界，吸收足够对抗类元素造成误分类，但人眼仍读作真类；$\mu=0$ 时退化为普通 AdvClass。直觉是——既然 $\bar{p}(x_t|y)$ 本就不是真实边际分布、而是"对学过流形的网络所提供引导的放大"，那就索性多榨取这个网络的信息，让它把对抗类的结构线索画进图里，而不是靠脆弱的像素扰动。

**2. 削弱约束对抗梯度（Reducing Adversarial Gradient）：用图像变换"磨掉"扰动捷径。** 约束对抗攻击对旋转、裁剪、平移等变换很脆弱，NatADiff 反向利用这点：对当前 $\hat{x}_0$ 估计施加一组可微变换 $T=\{T_1,T_2,\dots\}$ 后再求对抗梯度，把局部的扰动信号"平均掉"，逼真正的对抗类语义特征显形。归一化后的梯度为

$$\nabla_{x_t}\log p(\tilde{y}|x_t) = g(x_t)/\|g(x_t)\|_2, \quad g(x_t)=\nabla_{x_t}\log\sigma_{\tilde{y}}\!\Big(\tfrac{1}{|T|}\textstyle\sum_{i=1}^{|T|}h(T_i(\hat{x}_0(x_t)))\Big),$$

$h$ 返回受害分类器 logits、$\sigma_{\tilde{y}}$ 给出目标类概率。同时为缓解"现成分类器没在带噪样本上训过"的问题，用 Tweedie 公式 $\hat{x}_0(x_t)=(x_t-\beta(t)\epsilon_{\theta^\star}(x_t,t,y))/\alpha(t)$ 先估出干净图再喂分类器。

**3. Time-travel 采样：兜住画质防止掉出流形。** 单纯做上述强引导很容易把样本推离图像流形、产生 artifact。NatADiff 借鉴 RePaint/FreeDoM，在选定时间区间内对 $x_{t_i}$ 做 $R$ 次"前向加噪再反向去噪"的回退-重采，让扩散有机会从坏轨迹里恢复。为省算力只在子集步上启用。配合外层一个自适应搜索循环：若解码结果还没被判成 $\tilde{y}$，就增大 $\mu\!\to\!\mu+\delta_\mu$、$s\!\to\!s+\delta_s$ 重试，命中即提前停止。

**4. 相似度选靶（Similarity Targeting）：把方法推广到无目标攻击。** 无目标攻击通常更强，但需要动态挑"最容易得手的错类"。NatADiff 假设"从语义相近的类借对抗特征更容易"，于是用 CLIP 文本编码器 $C_{enc}$ 把类名映射到共享嵌入空间，选与真类余弦相似度最高的候选作为对抗靶：

$$\tilde{y} = \arg\max_{y\in Y_{cand}} \frac{C_{enc}(y_i)\cdot C_{enc}(y)}{\|C_{enc}(y_i)\|_2\,\|C_{enc}(y)\|_2}.$$

## 实验关键数据

设置：ImageNet 1000 类，SD1.5 为底座 + DDIM 200 步，单张 RTX 4090 约 103 秒/样本，每组生成 2000 个样本。受害模型横跨 CNN（RN-50/Inc-v3/RN-152/对抗训练的 AdvRes、AdvInc）与 Transformer（ViT-H/Max-ViT/Swin-B/DeIT）。指标：ASR（误分类率）、IS、相对 ImageNet-Val 的 FID-Val（画质/自然度）、相对 ImageNet-A 的 FID-A（与真实自然对抗样本的接近度）。

### 主实验表格（ASR %，节选 RN-50 与 ViT-H 代理）

| 代理 | 攻击 | 白盒* | 平均 ASR | IS↑ | FID-Val↓ | FID-A↓ |
|------|------|------|---------|-----|---------|--------|
| RN-50 | PGD | 99.4 | 17.6 | - | - | - |
| RN-50 | ACA | 78.8 | 52.9 | 23.9 | 65.0 | 77.9 |
| RN-50 | AdvClassᵁ | 99.9 | 45.7 | 38.5 | 50.2 | 92.7 |
| RN-50 | **NatADiffᵀ** | 96.9 | 56.8 | 26.0 | 66.5 | **77.3** |
| RN-50 | **NatADiffᵁ** | 99.3 | **68.2** | 43.2 | 51.4 | 95.9 |
| ViT-H | ACA | 75.8 | 53.2 | 25.5 | 64.2 | 80.9 |
| ViT-H | AdvClassᵁ | 98.7 | 42.8 | 39.2 | 48.5 | 98.8 |
| ViT-H | **NatADiffᵀ** | 98.5 | **73.2** | 15.3 | 88.0 | 93.5 |
| ViT-H | **NatADiffᵁ** | 99.6 | 69.7 | 31.9 | 53.9 | 96.2 |

（*白盒=代理与受害同模型；上标 T=随机目标，U=相似度无目标。）核心读法：NatADiff 的白盒 ASR 与 SOTA 持平，但**平均迁移 ASR 全面领先**（RN-50 上 68.2 vs ACA 52.9 / AdvClass 45.7），且对抗训练的 AdvRes/AdvInc 对它几乎没有额外鲁棒性。

### 消融实验表格

| 移除的部件 | 后果 |
|-----------|------|
| 去掉 $v_{y\cap\tilde{y}}$（$\mu=0$，退化为 AdvClass） | 轨迹无法走向自然对抗样本，迁移性骤降（App. G.2） |
| 不做图像变换、直接用原始对抗梯度 | 生成回退为约束对抗样本，可见对抗特征减少（App. G.1） |
| 关闭 time-travel 采样 | 画质退化 / 掉出流形 |

### 关键发现
- **迁移性来源被验证**：NatADiff 之所以高迁移，是因为它注入的是分类器无关的结构性对抗线索（图 2 中不同代理模型生成的对抗特征类似），而非依赖单一代理梯度——它是唯一不"只靠代理分类器梯度"的方法。
- **有目标 vs 无目标的画质-自然度权衡**：有目标 NatADiff 的 FID-A 更低（更像真实自然对抗样本），但 IS/FID-Val 更差；无目标版本反之——因为真实自然对抗样本本就混合异类特征，复刻这种"混搭"对扩散底座要求更高，易出 artifact。
- **ViT-H 是最硬的骨头**：作为最大、最现代的模型，它学到更鲁棒的特征表示，迁移 ASR 最低；且有目标 ViT-H 攻击会引入 artifact 虚抬 ASR，反衬相似度选靶在找"模型弱点"上的价值。

## 亮点与洞察
- **范式转换**：把对抗攻击从"加扰动"重新定义为"沿类边界做语义引导"，第一次把生成式攻击和 Hendrycks 的 natural adversarial example 现象（shortcut learning / 上下文线索）真正对接起来，理论叙事自洽。
- **$v_{y\cap\tilde{y}}$ 这一项很巧**：不需要新模型，仅靠提示词 `"A and B"` 就让 SD 自己合成类交界方向，把"对抗类语义"画进图里而非贴扰动——这是迁移性的根因。
- **FID-A 作为评测维度**：用"与真实 ImageNet-A 的距离"来量化"像不像真实 test-time error"，比单看 ASR 更能说明攻击的现实意义。

## 局限与展望
- **成本高**：每样本约 103 秒（200 步 DDIM + time-travel 多次回退 + 外层搜索循环），远慢于 PGD 类一次前向的攻击，难以大规模部署。
- **画质-自然度难两全**：有目标/无目标在 IS、FID-Val、FID-A 上系统性此消彼长，没有单一配置全面占优；ViT-H 有目标攻击甚至靠 artifact 虚抬成功率。
- **依赖底座先验**：生成质量受 SD1.5 流形与相似度选靶共同约束，对底座分布外的类交界可能造不出可信样本。
- **防御侧未深入**：论文指出自然对抗样本能绕过常见防御，但未系统给出"如何用 NatADiff 做对抗训练以提升鲁棒性"的闭环。

## 相关工作与启发
- **约束/无约束/自然对抗样本的层级**（$A_N\subseteq A_C\subseteq A_U$）来自 Szegedy、Song、Hendrycks 等，本文站在最强的 $A_N$ 一端。
- **生成式攻击谱系**：从 GAN 攻击 → AdvDiff/AdvClass（Dai 2024，直接注入分类器梯度）→ ACA（Chen 2023b，扰动 latent 但受源图语义约束）→ DiffAttack。NatADiff 的差异点是"无源图、自由合成 + 类交界引导"，攻击面更宽。
- **机理基础**：shortcut learning（Geirhos 2020）、invariant risk minimization（Arjovsky 2020）解释了"为何多个分类器共享同样的错误线索 → 高迁移"。
- **技术借用**：Tweedie 公式（Efron 2011）、time-travel/universal guidance（RePaint、FreeDoM、Bansal 2024）、CLIP（Radford 2021）。
- **启发**：把"对抗"从像素空间搬到"语义/类边界空间"的思路，对鲁棒性诊断、数据增广、以及构造更贴近真实失败模式的测试集都有迁移价值。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把"类交界语义引导"与 natural adversarial example 机理对接，$v_{y\cap\tilde{y}}$ 项设计简洁且有理论叙事，区别于一众"注梯度"式生成攻击。
- **实验充分度**: ⭐⭐⭐⭐ — 9 种受害模型横跨 CNN/Transformer、6 个 SOTA 基线、ASR+IS+FID-Val+FID-A 多维度，并有针对三个部件的消融；略欠的是缺对抗训练/防御闭环与更大底座的验证。
- **写作质量**: ⭐⭐⭐⭐ — 定义层级清晰、公式推导扎实、图 1/图 3 直观对比不同攻击形态，机理叙事连贯。
- **价值**: ⭐⭐⭐⭐ — 提供了生成"高迁移且贴近真实 test-time error"对抗样本的可用方案，对鲁棒性研究与安全评测有实际意义；成本偏高限制了即时落地。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GROW: Watermark Generation with Progressive Guidance for Diffusion Models](../../CVPR2026/ai_safety/grow_watermark_generation_with_progressive_guidance_for_diffusion_models.md)
- [\[ICLR 2026\] On the Interaction of Compressibility and Adversarial Robustness](on_the_interaction_of_compressibility_and_adversarial_robustness.md)
- [\[ICLR 2026\] SCOPED: Score–Curvature Out-of-Distribution Proximity Evaluator for Diffusion](scoped_scorecurvature_out-of-distribution_proximity_evaluator_for_diffusion.md)
- [\[ICLR 2026\] Concept-based Adversarial Attack: a Probabilistic Perspective](concept-based_adversarial_attack_a_probabilistic_perspective.md)
- [\[ICLR 2026\] FERD: Fairness-Enhanced Data-Free Adversarial Robustness Distillation](ferd_fairness-enhanced_data-free_adversarial_robustness_distillation.md)

</div>

<!-- RELATED:END -->
