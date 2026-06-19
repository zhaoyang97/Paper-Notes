---
title: >-
  [论文解读] Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation
description: >-
  [CVPR 2026][图像生成][一步生成] 针对 MeanFlow 一步生成在加大"轨迹样本"比例时训练崩坏的问题，本文诊断出根因是不同时间尺度的梯度方差严重失衡，提出"时间均衡加权 + 动态边界调度"两个零额外推理开销的改动，把 ImageNet 256×256 的 1-NFE FID 刷到 2.62，超过所有扩散/流式一步方法。
tags:
  - "CVPR 2026"
  - "图像生成"
  - "一步生成"
  - "MeanFlow"
  - "Flow Matching"
  - "梯度方差均衡"
  - "边界调度"
---

# Temporal Equilibrium MeanFlow: Bridging the Scale Gap for One-Step Generation

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Tu_Temporal_Equilibrium_MeanFlow_Bridging_the_Scale_Gap_for_One-Step_Generation_CVPR_2026_paper.html)  
**代码**: 无（项目页 https://temf.github.io）  
**领域**: 扩散模型 / 图像生成  
**关键词**: 一步生成, MeanFlow, Flow Matching, 梯度方差均衡, 边界调度

## 一句话总结
针对 MeanFlow 一步生成在加大"轨迹样本"比例时训练崩坏的问题，本文诊断出根因是不同时间尺度的梯度方差严重失衡，提出"时间均衡加权 + 动态边界调度"两个零额外推理开销的改动，把 ImageNet 256×256 的 1-NFE FID 刷到 2.62，超过所有扩散/流式一步方法。

## 研究背景与动机
**领域现状**：从噪声一步映射到图像（one-step generation）是把扩散/流模型推向实用的关键。Flow Matching 直接回归"瞬时速度场" $v(x_t,t)$ 来搬运分布，但采样要几十上百次网络评估（NFE）。MeanFlow 是近期最有代表性的一支：它定义"平均速度场" $u(x_t,r,t)=\frac{1}{t-r}\int_r^t v(x_\tau,\tau)\,d\tau$，并推出连接平均速度与瞬时速度的恒等式，从而**不靠蒸馏、不靠课程学习就能从零训练一步生成器**。

**现有痛点**：MeanFlow 有个很反直觉的实操限制——当训练里"轨迹样本"（即 $r\neq t$、跨越较长时间区间的样本）占比超过某个最优阈值时，性能不升反降，出现模糊样本和高 FID。这等于把"长程跳跃"这一对一步生成最有用的信息按住了不让用。

**核心矛盾**：作者把这归因为**时间尺度失衡（temporal scale imbalance）**。记区间长度 $\Delta=t-r$，训练梯度的方差里有两股反向的力：① 当 $\Delta\to 0$（贴近生成起点 $r\approx t$），需要强边界约束来保证稳定，但这一项的方差随路径曲率近似以 $O(1/\Delta)$ 爆炸；② 当 $\Delta$ 较大（长区间 $r\ll t$），轨迹建模带来的 JVP（Jacobian-Vector Product）项主导，方差按 $O(\Delta^2)$ 增长。MeanFlow 用一个**固定**的采样策略，无法同时安抚这两端。

**本文目标**：在不改网络结构、不增加推理开销的前提下，让训练在所有时间尺度上"梯度贡献均等"，从而能放心地多用长程轨迹样本。

**切入角度 / 核心 idea**：既然方差是 $\Delta$ 的函数且形状可写出来（$\frac{C_1}{\Delta}+C_2\Delta^2$），那就**用一个随 $\Delta$ 变化的权重把方差拍平**；再配一个**随训练进度演化的边界约束调度**——先稳边界、后练轨迹。一句话：用"按时间尺度自适应加权 + 两阶段边界调度"替代固定采样，化解一步生成里的时间尺度失衡。

## 方法详解

### 整体框架
TEMF（Temporal Equilibrium MeanFlow）不动 MeanFlow 的模型与恒等式，只在**训练目标和采样调度**上做两处改动，外加一个让 CFG 也能一步出图的细节。给定数据 $x_0\sim p_{data}$、先验 $x_1\sim\mathcal N(0,I)$ 与加噪过程 $x_t=\alpha_t x_0+\beta_t x_1$，MeanFlow 的原始目标是回归带 stop-gradient 的目标速度：

$$L_{MF}(\theta)=\mathbb E_{x_t,r,t}\big[\,\|u_\theta(x_t,r,t)-\mathrm{sg}(u_{tgt})\|_2^2\,\big],\quad u_{tgt}=v_t-(t-r)\big(v_t\partial_x u_\theta+\partial_t u_\theta\big)$$

其中 $v_t=\alpha_t'x_0+\beta_t'x_1$ 是条件速度，括号里是 JVP 项。TEMF 的核心是：先在每个样本的 loss 上乘一个**只依赖区间长度 $\Delta=t-r$ 的权重** $w(\Delta)$ 把梯度方差均衡掉（设计 1）；再让"$r\neq t$ 样本的采样比例 $p$"**随训练进度 $\tau$ 从低到高变化**，实现"先稳边界、后练轨迹"的两阶段训练（设计 2）；最后把 CFG 的引导速度也纳入训练目标，并用一个递减的"速度混合系数"压住大引导强度下的方差（设计 3）。推理仍是一步：$x_0=x_1-u_\theta(x_1,0,1)$，**零额外开销**。

整篇方法本质是对训练动力学的诊断与修正（加权函数 + 采样调度），不是多模块串行 pipeline，因此用公式而非框架图说明更清楚。

### 关键设计

**1. 时间均衡加权：用一个随 $\Delta$ 变化的权重把梯度方差拍平**

这一项直接针对"短区间方差 $O(1/\Delta)$ 爆、长区间方差 $O(\Delta^2)$ 涨"的矛盾。作者先把梯度方差形式化为定理 1：在温和的正则/曲率假设下，

$$\mathrm{Var}[\nabla_\theta L_{MF}]\le \frac{C_1}{\Delta}+C_2\Delta^2+O(\Delta^3),\qquad C_1,C_2>0$$

其中 $C_1/\Delta$ 来自 $\Delta\to0$ 时的边界约束方差（路径曲率诱导），$C_2\Delta^2$ 来自大 $\Delta$ 时 JVP 项的放大。要让加权后的条件方差对所有 $\Delta$ 都是常数，需要 $w(\Delta)^2\big(\frac{C_1}{\Delta}+C_2\Delta^2\big)=\text{const}$，理想解是 $w(\Delta)=\big(\frac{C_1}{\Delta}+C_2\Delta^2\big)^{-1/2}$。论文用一个有界、可调的参数化近似来实现它：

$$w(\Delta)=\frac{1}{(1+\lambda_1\Delta^{\beta_1})\sqrt{1+\lambda_2\Delta^2}}$$

第一项 $1/(1+\lambda_1\Delta^{\beta_1})$ 调节边界约束强度（管小 $\Delta$ 端），第二项 $1/\sqrt{1+\lambda_2\Delta^2}$ 专门压制大 $\Delta$ 的 JVP 放大。加权后的损失变成 $L_{TEMF}=\mathbb E\big[w(t-r)\|u_\theta-\mathrm{sg}(u_{tgt})\|_2^2\big]$。定理 2 进一步证明：在 $\Delta\ge\Delta_{min}>0$ 的截断采样下，加权后的梯度条件方差被一个有限常数 $K$ 一致界住——这正是"不同时间尺度需要区别对待"的理论落点。消融里"无加权 6.52 → 只 JVP 项 5.13 / 只边界项 4.86 → 完整加权 4.31"印证了两项缺一不可。

**2. 动态边界调度：让"短/长区间样本配比"随训练从稳边界滑向练轨迹**

加权解决了"同一时刻各尺度的方差"，但还有一层时间维度的矛盾：训练早期模型连边界条件 $u(x_t,t,t)=v_t$ 都没学好，此时硬塞长程轨迹样本只会更不稳；训练后期边界已稳，应该把重心移到轨迹精度上。作者定义**边界约束强度** $\Gamma(\tau)=\mathbb E\big[\|u_\theta(x_t,t,t)-v_t\|_2^2\big]$（$\tau\in[0,1]$ 为归一化训练进度），并在一个简化的误差分解 $E_{1step}=E_{bias}(\Gamma)+E_{var}(p)$ 下启发式地推出 $\Gamma$ 的最优轨迹近似为指数衰减 $\Gamma^*(\tau)=\Gamma_0 e^{-\lambda\tau}$（定理 3，作者明确说这是 intuition 而非精确刻画 ⚠️ 以原文为准）。

落地时，他们调度的是"$r\neq t$ 样本比例" $p$，用 sigmoid 平滑爬升：

$$p(\tau)=p_{min}+(p_{max}-p_{min})\,\sigma\big(\kappa(\tau-\tau_0)\big),\quad p_{min}=0.1,\ p_{max}=0.9,\ \kappa=8,\ \tau_0=0.6$$

于是训练分两阶段：$\tau<\tau_0$ 是**边界稳定期**（以 $r=t$ 样本为主，专攻边界条件），$\tau\ge\tau_0$ 是**轨迹优化期**（以 $r\neq t$ 样本为主，专攻长程轨迹）。消融显示调度形状很重要：静态 25%（即 MeanFlow 做法）5.72，线性 4.86，余弦 4.53，指数 4.31——指数调度最贴合定理 3 的衰减预测。

**3. CFG 速度混合：让无分类器引导也能一步出图且不放大方差**

一步生成要保质量离不开 CFG，但直接拿引导速度场 $v_{cfg}=\omega v(x_t,t|c)+(1-\omega)v(x_t,t)$ 当训练目标，会因高引导强度 $\omega$ 带来巨大方差。TEMF 训练时以 0.1 概率丢标签来同时学条件/无条件速度，并引入**速度混合**把目标做平滑：$v_{mix}=m\cdot v_t^{cfg}+(1-m)\cdot v_{pred}$，混合系数随训练递减 $m(\tau)=0.3(1-\tau)+0.1\tau$。早期 $m$ 大（更靠近真实引导目标、但模型预测尚不可信，靠它降方差），后期 $m$ 减小让模型更依赖自身预测。这样在保持 $x_0=x_1-u_\theta(x_1,0,1)$ 一步采样的同时拿到 CFG 的质量收益，且推理不增开销。消融：无混合 5.24 → 固定 $m{=}0.25$ 为 4.58 → 动态混合 4.31。

### 损失函数 / 训练策略
最终训练目标即加权后的 $L_{TEMF}$。训练流程（Algorithm 1）每步：按进度算 $p$ → 采 $t$（对高斯做 sigmoid 得 $t\in(0,1)$）、$r\sim U(0,1)$ 并保证 $t>r$ → 以概率 $p$ 给 $r$ 加小扰动制造 $r\neq t$ → 算 $x_t,v_t$、前向得 $u_{pred}$、用 JVP 求 $u_{tgt}$ → 算 $\Delta$ 与 $w$ → 反传 $w\|u_{pred}-\mathrm{sg}(u_{tgt})\|_2^2$。全程从零训练，无预训练/蒸馏/外部初始化。

## 实验关键数据

### 主实验：ImageNet 256×256 类条件生成（FID-50K，越低越好）
所有 TEMF 模型从零训练 240 epoch（XL+ 额外 +60），与同骨干同设定的一步方法对比。

| 方法 | 参数量 | NFE | FID↓ |
|------|--------|-----|------|
| MeanFlow-B | 131M | 1 | 6.17 |
| **Ours-B** | 131M | 1 | **4.31** |
| MeanFlow-L | 459M | 1 | 3.84 |
| **Ours-L** | 459M | 1 | **3.26** |
| MeanFlow-XL | 675M | 1 | 3.43 |
| SoFlow-XL | 675M | 1 | 3.35 |
| **Ours-XL** | 675M | 1 | **2.81** |
| **Ours-XL+** | 675M | 1 | **2.62** |
| **Ours-XL+** | 675M | 2 | **2.30** |
| DiT-XL（多步参考） | 675M | 250×2 | 2.27 |
| SiT-XL（多步参考） | 675M | 250×2 | 2.06 |

每个尺度都稳定压过 MeanFlow / SoFlow；1-NFE 的 2.62 是扩散/流式一步方法新最优，2-NFE 的 2.30 已逼近需要 500 NFE 的多步 DiT-XL/SiT-XL。CIFAR-10 无条件（同 55M U-Net、像素空间、且不用 EDM 预条件）也拿到 2.81 的新 SOTA（MeanFlow/SoFlow 均 2.92）。

### 消融实验（ImageNet 256×256，DiT-B/4 131M，FID 1-NFE）

| 维度 | 配置 | FID | 说明 |
|------|------|-----|------|
| 增量组件 | Baseline MeanFlow | 6.17 | 起点 |
| | + 时间均衡加权 | 5.26 | 加权先降一大截 |
| | + 动态边界调度 | 4.93 | 两阶段调度继续降 |
| | + CFG 集成 | 4.61 | 引导带来质量 |
| | Full TEMF | **4.31** | 三者协同 |
| 加权函数 | 无加权 / 只 JVP / 只边界 / 完整 | 6.52 / 5.13 / 4.86 / **4.31** | 两项缺一不可 |
| 边界调度 | 静态25% / 线性 / 余弦 / 指数 | 5.72 / 4.86 / 4.53 / **4.31** | 指数最优，合定理 3 |
| 速度混合 | 无 / 固定0.25 / 动态 | 5.24 / 4.58 / **4.31** | 早期降方差最关键 |
| $r\neq t$ 比例 | 0 / .25 / .5 / .75 / 1.0 | 5.95 / 4.87 / **4.31** / 4.45 / 4.68 | 均衡框架下 0.5 最佳 |

### 关键发现
- **加权和调度高度互补**：单看增量消融，加权贡献最大（6.17→5.26），调度与 CFG 接力把 FID 一路压到 4.31，去掉任一项都明显回退。
- **"轨迹样本越多越好"被纠偏**：原版 MeanFlow 对 $r\neq t$ 比例极敏感（图 2 显示比例一大就崩）；引入时间均衡框架后，最优比例从约 0.25 推到 0.5，证明加权确实"解锁"了长程信息。
- **超参有清晰甜点**：$\lambda_1{=}0.5$、$\lambda_2{=}1.0$、$\beta_1{=}1.0$、$m{=}0.3$ 处 FID 最低，偏离两侧都变差——说明加权确实在"约束不足"和"过约束"之间找平衡。
- **质量随容量单调变好**：B→XL 一步 FID 持续下降，与 DiT 缩放规律一致。

## 亮点与洞察
- **把"训练崩坏"翻译成一条方差公式**：作者没有停在"长程样本一多就坏"的现象，而是推出 $\mathrm{Var}\le C_1/\Delta+C_2\Delta^2$，于是修复方案（求 $w(\Delta)$ 让方差恒定）几乎是公式的直接推论——诊断与药方一脉相承，这是最"啊哈"的地方。
- **两处改动都零推理开销**：加权只动 loss、调度只动采样比例，模型结构和一步采样式 $x_0=x_1-u_\theta(x_1,0,1)$ 原封不动，部署侧零成本却换来跨尺度的稳定。
- **"边界约束随训练演化"而非静态**：把边界条件当作随 $\tau$ 指数衰减的量来调度，是对"先学站稳、再学跑"这一训练直觉的量化表达，可迁移到其他需要在稳定性与表达力间权衡的从零训练场景（如一致性模型、Flow Map）。
- **可复用 trick**：当某个目标项的方差是区间/尺度的已知函数时，"求一个使方差恒定的权重并用有界参数化近似"是一个通用配方。

## 局限与展望
- **理论多为启发式**：定理 3 的指数衰减、误差分解 $E_{bias}(\Gamma)=a\Gamma^2$、$E_{var}(p)=b/(1-p)$ 都是作者明示的"直觉用"近似（⚠️ 以原文为准），并非对真实训练动力学的精确刻画；$O(1/\Delta)$ 项也是经验拟合而非严格导出。
- **采样需截断 $\Delta\ge\Delta_{min}$**：定理 2 的方差有界依赖一个正的最小区间截断，太小的 $\Delta$ 仍可能不稳，这一阈值的设置未充分讨论。
- **超参不少**：$\lambda_1,\lambda_2,\beta_1$ 加调度的 $p_{min},p_{max},\kappa,\tau_0$ 与混合系数共同决定效果，迁移到新数据集/分辨率时的调参代价未知；实验主要集中在 ImageNet-256 与 CIFAR-10，更高分辨率/文生图尚未验证。
- **改进方向**：把启发式调度换成可学习/自适应的方差估计；把均衡加权推广到 Flow Map、一致性模型等其他"双时刻"目标上。

## 相关工作与启发
- **vs MeanFlow**：同一恒等式与架构，MeanFlow 用固定采样比例，TEMF 指出其梯度方差随 $\Delta$ 严重失衡并用加权+调度修复，在所有尺度上一致超越且不增开销。
- **vs 一致性模型（iCT / sCT / Shortcut / IMM）**：它们靠跨时刻的一致性/矩约束做一步生成，多需 1×2 NFE 或表现落后（IMM-XL 7.77、Shortcut-XL 10.60）；TEMF 用真·1-NFE 且 FID 2.62 大幅领先。
- **vs SoFlow / AlphaFlow**：同样在 MeanFlow 损失上做文章（计算效率 / 耦合项），TEMF 切入点是时间尺度的梯度方差均衡，逐尺度都略优于 SoFlow。
- **vs Flow Map（Boffi et al.）**：Flow Map 学的是区间内速度的路径积分（总位移），TEMF 仍用平均速度，但其"按区间长度自适应加权"的思路对 Flow Map 类目标同样有借鉴价值。

## 评分
- 新颖性: ⭐⭐⭐⭐ 把一步生成的训练失稳精确归因到时间尺度方差失衡，并给出对应的均衡加权，诊断到药方一气呵成。
- 实验充分度: ⭐⭐⭐⭐⭐ 四骨干尺度全覆盖 + 九张消融表 + CIFAR-10 交叉验证，结论自洽。
- 写作质量: ⭐⭐⭐⭐ 公式与动机衔接清晰，但多处理论自承为启发式，需读者留意。
- 价值: ⭐⭐⭐⭐⭐ 零额外开销把扩散/流式一步生成刷到 FID 2.62，逼近多步模型，实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] OMP: One-step Meanflow Policy with Directional Alignment](../../ICML2026/image_generation/omp_one-step_meanflow_policy_with_directional_alignment.md)
- [\[CVPR 2026\] Bridging Fidelity-Reality with Controllable One-Step Diffusion for Image Super-Resolution](bridging_fidelity-reality_with_controllable_one-step_diffusion_for_image_super-r.md)
- [\[CVPR 2026\] BiFM: Bidirectional Flow Matching for Few-Step Image Editing and Generation](bifm_bidirectional_flow_matching_for_few-step_image_editing_and_generation.md)
- [\[CVPR 2026\] Extending One-Step Image Generation from Class Labels to Text via Discriminative Text Representation](emf_meanflow_text_to_image.md)
- [\[CVPR 2026\] Improved Mean Flows: On the Challenges of Fastforward Generative Models](improved_mean_flows_on_the_challenges_of_fastforward_generative_models.md)

</div>

<!-- RELATED:END -->
