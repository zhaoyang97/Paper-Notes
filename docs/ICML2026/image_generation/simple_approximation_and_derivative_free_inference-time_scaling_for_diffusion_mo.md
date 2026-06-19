---
title: >-
  [论文解读] Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures
description: >-
  [ICML 2026][图像生成][推理时缩放] 作者把扩散模型的推理时 reward 引导从"粒子空间 SMC + 高阶导数"升级为"路径空间 SMC + Girsanov 似然比"，得到 URGE 算法：每条轨迹只需对 guidance $G$ 做一阶梯度并累加一个简单的 Itô 项当权重，完全不需要 reward $r$ 的导数 / Hessian / score 估计，在 GMM、逆问题和文生图三类任务上都打平或优于 FK-Corrector / AFDPS / FK-Steering。
tags:
  - "ICML 2026"
  - "图像生成"
  - "推理时缩放"
  - "Girsanov 定理"
  - "路径空间 SMC"
  - "无导数引导"
  - "reward-tilted 采样"
---

# Simple Approximation and Derivative Free Inference-Time Scaling for Diffusion Models via Sequential Monte Carlo on Path Measures

**会议**: ICML 2026  
**arXiv**: [2605.17850](https://arxiv.org/abs/2605.17850)  
**代码**: 未公布  
**领域**: 扩散模型 / 推理时引导 / Sequential Monte Carlo  
**关键词**: 推理时缩放、Girsanov 定理、路径空间 SMC、无导数引导、reward-tilted 采样

## 一句话总结
作者把扩散模型的推理时 reward 引导从"粒子空间 SMC + 高阶导数"升级为"路径空间 SMC + Girsanov 似然比"，得到 URGE 算法：每条轨迹只需对 guidance $G$ 做一阶梯度并累加一个简单的 Itô 项当权重，完全不需要 reward $r$ 的导数 / Hessian / score 估计，在 GMM、逆问题和文生图三类任务上都打平或优于 FK-Corrector / AFDPS / FK-Steering。

## 研究背景与动机

**领域现状**：扩散模型把生成看作一条 SDE $dX_t = v(X_t,t)dt + V(t)dW_t$，部署时常需要在不微调的前提下把样本"扭"向某个 reward $\mathbf{r}(x)$，目标分布是 reward-tilted 后验 $q(x)\propto p_\text{data}(x)\mathbf{r}(x)$。主流做法是 guidance——把 drift 改成 $v + V^2\nabla_x G$ 来逼近这个后验。

**现有痛点**：guidance 实际采样的并不是真正的 $q$，因为严格做法需要 Doob $h$-变换 $h(x,t)=\mathbb{E}[\mathbf{r}(X_T)\mid X_t=x]$，而 $h$ 要解一个 backward Kolmogorov 方程，高维下基本无法精确得到。近期的修正方案（FK-Corrector、AFDPS）走"粒子空间 SMC"路线，对每个粒子算一个无偏权重然后重采样——但权重里塞了 $\Delta_x r$、$\|\nabla_x r\|^2$、$\nabla_x \log p_t$ 等高阶量，需要对 reward 求二阶导，对 score 函数求值，碰到神经网络打分器（如 ImageReward、HPS）就直接卡壳。

**核心矛盾**："想要无偏的 reward-tilted 采样"与"实际能计算的权重项"之间的鸿沟：粒子空间的无偏修正天然涉及生成器 $\mathcal{L}^G$，而生成器作用在 $r$ 上必然引入二阶导。

**本文目标**：找到一个既保留 SMC 无偏性、又不需要 reward 导数的权重构造方式，让黑盒神经 reward 也能直接用。

**切入角度**：作者跳出"对每个时刻的粒子打权重"框架，转到"对整条轨迹打权重"——既然 guided SDE 和 reference SDE 之间的路径测度比可以通过 Girsanov 定理写出闭式表达，那直接在路径空间做 SMC 即可。

**核心 idea**：用 Girsanov 路径似然比 $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G$ 乘上 $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}=\exp(r(X_t)-r(X_0))$ 当作轨迹的重要性权重；权重里只出现 $\nabla_x G$（guidance 自身的梯度，本来就要算）和 $r$ 的差分，不含 $r$ 的任何导数。

## 方法详解

### 整体框架
URGE 要解的是"在不微调预训练扩散模型的前提下，把样本无偏地采到 reward-tilted 后验 $q(x)\propto p_\text{data}(x)e^{\mathbf{r}(x)}$"这件事，而它的转化思路是：不再像 FK-Corrector / AFDPS 那样在每个时刻给单个粒子打含二阶导的权重，而是整条轨迹只算一次 Girsanov 路径似然比当权重。具体地，给定 drift $v(x,t)$、guidance 势能 $G(x,t)$ 与 reward $r(x,t)$（需 $r(x,T)=\mathbf{r}(x)$），算法并行模拟 $N$ 条 guided 轨迹、每个 Euler-Maruyama 小步 $\Delta t$ 后给每条轨迹乘一个只依赖 $\nabla_x G$ 和 $r$ 差分的权重 $\beta^{(i)}$，再按 $\beta$ 归一化做 Categorical 重采样复制高权重、淘汰低权重粒子；走满 $K$ 步后终末粒子 $\{X_T^{(i)}\}$ 就是 $q$ 的近似样本，并在 $\Delta t\to 0,\,N\to\infty$ 下严格无偏。

### 关键设计

**1. 路径空间 Girsanov 权重：用一阶量打包二阶导，并把随机路径信息写进权重**

现有粒子空间修正（FK-Corrector / AFDPS）的权重里塞满 $\Delta_x r$、$\|\nabla_x r\|^2$、$\nabla_x\log p_t$，碰到 ImageReward / HPS 这类黑盒神经 reward 就因为求不出 Hessian 而卡死。URGE 的破法是把无偏性需求搬到路径测度上：对无 guidance 的 reference 测度 $\mathbb{P}$ 与 guided 测度 $\mathbb{P}^G$ 用 Girsanov 定理写出闭式比值 $\mathrm{d}\mathbb{P}/\mathrm{d}\mathbb{P}^G \propto \exp(-\int_0^t V(s)\nabla_x G^\top dW_s - \tfrac{1}{2}\int_0^t V^2\|\nabla_x G\|^2 ds)$，再乘上 reward-tilted 测度对 $\mathbb{P}$ 的密度 $\exp(r(X_t)-r(X_0))$，就得到目标权重 $\mathrm{d}\mathbb{Q}/\mathrm{d}\mathbb{P}^G$。Euler-Maruyama 离散化后它落地成 $\beta^{(i)}_{s,t}=\exp\!\big(r(X_t)-r(X_s) - V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)} - \tfrac{1}{2}V(s)^2\|\nabla_x G\|^2(t-s)\big)$，其中 $\xi^{(i)}$ 正是 EM 步本来就要抽的高斯噪声，直接复用、几乎零额外开销。整个表达式里只剩 $\nabla_x G$（guidance 自身梯度，本来就算）和 $r$ 的数值差分，彻底没有 reward 导数——这才让 URGE 第一次能直接接到只能黑盒求值的神经打分器上。更妙的是，权重第一项 $-V(s)\nabla_x G^\top\sqrt{t-s}\,\xi^{(i)}$ 本身是个 Itô 积分（连续形式 $\int_s^t -V(\tau)\nabla_x G^\top dW_\tau$），把每个 EM 步抽到的噪声 $\xi^{(i)}$ 反写进权重——而 AFDPS / FK-Corrector 的权重只是终点 $x$ 的确定性函数，默认"同一终点不管走哪条路径都同样可信"，丢掉了扩散过程里关键的随机性。于是在 URGE 里，同一终点上"顺着 guidance 漂过来"和"逆着噪声硬撞过来"的两条轨迹会拿到不同权重，重采样因此更精细，论文实测方差更低、对粒子数 $N$ 的 scaling 也更稳。

**2. 路径-粒子等价性定理：证明 URGE 是 AFDPS 的母体而非另一种近似**

为了说明换到路径空间不是又引入一层近似，作者定义瞬时强度 $\lambda(x,t):=\lim_{h\to 0}\tfrac{1}{h}\big(\mathbb{E}_{\mathbb{P}^G}[w^\text{URGE}_{t-h,t}\mid X_t=x]-1\big)$，借 Feynman-Kac 反向值函数推出 marginalized 生成器 $\mathcal{L}^\text{eff}_t = \mathcal{L}^G_t + \lambda(\cdot,t)$，并在 Theorem 3.3 中证明 $\lambda(x,t)\equiv w_\text{AFDPS}(x,t)$——也就是把 URGE 的路径权重对终点取条件期望，恰好回收 AFDPS 的全部二阶项。这一等价性一箭双雕：既让 URGE 继承 AFDPS 的无偏性，又点明 AFDPS 只是 URGE 条件期望后的特例，路径空间因此保留了更大的设计自由度（可换更高阶离散格式、更稀疏的时间网格）。

### 损失函数 / 训练策略
URGE 是纯推理时算法，**不需要任何额外训练**。超参只有粒子数 $N$、离散步数 $K$、guidance 强度（通常令 $G=r$ 或取文生图里的 CFG 项）。论文用 EM 离散化的最简版本（公式 7）即可工作，并指出权重构造可替换为任意更高阶格式，以在 $N$ 受限时换取精度。

## 实验关键数据

### 主实验

30 维 40-component GMM 玩具任务（reward 选为已知二次函数，便于解析对照）：

| 方法 | MMD↓ | SWD↓ | Mean $\ell_2$↓ | Cov Frob↓ |
|------|------|------|----------------|-----------|
| Pure Guidance | 0.17 | 1.68 | 7.14 | 469.09 |
| AFDPS | 0.10 | 1.04 | 5.07 | 335.19 |
| AFDPS+VCG | 0.08 | 0.83 | 4.13 | 246.61 |
| FK-Steering | 0.07 | 0.85 | 4.86 | 198.20 |
| **URGE** | **0.06** | **0.62** | **3.20** | **181.31** |

URGE 在 4 项指标上全部最优，特别是协方差 Frobenius 误差比 AFDPS+VCG 还低 26%，且不需要 VCG 那种额外学控制 drift 的步骤。

ImageNet-256 上 4 个逆问题（PSNR↑/LPIPS↓）：

| 方法 | Gaussian Deblur PSNR | Motion Deblur LPIPS | Super-Res PSNR | Box Inpaint LPIPS |
|------|----------------------|---------------------|----------------|--------------------|
| SGS-EDM | 22.09 | 0.526 | 15.43 | 0.298 |
| FK-Corrector | 18.36 | 0.601 | 18.58 | 0.714 |
| AFDPS-SDE | 22.43 | 0.520 | 21.03 | 0.307 |
| AFDPS-ODE | 22.57 | 0.503 | 19.60 | **0.275** |
| **URGE** | 22.38 | 0.525 | 21.00 | 0.305 |

URGE 与最强的 AFDPS 变体打平（用 $\nabla\log p_t$ 当 reward 时仍需 score，但路径权重本身仍是 derivative-free 构造），且远超 FK-Corrector。

### 消融实验

文生图（Stable Diffusion v1.5，50 prompts × 3 seeds）：

| 采样器 | CLIP-Score↑ | HPS↑ | ImageReward↑ | GenEval↑ |
|--------|-------------|------|--------------|----------|
| Base $N=1$ | 0.273 | 0.262 | 0.214 | 0.640 |
| 梯度引导 $N=1$ | 0.273 | 0.262 | 0.207 | 0.640 |
| FK-Steering $N=4$ | 0.290 | 0.285 | 0.840 | 0.720 |
| 梯度版 FK $N=4$ | 0.290 | 0.284 | 0.791 | 0.747 |
| **URGE $N=4$** | **0.300** | **0.293** | **0.996** | **0.780** |

ImageReward 从 base 的 0.21 跳到 0.996（×4.7），CLIP / HPS / GenEval 也全面领先；并且作者强调 SDv1.5 + URGE 在双对象 prompt 上经常追平甚至超过 SDXL baseline。

### 关键发现
- **derivative-free 不掉点**：把 $\Delta_x r$、$\|\nabla_x r\|^2$、$\nabla_x \log p_t$ 全砍掉后，URGE 在 GMM 和逆问题上反而比保留这些项的 AFDPS 更准——印证 Itô 路径项注入的随机信息抵消了二阶项的去除。
- **粒子 scaling 单调上升**：图 3 显示 ImageReward 随 $N$ 单调增长，而 FK-Steering 在 $N$ 大时出现 plateau；说明路径空间的权重方差更稳。
- **小模型反超大模型**：SDv1.5 + URGE ($N=4$) 的 ImageReward 高于 SDXL base，意味着同样算力下"加 SMC 重采样"可能比"换更大模型"更划算。
- **黑盒 reward 才是杀手锏**：FK-Corrector / AFDPS 在文生图设置里完全不能用（无法求神经 reward 的 Hessian），URGE 是唯一能直接接入 ImageReward / HPS 的 SMC 方案。

## 亮点与洞察
- **测度论视角的代换**：把"对每个粒子打权重"换成"对每条轨迹打权重"看似只是符号变化，但 Girsanov 定理直接把无穷小生成器里的所有二阶项打包成一个 Itô 积分，工程上变成"复用 EM 步的 $\xi^{(i)}$ 就行"，是典型的"用更深的数学换更简单的代码"。
- **等价性定理的解释力**：Theorem 3.3 同时充当"正确性证明"和"设计自由度论据"——既然 AFDPS 是 URGE 条件期望后的特例，那 URGE 显然有更多选项（更稀疏离散、更高阶格式、不同时间网格），未来方向自然打开。
- **可迁移技巧**：这种"路径权重 ≡ 粒子权重对终点取条件期望"的 Feynman-Kac 对偶论证可以套到任何基于 SMC 的扩散模型推理算法，例如 reward fine-tuning、分子构象采样、蛋白质 inverse folding 里的 guidance 校正。

## 局限与展望
- 论文实验只到 $N=4 \sim 16$，没系统测试 $N$ 大到 $\sim 100$ 时是否会因为重采样退化（particle degeneracy）而饱和，文生图里的 ImageReward 接近上限可能就是这种退化的前兆。
- 权重里仍含 $\nabla_x G$，所以"无导数"严格说是"无 reward 导数"——guidance 势能 $G$ 自身需要可微（CFG 中 $G$ 通常就是 classifier-free 项，本来就可微，但若 $G$ 也变成黑盒，URGE 退化为只剩 $r$ 差分，无偏性消失）。
- 离散步长 $\Delta t$ 必须足够小以保证 Girsanov 离散化稳定，作者没给"自适应 $\Delta t$"方案，对长 horizon 的视频扩散模型可能成为瓶颈。
- 等价性定理只在 $N\to\infty$ + 连续时间极限下成立，有限 $N$ 下 URGE 与 AFDPS 的方差差距有理论保证但论文未给闭式 bound。
- 未来工作：把高阶 SDE 离散格式（Heun / DPM-Solver-2）和 URGE 路径权重耦合；研究 reward 是非光滑（如 GenEval 这类离散指标的 surrogate）时的 URGE 变体；将 URGE 拓展到 jump diffusion 以支持 categorical / discrete diffusion。

## 相关工作与启发
- **vs FK-Corrector (Skreta et al., 2025) / AFDPS (Chen et al., 2025)**：它们在粒子空间用 $\mathcal{L}^G + w_\text{AFDPS}$ 修正生成器，权重含 $\Delta_x r$、$\|\nabla_x r\|^2$、$\nabla_x \log p_t$；URGE 在路径空间用 Girsanov 权重，二阶项全消失，理论上等价（Theorem 3.3）但工程上更简单且支持黑盒 reward。
- **vs FK-Steering (Singhal et al., 2025)**：FK-Steering 用 $r(X_{t+\Delta t})-r(X_t)$ 当权重，丢掉了 Girsanov 路径修正项，因此并非无偏；URGE 多了路径信息项，无偏性有保证且实验全面占优。
- **vs Doob $h$-transform 方法（DEFT / 各种 adjoint matching）**：$h$-transform 需要训练一个网络估计 $h$ 或解 backward Kolmogorov 方程；URGE 完全免训，只在推理时多花 $N$ 倍 forward。
- **vs VCG (Ren et al., 2025a)**：VCG 通过加权最小二乘学一个 control drift 来降方差；URGE 不学任何东西，却在 GMM 上比 AFDPS+VCG 还低 26% 协方差误差，说明路径空间天然就更稳。
- **思想史脉络**：本文延续了"用 SMC + 扩散做无偏推理时引导"的近年线索（Wu et al. 2023; Singhal et al. 2025; Skreta et al. 2025; Chen et al. 2025），将其推到"derivative-free + 路径空间"的简洁极致；进一步往前是经典 SMC（Del Moral 2004）和金融数学里的 Girsanov 重要性采样。

## 评分
- 新颖性: ⭐⭐⭐⭐ 路径空间 SMC + Girsanov 权重的组合在扩散模型推理时缩放里是新的，但 Girsanov 重要性采样在金融 / 分子模拟里早是经典工具，迁移而非全新发明。
- 实验充分度: ⭐⭐⭐⭐ 覆盖玩具 GMM + 4 个逆问题 + 文生图三档难度，但 $N$ 的范围偏小、缺乏对超长 horizon 视频扩散的验证、未给运行时间详细对比表。
- 写作质量: ⭐⭐⭐⭐ 把 Girsanov、Feynman-Kac、Kolmogorov backward 三件套梳理得清楚，Table 1 一眼看清各方法权重差异；术语"approximation-free"重复出现略冗余。
- 价值: ⭐⭐⭐⭐⭐ derivative-free + 黑盒 reward 友好意味着 ImageReward / HPS / 人类偏好打分器可以直接插，文生图、视频、3D 等任何用神经 reward 做引导的场景都能直接迁移；理论 + 工程双重清爽，可复用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] SURGE: Approximation and Training Free Particle Filter for Diffusion Surrogate](surge_approximation_and_training_free_particle_filter_for_diffusion_surrogate.md)
- [\[CVPR 2026\] Rethinking Prompt Design for Inference-time Scaling in Text-to-Visual Generation](../../CVPR2026/image_generation/rethinking_prompt_design_for_inference-time_scaling_in_text-to-visual_generation.md)
- [\[ICML 2025\] Performance Plateaus in Inference-Time Scaling for Text-to-Image Diffusion Without External Models](../../ICML2025/image_generation/performance_plateaus_in_inference-time_scaling_for_text-to-image_diffusion_witho.md)
- [\[NeurIPS 2025\] Inference-Time Scaling for Flow Models via Stochastic Generation and Rollover Budget Forcing](../../NeurIPS2025/image_generation/inference-time_scaling_for_flow_models_via_stochastic_generation_and_rollover_bu.md)
- [\[CVPR 2026\] Denoising as Path Planning: Training-Free Acceleration of Diffusion Models with DPCache](../../CVPR2026/image_generation/dpcache_denoising_path_planning_diffusion_accel.md)

</div>

<!-- RELATED:END -->
