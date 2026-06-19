---
title: >-
  [论文解读] Flatness-Aware Stochastic Gradient Langevin Dynamics
description: >-
  [ICML 2026][AI安全][SGLD] 本文提出 fSGLD：在标准 SGLD 更新里把梯度处的参数 $\theta$ 换成被高斯扰动过的 $\theta+\epsilon$，并将扰动尺度 $\sigma$ 与逆温度 $\beta$ 通过 $\sigma=\beta^{-(1+\eta)/4}$ 严格耦合，从而在不增加任何梯度/内存开销的前提下，让算法的不变测度逼近 Hessian-trace 正则化目标 $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 对应的 Gibbs 分布，并给出 Wasserstein-1 与超额风险的非渐近界…
tags:
  - "ICML 2026"
  - "AI安全"
  - "SGLD"
  - "平坦最小值"
  - "Hessian-trace 正则"
  - "Gibbs 分布"
  - "随机权重扰动"
---

# Flatness-Aware Stochastic Gradient Langevin Dynamics

**会议**: ICML 2026  
**arXiv**: [2510.02174](https://arxiv.org/abs/2510.02174)  
**代码**: https://github.com/youngsikhwang/Flatness-aware-SGLD (有)  
**领域**: 优化 / 贝叶斯采样 / 平坦最小值  
**关键词**: SGLD, 平坦最小值, Hessian-trace 正则, Gibbs 分布, 随机权重扰动

## 一句话总结
本文提出 fSGLD：在标准 SGLD 更新里把梯度处的参数 $\theta$ 换成被高斯扰动过的 $\theta+\epsilon$，并将扰动尺度 $\sigma$ 与逆温度 $\beta$ 通过 $\sigma=\beta^{-(1+\eta)/4}$ 严格耦合，从而在不增加任何梯度/内存开销的前提下，让算法的不变测度逼近 Hessian-trace 正则化目标 $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 对应的 Gibbs 分布，并给出 Wasserstein-1 与超额风险的非渐近界，在 CIFAR/WebVision/ViT 上取得与 SAM/ASAM 相当或更优、但训练时间近乎减半的效果。

## 研究背景与动机
**领域现状**：深度网络泛化与损失曲面的"平坦性"高度相关，主流做法是 SAM 系列（min-max 内层扰动 + 双梯度）与 Entropy-SGD/Entropy-MCMC（引入辅助变量做局部熵平滑），它们都能把训练推向低曲率盆地，但代价不小：SAM 每步两次梯度，Entropy 系列翻倍内存。

**现有痛点**：这些方法本质都是"局部"的——只用当前点周围一小圈的几何信息，因此在多模态、高度非凸的损失曲面上很难跳出尖锐盆地；理论保证也基本只到局部收敛。另一条线是 Langevin 类全局采样（SGLD），理论上能在足够低温下集中到全局极小，但它的不变测度 $\pi_\beta^{\text{SGLD}}\propto\exp(-\beta u)$ 完全由目标函数决定，对曲面几何无感，所以它能找到的是"任意一个"全局极小，而不是"平坦的"全局极小。

**核心矛盾**：现有体系里没有同时具备 (a) 全局探索能力、(b) 对低曲率区域的归纳偏置、(c) 与 SGD 同等计算/内存代价 这三点的算法。Entropy-MCMC 算是最接近的工作，但它需要辅助变量、内存翻倍，理论也只在强凸下成立。

**本文目标**：设计一个一阶、不增任何额外梯度或内存的 Langevin 算法，使其不变测度集中在"Hessian-trace 正则化目标" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 的全局极小（即"全局平坦最小值"），并在非凸设定下给出非渐近 Wasserstein 与超额风险界。

**切入角度**：作者敏锐地注意到，把 SGLD 里的梯度 $\nabla U(\theta,X)$ 换成在扰动点 $\theta+\epsilon$ 处求的扰动梯度 $\nabla U(\theta+\epsilon,X)$，其期望恰好是随机化平滑代理 $g_\epsilon(\theta)=\mathbb{E}[u(\theta+\epsilon)]$ 的梯度；而 $g_\epsilon$ 的二阶 Taylor 展开正好等于 $u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 加上一个高阶残差。换句话说，"扰动梯度 + Langevin 噪声"天然内嵌了 Hessian-trace 正则——只要能控住那个高阶残差。

**核心 idea**：用一条"$\sigma$–$\beta$ 耦合公式" $\sigma=\beta^{-(1+\eta)/4}$（$\eta$ 固定在 0.1）同时充当采样温度与扰动尺度的桥梁，使得当 $\beta$ 升高时残差恰好以可控速率消失，从而让 fSGLD 的不变测度严格逼近"平坦偏置 Gibbs 分布" $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$。

## 方法详解

### 整体框架
fSGLD 想解决的是"既要全局探索、又要偏向平坦盆地、还不能比 SGD 更贵"这个三难。它的做法出奇地小：和标准 SGLD 唯一的差别只是"在哪个点上算梯度"——把梯度从当前参数 $\theta_k$ 挪到一个被高斯扰动过的点 $\theta_k+\epsilon_{k+1}$，再配一条把扰动尺度 $\sigma$ 绑死到采样温度 $\beta$ 上的解析公式。输入是初始参数 $\theta_0$ 和数据分布，输出是参数链 $\{\theta_k\}$，既能像 SGLD 一样做后验平均得贝叶斯预测器，也能当普通优化器取末态参数。

### 关键设计

**1. 扰动梯度：在零额外开销下注入二阶曲率信息**

针对的痛点是 SAM 要算两次梯度、Hessian-penalty 要近似 Hessian-vector product，二阶信息向来很贵。fSGLD 把 SGLD 更新里的 $\nabla_\theta U(\theta_k,X_{k+1})$ 直接换成扰动点处的 $\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})$，其中 $\epsilon_{k+1}\sim\mathcal{N}(0,\sigma^2 I_d)$，配上标准 Langevin 噪声 $\xi_{k+1}\sim\mathcal{N}(0,I_d)$，整步写作 $\theta_{k+1}=\theta_k-\lambda\,\nabla_\theta U(\theta_k+\epsilon_{k+1},X_{k+1})+\sqrt{2\lambda\beta^{-1}}\,\xi_{k+1}$。

这一步看着只是"给权重加点噪声"，但它的期望藏着曲率：扰动梯度的期望恰是随机化平滑代理的梯度 $\mathbb{E}_{\epsilon,X}[\nabla_\theta U(\theta+\epsilon,X)]=\nabla g_\epsilon(\theta)$，而 $g_\epsilon$ 在高斯期望下的二阶 Taylor 展开正好是 $g_\epsilon(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))+\mathbb{E}[\mathcal{R}(\theta,\epsilon)]$。也就是说，一次高斯扰动就把 Hessian-trace 偷偷塞进了优化目标，省掉了显式上升梯度和 Hessian 近似，算法仍保持 SGLD 的单次梯度与 $O(d)$ 内存。

**2. $\sigma$–$\beta$ 耦合公式：让近似误差和平坦偏置同步衰减**

设计 1 留了个隐患：Taylor 展开有个残差 $\mathbb{E}[\mathcal{R}(\theta,\epsilon)]=O(\sigma^4 d^2)$，要是扰动尺度 $\sigma$ 当成独立超参乱设，要么残差炸掉、破坏 Hessian-trace 偏置，要么扰动太小、退化回普通 SGLD，两头不讨好且写不出统一的非渐近界。fSGLD 的解法是不让 $\sigma$ 自由，而用一条解析关系把它绑到采样温度上：$\sigma=\beta^{-(1+\eta)/4}$，$\eta$ 固定在 $0.1$。

这条公式不是调出来的而是反推出来的。Proposition 3.4 证明在 $\eta\in(0,1)$ 时 $W_2(\pi^{\text{fSGLD}}_\beta,\pi^\star_{\beta,\sigma})=O(\beta^{-\eta/4}\sqrt d+\beta^{-\eta/2}d+\beta^{-(1+\eta)/2}d^2)$，靠增大 $\beta$ 就能把近似误差压到任意小；而同一条 $\sigma=\beta^{-(1+\eta)/4}$ 又保证 $\beta\to\infty$ 时平坦偏置不会太快消失，于是存在一个有限 $\beta$ 的"甜区"。换句话说，耦合把"近似精度 vs 平坦偏置强度"的 trade-off 压成了一条单参数曲线——对用户暴露的超参数因此和 SGLD 完全一样，只需调 $\beta$ 和步长 $\lambda$。

**3. 平坦偏置 Gibbs 分布：把"找平坦极小"升级成可证明的采样目标**

前两个设计回答了"怎么做"，这一点回答"做到了什么"。fSGLD 把启发式的"找平坦盆地"明确成一个概率测度 $\pi^\star_{\beta,\sigma}\propto\exp(-\beta v(\theta))$，其中 $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 就是 Hessian-trace 正则化目标，并围绕它给出非渐近保证。

在标准 SGLD 假设（四阶可微 + 数据相关 Lipschitz + 耗散性）下，Theorem 3.5 给出 $W_1(\mathcal{L}(\theta_k^{\text{fSGLD}}),\pi^\star_{\beta,\sigma})\le D_1 e^{-\dot c\lambda k/2}+(D_2+D_3)\sqrt\lambda+\underline{D}$，三项分别是过阻尼 Langevin 的指数混合、Euler–Maruyama 的离散误差 $O(\lambda^{1/2})$、以及不变测度偏差；Theorem 3.8 进一步把这翻译成超额风险界 $\mathbb{E}[v(\theta_k)]-\inf v\le D_1^\diamond e^{-\dot c\lambda k/4}+D_2^\diamond\lambda^{1/4}+D_3^\diamond$。它的意义在于：以往 Langevin 全局收敛理论都瞄准原目标 $u$ 的极小，这是第一次把目标换成平坦目标 $v$，证明算法偏置不是"跑出来的"而是被刻画出来的"平坦极小的全局采样"，且离散化速率与最优的标准 SGLD 分析（Zhang et al., 2023）持平——加平坦偏置没有牺牲收敛速率。

### 损失函数 / 训练策略
作者没有显式改损失函数——优化的"有效目标" $v(\theta)=u(\theta)+\tfrac{\sigma^2}{2}\mathrm{tr}(H(\theta))$ 是由算法动力学隐式定义的。实现上只需把 SGLD 里"梯度处的参数"加一次高斯扰动即可；$\eta=0.1$ 全程固定，$\beta$ 与步长 $\lambda$ 按各 benchmark 的标准 SGLD 调度。理论上还要求 $\beta$、$\lambda$、迭代数 $k$ 满足公式 (63)–(65) 的下/上界，以保证 $W_1$ 误差 $\le\bar\delta$。

## 实验关键数据

### 主实验

ResNet-18 上的贝叶斯图像分类（贝叶斯模型平均，结果取 3 个随机种子均值±std；除 fSGLD 与 ASAM 外其他基线引自 Entropy-MCMC 原文）：

| 数据集 | 指标 | fSGLD | 之前 SOTA | 提升 |
|--------|------|-------|-----------|------|
| CIFAR-10 | ACC % ↑ | **95.73** | Entropy-MCMC 95.69 | +0.04 |
| CIFAR-10 | NLL ↓ | **0.144** | ASAM 0.150 | -0.006（≈ 4% 相对） |
| CIFAR-100 | ACC % ↑ | 78.53 | Entropy-MCMC **79.16** | -0.63（第三） |
| CIFAR-100 | NLL ↓ | **0.810** | ASAM 0.814 | -0.004 |
| CIFAR-10→SVHN OOD | AUROC % | **98.91** | Entropy-SGD 98.71 | +0.20 |
| CIFAR-100→SVHN OOD | AUPR % | **88.01** | ASAM 87.93 | +0.08 |

ResNet-34/50 在带噪声标签的 CIFAR-N 和 WebVision 上从头训练（5 个种子均值；s/epoch 在 CIFAR-10N 上测得）：

| Model | Optimizer | CIFAR-10N | CIFAR-100N | WV-1 | WV-5 | s/epoch |
|-------|-----------|-----------|------------|------|------|---------|
| ResNet-34 | SGD | 89.31 | 58.47 | 71.87 | 89.33 | 22.0 |
| ResNet-34 | SAM | 91.53 | 59.18 | 73.49 | **90.32** | 41.3 |
| ResNet-34 | ASAM | **91.73** | 60.79 | 73.46 | 90.14 | 41.4 |
| ResNet-34 | **fSGLD** | 91.37 | **61.51** | **73.95** | 90.03 | **23.7** |
| ResNet-50 | SAM | 90.88 | 59.01 | 72.52 | 89.53 | 60.7 |
| ResNet-50 | ASAM | **91.25** | 60.47 | 71.92 | 88.48 | 60.9 |
| ResNet-50 | **fSGLD** | 90.86 | **61.26** | **73.54** | **90.34** | **34.1** |

ViT-B/16 微调：fSGLD 在 CIFAR-100N 上 75.67，超过 ASAM 的 74.86，而单 epoch 用时 345.8s（SAM 656.7s、ASAM 662.5s），近乎减半。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|----------|------|
| 耦合 $\sigma=\beta^{-(1+\eta)/4}$，$\eta\in(0,1)$ | 性能稳定在峰值 | 推荐 $\eta=0.1$ |
| 固定 $\beta=10^8$ 扫 $\sigma$ | 隐含 $\eta\notin(0,1)$ 时显著掉点 | 验证扰动尺度不能脱离温度独立设置 |
| 固定 $\sigma=10^{-3}$ 扫 $\beta$ | 同上 | 反向验证：温度同样不能脱离扰动独立设置 |
| Hessian 谱比较 (ResNet-34 / CIFAR-10N) | fSGLD 的 $\lambda_{\text{top}}$ 与 $\mathrm{tr}(H)$ 都明显小于 SGD/SGLD | 直接证实 fSGLD 收敛到更平坦的极小 |

### 关键发现
- 与 SAM/ASAM 相比：fSGLD 在 CIFAR-100N、WebVision Top-1 这种更"难"（高噪 + 多类）的任务上反超，**且训练时间约为 SAM/ASAM 的一半**——证明"用扰动梯度替代显式二阶"既省又好。
- 与 Entropy-MCMC 相比：fSGLD 不需要辅助变量，内存减半，性能在 CIFAR-10 上反超，CIFAR-100 上略低 0.6%（但 NLL 更优）。
- $\eta$ 在 $(0,1)$ 内对性能几乎不敏感（图 1 在很宽范围内保持峰值平台），说明耦合公式既必要又稳健，工程上调一个 $\beta$ 就够。
- Hessian 谱实验给出"算法机制 → 几何效果"的闭环验证：理论说 fSGLD 隐式正则 $\mathrm{tr}(H)$，实验上 $\mathrm{tr}(H)$ 真的明显变小。

## 亮点与洞察
- **"随机化平滑 = 隐式 Hessian-trace 正则"** 这个等价被用得很干净：作者没有引入任何辅助变量、Hessian-vector 估计或双梯度，把 SAM/Hessian-penalty 想要的东西全部塞进 SGLD 的一次扰动里。
- **把超参数耦合做成理论结论而不是工程 trick**：$\sigma=\beta^{-(1+\eta)/4}$ 不是经验调出来的，而是由 Wasserstein 界和 Taylor 残差量级反推出来的最优耦合速率；正因如此，作者敢声明"fSGLD 暴露给用户的超参数和 SGLD 一样多"。
- **可迁移设计**：任何基于 Langevin/扩散的优化器（如训练扩散生成模型、Bayesian fine-tuning）都可以套用"扰动点处求梯度 + 温度耦合扰动"这一招来无痛获得平坦偏置；作者也在结论里点名扩散生成是下一步方向。
- **理论范式的小升级**：从"对原目标 $u$ 的 Wasserstein 收敛"切到"对平坦目标 $v$ 的 Wasserstein 收敛"，第一次给出"采样收敛到平坦极小"的非渐近、全局结果——之前这条路只有局部 PAC-Bayes 界。

## 局限与展望
- 作者承认的局限：常数 $D_1,D_3$ 对维度 $d$ 与温度 $\beta$ 是指数依赖（继承自 Eberle 等的耦合论证），这是当前 SGLD 理论的天花板；以及现有分析需要 $u$ 满足 Assumption 3.2 的全局 Lipschitz，semiconvex 情形留待后续。
- 自己发现的局限：理论上的 $\beta$、$\lambda$、$k$ 选取（公式 63–65）涉及与 $d^2$ 同阶的常数，工程上无法直接套用，实际还是按 SGLD 经验调 $\beta$；实验全部集中在 ResNet/ViT 图像分类，没有验证 NLP、检测、扩散生成等更高维的任务，能否扩展到现代 LLM/扩散模型的训练规模还是开放问题。
- 改进思路：(i) 把 $\eta$ 做成 schedule（前期大、后期小）以兼顾探索与精度；(ii) 与 preconditioned/replica-exchange SGLD 结合，缓解高维下的指数常数；(iii) 实证扩展到训练扩散生成模型，验证作者预言的"更平坦 → 更多样/高质量样本"。

## 相关工作与启发
- **vs SAM/ASAM**：SAM 用 min-max 在邻域内取最坏点做梯度，需要双梯度；fSGLD 用高斯期望取邻域平均，只需单梯度，且天然有全局采样属性（Langevin 噪声），不会困在局部尖谷。实验上 fSGLD 在高噪/大类任务反超，且训练时间减半。
- **vs Entropy-SGD / Entropy-MCMC**：两者都引入辅助变量近似局部熵 / 平坦后验，内存翻倍且 Entropy-MCMC 的理论只在强凸下成立。fSGLD 没有辅助变量、内存与 SGLD 一致，且在一般非凸+耗散假设下给出非渐近 Wasserstein 界。
- **vs 标准 SGLD（Welling-Teh / Raginsky / Zhang 2023）**：标准 SGLD 的 Gibbs 测度对几何无感，只能集中到 $u$ 的全局极小；fSGLD 把目标换成 $v=u+\tfrac{\sigma^2}{2}\mathrm{tr}(H)$，提供了第一个"采样到平坦极小"的非渐近全局结果，且离散化误差速率 $O(\lambda^{1/2})$ 与最优 SGLD 分析持平。
- **vs Random Weight Perturbation (RWP, Ahn 2024 等)**：RWP 通常把扰动尺度当独立超参，缺乏全局收敛保证；fSGLD 可视为"SGLD + 强制耦合的 RWP"，并把 RWP 的几何作用首次纳入 Langevin 全局非渐近分析框架。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一次把"扰动梯度 + 耦合温度"做成证明可行的平坦偏置 SGLD，理论与工程同时给出干净答案。
- 实验充分度: ⭐⭐⭐⭐ 贝叶斯分类/不确定性/OOD/有噪标签/ViT 微调全覆盖，且做了 $\beta$-$\sigma$ 解耦消融与 Hessian 谱可视化；缺点是只在视觉分类，没有 NLP 与生成任务。
- 写作质量: ⭐⭐⭐⭐ 概念递进清晰（动机→randomized smoothing→耦合→非渐近界→实验），公式略密但每步都给了直观说明；对相关工作交代到位。
- 价值: ⭐⭐⭐⭐⭐ 一阶、单梯度、无额外内存就能拿到 SAM/ASAM 量级或更优的平坦偏置，且训练时间减半，可作为通用 SGD 替换并搭载到任意 Bayesian 流程上，性价比极高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] FlowHijack: A Dynamics-Aware Backdoor Attack on Flow-Matching Vision-Language-Action Models](../../CVPR2026/ai_safety/flowhijack_a_dynamics-aware_backdoor_attack_on_flow-matching_vision-language-act.md)
- [\[ICML 2026\] FedHPro: Federated Hyper-Prototype Learning via Gradient Matching](fedhpro_federated_hyper-prototype_learning_via_gradient_matching.md)
- [\[ICML 2026\] Hidden in Plain Tokens: Simply Robust, Gradient-Free Watermark for Synthetic Audio](hidden_in_plain_tokens_simply_robust_gradient-free_watermark_for_synthetic_audio.md)
- [\[ICML 2026\] Gradient Transformer: Learning to Generate Updates for LLMs](gradient_transformer_learning_to_generate_updates_for_llms.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)

</div>

<!-- RELATED:END -->
