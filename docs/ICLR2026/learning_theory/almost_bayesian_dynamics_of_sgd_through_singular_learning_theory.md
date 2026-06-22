---
title: >-
  [论文解读] Almost Bayesian: Dynamics of SGD Through Singular Learning Theory
description: >-
  [ICLR 2026][学习理论][SGD动力学] 本文把长时间运行后的 SGD 描述为奇异损失地形上的多孔介质扩散，用局部学习系数刻画可达低损失区域的几何复杂度，并推导出 SGD 稳态分布近似等于经过可达性温度修正的贝叶斯后验。 领域现状：深度学习理论里有两条长期并行的线索。一条从优化动力学出发，把 SGD 看成带噪声的梯…
tags:
  - "ICLR 2026"
  - "学习理论"
  - "SGD 动力学"
  - "SGD动力学"
  - "奇异学习理论"
  - "局部学习系数"
  - "分数Fokker-Planck方程"
  - "贝叶斯后验"
---

# Almost Bayesian: Dynamics of SGD Through Singular Learning Theory

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=5ebDXlue3d](https://openreview.net/forum?id=5ebDXlue3d)  
**代码**: 未见开源代码  
**领域**: 学习理论 / SGD 动力学  
**关键词**: SGD动力学, 奇异学习理论, 局部学习系数, 分数Fokker-Planck方程, 贝叶斯后验  

## 一句话总结
本文把长时间运行后的 SGD 描述为奇异损失地形上的多孔介质扩散，用局部学习系数刻画可达低损失区域的几何复杂度，并推导出 SGD 稳态分布近似等于经过可达性温度修正的贝叶斯后验。

## 研究背景与动机
**领域现状**：深度学习理论里有两条长期并行的线索。一条从优化动力学出发，把 SGD 看成带噪声的梯度流，常用 Langevin 方程或 Fokker-Planck 方程描述参数分布如何随时间移动；另一条从贝叶斯统计出发，用奇异学习理论（Singular Learning Theory, SLT）解释神经网络这种非可辨识、Hessian 退化的模型为什么仍能泛化。

**现有痛点**：传统的 SGD-Bayes 联系通常依赖近似二次的局部极小值，例如把 SGD 近似成 Ornstein-Uhlenbeck 过程，再得到某种高斯后验解释。但神经网络的损失盆地通常高度退化：许多参数化对应同一个函数，Hessian 有大量接近 0 的方向，局部形状并不是规则抛物面。这样一来，正则模型里的 BIC、二次近似和普通 Brownian 扩散都很难解释真实 SGD 的后期轨迹。

**核心矛盾**：经验上，SGD 找到的解似乎和贝叶斯后验有关系，但 SGD 又不是在完整参数空间里自由采样。它受到初始化、噪声尺度、低损失连通结构和退化方向的限制，只能访问某些局部区域。问题就变成：如果神经网络是奇异模型，SGD 到底是在采样贝叶斯后验，还是在采样一个被动力学可达性扭曲过的版本？

**本文目标**：作者希望给出一个能同时回答三件事的理论框架：第一，为什么 SGD 后期常表现为次扩散而不是普通 Brownian 扩散；第二，奇异学习理论里的局部学习系数如何进入 SGD 轨迹；第三，在什么条件下 SGD 的稳态分布可以和贝叶斯后验建立显式关系。

**切入角度**：本文的关键观察是，退化损失地形可以类比为多孔介质。低损失参数不是一个规则欧氏球，而像被许多狭窄通道、平坦方向和瓶颈连接起来的可达区域。局部学习系数 $\lambda(w)$ 负责描述这一带“好参数体积”如何随误差阈值收缩，谱维度 $d_s$ 则描述 SGD 扩散过程实际能以多快速度探索这些可达状态。

**核心 idea**：用分数 Fokker-Planck 方程刻画 SGD 后期的异常扩散，再把 SLT 的局部学习系数转化为有效扩散系数，从而说明 SGD 稳态是贝叶斯后验的一个可达性加权版本。

## 方法详解

### 整体框架
本文不是提出一个新优化器，而是提出一个解释 SGD 长时间行为的理论模型。整体路线可以概括为：先从标准 SGD 的噪声动力学出发，指出普通 Langevin/Brownian 模型无法解释后期次扩散；再用分数 Fokker-Planck 方程描述带记忆的异常扩散；随后用奇异学习理论里的局部学习系数 $\lambda(w)$ 描述低损失区域的多孔几何；最后把几何维度和谱维度 $d_s$ 合成有效扩散系数 $D_\xi(w)$，解出局部稳态分布，并把它和贝叶斯后验联系起来。

更直白地说，论文把“SGD 会走到哪里”拆成两个问题。几何问题问：某个局部区域里到底有多少近似等价的低损失参数？动力学问题问：SGD 的噪声和梯度在有限时间内能访问这些参数中的多少？前者由局部学习系数控制，后者由谱维度和步行维度控制，两者合在一起决定 SGD 在不同损失盆地里的稳态权重。

这个框架关心的是训练后期，而不是初始化附近的剧烈漂移。作者承认 SGD 早期可能出现超扩散，但只要概率质量没有在训练中消失，稳态解主要由长时间的次扩散阶段决定。因此模型重点放在小学习率、大 batch、接近临界点的 regime：此时梯度噪声不再主导一切，退化低损失结构开始决定参数分布。

### 关键设计
**1. 分数 Fokker-Planck：把 SGD 后期从普通扩散改写成带记忆的次扩散**

标准 Langevin 视角会把 SGD 写成类似 $dw/dt=-\gamma \nabla L(w)+\Sigma_w$ 的随机微分方程，对应的位移尺度通常是 Brownian 型 $R(t)\propto t^{1/2}$。但论文和既有实验都观察到，神经网络训练早期可能超扩散，后期则常常变成 $R(t)\propto t^{1/d_{walk}}$ 且 $d_{walk}>2$ 的次扩散，甚至可接近 $R(t)\propto \log t$ 的超慢扩散。普通 Fokker-Planck 方程无法自然表达这种记忆效应。

作者因此把时间导数换成 Caputo 分数导数 $D_t^\alpha$，得到 SGD 的分数 Fokker-Planck 方程：

$$
D_t^\alpha p(w,t)=\nabla\cdot\left(D(w,t)\nabla p(w,t)-\gamma p(w,t)\nabla L_m[w]\right).
$$

这里 $p(w,t)$ 是参数分布，$D(w,t)$ 是扩散系数，$\gamma$ 是类似摩擦或学习率尺度的系数，$L_m$ 是经验损失。分数导数的作用不是装饰公式，而是让当前变化依赖过去轨迹的幂律记忆；这正好对应 SGD 在退化盆地中被瓶颈、平台和局部相变拖慢的现象。

**2. 局部学习系数：用 SLT 的有效维度替代规则模型的参数维度**

在规则统计模型中，BIC 的复杂度项来自 Hessian 非退化极小值附近的二次体积，复杂度基本由参数维度 $d/2$ 控制。但神经网络是奇异模型，Hessian 常常退化，等价参数化很多，低损失集合的体积不能用普通椭球体积描述。SLT 的局部学习系数（local learning coefficient, LLC）正是为这个问题服务。

论文使用局部奇异积分来刻画某个参数 $w^*$ 附近低损失区域的体积：

$$
V(\epsilon)=\int_{B_r(w^*,\epsilon)}\rho(w)dw,
$$

并用近似尺度关系 $V(\epsilon)\propto \epsilon^{\lambda(w^*)}$ 定义局部学习系数。直觉上，$\lambda(w)$ 越小，说明在该区域内低损失参数体积越“厚”、退化越强、模型局部有效复杂度越低。本文的关键转译是：把 $\lambda(w)$ 看成多孔介质里的局部质量维度，也就是低损失孔隙体积如何随误差高度收缩。

这个解释把 SLT 从静态泛化理论接到了 SGD 动力学上。LLC 不再只是“这个解复杂不复杂”的指标，而是变成了 SGD 在局部能否移动、能否停留、能否访问其他低损失状态的几何约束。

**3. 谱维度与步行维度：区分“有多少好参数”和“SGD 能访问多少”**

只有 $\lambda(w)$ 还不够，因为低损失区域体积大并不等于 SGD 能轻松走遍它。一个盆地可以很宽，但内部通道狭窄、瓶颈多、路径复杂，导致扩散速度很慢。论文因此引入谱维度 $d_s$，用它描述扩散过程在时间 $t$ 内实际占据状态体积的增长：

$$
V_s(t)\sim t^{d_s/2}.
$$

然后用步行维度 $d_{walk}$ 描述位移尺度：

$$
R(t)\sim t^{1/d_{walk}}.
$$

在接近临界点、局部多孔结构足够稳定时，作者借用 Alexander-Orbach 类型关系，把两者和 LLC 连起来：

$$
d_{walk}(t)=\frac{2\lambda(w_t)}{d_s}.
$$

这条关系的意思很重要：LLC 负责“低损失区域的几何容量”，谱维度负责“SGD 动力学看见的可达容量”。如果 $d_s$ 很小，SGD 即使身处大而平的区域，也可能只是缓慢爬行；如果相对谱维度更高，则说明它能在同一低损失区域里更充分地探索。

**4. 可达性 tempering：把 SGD 稳态写成贝叶斯后验的动力学修正版**

为了求稳态，论文还需要把位置相关、各向异性的扩散张量简化成可处理的标量扩散系数。作者给出的理由是，在大 batch、小学习率、训练后期的 regime 下，Hessian 和扩散张量的大多数特征值接近 0，有效扩散张量可近似为低秩乃至标量函数。再选择一个粗粒化尺度 $\xi$，可得到有效扩散系数：

$$
D_\xi(w)=\xi^{2-2\lambda(w)/d_s}.
$$

当某个局部区域 $W$ 内 $D_\xi$ 近似常数时，分数 Fokker-Planck 方程在稳态退化为普通稳态 Fokker-Planck 方程，解为：

$$
p_s(w)\propto \exp\left(-\frac{\gamma L_m[w]}{D_\xi}\right).
$$

若 $L$ 是 log loss 且为简化取 $\gamma=1$，作者进一步得到：

$$
p_s(w)^{mD_\xi}\propto p(X_m|w),
$$

从而

$$
p(w|X_m)=\frac{\rho(w)p_s(w)^{mD_\xi}}{Z_{mD_\xi}}.
$$

这就是标题里 “Almost Bayesian” 的来源。SGD 不是朴素地采样贝叶斯后验，而是先产生一个受局部可达性限制的稳态分布；把这个稳态分布按 $mD_\xi$ 做温度修正后，才和贝叶斯似然/后验对齐。低 LLC 区域会更容易吸引 SGD 解，但最终概率还要经过谱维度和粗粒化尺度决定的可达性校正。

### 一个完整示例
可以把论文的模型想成一个二维 moons 分类任务上的许多相同网络。每个网络从不同初始化出发，用 SGD 训练到低损失区域。训练结束后，这些解会落入若干参数空间 cluster：有的 cluster 被 SGD 频繁访问，有的 cluster 在贝叶斯后验里概率不低，但 SGD 从常见初始化和噪声尺度下很难到达。

传统说法可能会问：“SGD 样本和 SGLD 近似出来的贝叶斯后验是不是一样？”本文的回答更细：先估计每个 SGD 解附近的 LLC，观察 SGD 是否偏向低 LLC 区域；再用 SGLD 从低 loss、低 LLC 的解附近采样近似局部贝叶斯后验；最后按 $D_\xi$ 对 SGD 稳态概率做 tempering。实验中，当选择 $\xi=0.5$ 时，tempered SGD 分布与 SGLD 近似后验在 cluster 浓度上几乎重合，说明差异主要来自动力学可达性，而不是两者毫无关系。

这个例子也解释了为什么“SGD 是不是贝叶斯采样器”不能简单回答 yes/no。SGD 的原始样本会偏向某些更容易被动力学访问的 basin；但如果知道这些 basin 的局部几何和可达尺度，就可以把这种偏置校正回一个接近贝叶斯后验的分布。

### 损失函数 / 训练策略
本文没有提出新的训练 loss。理论分析默认使用经验损失 $L_m[w]$，在和贝叶斯后验相连时主要考虑 log loss 或等价的 KL divergence，因为此时 $e^{-mL_m[w]}$ 可以解释成似然 $p(X_m|w)$。

训练和估计策略服务于验证理论假设。LLC 使用 Lau 等工作和 devinterp 工具链里的估计器，核心形式可写为：

$$
\hat{\lambda}(w^*)=\frac{n}{\log n}\left(E_{w\mid B_r(w^*)}[L_n(w)]-L_n(w^*)\right).
$$

谱维度则从权重位移的幂律拟合中估计。作者记录总位移 $R(t)$，再用

$$
\log R(t)=\frac{d_s}{2\lambda(w)}\log t+c
$$

做线性回归，得到 $d_s$ 和拟合质量 $r^2$。因此实验中的“训练策略”本质上是：让模型进入足够长的后期训练阶段，周期性估计 LLC 与位移，再检查 $d_s\leq \bar{\lambda}$、位移预测和 tempered posterior 是否成立。

## 实验关键数据

### 主实验
作者在三类设置中验证理论：MNIST 上的全连接 ReLU 网络、TinyStories 上的小语言模型、Tiny ImageNet 上的视觉模型微调。主结果关注两个问题：次扩散预测是否拟合权重位移，以及谱维度是否被平均 LLC 上界约束。

| 模型 / 设置 | $\lambda$ | $d_s$ | $\alpha$ | $r^2$ | 说明 |
|--------|------|------|------|------|------|
| TinyStories-1M | 32 | 21.422 | 0.33 | 0.98 | 小语言模型继续训练，次扩散拟合较好 |
| TinyLlama-15M | 76.1 | 48.3 | 0.32 | 0.98 | 更大语言模型，仍满足 $d_s<\lambda$ |
| TinyStories-33M | 39.3 | 38.7 | 0.49 | 0.98 | $d_s$ 接近但略低于 LLC，拟合稳定 |
| ResNet18 | 72.05 | 0.57 | 0.004 | 约 1 | 先 Adam 后低学习率 SGD 微调，SGD 后期近乎精确拟合 |
| ResNet34 | 73.5 | 0.62 | 0.004 | 约 1 | 视觉微调结果与理论一致 |
| VGG16 | 159.7 | 0.14 | 0.001 | 约 1 | LLC 很高但谱维度很低，说明可达动态很受限 |

第二组主实验检验 “SGD 稳态 vs 贝叶斯后验” 的关系。作者在 moons 数据集上训练 500 个相同全连接网络，用 SGD 得到解簇，再用 SGLD 近似局部贝叶斯后验，并比较 tempered SGD 分布和贝叶斯分布。

| 指标 | Bayes vs Tempered SGD | 含义 |
|------|---------|------|
| $K(\mathrm{Bayes}\Vert\mathrm{Tempered\ SGD})$ | 0.009 | KL divergence 很小，说明两种 cluster 概率接近 |
| Wasserstein distance | 0.002 | 分布质量搬运距离很小 |
| Jensen-Shannon divergence | 0.003 | 对称分布差异很小 |
| 最佳粗粒化尺度 | $\xi=0.5$ | 该尺度下 tempered SGD 与 SGLD 后验最接近 |

### 消融实验
论文的消融主要看优化器和超参数如何影响谱维度、LLC 与性能。MNIST 全连接网络上，SGD 与 Adam 呈现出不同的动力学结构：SGD 更符合本文用原始参数度量建立的 LLC-位移关系，Adam 因为自适应预条件改变了几何度量，表现更复杂。

| 配置 | $d_s$ 均值 | $d_s$ 标准差 | 最终 $\lambda$ 均值 | 最终 $\lambda$ 标准差 | 测试准确率均值 |
|------|---------|------|------|------|------|
| Adam | 0.4061 | 0.9068 | 3.0957 | 5.7533 | 90.4297 |
| SGD | 7.8165 | 10.2494 | 12.5270 | 11.8393 | 94.0592 |

作者还报告了若干超参数趋势。对 SGD 来说，最终位移和平均 LLC 在大 batch、低学习率 regime 下强相关；学习率和谱维度 $d_s$ 的相关性比和 $\lambda$ 的相关性更明显，这符合“学习率改变动力学可达性，而 LLC 更多刻画局部几何”的解释。对 Adam 来说，谱维度有时比 LLC 更能预测表现，因为 Adam 的自适应预条件等价于改变参数空间的 Riemannian 度量。

### 关键发现
- SGD 后期的权重位移通常不是普通 Brownian 扩散，而是能被 $R(t)\sim t^{1/d_{walk}}$ 描述的次扩散；语言模型和视觉模型上的 $r^2$ 多在 0.98 到约 1，说明幂律模型很有解释力。
- 实验支持 $d_s\leq \bar{\lambda}$：谱维度被平均局部学习系数上界约束，符合“SGD 实际可访问状态不能超过低损失几何容量”的理论图景。
- SGD 解会偏向较低 LLC 的区域，这与“低 LLC 对应更简单、更平坦、更可能泛化的局部结构”一致。
- 原始 SGD 分布和贝叶斯后验并不完全一样，但经过 $D_\xi$ tempering 后，在 moons 实验中的 KL、Wasserstein 和 JS 距离都很小。
- Adam 的结果更复杂，不是完全失败，而是提示本文理论主要适用于 vanilla SGD 或后期低学习率 SGD；自适应优化器可能需要重新定义与其度量匹配的 LLC。

## 亮点与洞察
- 把 SLT 的 LLC 从“泛化复杂度指标”推进到“SGD 可达几何指标”。这一步很有价值，因为它让 developmental interpretability 里常用的 LLC 曲线不只是观测量，而能进入动力学方程。
- 标题里的 “Almost Bayesian” 很准确。论文没有夸张地说 SGD 就是贝叶斯采样，而是指出两者之间隔着一个由局部几何、谱维度和粗粒化尺度决定的动力学温度。
- 多孔介质类比抓住了神经网络损失地形的本质：低损失区域不是规则盆地，而是高维退化结构里的可达通道。这个类比比“平坦极小值”更细，因为它同时关心体积和连通/访问速度。
- 谱维度 $d_s$ 是一个值得关注的训练动力学指标。它可能比单看 loss、Hessian 或 LLC 更能说明优化器当前是在探索、局部化，还是被瓶颈困住。
- 对迁移学习和学习率调度也有启发：如果初始化点附近 LLC 低但 $d_s$ 也低，模型可能在宽盆地里移动不足；调大学习率或减小 batch 可能不是为了“泛化玄学”，而是在改变可达谱维度。

## 局限与展望
- 理论依赖后期近似稳态。真实 SGD 可能因为标签噪声、非平稳数据或学习率 schedule 产生非平衡概率流，此时 $D_t^\alpha p=0$ 的局部稳态假设不一定成立。
- 标量扩散系数近似需要大 batch、小学习率和训练后期等条件。若梯度噪声强烈各向异性，或者模型处在早期超扩散/相变阶段，扩散张量可能不能被简单的 $D_\xi(w)$ 概括。
- Adam 等自适应优化器还没有被理论完整覆盖。论文的解释是 Adam 改变了参数空间度量，因此原始 LLC 与动力学的对应会变弱；未来需要在优化器诱导的 metric 下重新建立 SLT 指标。
- 粗粒化尺度 $\xi$ 的选择仍带有经验性。moons 实验里 $\xi=0.5$ 效果好，但不同模型、数据集和训练阶段如何自动选择 $\xi$ 仍是开放问题。
- 实验虽然覆盖 MNIST、TinyStories、Tiny ImageNet 和 CIFAR 附加结果，但整体仍偏小模型/可控实验。要证明该理论能解释大规模 foundation model 的长期训练，还需要更廉价、稳定的 LLC 与谱维度估计方法。

## 相关工作与启发
- **vs Mandt et al. 的 SGD-as-Bayesian-inference**: 传统近似把 SGD 放在非退化二次极小值附近，得到近似高斯后验；本文处理的是神经网络更常见的奇异、退化、非二次损失地形，因此把后验关系改写成可达性 tempering。
- **vs Watanabe 的奇异学习理论**: SLT 原本主要解释贝叶斯泛化误差和 WBIC，本文把局部学习系数嵌入 SGD 扩散方程，让静态复杂度指标参与训练动力学建模。
- **vs Chen et al. 的异常扩散观测**: 既有工作观察到深度网络训练中存在从超扩散到次扩散的现象，本文进一步给出分数 Fokker-Planck 与分形维度解释，并用实验验证定量关系。
- **vs flat minima / Hessian 谱分析**: 平坦极小值通常关注曲率大小，本文更进一步区分低损失体积、可达通道和扩散速度。LLC 可以看作比单纯 Hessian 特征值更适合奇异模型的局部复杂度度量。
- **对训练诊断的启发**: 未来可以把 $\lambda(t)$、$d_s(t)$ 和位移曲线作为训练阶段监控信号，用来识别 emergence、grokking、迁移学习中的瓶颈，甚至设计更结构化的 learning-rate schedule。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把分数扩散、多孔介质、谱维度和 SLT 后验联系起来，理论组合很有辨识度。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 toy posterior、MNIST、语言模型和视觉微调，足以支持主要 claim，但大规模模型验证仍有限。
- 写作质量: ⭐⭐⭐⭐ 主线清晰，附录给出较多直觉和证明；部分符号和假设切换较快，对不熟悉 SLT 的读者门槛较高。
- 价值: ⭐⭐⭐⭐⭐ 这篇论文为“SGD 为什么 almost Bayesian”提供了更适合神经网络奇异性的解释，也给 LLC 在训练动力学中的使用打开了新方向。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Optimizing Data Augmentation through Bayesian Model Selection](optimizing_data_augmentation_through_bayesian_model_selection.md)
- [\[ICLR 2026\] Learning Admissible Heuristics for A*: Theory and Practice](learning_admissible_heuristics_for_a_theory_and_practice.md)
- [\[ICLR 2026\] Reshaping Reasoning in LLMs: A Theoretical Analysis of RL Training Dynamics through Pattern Selection](reshaping_reasoning_in_llms_a_theoretical_analysis_of_rl_training_dynamics_throu.md)
- [\[ICLR 2026\] "Noisier" Noise Contrastive Estimation is (Almost) Maximum Likelihood](noisier_noise_contrastive_estimation_is_almost_maximum_likelihood.md)
- [\[ICLR 2026\] Sharp Asymptotic Theory for Q-Learning with LD2Z Learning Rate and Its Generalization](sharp_asymptotic_theory_for_q-learning_with_textttld2z_learning_rate_and_its_gen.md)

</div>

<!-- RELATED:END -->
