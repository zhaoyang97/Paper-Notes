---
title: >-
  [论文解读] Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization
description: >-
  [ICLR 2026][优化/理论][锐度感知最小化] 这篇论文从理论上证明了锐度感知最小化（SAM）之所以能缓解深度网络的"过自信"，本质是在隐式地对预测分布的负熵做正则（等价于隐式最大熵），并据此提出一个改进版 CSAM，专门压制过自信样本，在多个数据集（含 ImageNet-1K）上取得比 SAM 和各类校准方法更低的校准误差。
tags:
  - "ICLR 2026"
  - "优化/理论"
  - "锐度感知最小化"
  - "模型校准"
  - "最大熵正则"
  - "过自信"
  - "CSAM"
---

# Towards Understanding the Calibration Benefits of Sharpness-Aware Minimization

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=c0ERcCz6lD](https://openreview.net/forum?id=c0ERcCz6lD)  
**代码**: 无  
**领域**: 优化理论 / 模型校准 / SAM  
**关键词**: 锐度感知最小化, 模型校准, 最大熵正则, 过自信, CSAM

## 一句话总结
这篇论文从理论上证明了锐度感知最小化（SAM）之所以能缓解深度网络的"过自信"，本质是在隐式地对预测分布的负熵做正则（等价于隐式最大熵），并据此提出一个改进版 CSAM，专门压制过自信样本，在多个数据集（含 ImageNet-1K）上取得比 SAM 和各类校准方法更低的校准误差。

## 研究背景与动机

**领域现状**：现代深度网络（ResNet、DenseNet、ViT 等）在安全攸关场景（自动驾驶、医疗诊断）中越来越常用，但人们早就发现它们普遍"校准很差"——模型给出的置信度（softmax 最大值）远高于其真实正确率，也就是**过自信**。一个理想的良好校准模型，应该满足"我说有 80% 把握，那么这批样本里就应该有 80% 真的对"。与此并行，SAM（Foret et al., 2021）作为一种把解推向损失曲面平坦区域的优化器，因为显著提升泛化而广受关注：它每步先沿梯度方向爬升一个半径 $\rho$ 的扰动 $\tilde\theta = \theta + \rho\,\nabla L/\|\nabla L\|$，再在扰动点求梯度做下降。

**现有痛点**：之前已有零星工作（Zheng et al., 2021; Möllenhoff & Khan, 2023）观察到"SAM 训出来的模型更好校准"这个现象，但**没有人正式回答"为什么"**。同时，专门做校准的方法各有短板：focal loss、label smoothing 这类显式惩罚置信度的做法会损害精度，而且会压低后处理（temperature scaling）阶段的提升空间；post-hoc 方法（温度缩放、isotonic 回归）则需要额外的验证集且只是事后补救。

**核心矛盾**：过自信主要来自过拟合与过参数化——网络在训练后期不断把真标签的概率推向 1。已有校准方法要么牺牲精度，要么治标不治本；而 SAM 明明能同时提升精度又改善校准，但它的这一"副作用"缺乏机制解释，也就无法被进一步利用和增强。

**本文目标**：（1）给出 SAM 改善校准的理论证明；（2）刻画这种校准收益在分布漂移（OOD）下是否仍然成立；（3）顺着理论设计一个更强的校准优化器。

**切入角度**：作者直接去比较 SAM 扰动前后对真标签的置信度 $p_y$ 与 $\tilde p_y$。直觉上，SAM 在扰动点（损失更差的邻域点）上评估，真标签概率应该被压低，这恰好对应"不让置信度无脑冲到 1"。

**核心 idea**：证明最小化扰动损失 $\ell_{\tilde\theta}$ 等价于在原损失上叠加一个对预测分布的**最大熵正则**（与 focal loss 同源），从而把 SAM 的校准收益归结为"隐式熵正则"；再把这个正则在过自信样本上加权放大，得到 CSAM。

## 方法详解

### 整体框架
全文是一条"理论先行、方法跟进"的链路：先用两条引理证明 SAM 的扰动会把真标签置信度按指数压低，再用两条定理把"压低置信度"翻译成"隐式最大熵正则"，最后基于这个正则在训练后期才发力的观察，改造 SAM 的外层损失得到 CSAM。整条链路不涉及多模块 pipeline，而是围绕一个核心量——扰动前后真标签概率之比展开的推导，因此这里用公式而非框架图来讲清。

记 $p_y = [f_\theta(x)]_y$ 为当前权重 $\theta$ 下真标签 $y$ 的置信度，$\tilde p_y = [f_{\tilde\theta}(x)]_y$ 为扰动权重 $\tilde\theta$ 下的置信度。分类损失取最常用的交叉熵，单样本写作 $\ell_\theta(z) = -\log p_y$。校准用期望校准误差（ECE）度量：把样本按 top 置信度分到 $M$ 个桶，桶内算平均置信度与平均准确率之差再加权平均，

$$\widehat{\text{ECE}} = \sum_{i=1}^{M}\frac{|B_i|}{n}\,\big|\,\text{acc}(B_i) - \text{conf}(B_i)\,\big|.$$

整个方法要回答的就是：为什么 SAM 训出的模型让 $\text{conf}(B_i)$ 紧贴 $\text{acc}(B_i)$。

### 关键设计

**1. 扰动把真标签置信度指数压低：SAM 的"反过自信"来源**

这一步针对"为什么 SAM 不会过自信"的根问题。作者证明（Lemma 1，1-SAM 即 mini-batch 为 1 的情形）：在每步梯度非零、且 Hessian 最小特征值满足有界假设 $\kappa_{\min}(\nabla^2\ell_{\theta'}(z)) \ge -\|\nabla_\theta\ell(z)\|/\rho$（对插值点 $\theta' = (1-t)\theta + t\tilde\theta,\ t\in[0,1]$ 成立）的条件下，扰动点的真标签置信度被乘性压缩：

$$\tilde p_y \le e^{-\rho\|\nabla_\theta\ell(z)\|/2}\,p_y.$$

也就是说 $\tilde p_y$ 随扰动半径 $\rho$ 和梯度范数 $\|\nabla_\theta\ell(z)\|$ **指数衰减**，越是"自信、梯度大"的样本被压得越狠。作者还在 ResNet-56/CIFAR-10 上验证了 $\kappa_{\min}$ 的有界假设确实近似成立，且 $\kappa_{\min}$ 沿 $\theta\to\tilde\theta$ 近似线性变化（Figure 2），这也解释了为何 $\rho$ 不能取太大——太大时有界假设容易被破坏。Lemma 2 把结论推广到 m-SAM：此时 $p_y, \tilde p_y$ 改用一个 mini-batch 内 $m$ 个样本概率的几何平均 $\big(\prod_i p_{y_i}\big)^{1/m}$，同样有 $\tilde p_y \le e^{-\rho\|\nabla L_\Omega(\theta)\|/2}p_y$。

**2. SAM 等价于隐式最大熵正则：和 focal loss 同源但不掉精度**

有了置信度被压低的结论，作者把它翻译成正则化语言（Theorem 1，1-SAM）。令 $\lambda = (1-\tilde p_y)/(1-p_y)$，则

$$\ell_{\tilde\theta}(z) \ge \ell_\theta(z) - \lambda H(p_y) + H(\tilde p_y),$$

其中 $H(p) = -p\log p - (1-p)\log(1-p)$ 是二元熵。由 Lemma 1 知 $\lambda > 1$，所以最小化扰动损失 $\ell_{\tilde\theta}$ 实际上**更看重最大化 $H(p_y)$**（系数 $\lambda$ 大于压低 $H(\tilde p_y)$ 的系数 1）：当 $p_y$ 接近 1 时把它往下拉、接近 0 时往上推，恰好就是在做最大熵正则、抵消过自信。把 $p_y$ 用 $e^{\rho\|\nabla\ell\|/2}\tilde p_y$ 代入还能得到对 $\lambda$ 的下界 $\lambda_0 = (1-\tilde p_y)/(1 - e^{\rho\|\nabla\ell\|/2}\tilde p_y)$。Theorem 2 把它推广到 m-SAM（$p_y$ 为几何平均）。作者特别指出：这个熵惩罚在**训练后期更强**（此时 $\tilde p_y$ 高，Figure 2(c)），所以越是容易过自信的架构，SAM 校准得越好——这与 focal loss 的隐式最大熵机制同源，但 SAM 不以牺牲精度为代价。

**3. CSAM：在过自信样本上放大熵正则**

第 2 点同时暴露了一个可改进点——SAM 主要在训练后期、$\tilde p_y$ 已经偏高时才开始惩罚预测分布。于是作者想：能不能让过自信样本"显得"更自信，从而触发更强的熵惩罚？做法是改写 SAM 外层（下降步）的单样本损失：

$$\tilde\ell_{\tilde\theta}(z) = \begin{cases} -\log\tilde p_y, & \tilde p_y \le 1/2,\\ -(1+\tilde p_y)^{-\gamma}\log\tilde p_y, & \text{否则,}\end{cases}$$

其中 $0\le\gamma\le 2$ 是超参，$\gamma=0$ 退化为标准 SAM。Theorem 3 证明：当 $\tilde p_y > 1/2$ 时，

$$\tilde\ell_{\tilde\theta}(z) \ge \ell_\theta(z) - \lambda H(p_y) + (1-\gamma/2)H(\tilde p_y).$$

与 Theorem 1 相比，$H(\tilde p_y)$ 前多了系数 $(1-\gamma/2)$：只要 $1-\gamma/2 > 0$，对 $H(p_y)$ 的隐式惩罚就被进一步放大；要求 $\gamma\le 2$ 是为了保证优化方向仍像 SAM 一样朝减小 $\ell_{\tilde\theta}$ 走。这样 CSAM 只在过自信（$\tilde p_y>1/2$）样本上额外加力，对欠自信样本保持不变，从而在不损害泛化的前提下进一步降低校准误差。

### 损失函数 / 训练策略
训练损失默认交叉熵（CE）。CSAM 只改 SAM 外层下降步用的逐样本损失为上式分段形式，超参 $\gamma\in\{0.5, 1.0, 2.0\}$。扰动半径 $\rho$ 按数据集/架构设置：CIFAR-10 用 0.05、CIFAR-100 用 0.2、ImageNet 上 ResNet 用 0.05、ViT 用 0.2；基础优化器为 SGD（动量 0.9）或 AdamW，余弦学习率衰减。

## 实验关键数据

### 主实验

ImageNet-1K（ID 指标 + ImageNet-C 三种损坏的 OOD 指标，TCE 为温度缩放后的 ECE）：

| 模型 | 方法 | Test Acc ↑ | ECE ↓ | TCE ↓ | AdaECE ↓ |
|------|------|-----------|-------|-------|----------|
| ResNet-50 | SGD | 76.97 | 3.39 | 1.80 | 3.31 |
| ResNet-50 | SAM | 77.32 | 1.52 | 1.54 | 1.44 |
| ResNet-50 | **CSAM** | **77.95** | **1.18** | **1.09** | **1.19** |
| ViT-S/16 | AdamW | 71.35 | 9.72 | 3.66 | 9.72 |
| ViT-S/16 | SAM | 75.42 | 1.76 | 1.66 | 1.73 |
| ViT-S/16 | **CSAM** | **75.91** | **1.58** | **1.34** | **1.54** |

CIFAR-10 上与多种校准专用方法对比（WideResNet-28-10，3 个随机种子均值）：

| 方法 | Test Acc ↑ | ECE ↓ | AdaECE ↓ | TCE ↓ |
|------|-----------|-------|----------|-------|
| CE | 95.83 | 2.36 | 2.04 | 1.06 |
| Focal Loss | 95.91 | 1.16 | 1.42 | 1.01 |
| AdaFocal | 95.78 | 0.91 | 0.65 | 0.97 |
| MIMO | 95.96 | 0.88 | 0.73 | 0.74 |
| bSAM | 96.45 | 1.82 | 1.78 | 0.70 |
| SAM | 96.91 | 0.86 | 0.84 | 0.52 |
| **CSAM** | **96.97** | **0.50** | **0.48** | **0.47** |

### 消融与分析实验

OOD（ResNet-18 / CIFAR-10 训练，迁移到 SVHN、CIFAR-10/100-C，配合 MC-Dropout、Ensemble）：

| 优化器 | 配置 | ECE ↓ | 说明 |
|--------|------|-------|------|
| SGD | Vanilla | 5.76 | 过自信最严重 |
| SAM | Vanilla | 3.24 | 比 SGD 校准约好 1.8 倍 |
| CSAM | Vanilla | 2.55 | 比 SAM 再降 |
| SGD | Ensemble | 1.84 | 集成能降 ECE |
| SAM | Ensemble | 1.09 | 集成对 SAM 同样有效 |
| CSAM | Ensemble | **0.86** | 集成 + CSAM 最低 |

### 关键发现
- **SAM 的未校准 ECE 常常比 SGD 经温度缩放后的 ECE 还低**：例如 ResNet-56 上 SGD 的 ECE 约为 SAM 的 6 倍，说明 SAM 自身就产出可靠预测，几乎不依赖后处理。也正因 SAM 已在训练中压制过自信，温度缩放对它的额外提升（TCE 改善）远不如对 SGD 明显。
- **OOD 下校准收益依然保持**：SGD 的 ECE 大致仍是 SAM 的两倍，且 SAM 在 OOD 上的泛化也明显更好；集成（Ensemble）对两者都有效且对 OOD 提升更突出。一个反直觉发现是 **MC-Dropout 会损害 OOD 表现**，对 SAM 尤甚——可能是 Dropout 与 SAM 叠加过度抬高了不确定性。
- **CSAM 在所有 baseline 中校准误差最低**：focal loss、Rank1-BNN 等虽降 ECE 但常损精度；贝叶斯变体 bSAM 因引入额外超参反而不如 SAM；CSAM 则在不掉精度（甚至略升）的同时把 ECE 进一步压到最低。
- **架构越容易过自信，SAM 收益越大**：ViT（AdamW）未用 SAM 时校准比 ResNet 还差（ECE 9~10%），用 SAM 后掉到约 1.5~3%、与 ResNet 差距几乎消失，印证了"后期熵惩罚对过自信架构更强"的理论。

## 亮点与洞察
- **把"优化器副作用"还原成一条干净的正则项**：从 $\tilde p_y \le e^{-\rho\|\nabla\ell\|/2}p_y$ 一步步推到 $\ell_{\tilde\theta} \ge \ell_\theta - \lambda H(p_y) + H(\tilde p_y)$，把 SAM 的校准收益精确归结为隐式最大熵正则，并指出它与 focal loss 同源——这种"用一个标量比值 $\lambda$ 串起置信度与熵"的推导很优雅。
- **改进点直接长在理论缝隙上**：理论发现"熵惩罚只在后期、$\tilde p_y$ 高时才发力"，CSAM 就恰好在 $\tilde p_y>1/2$ 处加权，改动极小（只换一个分段损失），却把这个时机利用满，是"先理解再改进"的范例。
- **梯度范数的角色**：Lemma 1 揭示梯度范数越大、SAM 越有效，且允许取更大的 $\rho$——这与"SAM 在 ViT+AdamW 上 ImageNet 提升 5%+"的经验观察对得上，可迁移到"何时该用大 $\rho$"的实践判断。

## 局限与展望
- **理论依赖有界性假设**：核心引理建立在 Hessian 最小特征值有界 $\kappa_{\min}\ge -\|\nabla\ell\|/\rho$ 之上，作者承认该不等式不一定对所有插值点 $\theta'$ 成立，严格化需要逐步变化 $\rho$（最坏情况 $\rho\to 0$），与实践中常用常数 $\rho$ 有差距，只能靠实验近似验证。
- **只覆盖交叉熵损失**：分析与主实验都基于 CE，作者只给了初步证据（附录）表明 SAM/CSAM 能与其他损失结合，泛化到非 proper scoring rule 的损失尚未证明。
- **CSAM 引入新超参 $\gamma$**：虽然只一个超参且范围明确（$0\le\gamma\le 2$），仍需在 $\{0.5,1.0,2.0\}$ 间挑选；分段阈值固定在 $1/2$ 也较启发式，是否最优未深究。
- **可改进方向**：把有界假设替换成可验证的轨迹级条件、将熵正则推广到多类联合熵（当前用二元熵 $H(p_y)$ 近似）、以及自适应调节 $\gamma$ 随训练阶段变化，都值得探索。

## 相关工作与启发
- **vs Focal Loss / AdaFocal**: 它们显式/隐式惩罚置信样本来降校准误差，但常牺牲精度且压低后处理空间；本文证明 SAM 做的是同源的隐式最大熵正则，却不掉精度，CSAM 还把这个正则在过自信样本上放大，实验上 ECE 更低。
- **vs 温度缩放等 post-hoc 方法**: 它们是训练后补救、需额外验证集；SAM 是训练中就把过自信压住，未校准 ECE 甚至低于 SGD 的后处理结果，二者可叠加但 SAM 对后处理的依赖更小。
- **vs bSAM（贝叶斯 SAM）**: bSAM 引入多个额外超参、难调，校准上并不优于 SAM；CSAM 改动最小（一个分段损失 + 单超参 $\gamma$），却取得更低校准误差。
- **vs Andriushchenko & Flammarion (2022) 等 SAM 理论**: 已有理论多聚焦 SAM 的泛化/隐式偏置（Hessian 谱、SDE 视角），本文首次把 SAM 与**模型校准**用熵正则正式连接，补上了"为什么 SAM 校准更好"这一空白。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次为"SAM 改善校准"给出隐式最大熵正则的理论解释，并顺势导出改进版 CSAM
- 实验充分度: ⭐⭐⭐⭐ ID/OOD/ImageNet-1K + 十余个校准 baseline 对比充分，但局限于交叉熵损失
- 写作质量: ⭐⭐⭐⭐⭐ 理论推导清晰、图示（可靠性图、$\kappa_{\min}$ 验证）与结论紧密对应
- 价值: ⭐⭐⭐⭐ 给安全攸关场景提供"训练即校准"的免后处理方案，理论与实用兼具

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] MASAM: Multimodal Adaptive Sharpness-Aware Minimization for Heterogeneous Data Fusion](masam_multimodal_adaptive_sharpness-aware_minimization_for_heterogeneous_data_fu.md)
- [\[ICLR 2026\] Minor First, Major Last: A Depth-Induced Implicit Bias of Sharpness-Aware Minimization](minor_first_major_last_a_depth-induced_implicit_bias_of_sharpness-aware_minimiza.md)
- [\[ICML 2026\] Stability Analysis of Sharpness-Aware Minimization](../../ICML2026/optimization/stability_analysis_of_sharpness-aware_minimization.md)
- [\[ICLR 2026\] Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models](bi-lora_efficient_sharpness-aware_minimization_for_fine-tuning_large-scale_model.md)
- [\[ICML 2025\] Tilted Sharpness-Aware Minimization](../../ICML2025/optimization/tilted_sharpness-aware_minimization.md)

</div>

<!-- RELATED:END -->
