---
title: >-
  [论文解读] NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering
description: >-
  [ICLR 2026][自监督学习][Test-Time Adaptation] NEO 发现输入分布偏移会让倒数第二层 embedding 整体产生一个跨样本/跨类别共享的平移，于是只用一个全局质心向量把测试特征重新居中到原点，就能在**零优化、零超参、几乎零额外开销**下超过 7 个主流 TTA 方法。
tags:
  - "ICLR 2026"
  - "自监督学习"
  - "Test-Time Adaptation"
  - "Neural Collapse"
  - "Latent Re-Centering"
  - "Distribution Shift"
  - "Transformer"
  - "Edge Inference"
---

# NEO — No-Optimization Test-Time Adaptation through Latent Re-Centering

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=mVlIKLiizr](https://openreview.net/forum?id=mVlIKLiizr)  
**代码**: 公开（论文声明已开源，替换 `nn.Linear` 为自定义 NEO 层）  
**领域**: 测试时适应 / 自监督表示 / 分布偏移鲁棒性  
**关键词**: Test-Time Adaptation, Neural Collapse, Latent Re-Centering, Distribution Shift, Vision Transformer, Edge Inference  

## 一句话总结
NEO 发现输入分布偏移会让倒数第二层 embedding 整体产生一个跨样本/跨类别共享的平移，于是只用一个全局质心向量把测试特征重新居中到原点，就能在**零优化、零超参、几乎零额外开销**下超过 7 个主流 TTA 方法。

## 研究背景与动机
**领域现状**：测试时适应（TTA）让模型在部署阶段用无标签测试数据自我调整，以对抗训练/部署间的协变量偏移（雪、雾、运动模糊等 corruption）。主流做法如 TENT/SAR/EATA 通过最小化预测熵来更新 BatchNorm 仿射参数，CoTTA 做持续适应，T3A 用原型修正分类器。

**现有痛点**：这些方法几乎都依赖反向传播或多次前向，带来三大代价——① 显存与延迟显著上升（边缘设备难以承受）；② 对超参数极度敏感，调不好会灾难性遗忘、精度归零；③ 很多方法假设有 BN 层、或需要大 batch / 大量样本才能算出稳定统计量。对于"车载红绿灯识别"这类必须本地、实时、低功耗的场景，现有 TTA 基本不可用。

**核心矛盾**：TTA 的有效性几乎总是用"更多计算/更多样本/更多调参"换来的，**精度提升与部署可行性之间存在根本张力**。

**本文目标**：做一个 fully TTA（不碰源数据）、**无优化、无超参、无显著额外开销**的方法，且能在单样本/单类别/持续偏移下都稳健。

**核心 idea**：**【几何洞察 + 全局重居中】** 作者发现分布偏移在 latent 空间里主要表现为一个**跨样本、跨类别共享的全局平移**——而且这个平移只集中在极少数高幅值维度上。借助 neural collapse 理论可证明：训练到收敛的模型其源数据全局质心恰好在原点（$\mu_G=0_d$），因此"把腐蚀样本的全局质心对齐回源"等价于"把测试特征重新居中到原点"。于是适应只需在线估计一个全局质心向量 $\tilde\mu_G$ 并从特征里减掉它，无需任何梯度。

## 方法详解

### 整体框架
NEO 把模型拆成编码器 $h$ 与线性分类头 $\theta$（$f=\theta\circ h$）。它不改任何权重，只在 $h$ 和 $\theta$ 之间插入一个"重居中"操作：在线维护腐蚀 embedding 的全局均值 $\tilde\mu_G$，对每个测试特征做 $h(\tilde x)-\tilde\mu_G$ 后再喂给分类头。实现上只需把 ViT 的 `nn.Linear` 换成一个自定义层即可，存储成本是单个向量。

```mermaid
flowchart LR
    A[腐蚀输入 x̃] --> B[编码器 h]
    B --> C["embedding h(x̃)"]
    C --> D["在线更新全局质心 μ̃_G<br/>(累积平均)"]
    C --> E[重居中 h(x̃) − μ̃_G]
    D --> E
    E --> F[原始线性分类头 θ]
    F --> G[预测 y]
```

### 关键设计

**1. 分布偏移的几何刻画：偏移是少数维度上的全局共享平移**，这是 NEO 一切的实证起点。作者把每个样本写成 $h(\tilde x)=h(x)+\Delta_G+\Delta_c+\delta$，其中 $\Delta_G=\tilde\mu_G-\mu_G$ 是全局质心偏移、$\Delta_c$ 是去掉全局后的类别偏移、$\delta$ 是样本残差。在 50000 个 ImageNet-C 样本上统计 $h(x)-h(\tilde x)$ 的最大幅值维度，发现 contrast 这类腐蚀里 95% 样本的最大差异只落在不到 20/768 个维度上、所有腐蚀下 80% 样本最大差异维度少于 50 个。更关键的是逐级对齐的对比实验：只做全局对齐就把源/腐蚀 embedding 的余弦相似度从 −0.44 拉到 0.51 并把 L2 范数差从 4.33 降到 3.65（贡献最大），而再做类别对齐余弦只微涨且**范数差反而变大**。这说明"全局共享的那一个平移"才是偏移的主体，逐类细修得不偿失。

**2. 用 neural collapse 把"全局对齐"落地为"原点居中"**，解决了 fully TTA 下源质心不可知的难题。前述 $\Delta_G$ 需要源数据才能算，但 Proposition 4.2 在 neural collapse + 交叉熵 + unconstrained features + 类别平衡假设下证明源全局质心 $\mu_G=0_d$，于是 $\Delta_G=\tilde\mu_G-\mu_G=\tilde\mu_G$——**腐蚀数据的全局质心本身就是要减掉的偏移量**，完全不需要源数据。实证上（图 3b 最后一行）把 $h(\tilde x)$ 减 $\tilde\mu_G$ 得到的余弦相似度和范数差，与减真实 $\Delta_G$ 几乎一致，验证了这个等价。Proposition 4.1 进一步说明：neural collapse 下线性分类结果只由 $\cos(w_c,h(x))$ 决定（$\arg\max_c \|w_c\|\,\|h(x)\|\cos(w_c,h(x))$），所以只要恢复余弦相似度就能恢复精度，而范数差则解释了置信度/校准的变化——这就把"为什么重居中足够"讲清楚了。

**3. 在线累积均值 + 持续变体**，让方法零优化又能应对演化偏移。NEO 核心算法（Algorithm 1）极简：初始化 $\tilde\mu_G\leftarrow 0_d$，每来一个 batch 用累积平均 $\tilde\mu_G\leftarrow\frac{i-1}{i}\tilde\mu_G+\frac{1}{i}\mathrm{Avg}(h(B))$ 更新质心，然后输出 $\theta(h(B)-\tilde\mu_G\mathbf{1}_b^T)$。整个过程只有平均、加法、标量乘和一次向量广播，没有任何梯度或矩阵分解。因为它对所有样本一视同仁、用全部样本估计同一个全局量，估计极稳健（对比 T3A 等按类估计因每类样本太少而不可靠），且 batch size 无关、不依赖伪标签、不改权重故无灾难性遗忘。当偏移会随时间变化时，作者提出 **NEO-Continual**，把累积平均换成指数移动平均 $\tilde\mu_G\leftarrow(1-\alpha)\tilde\mu_G+\alpha\,\mathrm{Avg}(h(B))$，仅引入一个语义清晰的超参 $\alpha$ 即可跟踪漂移的特征均值。

## 实验关键数据

### 主实验表格
ViT-Base 在 ImageNet-C 15 种腐蚀上、用于适应的 512 样本上的精度（%，节选 + 汇总）：

| 方法 | Contrast | Fog | Glass | ImageNet-C 平均 | CIFAR-10-C | ImageNet-R | ImageNet-S |
|------|---------|-----|-------|-----------------|-----------|-----------|-----------|
| No Adapt | 32.6 | 65.8 | 35.3 | 55.6 | 80.4 | 59.2 | 45.4 |
| TENT | 36.9 | 62.4 | 36.8 | 56.3 | 81.3 | 59.4 | 45.7 |
| FOA | 54.5 | 70.7 | 36.8 | 58.4 | 80.9 | 60.2 | 46.3 |
| Surgeon | 31.7 | 63.0 | 36.8 | 56.1 | **82.7** | 60.2 | 47.0 |
| **NEO** | **58.2** | **71.2** | **37.9** | **59.2** | 82.4 | **60.3** | **47.2** |

- 15 种腐蚀里 NEO 在 12 种取得最高、其余 3 种第二（仅被开销大得多的 Surgeon 超过）；平均提升 3.6%，contrast 上几乎翻倍（32.6→58.2）。
- **NEO 在任何一种腐蚀上都没有让精度下降**——这与其他 TTA 常因调参不当而崩溃形成鲜明对比。
- ImageNet-C / ImageNet-R / ImageNet-S 上全面第一，CIFAR-10-C 上 7 个方法里赢 6 个。

### 消融实验表格
不同对齐方式对源/腐蚀 embedding 对齐质量的影响（ImageNet-C 严重度 5 高斯噪声，ViT-Base，50000 样本平均）：

| 对齐方式 | 与 $h(x)$ 余弦相似度 ↑ | L2 范数差 ↓ |
|---------|----------------------|------------|
| $h(\tilde x)$（不对齐） | −0.44 | 4.33 |
| $-\Delta_G$（全局对齐） | 0.51 | 3.65 |
| $-\Delta_G-\Delta_c$（再加类别对齐） | 0.64 | 5.47 |
| $-\Delta_G-\Delta_c-\delta$（理想全对齐） | 1.00 | 0.00 |
| $-\tilde\mu_G$（**原点居中，可计算**） | 0.49 | **3.64** |

关键信号：原点居中（NEO 实际用的、唯一在 fully TTA 下可算的方案）在余弦相似度上几乎追平全局对齐，且范数差最小——印证了 $\Delta_G=\tilde\mu_G$ 的理论。类别对齐虽提升余弦却恶化范数差，说明"只修全局"是正确取舍。

### 关键发现
- **极少样本即可适应**：仅 1 个样本就让 ImageNet-C 精度涨 1.5%，1 个 batch（64 样本）后再加样本收益已很小（55.6→59.2）。
- **跨类别泛化**：只用 1 个类别的数据适应，能让其余 999 个类别精度提升超 3%。
- **校准更好**：12 个设置中 9 个 ECE 持平或改善；ViT-S/ImageNet-C 上 ECE 低于所有对比 TTA 方法。
- **边缘效率**：Jetson Orin Nano 上 NEO 是唯一不增加推理时间、不增加峰值显存的方法；论文摘要报告相对 baseline 推理时间降 63%、显存降 9%。
- **持续适应**：NEO-Continual 在 15 种随机排序腐蚀上优于 CoTTA，仅次于资源消耗大得多的 Surgeon。
- **质心可迁移**：噪声类、模糊类腐蚀之间 $\tilde\mu_G$ 余弦相似度高、跨域适应精度也高，意味着可复用质心、省去对每个域单独适应。

## 亮点与洞察
- **把 TTA 从"优化问题"降维成"减一个向量"**：当业界普遍在 backprop-free 里卷"仅前向迭代优化器"时，NEO 直接给出解析、无优化的方案，且有 neural collapse 的理论支撑而非纯启发式。
- **"原点居中"的优雅等价**：$\mu_G=0_d$ 这一 neural collapse 结论让"对齐源质心"在无源数据时变得可执行，是全文最漂亮的一跳。
- **工程落地极低门槛**：一行代码替换 `nn.Linear`，存储一个向量，天然 batch-size 无关、不遗忘、不依赖伪标签——对真实边缘部署友好。
- **诊断性实验扎实**：少数维度承载偏移、逐级对齐分解、跨腐蚀质心相似度热图，把"为什么全局重居中就够"层层讲透。

## 局限与展望
- **架构覆盖窄**：只在 ViT（ViT-S/B/L）上验证，CNN、ConvNeXt、混合架构等是否同样有"全局共享平移"未知。
- **只看倒数第二层**：洞察局限于 penultimate-layer 激活，是否对其他层、其他 representation 成立留待未来。
- **理论假设较强**：$\mu_G=0_d$ 依赖 neural collapse + 类别平衡 + unconstrained features，类别严重不平衡或未充分训练的模型上等价性可能松动（实验虽用真实模型，但理论保证会打折）。
- **单一全局偏移的假设边界**：NEO 假设所有测试样本来自同一分布；混合多源/快速漂移场景靠 NEO-Continual 的 EMA 缓解，但 $\alpha$ 仍需人工设定，且只校正"平移"而非更复杂的协方差形变。

## 相关工作与启发
- **熵最小化 TTA**（TENT/SAR/EATA）与 NEO 形成对照：前者更新 BN 仿射参数、需反向传播且对超参敏感；NEO 不动任何权重。
- **原型/输出型解析 TTA**（T3A、LAME、FOA、Surgeon）同属无/弱优化阵营，但 T3A 按类估计原型样本不足、FOA 需源数据算统计量；NEO 用单一全局量规避了这些脆弱点。
- **Neural Collapse**（Papyan 2020、Súkeník 2023）此前多用于域泛化与 OOD 检测，本文首次把它引入 TTA 来论证"全局重居中足够"，提供了一条"用收敛态几何指导部署期适应"的新思路。
- 启发：很多分布偏移问题或许并不需要重新优化模型，而只需在**表示空间做一次几何校正**——这对低功耗、隐私敏感的端侧持续学习是很有价值的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐⭐ 把 neural collapse 的 $\mu_G=0$ 转化为可执行的原点居中，将 TTA 从优化降维成减一个向量，视角新颖且理论实证闭环。
- **实验充分度**: ⭐⭐⭐⭐ 4 数据集 × 3 ViT × 7 baseline，覆盖精度/校准/单样本/单类别/持续/边缘设备，诊断实验扎实；扣分在仅限 ViT 与倒数第二层。
- **写作质量**: ⭐⭐⭐⭐⭐ 从几何观察→理论命题→极简算法层层递进，图 3 的分解实验把动机讲得非常清楚。
- **价值**: ⭐⭐⭐⭐⭐ 零超参、零优化、近零开销且不退化，一行代码可落地，对边缘/实时部署有直接实用价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] ZeroSiam: An Efficient Asymmetry for Test-Time Entropy Optimization without Collapse](zerosiam_an_efficient_asymmetry_for_test-time_entropy_optimization_without_colla.md)
- [\[ICLR 2026\] Architecture-Agnostic Test-Time Adaptation via Backprop-Free Embedding Alignment](architecture-agnostic_test-time_adaptation_via_backprop-free_embedding_alignment.md)
- [\[ICLR 2026\] Test-Time Efficient Pretrained Model Portfolios for Time Series Forecasting](test-time_efficient_pretrained_model_portfolios_for_time_series_forecasting.md)
- [\[CVPR 2026\] Energy Waveify and Redistribution for Test-Time Adaptation: A Control System Perspective](../../CVPR2026/self_supervised/energy_waveify_and_redistribution_for_test-time_adaptation_a_control_system_pers.md)
- [\[ICLR 2026\] Bayesian Test-Time Adaptation via Dirichlet feature projection and GMM-Driven Inference for Motor Imagery EEG Decoding](bayesian_test-time_adaptation_via_dirichlet_feature_projection_and_gmm-driven_in.md)

</div>

<!-- RELATED:END -->
