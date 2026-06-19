---
title: >-
  [论文解读] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders
description: >-
  [ICML 2026][可解释性][稀疏自编码器] 本文指出 SAE 中"死特征"问题的真正根源不是训练动力学而是激活分布的几何性质——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 量化"维度级离群"严重程度，从初始化就解析地预测死率（454 个模型-层组合上 Spearman $\rho=0.82\sim0.89$），并证明仅用 mean-centering 就能把 AlphaFold3/ESM3 等高 $\gamma$ 模型的死率从 70%+ 降到接近零。
tags:
  - "ICML 2026"
  - "可解释性"
  - "稀疏自编码器"
  - "feature death"
  - "激活离群"
  - "mean-centering"
  - "TopK SAE"
---

# On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders

**会议**: ICML 2026  
**arXiv**: [2605.31518](https://arxiv.org/abs/2605.31518)  
**代码**: 无  
**领域**: 可解释性  
**关键词**: 稀疏自编码器, feature death, 激活离群, mean-centering, TopK SAE

## 一句话总结
本文指出 SAE 中"死特征"问题的真正根源不是训练动力学而是激活分布的几何性质——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 量化"维度级离群"严重程度，从初始化就解析地预测死率（454 个模型-层组合上 Spearman $\rho=0.82\sim0.89$），并证明仅用 mean-centering 就能把 AlphaFold3/ESM3 等高 $\gamma$ 模型的死率从 70%+ 降到接近零。

## 研究背景与动机

**领域现状**：稀疏自编码器（SAE）是机制可解释性的主流工具，把神经网络激活映射到高维稀疏字典空间（$n>d$），每个字典方向代表一个可解释概念。架构上有 ReLU-SAE、TopK-SAE、JumpReLU-SAE 等变体，本文以 TopK-SAE 为主。

**现有痛点**：同一套 SAE 配置（架构、字典大小、稀疏度、AuxK 都一致）在 GPT-2 上死特征率 <5%，到 AlphaFold3 上却高达 72%。即使在 ESM3 单个模型内部，不同层的死率也在 20%–80% 之间剧烈波动。死特征意味着字典容量被严重浪费，幸存特征不得不"挤"进更多概念，反而让 SAE 本想消除的 superposition 回流。

**核心矛盾**：以往的复活技术（AuxK、Ghost Gradient、Resampling）都把死特征视为"训练动力学问题"——既然卡住了就用各种 trick 推一把。但在最严重的模型上这些 trick 全部失效，说明问题根本不在训练。

**本文目标**：(1) 找到一个能跨模态、跨模型预测死率的可解释诊断量；(2) 解释为什么 AuxK 等复活方法在高死率模型上失效；(3) 给出原则性的预处理方案并说明何时必须使用。

**切入角度**：作者发现高死率层都有一个共同的激活模式——少数维度的均值显著大于 per-token 标准差（"高均值低方差"的维度级离群），这与量化领域研究的 token 级离群（个别 token 的尖峰）完全不同。这种几何性质在初始化时就已经决定了大部分特征的命运。

**核心 idea**：死特征是激活几何问题而非训练问题——用 $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 一个标量就能预测死率，而 mean-centering（用激活均值初始化 bias）就能从根上消除离群导致的死。

## 方法详解

### 整体框架

这篇论文要回答的是"为什么同一套 SAE 在某些模型上 70%+ 的特征会死"，它的答案不在训练，而在激活分布本身的几何形状。围绕这条线，论文做了三件事：先用一个训练前就能算出的标量 $\gamma$ 解析地预测死率，再把训练中复活机制拆成快慢两条路径来解释 AuxK 为何在最难的模型上失灵，最后给出一个不增加任何推理开销的预处理——mean-centering。它的输入是任意预训练网络（GPT-2、Pythia、DINOv3、ESM3、AlphaFold3、Evo2 等）某一层的激活分布，产出则是 $\gamma$ 诊断值、由 $\gamma$ 直接给出的初始死率公式，以及一行修改 SAE bias 初始化的代码。

分析都建立在 TopK-SAE 的标准结构上：$\mathbf{z}_{\text{pre}}=\mathbf{W}_{\text{enc}}(\mathbf{x}-\mathbf{b})+\mathbf{b}_{\text{enc}}$，$\mathbf{z}=\text{TopK}(\text{ReLU}(\mathbf{z}_{\text{pre}}))$，$\hat{\mathbf{x}}=\mathbf{W}_{\text{dec}}^{\top}\mathbf{z}+\mathbf{b}$。一个特征"死掉"有两条互不相同的路径：**dead-by-ReLU**（pre-activation 在任何输入上都为负，永远被 ReLU 截断）和 **dead-by-TopK**（pre-activation 为正但永远挤不进 top-$k$）。后文的诊断量、复活分析和预处理都是顺着这两条路径展开的。

### 关键设计

**1. $\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 诊断量与解析死率公式：训练前一个标量预测死率**

痛点在于以往要么用 token 级离群指标（如 kurtosis），要么干脆没有诊断量，都解释不了死率为何跨模型剧烈波动。本文把单 token 激活分解为 $\mathbf{x}=\bm{\mu}+(\mathbf{x}-\bm{\mu})$，于是 pre-activation 自然拆成一个常数 shift 项 $\mathbf{w}_i\cdot\bm{\mu}$ 和一个随输入变化的 signal 项 $\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})$；$\gamma=\|\bm{\mu}\|/\|\bm{\sigma}\|$ 正是量化"均值"相对"每 token 标准差"占比的标量（计算前先对激活做 per-token LayerNorm，剥掉不同 token 间的尺度差异）。当 $\gamma$ 很大时 shift 主导一切：与 $\bm{\mu}$ 反向对齐的特征 pre-activation 恒为负而 dead-by-ReLU，与 $\bm{\mu}$ 强正向对齐的特征则在每个输入上都激活，只有与 $\bm{\mu}$ 近似正交的特征才真正响应输入。把 shift 与 signal 都看成随机单位向量在固定方向上的投影，用高维概率近似（推导见 Appendix B）即得 $P(\text{dead-by-ReLU})=\Phi(-C/\gamma)$，其中 $C=\Phi^{-1}(1-1/N)\approx 4.26$（$N=10^5$ 评估样本）；TopK 情形把生存门槛抬高到 shift 分布的 $(1-k/n)$ 分位数 $t_k=\Phi^{-1}(1-k/n)$，得到 $P(\text{dead-by-TopK})\approx \Phi(t_k-C/\gamma)$。它有效的关键在于 $\gamma$ 抓住的是真正"维度级"的几何量、且公式不含任何拟合参数——在 454 个跨模态模型-层组合上 Spearman $\rho$ 仍高达 0.89（dead-by-TopK）/ 0.82（dead-by-ReLU），实践者因此可以在投入算力训练 SAE 之前就预判是否会出现严重死特征。

**2. 两条复活路径与"bias 学 $\bm{\mu}$ 是瓶颈"：解释 AuxK 为何在高 $\gamma$ 下失效**

以往工作隐式把死特征当成"训练动力学坏了"，于是一直在想办法给死特征注入更多梯度。本文在合成数据上对 bias 冻结/不冻结、加/不加 AuxK 做消融，第一次把复活机制按死亡路径解耦。**dead-by-TopK** 的复活靠 alive feature 在收敛后主动下调自己的激活幅度，把原本卡在第 $k+1$ 名的特征腾进来——这条路径约 200K 步即可完成，且 bias 冻结也不受影响；**dead-by-ReLU** 的复活则只能依赖 bias 慢慢吸收 $\bm{\mu}$，因为只有 bias 能把恒为负的 pre-activation 抬到零以上。麻烦的是 bias 学 $\bm{\mu}$ 的速度严重依赖 $\gamma$：$\gamma\le 5$ 时 200K 步就学到 99%，$\gamma\approx 20$ 时 2M 步才到 90%，$\gamma\ge 30$ 时 2M 步也只到 50–70%。直观原因是特征权重作用于输入、效果随输入幅度放大，而 bias 是直接相加、不被输入放大，所以 $\|\bm{\mu}\|$ 越大 bias 越追不上，alive feature 一旦学到 $\bm{\mu}$ 又会进一步压低 bias 的梯度。顺着这条逻辑也就看清了 AuxK 的真实角色：TopK 复活过程中部分 alive feature 在缩小激活时被压到零以下、变成新的 dead-by-ReLU，AuxK 给 dead-by-TopK 提供梯度让它们稳住、不滑入 dead-by-ReLU，本质是在抑制"附带死亡"而非真复活；它完全不加速 bias 学习，所以面对从初始化就 dead-by-ReLU 的特征束手无策，这正解释了 AuxK 在 $\gamma$ 中等时有效、$\gamma$ 高时失效。结论是高 $\gamma$ 下根本不需要更好的复活技术，只要让 bias 一开始就处在 $\bm{\mu}$ 的位置即可。

**3. Mean-centering：用激活均值初始化 bias，从根上消掉 shift 项**

既然瓶颈是 bias 追不上 $\bm{\mu}$，最直接的做法就是把 bias 初始化为激活均值。令 $\mathbf{b}=\bm{\mu}$ 后，pre-activation 退化为 $z_i=\mathbf{w}_i\cdot(\mathbf{x}-\bm{\mu})+b_{\text{enc}}$，shift 项 $\mathbf{w}_i\cdot\bm{\mu}$ 直接消失，所有特征的 pre-activation 都围绕零分布、只随输入变化，从初始化就不再有离群导致的死。默认取 geometric median 而非 arithmetic mean，因为部分模型激活分布偏斜较重（逐模型对比见 Appendix D.5）；这等价于在 runtime 做 mean subtraction，但折进 bias 初始化后没有任何额外推理开销。需要注意它只消除"离群型死"，对极少数方差集中在小维度子空间的层（蛋白/基因模型的少数层）仍有残留死，需再叠加 PCA whitening（Appendix E）。它真正的价值在于把一个早就零散出现过（Bricken 2023b、Gao 2024）却用得不一致、没有判据的 trick 原则化——$\gamma$ 恰好给出"何时必须 center"的判据：高 $\gamma$ 必做、低 $\gamma$ 可选，从而把经验做法升级为有解析依据的标准预处理。

### 损失函数 / 训练策略

训练目标仍是标准 TopK-SAE（重构 MSE + TopK 稀疏化），$k$、字典大小、学习率等超参在跨模型对比中保持一致，mean-centering 不动 loss、只改 bias 的初始化。合成实验均取 10 个种子平均，真实数据则对 454 个模型-层组合统一选 mid-network 层训练。

## 实验关键数据

### 主实验：$\gamma$ 在合成与真实数据上预测死率

| 数据 | 指标 | dead-by-ReLU | dead-by-TopK | 备注 |
|------|------|--------------|--------------|------|
| 合成激活（控制 $\gamma$） | Spearman $\rho$ | 1.0 | 1.0 | $\Phi(-C/\gamma)$ 曲线几乎完美对齐 |
| 454 个真实模型-层（语言/视觉/蛋白/基因） | Spearman $\rho$ | 0.82 | 0.89 | 无任何拟合参数 |
| AlphaFold3 mid layer | 死特征率 | — | 98% → <5% | mean-centering 后 |
| ESM3 mid layer | 死特征率 | — | 83% → ≈0 | mean-centering 后 |

### 消融：mean-centering vs baseline vs AuxK（ESM3 L24，$\gamma\approx 8$）

| 配置 | 训练末死特征率 | 可解释生物概念数 |
|------|----------------|------------------|
| baseline | ≈75% | 73 (dict=8192) |
| baseline + AuxK | ≈25%（plateau） | — |
| LayerNorm + $\sqrt{d}$ rescale | ≈20% | 少于 baseline |
| mean-centering（dict=2048） | ≈0 | **100** |
| mean-centering（dict=8192） | ≈0 | 更高 |

### 合成 ground-truth feature 恢复（$\gamma=40$）

| 配置 | MMCS（最大余弦相似度均值） |
|------|----------------------------|
| baseline | 0.38 |
| mean-centering | 0.97 |

### 关键发现

- **$\gamma$ 是真正的"训练前可算"诊断量**：在 454 个真实模型-层上无拟合就能达到 $\rho\approx 0.89$，意味着可以先算 $\gamma$ 再决定是否投入算力训练 SAE。
- **bias 学习是高 $\gamma$ 下复活的瓶颈**：$\gamma\ge 30$ 时 bias 在 2M 步内只能学到 $\bm{\mu}$ 的 50–70%，dead-by-ReLU 因此长期卡在 75–90%。
- **AuxK 真正在做的事是抑制 collateral death**：它给 dead-by-TopK 提供梯度让其稳住，而不是真的复活初始化时就 dead-by-ReLU 的特征。
- **mean-centering 用 4× 更小的字典超过 baseline**：ESM3 上 dict=2048 的 mean-centered SAE（100 个概念）已超过 dict=8192 的 baseline（73 个概念），训练算力大幅下降。
- **mean-centering 还稳定了对学习率的敏感性**：baseline 在 LR sweep 中死率方差极大，mean-centered 几乎保持一致的低死率。
- **理论稍微高估 dead-by-ReLU**：当激活分布重尾（用 per-dim kurtosis 可以直接诊断），实际中最大 signal 可以超过 Gaussian 假设下的 $C\approx 4.26$，从而救活一部分本应被判死的特征。

## 亮点与洞察

- **把"训练问题"重新框定为"几何问题"**：长期以来 SAE 社区都在调 AuxK / Ghost Gradient / Resampling，这篇论文证明在最难的模型上这些 trick 全部失效，本质是因为大部分特征在初始化时就死了——这是研究框架的转向，不只是一个新方法。
- **一个 0 参数的解析公式打败一堆经验诊断**：之前社区用 kurtosis 等指标都是 token 级的，而 $\gamma$ 抓住了 dimension-level outlier 这个真正的来源，公式纯粹从高维几何推出来不含拟合参数，却跨四个模态都能预测死率。这种"先验+几何"的范式可以迁移到其他需要诊断初始化问题的场景。
- **bias 与 weight 的学习速度不对称**：权重作用于输入、效果随输入幅度放大，而 bias 直接相加。这个观察可以解释很多归一化/中心化技术为何有效，是个可复用的直觉。
- **AuxK 的真实作用被重新解释**：之前都以为 AuxK 在"复活"特征，本文细分后发现它其实在"防止 alive 特征滑入 dead"，这种"现象→机制重判"的细致拆解值得借鉴。
- **MMCS 从 0.38 → 0.97**：合成数据上字典几乎完美对齐 ground-truth，说明 mean-centering 不只是降死率，而是真的让 SAE 学到了对的方向，从可解释性角度直接受益。

## 局限与展望

- **极少数层 mean-centering 不够**：蛋白和基因模型的部分层即使 center 后仍有残留死特征，因为方差集中在少数方向，需要再做 PCA whitening（Appendix E），这意味着 mean-centering 只是"必要而非充分"的预处理。
- **理论假设了 Gaussian signal**：实际激活重尾时 $\Phi(-C/\gamma)$ 会高估死率，作者用 per-dim kurtosis 兜底，但还没有一个统一的公式同时容纳重尾激活。
- **仅在 mid-network 单层做了系统对比**：跨层、跨任务的迁移性虽然有 Appendix 数据但不够全面，尤其是不同层 $\gamma$ 差异很大时如何统一选预处理仍开放。
- **未与最新的"low-rank attention"假说（Wang 2025）做正面对比**：Wang 等把死率归因于注意力激活的低秩结构，本文归因于维度级离群，两种几何因子的相互作用没有系统消融。
- **应用层面的延伸方向**：可以把 $\gamma$ 反过来用作"激活归一化"或"模型架构正则"的目标——直接把训练时的 $\gamma$ 压低，可能从源头减少下游 SAE 训练成本，甚至有助于量化。

## 相关工作与启发

- **vs AuxK / Ghost Grad / Resampling (Gao 2024; Bricken 2023b)**：它们都试图给死特征"打梯度"复活；本文证明在高 $\gamma$ 下根本不是梯度问题而是 bias 距离问题，AuxK 真正在做的事是抑制 collateral death 而非真复活。本文优势是给出原则性的预处理，劣势是不能处理"非离群型"残留死。
- **vs token-level outlier 研究 (Sun 2024; Dettmers 2022)**：他们关注的是量化中个别 token 的尖峰；本文关注的是 dimension-level outlier——在每个 token 上都偏离零的固定维度，几何性质完全不同。
- **vs Lu et al. 2025（ESMFold 维度离群）/ Wang et al. 2025（低秩死特征）**：Lu 等首先观察到 ESMFold 的同类离群但未给出诊断/解决方案；Wang 等把死率归到 attention 激活的低秩结构。本文是第一次给出跨模态可预测的诊断量 + 解析公式 + 极简解决方案。
- **vs 早期 SAE 工作（Bricken 2023b; Gao 2024 用过 mean-centering）**：他们用得不一致，没有判据。本文用 $\gamma$ 把"何时必须 center"原则化，把它从经验 trick 升级为理论支持的标准流程。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把死特征从"训练动力学问题"重新框定为"激活几何问题"，从高维概率推出无拟合参数的死率公式，是范式级的视角转变。
- 实验充分度: ⭐⭐⭐⭐⭐ 454 个真实模型-层 + 控制 $\gamma$ 的合成实验 + 多模态（语言/视觉/蛋白/基因）+ bias 冻结消融，证据链非常完整。
- 写作质量: ⭐⭐⭐⭐⭐ 公式推导和 figure 配合极佳，关键洞察用一句话能讲清，附录补全了所有可质疑的细节（重尾激活、几何中位数、跨层迁移）。
- 价值: ⭐⭐⭐⭐⭐ 直接给出可立即落地的 mean-centering 方案 + 训练前诊断 $\gamma$，对所有训 SAE 的可解释性研究者来说是必读的工程改进。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding](polysae_modeling_feature_interactions_in_sparse_autoencoders_via_polynomial_deco.md)
- [\[ICML 2026\] Sparse Autoencoders are Topic Models](sparse_autoencoders_are_topic_models.md)
- [\[ICML 2026\] Ensembling Sparse Autoencoders](ensembling_sparse_autoencoders.md)
- [\[ICLR 2026\] Toward Faithful Retrieval-Augmented Generation with Sparse Autoencoders](../../ICLR2026/interpretability/toward_faithful_retrieval-augmented_generation_with_sparse_autoencoders.md)
- [\[NeurIPS 2025\] A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders](../../NeurIPS2025/interpretability/a_is_for_absorption_studying_feature_splitting_and_absorption_in_sparse_autoenco.md)

</div>

<!-- RELATED:END -->
