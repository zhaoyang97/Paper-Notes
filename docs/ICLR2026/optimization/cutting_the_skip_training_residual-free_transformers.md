---
title: >-
  [论文解读] Cutting the Skip: Training Residual-Free Transformers
description: >-
  [ICLR2026][优化/理论][残差连接] 本文从 Transformer Jacobian 的条件数出发，揭示残差（skip）连接的本质作用是改善网络条件数，并据此提出一套**只改初始化、不改架构**的方案，让完全去掉残差连接的 ViT 第一次能和带残差的模型一样快地训练，同时在密集预测任务上学到更抽象、层级更清晰的表征、反超带残差的基线。
tags:
  - "ICLR2026"
  - "优化/理论"
  - "残差连接"
  - "网络初始化"
  - "Jacobian 条件数"
  - "二阶优化"
  - "Transformer"
---

# Cutting the Skip: Training Residual-Free Transformers

**会议**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=iJl3L059s6](https://openreview.net/forum?id=iJl3L059s6)  
**代码**: 无（论文称接收后发布）  
**领域**: 网络优化 / Transformer 初始化  
**关键词**: 残差连接, 网络初始化, Jacobian 条件数, 二阶优化, Vision Transformer

## 一句话总结
本文从 Transformer Jacobian 的条件数出发，揭示残差（skip）连接的本质作用是改善网络条件数，并据此提出一套**只改初始化、不改架构**的方案，让完全去掉残差连接的 ViT 第一次能和带残差的模型一样快地训练，同时在密集预测任务上学到更抽象、层级更清晰的表征、反超带残差的基线。

## 研究背景与动机
**领域现状**：现代 Transformer 之所以能堆到很深还能训得动，几乎完全依赖残差连接（skip / residual connection）。从 ResNet 到 Vaswani et al. 的 Transformer，残差被视为深网络可训练性的基石，普遍解释是它改善了网络 Jacobian 的条件数、让梯度顺畅流动。

**现有痛点**：残差连接虽然稳住了优化，却同时破坏了深网络本该有的**层级抽象结构**。深网络的本意是逐层把表征变得越来越抽象，但残差不断把浅层信息重新注入深层，等于一直在"抄近路"。结果是：带残差的网络表现得像一堆浅子网络的集成（Veit et al.），Transformer 收敛后很多深层贡献极小、甚至能被剪掉而几乎不掉点（Gromov et al.）。也就是说，残差让"名义深度"和"有效深度"严重脱节，掩盖了深度本该带来的表征收益。

**核心矛盾**：残差对优化不可或缺，但对表征层级有害——这是一个"稳定性 vs. 层级抽象"的内在冲突。想拿到真正的深层级表征，就得去掉残差；可一旦去掉，skipless Transformer 在随机初始化下 Jacobian 条件数极差，根本训不动。

**本文目标**：在**不修改标准 Transformer 架构**的前提下，让完全没有残差连接的 Transformer 也能稳定、高效地训练，从而第一次能系统地研究"真正很深"的 ViT。

**切入角度**：唯一一篇训 skipless Transformer 的前作（He et al. 2023）是去**改自注意力块**来维持良态的前向核，但这破坏了标准架构、和 FlashAttention 不兼容，而且收敛仍明显更慢。本文换个角度：不去碰前向核，而是直接盯着**网络 Jacobian 的条件数**——既然残差的作用就是改善条件数，那能不能用一个"原理上正确的初始化"把这份好处补回来？

**核心 idea**：分析发现残差在 Jacobian 里相当于加了个单位阵 $I$，把自注意力子块的奇异谱从 0 处抬开、正则化了最小奇异值；于是只要用一套**精心设计的权重初始化**，让自注意力子块在初始时就天然良态，就能在不加残差、不改架构的情况下复现这份稳定性。

## 方法详解

### 整体框架
方法的目标很纯粹：去掉所有残差连接后，标准 Transformer 块在初始化时的 Jacobian 条件数会爆炸，导致训不动；本文要做的就是**只通过初始化**把这个条件数压回良态。整条逻辑分三步走：先对去残差 Transformer 块的 Jacobian 做理论分解，看清残差到底改善了哪一项的条件数（4.2–4.3 节）；再据此推导出对自注意力四个投影矩阵 $W^Q,W^K,W^V,W^O$ 的初始化方案，让自注意力子块的导数矩阵 $K_\ell$ 在初始时就良态（5.1–5.2 节）；最后配一个二阶优化器 SOAP，把"良好初始条件"在整个训练过程中维持住，使 skipless ViT 的收敛速度追平甚至超过带残差基线。

这是一篇分析驱动的方法论文——核心是矩阵条件数的推导和初始化构造，而非多模块串行的 pipeline，所以下面用公式而非框架图来讲清。整套方法不引入任何新模块、不改前向计算图，与 FlashAttention 完全兼容。

记 Transformer 块为自注意力块（SAB）+ 前馈网络（FFN）：带残差时 $X_\ell=\hat X_{\ell-1}+\mathrm{SA}(\hat X_{\ell-1})$、$\hat X_\ell=X_\ell+\mathrm{MLP}(X_\ell)$；skipless 就是把这两个 $+\hat X_{\ell-1}$、$+X_\ell$ 删掉。其中 $\mathrm{SA}(X)=AVW^O$，注意力 $A=\eta(QK^\top)$，$\eta$ 为 softmax。

### 关键设计

**1. Jacobian 分解：看清残差到底改善了哪一项的条件数**

把网络写成 $f(x;\theta)$（去掉 token embedding 与输出头，只看块内交互），网络 Jacobian $J=\partial F/\partial\theta$ 按 SAB、FFN 子块拆成块列。作者沿用一个简化假设：整网条件数被最差的子块条件数控制，$\kappa(J)\le\max_\ell\{\kappa(J_\ell),\kappa(\hat J_\ell)\}$，并且已知自注意力子块 $J_\ell$ 的条件数远差于 FFN 子块 $\hat J_\ell$——所以问题的瓶颈锁定在自注意力。

关键对比在于：记 $K_\ell$、$\hat K_\ell$ 分别为 SA、MLP 输出对各自输入的导数。带残差时，第 $\ell$ 层 SA 参数的导数里出现的是 $(\hat K_i+I_{nd})(K_i+I_{nd})$ 这类**加了单位阵**的连乘；去残差后就变成 $\hat K_i K_i$ 的裸连乘。这个 $+I_{nd}$ 正是残差的全部魔力：它把本就病态（最小奇异值贴近 0）的 $K_\ell$ 的谱整体平移、正则化掉那些接近零的最小奇异值，从而让条件数变好。由此自然引出本文要回答的问题——**有没有别的办法，让 skipless 下也有 $\kappa(K_\ell)\approx\kappa(K_\ell+I)$？**

**2. $W^V W^O$ 缩放正交初始化：让"公共因子"条件数恰好为 1**

把自注意力子块的导数写开：

$$K_\ell=(\hat X_{\ell-1}W^V_\ell W^O_\ell\otimes I_n)^\top A'_\ell+(W^V_\ell W^O_\ell)^\top\otimes A_\ell$$

可以看到乘积 $W^V_\ell W^O_\ell$ 同时出现在两项里，是个"公共因子"。要让 $K_\ell$ 良态，这个乘积本身必须良态；最理想是它为（缩放）正交矩阵——这样所有奇异值相等、$\kappa(W^V_\ell W^O_\ell)=1$。做法很直接：先采一个零均值单位方差的随机方阵 $Q\in\mathbb R^{d\times d}$，做 SVD $Q=USV^\top$，然后令 $W^V_\ell=c\cdot U$、$W^O_\ell=c\cdot V^\top$（$c$ 为缩放常数）。这样 $W^V_\ell W^O_\ell$ 就是缩放正交阵，把 $K_\ell$ 第二项里的这个因子钉死在最优条件数上。

**3. $W^Q W^{K\top}$ 对角占优初始化：把注意力图从"均匀矩阵"救回良态**

光有正交的 $W^V W^O$ 还不够，$K_\ell$ 里还有注意力矩阵 $A_\ell=\mathrm{softmax}(M_\ell)$，$M_\ell=\hat X_{\ell-1}W^Q_\ell W^{K\top}_\ell\hat X^\top_{\ell-1}$，它的条件数取决于 logits $M_\ell$ 的结构。作者用 Proposition 1 点明两种极端：若每行 logits 的极差 $\Delta\ll\tau$（"弥散行"），softmax 趋近秩 1 的均匀矩阵 $\frac1n\mathbf{1}\mathbf{1}^\top$，$\kappa\gtrsim\tau/\Delta$ 且随 token 数 $n$ 增大而更糟；反之若 $M_\ell$ **对角占优**（$M_{ii}-\max_{j\ne i}M_{ij}\ge\gamma>0$），softmax 趋近单位阵、条件数良好。问题在于：随机初始化下 logits 恰恰是"弥散"的，注意力图接近均匀矩阵——这正是 $K_\ell$ 病态的主因。

对策是把 query/key 投影初始化成 $W^Q_\ell W^{K\top}_\ell=\alpha Z+\beta I$，其中 $Z_{ij}\sim\mathcal N(0,1/d)$，$\alpha,\beta$ 为标量常数。这种"mimetic 初始化"经验上能提升收敛与性能，而本文的贡献是给它一个**理论动机**：单位项 $\beta I$ 鼓励 $W^Q_\ell W^{K\top}_\ell$ 对角占优，进而让初始注意力算子良态。作者也诚实指出，$W^Q W^{K\top}$ 对角占优并不自动等于投影后 $X^\top W^Q W^{K\top}X$ 也对角占优，附录给了它在何种条件下能传递过去的讨论。两套初始化合在一起（Proposition 2），就能保证整个 $K_\ell$ 良态——直觉是扰动项 $E_\ell$ 的最大奇异值小于主导项 $B_\ell$ 的最小奇异值，于是 $\kappa(K_\ell)\approx\kappa(B_\ell)$，从根上拆掉了历来阻碍 skipless Transformer 训练的障碍。

**4. 搭配二阶优化器 SOAP：把初始良态在全程维持住**

良好的初始条件只是起点，训练过程中条件数还会漂移。本文把上述初始化与二阶优化器 SOAP（Vyas et al. 2025）配套使用：实验显示，单靠 AdamW 时 skipless 即便有好初始化也只能恢复大部分性能，而换成 SOAP 后，skipless ViT 能在标准 300 epoch 内**收敛得和带残差的 ViT 一样快**，并最终反超。可以理解为：初始化负责把训练拉到良态起点，二阶优化器负责在含病态自注意力的优化地形里持续保持良态、不让条件数重新恶化——二者缺一不可。

### 损失函数 / 训练策略
方法不引入新损失。监督实验用 ViT-Base（12 层、12 头、头维 64、token 维 768），skipless 模型去掉 SAB 与 FFN 的全部残差，自注意力权重用本文初始化（$\alpha=2,\beta=0.6,c=3$），MLP 参数用 scale-corrected uniform orthogonal 初始化，并关掉 skipless 不适用的 drop path；在 ImageNet-1k 上对比 AdamW 与 SOAP。自监督实验用 DINO 自蒸馏框架 + ViT-Small（12 层、6 头、token 维 384，$\alpha=1.8,\beta=1,c=3$）。作者注明初始化超参 $(\alpha,\beta,c)$ 不敏感。

## 实验关键数据

### 主实验
ImageNet-1k 上 ViT-Base 的验证集准确率（监督）：去掉残差后 AdamW 直接崩到 61.4%，本文初始化把它救回 78.1%；配 SOAP 后 skipless 达 **80.8%**，反超带残差基线 0.5 个点。

| 配置 | 优化器 | 准确率 |
|------|--------|--------|
| 带残差 ViT-Base | AdamW | 80.3% |
| 带残差 ViT-Base | SOAP | 80.1% |
| skipless（无初始化） | AdamW | 61.4% |
| skipless（无初始化） | SOAP | 77.0% |
| skipless + 本文初始化 | AdamW | 78.1% |
| skipless + 本文初始化 | SOAP | **80.8%** |

自监督（DINO ViT-Small，300 epoch）密集线性探针分割 mIoU 与 TokenCut 目标发现，skipless 在密集任务上整体反超带残差：

| 任务 / 数据集 | 评估方式 | 优化器 | 带残差 | skipless |
|---------------|----------|--------|--------|----------|
| 分割 VOC2012 | 单层特征 | AdamW | 56.3 | **62.3** |
| 分割 VOC2012 | 多尺度 | AdamW | 61.6 | **65.4** |
| 分割 COCOStuff | 单层特征 | AdamW | 24.6 | **24.9** |
| 分割 ADE20K | 单层特征 | AdamW | 23.7 | 22.5 |
| 分割 ADE20K | 多尺度 | AdamW | 26.0 | **26.3** |
| 目标发现 VOC2012 | TokenCut | SOAP | 49.4 | **63.2** |
| 目标发现 COCO20k | TokenCut | SOAP | 27.5 | **46.7** |

### 消融实验

| 配置 | 关键指标（ImageNet acc.） | 说明 |
|------|---------------------------|------|
| skipless 裸训（AdamW） | 61.4% | 去残差且无初始化，几乎训崩 |
| + 本文初始化（AdamW） | 78.1% | 仅靠初始化就恢复大部分性能 |
| + 本文初始化 + SOAP | 80.8% | 初始化与二阶优化器缺一不可，反超基线 |

### 关键发现
- **初始化是 skipless 能训的前提**：没有它，AdamW 下 skipless 掉到 61.4%；有了它直接拉回 78.1%——这是单点贡献最大的一环。
- **初始化 × 二阶优化器是互补的**：只加初始化（AdamW）追平不了带残差，配上 SOAP 才在 300 epoch 内收敛同速并反超 0.5 点，说明"良好起点"还需优化器在全程维持。
- **单层 vs. 多尺度揭示层级差异**：skipless 在场景复杂的 ADE20K 单层评估下略逊，作者归因于带残差模型能隐式跨层混合、自带多尺度上下文；而 skipless 强制更严格的层级结构、每层特征更抽象，所以一旦在评估时**显式聚合多尺度层特征**，skipless 又重新反超。
- **更浅却更强，凸显效率**：深度分析中，skipless 仅 9 层就在目标发现上超过带残差的 12 层 ViT；10 层 skipless 在分割上与带残差相当——印证 skipless 设计的参数/深度效率。
- **表征更干净**：对第 11 层特征做 PCA 投影成 RGB，带残差模型因浅层信息反复注入而呈斑驳噪声，skipless 则物体边界清晰、同物体内颜色一致，语义更连贯。

## 亮点与洞察
- **把"残差为什么有用"翻译成一项可被替代的数学事实**：作者证明残差在 Jacobian 里就是那个 $+I_{nd}$ 项，作用是把自注意力子块病态的最小奇异值正则化掉。既然只是改善条件数，那就能用初始化把这份好处搬过来——这是全文最漂亮的"啊哈"。
- **不改架构、兼容 FlashAttention**：和前作 He et al. 去改注意力块不同，本文只动权重初始化，标准 Transformer 块原封不动，工程上即插即用、能直接吃 FlashAttention 的加速，这是落地性的关键差异。
- **"公共因子"视角下的正交初始化**：识别出 $W^V W^O$ 是 $K_\ell$ 两项里的公共因子，于是用 SVD 把它初始化成 $\kappa=1$ 的缩放正交阵——这种"先找出条件数瓶颈在哪个乘积上、再把它钉到最优"的思路，可迁移到其他需要控制 Jacobian 条件数的结构（如其他注意力变体、深 MLP 堆叠）。
- **给经验性的 mimetic 初始化补了理论**：$\alpha Z+\beta I$ 此前是经验 trick，本文用"对角占优 → softmax 趋近单位阵 → 良态"把它讲圆了，并诚实标注对角占优在投影后不一定保持，附带条件分析——这种自我设限让结论更可信。

## 局限与展望
- **规模受限**：实验只到约 100M 参数的 ViT，作者明确未验证十亿级模型上 skipless 训练是否依然成立，扩到大模型是留给未来的方向。
- **只在视觉 ViT 上验证**：选 ViT 是因为视觉任务天然层级、便于分析与可视化；但前作 He et al. 做的是语言 Transformer，本文未在语言模型上检验该初始化是否同样有效。
- **理论假设未显式验证**：条件数分析依赖"整网条件数被最差子块控制"以及温和的 block-incoherence 假设，作者承认没有显式验证，只用经验上跨深度/跨任务的稳定表现来旁证其实用性。
- **依赖二阶优化器才能完全追平**：纯 AdamW 下 skipless+init 仍逊于带残差，需配 SOAP 才反超，二阶优化器的额外开销与可扩展性是隐含成本。

## 相关工作与启发
- **vs He et al. 2023（唯一训 skipless Transformer 的前作）**：他们从前向核（kernel）出发、改自注意力块来防止核矩阵塌到秩 1；本文从**网络 Jacobian 的条件数**出发，只改权重初始化、不动架构，且在视觉模型而非文本上验证。本文优势是兼容标准实现与 FlashAttention、收敛更快，劣势是尚未在语言模型上验证。
- **vs ResNet/Transformer 的残差范式（He et al. 2016; Vaswani et al. 2017）**：主流把残差当深网络可训练性的必需品；本文反过来论证残差对优化必需、但对层级表征有害，并给出"去掉它也能训"的存在性证明，主张 skip 并非训练 ViT 的根本前提。
- **vs 残差让网络"变浅"的研究（Veit et al. 2016; Gromov et al. 2025）**：这些工作指出残差使深网络行为像浅子网络集成、深层可剪；本文把这个观察转化为正面动机——去掉残差以恢复真正的深层级抽象，并用 PCA 可视化与多尺度反超佐证 skipless 学到了更抽象的层级表征。
- **vs 条件化初始化系列（Ji et al. 2025a/b; Saratchandran & Lucey 2025）**：同样以"改善 Transformer 条件数"为线索，但本文聚焦的是**去残差**这一更激进的设定，并具体到 $W^V W^O$ 正交、$W^Q W^K$ 对角占优两条可操作的初始化规则。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 第一个不改架构、纯靠初始化就能稳定训练完全去残差 Transformer 的方法，且把"残差=改善条件数"讲成可替代的数学事实。
- 实验充分度: ⭐⭐⭐⭐ 监督+自监督、多数据集、多优化器、深度分析与 PCA 可视化都有，但规模止于 ~100M 且仅视觉。
- 写作质量: ⭐⭐⭐⭐⭐ 从 Jacobian 分解到初始化构造逻辑严密，命题清晰，且对假设与不保证之处坦诚标注。
- 价值: ⭐⭐⭐⭐ 让"真正很深的 skipless ViT"第一次可被系统研究，为层级表征学习开了新口子，工程上即插即用。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] WSM: Decay-free Learning Rate Schedule via Checkpoint Merging for LLM Pre-training](wsm_decay-free_learning_rate_schedule_via_checkpoint_merging_for_llm_pre-trainin.md)
- [\[ICLR 2026\] Learning to Recall with Transformers Beyond Orthogonal Embeddings](learning_to_recall_with_transformers_beyond_orthogonal_embeddings.md)
- [\[NeurIPS 2025\] Covariances for Free: Exploiting Mean Distributions for Training-free Federated Learning](../../NeurIPS2025/optimization/covariances_for_free_exploiting_mean_distributions_for_training-free_federated_l.md)
- [\[ICLR 2026\] Neural Sum-of-Squares: Certifying the Nonnegativity of Polynomials with Transformers](neural_sum-of-squares_certifying_the_nonnegativity_of_polynomials_with_transform.md)
- [\[ICLR 2026\] Markovian Transformers for Informative Language Modeling](markovian_transformers_for_informative_language_modeling.md)

</div>

<!-- RELATED:END -->
