---
title: >-
  [论文解读] Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings
description: >-
  [ICML2026][预训练][位置编码] 作者指出主流位置编码 RoPE 把"内容(what)"和"位置(where)"耦合进了同一个相位里，从而在需要"按位置找内容 / 按内容定位置"的任务上表现糟糕；他们提出 PoPE，用 softplus 把幅度(管 what)和纯位置相位(管 where)拆开，仅是 RoPE 的小改动，却在诊断任务、音乐/基因组/语言建模上一致更优，并且**零微调就能外推到 10 倍训练长度**，超过专门做外推的 YaRN。
tags:
  - "ICML2026"
  - "预训练"
  - "位置编码"
  - "RoPE"
  - "极坐标"
  - "长度外推"
  - "自回归序列建模"
---

# Decoupling the "What" and "Where" With Polar Coordinate Positional Embeddings

**会议**: ICML2026  
**arXiv**: [2509.10534](https://arxiv.org/abs/2509.10534)  
**代码**: https://github.com/agopal42/pope  
**领域**: LLM预训练 / 位置编码  
**关键词**: 位置编码, RoPE, 极坐标, 长度外推, 自回归序列建模

## 一句话总结
作者指出主流位置编码 RoPE 把"内容(what)"和"位置(where)"耦合进了同一个相位里，从而在需要"按位置找内容 / 按内容定位置"的任务上表现糟糕；他们提出 PoPE，用 softplus 把幅度(管 what)和纯位置相位(管 where)拆开，仅是 RoPE 的小改动，却在诊断任务、音乐/基因组/语言建模上一致更优，并且**零微调就能外推到 10 倍训练长度**，超过专门做外推的 YaRN。

## 研究背景与动机

**领域现状**：Transformer 的自注意力本身对位置不敏感（在因果掩码下是平移等变 + 排列不变的），因此必须额外注入位置信息。当前几乎所有前沿大模型（Llama 3、DeepSeek-v3、Gemma 3、Qwen3）都采用 RoPE：把 query/key 的 $d$ 维向量切成 $d/2$ 个二维分量，每个分量在 2D 平面里按位置旋转一个角度 $t\theta_c$，于是 query 在位置 $t$、key 在位置 $s$ 做点积时，旋转只剩相对位置 $(s-t)\theta_c$，天然实现"相对位置感知"。

**现有痛点**：作者把 RoPE 的注意力分数从笛卡尔坐标改写成极坐标后，发现了一个被长期忽视的结构性缺陷。每个二维分量 $\bm{q}_{tc}$ 可写成幅度 $\mu_{q_{tc}}$ 和初相 $\phi_{q_{tc}}$，于是注意力分数变成

$$a_{ts}^{\text{RoPE}}=\sum_{c=1}^{d/2}\mu_{q_{tc}}\mu_{k_{sc}}\cos\big((s-t)\theta_c+\phi_{k_{sc}}-\phi_{q_{tc}}\big).$$

**核心矛盾**：上式里的相位项 $\phi_{k_{sc}}-\phi_{q_{tc}}$ 就是病灶——这个相位由 query/key 的**内容**决定，却又被加进了**相对位置**的余弦参数里。也就是说，"匹配什么内容"会动态地平移"在哪个相对位置响应最强"。what 和 where 被搅在一起：模型想纯按位置索引，或纯按内容定位，都会被这个交叉项干扰。

**本文目标**：在保留 RoPE 全部优点（相对位置、平移等变、可高效实现）的前提下，**消除 $\phi_{k_{sc}}-\phi_{q_{tc}}$ 这个 what-where 交叉项**，让 key-query 匹配能写成"一个 what 匹配 ∧ 一个 where 匹配"的合取。

**核心 idea**：换一种极坐标表示——让幅度纯粹编码内容（用 softplus 保证非负，可解释为幅度），让相位纯粹由位置决定（$t\theta_c$），点积的余弦里只剩 $(s-t)\theta_c$，干净地把内容和位置解耦。

## 方法详解

### 整体框架
PoPE（Polar Coordinate Positional Embedding）的目标只有一句话：把 RoPE 注意力分数里那个"内容污染位置"的相位交叉项去掉。做法是把 $d$ 维实值 query/key 重新解释成 $d$ 维**复向量** $\tilde{\bm{q}}_t,\tilde{\bm{k}}_s\in\mathbb{C}^d$：每个复元素的**幅度**来自原始实值特征经 softplus（非负，承载 what），每个复元素的**相位**纯由序列位置给出（$t\theta_c$，承载 where）。注意力分数取两个复向量共轭内积的实部，自然落成

$$a_{ts}^{\text{PoPE}}=\sum_{c=1}^{d}\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}\cos\big((s-t)\theta_c\big),$$

余弦里干干净净只剩相对位置。与 RoPE 相比有两点差异：① 下标 $c$ 现在遍历**单个**元素而非二维对，于是频率数从 $d/2$ 翻倍到 $d$；② 内容→相位的交叉项被彻底删除。在此基础上再为每个频率加一个**可学习但固定**的相位偏置 $\delta_c$，作为对 RoPE 交叉项的"良性替身"。整条路径不引入新模块，只是把"幅度/相位各管一摊"这个归纳偏置硬编码进注意力，因此是纯机制改进，没有多阶段 pipeline。

### 关键设计

**1. 极坐标解耦：幅度管内容、相位管位置**

这是全文的根。RoPE 的问题在于把二维分量当复数后，内容信息会通过初相 $\phi$ 渗进相对位置的余弦参数里。PoPE 改成：复元素 $c$ 的幅度只由原始实值特征决定 $\mu_{\tilde k_{sc}}=\sigma(k_{sc})$、$\mu_{\tilde q_{tc}}=\sigma(q_{tc})$，其中 $\sigma(x)=\ln(1+e^x)$ 是 softplus；复元素的相位只由位置决定 $\phi_{\tilde k_{sc}}=s\theta_c$、$\phi_{\tilde q_{tc}}=t\theta_c$。softplus 在这里不是随手选的激活——它保证幅度非负，这样"复数的模"才能被合法解释为内容强度（一个特征"在不在"）。于是共轭内积的实部 $\Re[\tilde{\bm q}_t^H\tilde{\bm k}_s]$ 展开后，相位差恰好等于纯位置差 $(s-t)\theta_c$，内容只进入幅度乘积 $\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}$。这正是"what 匹配 × where 匹配"的合取：幅度负责内容是否对上，余弦负责相对位置是否对上，两者再不互相平移。

**2. 频率定义与翻倍：把旋转施加到每个元素而非每对元素**

RoPE 把向量切成 $d/2$ 个二维对、每对共享一个频率 $\theta_c=\theta^{-2(c-1)/d}$；PoPE 因为是对**每个**复元素单独定相位，频率改为 $\theta_c=\theta^{(c-1)/d}$，$c=1,\dots,d$，频率个数翻倍。这不只是记号差异：频率使用分析（见实验）显示，RoPE 只在稀疏的少数低频通道上保持高范数、高频通道被压成近似噪声；而 PoPE 因频率更密、内容不再污染位置，能在几乎所有层、整个频率范围上更均匀地利用特征。更密的频率谱是 PoPE 长度外推更稳的物理基础之一。

**3. 可学习相位偏置 $\delta_c$：给被删掉的交叉项一个良性替身**

直接删除 $\phi_{k}-\phi_{q}$ 会损失一点灵活性——有时"最佳相对偏移"确实不该是 0。作者于是给每个频率加一个标量偏置：

$$a_{ts}^{\text{PoPE}}=\sum_{c=1}^{d}\mu_{\tilde q_{tc}}\mu_{\tilde k_{sc}}\cos\big((s-t)\theta_c+\delta_c\big),$$

其中 $\delta_c\in\mathbb{R}$ 可学但与内容无关，单纯为每个频率调一个最优相对偏移。它被约束在 $[-2\pi,0]$ 内（实测提升稳定性），初始化有两种好选择：$\delta_c=0$ 或 $\delta_c\sim\text{Uniform}(-2\pi,0)$。关键观察是：**零初始化对长度外推更重要**，而均匀初始化在分布内性能略好。与 RoPE 交叉项的本质区别在于——$\delta_c$ 是"固定 per-frequency 偏置"，不随 query/key 内容动态变化，因此既补回了灵活性，又没把 what 重新拽回 where。

**4. 复数 Flash Attention 高效实现：几乎零额外开销**

PoPE 看似把实数注意力换成复数注意力，但作者基于 Flash Attention 2 写了 Triton kernel，让复数乘法在 kernel 内完成、绝不显式物化 $d\times d$ 的复值 query-key 矩阵。把 $\tilde q_{tc}=\mu(\cos\phi+i\sin\phi)$ 拆成实部虚部后，共轭内积的实部就是 $\sum_c x_{\tilde q}x_{\tilde k}+y_{\tilde q}y_{\tilde k}$，相比标准注意力只多**一次乘法**。代价是需要 2 倍显存/带宽来存取复值 key/value，但这可以通过"只载入幅度、在 kernel 内做旋转"来消除，从而把开销压回到仅一次额外乘法。论文为了方便原型化不同 PoPE 变体，选了较慢但通用的版本。

### 损失函数 / 训练策略
方法本身不改训练目标：实验全程用 decoder-only Transformer + 因果掩码做自回归建模（标准交叉熵 / 下一 token 预测），唯一把 LayerNorm 换成 RMSNorm 以对齐当代前沿模型。两组对比模型架构与超参完全一致，**只差位置编码**（RoPE vs PoPE）。长度外推实验里，PoPE+ft 仅在更长序列上微调 500 步、不做任何频率插值。

## 实验关键数据

### 主实验

诊断任务 **Indirect Indexing**（给定源串、源字符和相对偏移，要求输出目标字符，需要独立操纵内容与位置 + 指针算术）最能暴露 what-where 耦合问题：

| 任务 | 位置编码 | 准确率 (3 seeds) |
|------|----------|------------------|
| Indirect Indexing | RoPE | 11.16 ± 2.45 |
| Indirect Indexing | PoPE | **94.82 ± 2.91** |

RoPE 几乎学不会（11%≈随机），PoPE 近乎完美解决（95%），直接验证"解耦"假设。

序列建模（NLL，越低越好）与语言建模困惑度（越低越好）：

| 数据集 / 规模 | 指标 | RoPE | PoPE |
|---------------|------|------|------|
| JSB (音乐) | NLL | 0.5081 | **0.4889** |
| MAESTRO (音乐) | NLL | 1.501 | **1.486** |
| Human Ref. Genome | NLL | 4.217 | **4.152** |
| OpenWebText 124M | PPL | 21.55 | **21.33** |
| OpenWebText 253M | PPL | 18.88 | **18.55** |
| OpenWebText 774M | PPL | 15.85 | **15.45** |

语言建模上 PoPE 在 124M→774M 三档规模上一致更低，且性能差距随规模稳住甚至略增。零样本下游任务（6 个 benchmark 平均准确率）：124M 45.33→46.19、253M 48.76→48.78、774M 51.80→**52.46**，三档全胜。

### 消融实验

在 OpenWebText 上拆掉 PoPE 各组件（困惑度，越低越好）：

| 配置 | 124M | 253M | 说明 |
|------|------|------|------|
| PoPE w/o $\sigma()$ (softplus) | 21.57 | 18.93 | 去掉幅度非负约束，掉到比 RoPE 还差 |
| PoPE 用 ReLU 代替 $\sigma()$ | 21.55 | 18.90 | 退化到 ≈ RoPE 水平 |
| PoPE w/o $\bm{\delta}$ | 21.42 | 18.57 | 去掉可学相位偏置，仍优于 RoPE |
| **Full PoPE** | **21.33** | **18.55** | 完整模型 |

### 关键发现
- **softplus 是关键支柱**：去掉它（或换成 ReLU）困惑度直接退回甚至劣于 RoPE，说明"幅度必须非负、可解释为内容强度"不是装饰，而是解耦能成立的前提。
- **零样本长度外推是最亮点**：模型在 1024 上下文训练，测到 PG-19 最长 10240（10×）token。RoPE 性能随长度急剧崩坏；YaRN 只在不超过其微调长度(4096)时有效，超过就同样崩；**PoPE 完全不微调、不做频率插值，开箱即外推 10×，还反超专门为外推设计的 YaRN**。
- **外推随规模的反差**：RoPE 的外推能力随模型增大而**变差**，PoPE 则基本稳定。作者归因于 RoPE 的 what-where 交叉项会动态平移分量的位置调谐，这种平移在低频分量上、上下文窗口扩大时才暴露破坏性。
- **频率使用更均匀**：RoPE 只在稀疏低频通道保持高范数（高频近噪声被压低），PoPE 在除第一层外几乎所有层、整个频率范围都有高范数，利用更分散——这与频率翻倍 + 解耦一致。

## 亮点与洞察
- **把已知方法换坐标系就看出病灶**：作者没有发明全新机制，而是把 RoPE 从笛卡尔改写成极坐标，一眼看出 $\phi_k-\phi_q$ 这个内容污染位置的交叉项——"换个表示看清结构缺陷"本身就是很漂亮的分析。
- **极小改动 + 强归纳偏置**：PoPE 只是 RoPE 的微调版（幅度走 softplus、相位走纯位置、频率翻倍、加 $\delta_c$），却引入了"what∧where 合取匹配"这一强先验，同时改善数据效率、渐近精度和长度泛化。
- **诊断任务设计得极有说服力**：Indirect Indexing 把"按位置取内容 / 按内容定位置"逼到极致，11% vs 95% 的对比让"解耦"假设无可辩驳，可迁移为评估任意位置编码 what-where 纠缠程度的探针。
- **可落地**：复数 Flash Attention 仅多一次乘法、显存开销可消除，意味着 PoPE 不是纸面方法，有望直接替换前沿 LLM 里的 RoPE。

## 局限与展望
- **语言domain增益偏小**：作者自己承认，music/genome 之所以选是因为它们明确需要位置与内容分离 + 精确位置，而人类语言是否真有这些性质并不清楚；OpenWebText 上困惑度提升只有 0.2~0.4，远不如诊断任务那么戏剧化。
- **显存权衡留了尾巴**：通用实现需要 2× 显存/带宽存取复值 key/value，论文为原型方便没启用"kernel 内旋转"的省显存版，真要上大规模仍需工程收尾。
- **$\delta_c$ 初始化是 trade-off**：零初始化利于外推、均匀初始化利于分布内，两者不可兼得，说明还没找到统一最优，留有调参负担。
- **对比基线相对单一**：主表主要对打 RoPE/YaRN，正文坦言更广泛的 sinusoidal / learnable / ALiBi / 其他 RoPE 变体比较放在附录，主线证据集中在与 RoPE 的两两对照。

## 相关工作与启发
- **vs RoPE**: 同样用旋转编码相对位置、同样平移等变，但 RoPE 允许内容通过初相平移位置调谐（what-where 耦合），PoPE 用 softplus 幅度 + 纯位置相位把两者解耦，并把频率数翻倍；代价是复数表示，但 Flash Attention 实现后几乎零额外算力。
- **vs YaRN**: YaRN 是 RoPE 的外推补丁，靠对低频做频率插值 + 在长序列上微调来续命，但只能外推到微调长度内、超出仍崩；PoPE 不插值不微调即可外推 10×，从根因（去交叉项）而非打补丁层面解决问题，反超 YaRN。
- **vs 长度外推系列 (Chen 2023 / ALiBi / Sun 2023 / Wang 2024)**: 这些方法或改旋转频率、或加 ALiBi 式衰减与块掩码、或把波长取整以消除环绕偏移，多是在 RoPE 框架内做事后修补；PoPE 把"为什么外推会崩"归到 what-where 交叉项这一结构原因，提供了更本质的视角。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用极坐标改写揭示 RoPE 结构缺陷并以极小改动消除，视角新且解释清晰
- 实验充分度: ⭐⭐⭐⭐ 诊断+音乐+基因组+语言+下游+外推+频率分析覆盖全面，但语言增益偏小、广义基线在附录
- 写作质量: ⭐⭐⭐⭐⭐ 从 RoPE 推导到 PoPE 一气呵成，公式与动机咬合紧密
- 价值: ⭐⭐⭐⭐⭐ 若复现稳健，几乎零成本替换 RoPE 并显著改善长度外推，对 LLM 极有吸引力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Deconstructing Positional Information: From Attention Logits to Training Biases](../../ICLR2026/llm_pretraining/deconstructing_positional_information_from_attention_logits_to_training_biases.md)
- [\[CVPR 2025\] Seeing What Matters: Empowering CLIP with Patch Generation-to-Selection](../../CVPR2025/llm_pretraining/seeing_what_matters_empowering_clip_with_patch_generation-to-selection.md)
- [\[CVPR 2025\] The Scene Language: Representing Scenes with Programs, Words, and Embeddings](../../CVPR2025/llm_pretraining/the_scene_language_representing_scenes_with_programs_words_and_embeddings.md)
- [\[NeurIPS 2025\] ZEUS: Zero-shot Embeddings for Unsupervised Separation of Tabular Data](../../NeurIPS2025/llm_pretraining/zeus_zero-shot_embeddings_for_unsupervised_separation_of_tabular_data.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](../../ICCV2025/llm_pretraining/aceg_improving_generalization_of_scene_coordinate_regression.md)

</div>

<!-- RELATED:END -->
