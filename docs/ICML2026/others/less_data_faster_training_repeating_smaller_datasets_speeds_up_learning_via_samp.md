---
title: >-
  [论文解读] Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases
description: >-
  [ICML 2026][small-vs-large gap] 本文系统刻画并解释了"小数据多重复反而比大数据更快收敛"的 small-vs-large gap 现象：作者证明该加速既不能由 CSQ-SQ 差距、梯度方差减少、输入分布偏置三种已有理论解释，又通过 2-sparse parity 上的 2-layer 二次激活 MLP 给出闭式步数界 $T = O((Nd)^{1/4} \log(d/\varepsilon))$，并通过随机标签、初始化缩放、层间学习率等一系列干预实验验证：真正驱动加速的是"小数据集天然存在的 $O(N^{-1/2})$ 采样偏差通过加快第二层范数增长来加速第一层特征学习"。
tags:
  - "ICML 2026"
  - "small-vs-large gap"
  - "采样偏差"
  - "层间范数"
  - "特征学习"
  - "多轮重复训练"
---

# Less Data, Faster Training: Repeating Smaller Datasets Speeds Up Learning via Sampling Biases

**会议**: ICML 2026  
**arXiv**: [2605.20314](https://arxiv.org/abs/2605.20314)  
**代码**: 待确认  
**领域**: 优化 / 特征学习 / 训练动力学  
**关键词**: small-vs-large gap, 采样偏差, 层间范数, 特征学习, 多轮重复训练

## 一句话总结
本文系统刻画并解释了"小数据多重复反而比大数据更快收敛"的 small-vs-large gap 现象：作者证明该加速既不能由 CSQ-SQ 差距、梯度方差减少、输入分布偏置三种已有理论解释，又通过 2-sparse parity 上的 2-layer 二次激活 MLP 给出闭式步数界 $T = O((Nd)^{1/4} \log(d/\varepsilon))$，并通过随机标签、初始化缩放、层间学习率等一系列干预实验验证：真正驱动加速的是"小数据集天然存在的 $O(N^{-1/2})$ 采样偏差通过加快第二层范数增长来加速第一层特征学习"。

## 研究背景与动机
**领域现状**：深度学习的主流信条是"数据越多越好"，scaling law 和经典泛化理论都支撑这一点。然而最近一连串工作 (Charton & Kempe 2024; Zucchet et al. 2025; Kopiczko et al. 2026) 发现了一个反常现象：在固定计算预算 (步数 × batch size) 下，对一个较小的数据集反复训练，反而能比对大数据集做 fresh-sample 在线训练达到更好的测试性能；在 sparse parity 上这种 compute 节省可达两个数量级。这一现象作者统一称为 small-vs-large gap。

**现有痛点**：现有解释都站不住。(1) Dandi et al. 2024 提出"重复 batch 让 SGD 从 CSQ 算法升级为更强的 SQ 算法"，但这只在 SQ-CSQ 下界存在差距的任务 (如 single-index model) 上有意义，对 sparse parity、modular addition 等离散任务 SQ = CSQ，理论不适用；而且 full-batch GD 下大小数据都是"重复使用"，差距依旧存在，CSQ-SQ 分类完全失效。(2) 梯度方差减少 (Kotha 2025) 不能解释 full-batch 设定，因为 GD 根本无随机方差。(3) Cornacchia et al. 2025 的"输入分布偏置加速"理论给出的 Fourier 系数是 $O(N^{-k/2})$，对稀疏度 $k=6$ 已经几乎为零；而且作者实证去掉训练集的输入偏置 (强制 $\hat{\mathbb{E}}[x] = 0$) 后加速依旧存在，反向给大数据集人为注入小数据的统计偏置也无法补回 gap。

**核心矛盾**：现象普遍存在 (mini-batch & full-batch、SIM/parity/ICL/mod-add、MLP & Transformer)，但已有三套理论都各自解释不了其中至少一个设定，必须找一个跨设定的统一机制。

**本文目标**：(1) 在更广的任务/架构/优化器矩阵上系统验证 gap 存在；(2) 排除三类候选解释；(3) 给出新机制并配一个有闭式步数界的可分析模型；(4) 设计干预实验逐项验证机制。

**切入角度**：观察到 2 层 MLP 学 parity 时，第一层 (输入层) 才是特征学习层，而第二层的范数 $|a|$ 通过乘到 $\nabla_w L$ 上直接控制第一层的有效梯度；任何能让 $|a|$ 提前涨起来的力量都会加速第一层的特征学习。作者猜测：小数据集的"采样偏差"恰好就是这样一种力量。

**核心 idea**：small-vs-large gap 的本质不是"少看数据"或"重复使用"，而是"小数据集的经验矩 $\hat M = \frac{1}{N}\sum y x x^\top$ 偏离总体矩的方差是 $\Theta(N^{-1/2})$，比 $1/d$ 大得多，从而在训练早期把第二层范数推快一拍，间接加速第一层特征学习"——是一个被动诱发的层间增长不平衡，等价于一种隐式的层间学习率调度。

## 方法详解
本文方法层面分两块：(a) 在一个可分析的 toy 模型上给出步数复杂度定理；(b) 设计一组干预实验把"层间范数增长"作为可观测信号验证机制。

### 整体框架
- 任务集合：单指标模型 (SIM, Hermite link)、$(d,k)$-sparse parity、in-context 线性回归、$(N,p)$-modular addition；优化器涵盖 mini-batch SGD 与 full-batch GD；模型用 2 层 MLP (ReLU, 无残差) 与 2 层 Transformer (可选 QK 归一化)。
- 数据使用策略：除标准 single-set 重复外，引入 $T$-phase 训练 (Charton & Kempe 2024 的推广)，第 $i$ 阶段在子集 $\mathcal{S}_i \subset \mathcal{S}_{i+1}$ 上训练；启发式是前期小子集快速跑出非平凡训练性能，最终阶段用足够大的子集保泛化。
- 分析模型：$f(x) = a \sigma(w^\top x) - 1$，$\sigma(z) = \frac{1}{2}z^2$，correlation loss $\ell(y,y') = -yy'$，带投影更新：$a$ 被裁剪到 $[-1, 1]$，$w$ 每步归一化到单位球；2-sparse parity，$w^\star$ 仅在前两维非零。

### 关键设计

**1. 2-phase 训练的步数闭式界（Theorem 1）：把"小数据加速"翻译成可量化的步数上界**

要证明加速真实存在，得先在一个可分析的 toy 模型上算出步数。论文证明对 $d \le N \le d^2$，2-phase 训练只需 $O((Nd)^{1/4} \log(d/\varepsilon))$ 步即可把 $w$ 收敛到 $\|\hat w - w^\star\|_2 \lesssim \sqrt{\varepsilon}$，远小于直接全总体训练在宽度 $m \gg d^2$ 时所需的 $O(m^{1/2}\log(d/\varepsilon))$。第一阶段在大小 $N$ 的子集上跑投影 GD 直到 $|a| \ge a_\star$：关键不等式是 $a$ 的梯度幅度由 $q^{(t)} = (w^{(t)})^\top \hat M w^{(t)}$ 决定，而 $\hat M = \frac{1}{N}\sum y x x^\top$ 的反集中性给出 $|q^{(t)}| = \Theta(N^{-1/2})$，远大于总体梯度 $\Theta(1/d)$，于是 $a$ 在小数据下以 $N^{-1/2}$ 速率涨大，达到 $a_\star$ 只需 $T_1 \lesssim a_\star \sqrt{N}/\eta$ 步。第二阶段切到总体梯度做关于真矩阵 $M$ 的 power iteration，收敛率由 $\eta a_\star$ 控制，$T_2 \lesssim \frac{2}{\eta a_\star}\log(d/\varepsilon)$。两段相加再对 $a_\star$ 取最优即得 $(Nd)^{1/4}$ 的整体率。定理的妙处在于它把机制拆开了：$T_1$ 完全靠采样偏差驱动、与标签信号无关，$T_2$ 完全靠 $a_\star$ 的大小——这意味着任何能在早期把 $|a|$ 推大的手段都该带来等价加速，为后续两组干预实验埋下理论预测。

**2. 随机标签验证机制（Corollary 2 + 实验）：把"采样偏差驱动"和"任务信号驱动"分离**

如果加速来自任务信号或输入分布偏置，用随机标签训练就不该有加速——这是验证机制最干净的差分实验。把 Theorem 1 的第一阶段替换成"在小数据集上用均匀采样的 $\pm 1$ 随机标签训练"，理论上 $|a|$ 仍有 $\Theta(N^{-1/2})$ 的早期增长率，整体步数复杂度 $T = O(\sqrt{N}/(\eta\sqrt{d}) + \sqrt{d}\log(d/\varepsilon)/\eta)$。实验在 MLP-parity、MLP-SIM、Transformer-mod add 上都看到：第一阶段用随机标签的曲线（绿）与小集真标签曲线（黄）几乎重合，且都明显快于大集训练（蓝），实测的 $\|a\|_2 / \|W\|_F$ 比值也在小集/随机标签下更快上升。既然随机标签也有加速、且幅度与真标签相当，就反证"采样偏差 → 第二层快速增长"才是关键路径，标签信号并不重要。

**3. 层间初始化与学习率干预（Section 5.2）：主动复现层间不平衡来消除 gap**

如果机制为真，那么只要人为复现"采样偏差等价制造的层间增长不平衡"，就应该能在大数据上消除 gap。作者在 MLP 与 Transformer 上做三组干预：增大第二层初始化尺度让 $|a^{(0)}|$ 起点更大；用层间学习率给第二层更高的 $\eta_a$；在 Transformer 上观察 QK 归一化是否扮演类似角色。结果上述任一干预都能显著缩小甚至消除大数据相对小数据的差距（QK 归一化效应 nuanced，提示其对优化有未被充分理解的层间作用）。这背后的理论依据是 Theorem 1 第二阶段的收敛率正比于 $\eta a_\star$，即"层间相对增长速度"——任何把这一相对速度补齐的工程手段都该等效于"用小数据集"。这把现象从"经验观察"提升成"可被参数化控制的优化效应"。

### 训练策略
所有 MLP/Transformer 都默认 PyTorch 初始化 ($W_{ij} \sim \text{Unif}[-1/\sqrt{d_{\text{in}}}, 1/\sqrt{d_{\text{in}}}]$)、MLP 用 SGD、Transformer 用 AdamW，对每个设定独立 sweep 学习率；性能在固定 compute = batch × steps 下取多个随机种子平均，用以表示成功概率。

## 实验关键数据

### 主实验

| 任务 / 设定 | 数据集大小对比 | 观察到的 compute 节省 | 说明 |
|------------|---------------|-----------------------|------|
| (20,6)-sparse parity (mini-batch SGD, 2-layer Transformer) | 小集 vs 在线 | 黄色明显早于蓝色收敛 | Fig.1，多任务通用现象 |
| (20,6)-sparse parity (full-batch GD, 2-layer MLP) | $N = 2^{14}$ vs $N = 2^{20}$ | 约 100× compute 加速 | Fig.2，破除 SQ-CSQ 与方差减少假说 |
| SIM ($d=40$, full-batch GD) | 小集 vs 总体 | 每一步都更快 | 同 Fig.2 |
| ICL 线性回归 / mod addition (Transformer) | 多 phase 训练 | 显著加速 | Fig.1，跨架构通用 |

### 消融与机制实验

| 干预 | 关键指标 | 结论 |
|------|---------|------|
| 强制 $\hat{\mathbb{E}}[x]=0$、$\hat{\mathbb{E}}[y]=0$ | small-set 仍快 | 输入分布偏置不是主因 |
| 给大集人工注入小集统计偏置 ($m \in \{4..12\}$) | 仅 $m=5$ 时勉强追平 | 偏置幅度需小到不能学才匹敌，与 Cornacchia 理论一致 |
| 第一阶段用随机标签的小集 | 加速幅度与真标签相当 | 标签信号无关，采样偏差为主 |
| 第二层初始化放大 / 层间 $\eta_a$ 提升 | gap 大幅减小或消失 | 直接验证层间增长机制 |
| Transformer QK 归一化开关 | 效应 nuanced | 隐式调节层间动力学，值得专门研究 |

### 关键发现
- gap 在 full-batch GD 下依旧存在，是排除一切"随机性带来的加速"假说的最干净证据。
- $\|a\|_2 / \|W\|_F$ 比值是该机制的可观测代理变量：小数据 / 随机标签 / 大初始化第二层都对应更快的比值上升 (Fig.12, Fig.4)。
- 多阶段 (multi-phase) 训练只要前期用小子集即可，后期再切大子集保泛化，给出一个简单可用的训练 schedule 模板。
- $a_\star$ 的最优选择导出 $(Nd)^{1/4}$ 复杂度，提示对 reasoning 任务 (本质是离散组合) 走小数据多重复可能比堆数据更高效。

## 亮点与洞察
- "小数据加速 = 隐式层间学习率"是一个非常具有迁移性的视角：把数据策略与优化器策略统一到"层间相对增长速度"这一根本变量上，给了"为什么 µP / Tensor Programs / 层间 LR 重要"一个新角度。
- 用 2-sparse parity + 二次激活作 toy 模型可以同时算出 $T_1, T_2$ 两段的闭式上界，且预测能精准对应到层间范数比这一可观测变量——这种"理论可量化 + 实验有 proxy"的配对是分析训练动力学论文的范本。
- 随机标签训练也能起到与真实预训练等价的"层间预热"作用，提示某些"看似无意义"的预热步骤 (如用噪声 batch、随机标签 warm-start) 可能比想象中更有意义。

## 局限与展望
- 理论只覆盖 2-sparse parity + 2 层二次激活 MLP + correlation loss + 投影更新这一非常受控的设定；推广到一般 ReLU、深层、cross-entropy 仍是开放问题。
- 实验集中在合成任务 (parity / SIM / ICL / mod-add)，虽借用了 Kopiczko 2026 的 LLM 后训练观察作为旁证，但作者自己未在真实 LLM/ViT 上系统复现。
- 机制围绕"两层网络的层间不平衡"展开，对更深网络的"层间相对增长"是否仍是关键，以及与 LayerNorm/RMSNorm/QK Norm 的交互需要专门工作。
- "小数据多重复"在过参数化或小模型下的过拟合风险论文未深入讨论，工业上要落地仍需在记忆与泛化之间小心调度。

## 相关工作与启发
- **vs Dandi et al. 2024 / Lee et al. 2025 (CSQ → SQ 论)**: 他们解释了 SIM 上 batch SGD 的加速，但被本文用 full-batch GD 与离散任务 (parity/mod-add) 反例否决；本文机制覆盖更广的 batch/full-batch、连续/离散任务。
- **vs Kotha et al. 2025 (梯度方差减少论)**: 同样能解释一部分 mini-batch 现象，但 full-batch 无方差仍有加速，证明方差并非唯一关键。
- **vs Cornacchia et al. 2025 (输入分布偏置论)**: 给出 $O(\eta^k)$ 的偏置信号，相对于本文 $O(N^{-1/2})$ 采样偏差小一个数量级；本文用偏置注入实验直接反驳。
- **vs Charton & Kempe 2024 / Kopiczko 2026**: 这些工作给出现象与启发式 schedule (混合两个数据集 / 多 epoch fine-tune)；本文把它们的有效性追溯到统一的层间增长机制，并给出可解释的 ablation 路径。
- **vs µP / Tensor Programs (Yang & Hu 2020; Everett 2024)**: 那一线工作通过参数化与缩放显式控制层间增长；本文则展示"数据规模本身"通过采样偏差也起到类似作用，是同一原理的"数据侧版本"。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 small-vs-large gap 这一散见的反常现象拼成一个统一机制，并用反例否决三套已有理论，立题与切入都非常硬。
- 实验充分度: ⭐⭐⭐⭐ 干预实验设计完整、tasks/optimizers 矩阵覆盖广，但缺真实 LLM/视觉规模化验证。
- 写作质量: ⭐⭐⭐⭐ 论证链条清晰，Section 4 排除假说与 Section 5 干预证据形成漂亮的"否证-肯证"结构；定理证明草图直击核心，附录细节较重。
- 价值: ⭐⭐⭐⭐ 不仅给出新的训练直觉 ("少数据多 epoch 不只是 fallback"), 也提示对深层模型设计层间学习率与初始化时一个新的思考维度。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[AAAI 2026\] Forget Less by Learning from Parents Through Hierarchical Relationships](../../AAAI2026/others/forget_less_by_learning_from_parents_through_hierarchical_relationships.md)
- [\[ICML 2026\] Coupled Training with Privileged Information and Unlabeled Data](coupled_training_with_privileged_information_and_unlabeled_data.md)
- [\[ACL 2025\] FastMCTS: A Simple Sampling Strategy for Data Synthesis](../../ACL2025/others/fastmcts_a_simple_sampling_strategy_for_data_synthesis.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](../../AAAI2026/others/sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[ICLR 2026\] ANO: Faster is Better in Noisy Landscapes](../../ICLR2026/others/ano_faster_is_better_in_noisy_landscape.md)

</div>

<!-- RELATED:END -->
