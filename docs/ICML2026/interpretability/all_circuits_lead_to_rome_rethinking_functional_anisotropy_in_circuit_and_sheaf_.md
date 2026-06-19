---
title: >-
  [论文解读] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs
description: >-
  [ICML 2026][可解释性][circuit discovery] 这篇论文用 Overlap-Aware Sheaf Repulsion (OASR) 算法系统性地证伪了机理可解释性领域的隐含假设——"一个 LLM 能力对应一个独特的电路"——发现同一任务可被多个几乎不重叠 (IoU ~4–11%) 但都满足 faithful/sparse/complete 的电路或 sheaf 支撑，并给出"分布式稠密电路假设"作为理论解释。
tags:
  - "ICML 2026"
  - "可解释性"
  - "circuit discovery"
  - "sheaf discovery"
  - "functional anisotropy"
  - "superposition"
  - "IOI"
---

# All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs

**会议**: ICML 2026  
**arXiv**: [2605.12671](https://arxiv.org/abs/2605.12671)  
**代码**: <https://github.com/TonyXiChen/OASR>  
**领域**: LLM 可解释性 / 机理可解释性 (Mechanistic Interpretability) / Circuit & Sheaf Discovery  
**关键词**: circuit discovery, sheaf discovery, functional anisotropy, superposition, IOI

## 一句话总结
这篇论文用 Overlap-Aware Sheaf Repulsion (OASR) 算法系统性地证伪了机理可解释性领域的隐含假设——"一个 LLM 能力对应一个独特的电路"——发现同一任务可被多个几乎不重叠 (IoU ~4–11%) 但都满足 faithful/sparse/complete 的电路或 sheaf 支撑，并给出"分布式稠密电路假设"作为理论解释。

## 研究背景与动机

**领域现状**：电路与 sheaf 发现 (CSD) 是机理可解释性的主流路线——从 ACDC、EAP 到 EP、DiscoGP，都把 LLM 编码成残差流计算图 (DAG)，然后用启发式或基于梯度的边掩码优化找出能在 IOI、Docstring、BLiMP 等任务上保持准确率的稀疏子图。

**现有痛点**：现有方法的评测范式默认"唯一最小电路"——要么用 Tracr 等基准比对预设 ground-truth，要么追求"最少边数 + 保持性能"，隐式地假设每个任务被一个结构上独特的子机制实现。但作者发现，按现有方法找两个 sheaf 时，它们的边几乎不重合却都能 100% 完成任务。

**核心矛盾**：作者把这一隐含假设命名为**功能各向异性假设 (Functional Anisotropy Hypothesis)**——"模型能力被各向异性地局部化在某个子机制上"。但若多个结构差异巨大的 sheaf 都能独立支撑同一任务，那么"找到唯一电路" 这件事本身就失去了机制解释意义；先前的 backup head / hydra effect 等冗余现象只是"消融后才显形" 的备用机制，无法解释"正常推理中多电路并存"。

**本文目标**：(1) 设计一个能主动发现"互不重叠但都 faithful"的多电路的算法；(2) 在多个常用基准上系统证伪 anisotropy 假设；(3) 给出理论解释为什么这种非唯一性在 LLM 中天然存在。

**切入角度**：在 DiscoGP 的可微 sheaf 优化目标里加入一个"排斥已发现电路"的正则项，把"找下一个 sheaf" 显式转化为"在结构空间里远离上一个"。

**核心 idea**：用 Overlap-Aware Sheaf Repulsion (OASR) 把 CSD 从"找唯一最小子图"重构为"在功能等价类里枚举多个解"，再用高维 superposition 论证这些解必然多到无法被还原成单一规范电路。

## 方法详解

### 整体框架
要证伪"一个任务对应一个唯一电路"，光靠随机重跑 DiscoGP 还不够——优化器往往陷回同一个吸引子，找出来的 sheaf 仍高度重叠。OASR 的思路是把"找下一个 sheaf"显式改写成"在结构空间里远离已有的所有 sheaf"：沿用 DiscoGP 的可微框架，给残差流计算图每条边 $e$ 配一个可学习 logit $l_e$，经 Gumbel-Sigmoid 松弛得连续分数 $s_e=\sigma((l_e-\log(\log\mathcal{U}_1/\log\mathcal{U}_2))/\tau)$ 并用 straight-through estimator 转成硬掩码，再在原有 fidelity / sparsity / completeness 三项目标上叠一个排斥项，循环跑 $K$ 次，每次都要求新解几乎不复用旧解的边，从而主动枚举出一组互不重叠却都 faithful 的电路。

### 关键设计

**1. Overlap-Aware Sheaf Repulsion 排斥项：把非唯一性从被动现象变成主动发现**

现有 CSD 方法的痛点在于它们都奔着"唯一最优解"去，即便重新随机初始化重跑，得到的 sheaf 仍可能高度重叠，无法证明"任务能力可被多个结构迥异的子图独立承载"。OASR 在每轮发现时显式惩罚对已发现边的复用：设已发现 sheaf 的边集为 $R$，新一轮总损失为 $\mathcal{L}=\mathcal{L}_{fidelity}+\lambda_s\mathcal{L}_{sparsity}+\lambda_c\mathcal{L}_{complete}+\lambda_o\mathcal{L}_{overlap}(R)$，其中排斥项 $\mathcal{L}_{overlap}(R)=\frac{1}{|E|}\sum_{e\in R}\sigma(l_e)$ 只对 $R$ 里的边施加期望激活惩罚，相当于在 logit 空间里持续往"远离已有解"的方向加梯度。重复执行就得到 $K$ 个互不重叠的 sheaf $\{R_1,\dots,R_K\}$，其两两 IoU 远低于随机重跑的水平——这正是"非唯一性是 LLM 本质而非优化噪声"的直接证据。

**2. 互补/复杂验证协议：排除"电路太大随便切都不重叠"的平凡解释**

仅靠"任务 acc 一致 + IoU 低"不足以反驳 anisotropy，因为两个 sheaf 完全可能只是同一规范电路的旋转或重参数化，看似不同实则同源。为此作者设计了一套多维验证：先测每个 sheaf 的 IOI accuracy 确认任务保持率，再测 complement accuracy——把发现的边集 $E_A$ 整体抠掉、看剩下的图能否完成任务，以此验证 $E_A$ 确实是必要的而非冗余;同时报告 edge density (选中边占总边比例) 与 edge count，并与"随机初始化重跑"基线对照；最后对极端稀疏的三条边 sheaf 做逐边消融，检验是否任何一条都不可或缺。配合 layer-wise 分布分析，作者得以证实多 sheaf 之间的差异是结构性的，而不是表层的边重排。

**3. Distributive Dense Circuit Hypothesis：用 superposition 解释非唯一性为何必然**

实验观察到的非唯一性若只是 DiscoGP 的优化伪影，结论就站不住，所以作者给出了一个理论假设来说明它是 LLM 表征的结构性后果。论证基于 Elhage 等的 superposition 理论：高维残差流用近正交方向叠加表示多组特征，任何具体的"计算实现"都是把这些方向以某种线性组合路由到下游；给定任务行为 $b$，满足 fidelity 的子图集合会随模型深度/宽度组合膨胀，使"既 sparse 又 faithful 的解"以指数级增长。由此得到形式化命题——在 mild assumption 下，对足够大的模型存在 $\Omega(\exp(d))$ 个互不相交的 faithful sheaf。这个视角把 backup heads、hydra effect 等"局部冗余"现象统一到了"全局分布式稠密电路"的框架之下：冗余不是异常，而是默认状态。

### 损失函数 / 训练策略
四项损失加权求和 $\mathcal{L}=\mathcal{L}_{fid}+\lambda_s\mathcal{L}_{sp}+\lambda_c\mathcal{L}_{comp}+\lambda_o\mathcal{L}_{overlap}$，主要实验对象为 GPT-2 small (12L × 12H)，超参沿用 DiscoGP 默认设置；为得到 20 个 sheaf，OASR 每轮固定上一轮发现的 $R$、重新初始化 logits 后再联合优化 fidelity / sparsity / completeness / overlap 目标。

## 实验关键数据

### 主实验
在 GPT-2 small 上对 IOI 任务用 OASR 发现两个 sheaf：

| Sheaf | IoI Acc | Comp. Acc | Edge Density | Edge # |
|-------|---------|-----------|--------------|--------|
| A | 100% | 46.20% | 3.56% | 1158 |
| B | 100% | 45.80% | 3.97% | 1289 |
| 重叠 | $\|A\cap B\|=96$ | $\|A\cup B\|=2351$ | **IoU = 4.1%** | — |

跨 9 个常用 CSD 基准任务，每个任务发现两个 sheaf：

| Task | Sheaf A Acc | Sheaf B Acc | IoU(A,B) |
|------|------------|-------------|----------|
| IOI | 100% | 100% | 4.1% |
| BLiMP | 96.8% | 92.6% | 5.1% |
| AGA | 96.0% | 95.3% | 6.2% |
| ANA | 98.0% | 91.3% | 5.3% |
| DNA | 100% | 96.2% | 5.8% |
| DNA-i | 100% | 99.0% | 6.2% |
| DNA-a | 98.5% | 97.0% | 7.5% |
| DNA-ia | 100% | 99.0% | 6.4% |
| Docstring | 98.9% | 100% | 11.0% |

### 消融实验
扩展到 20 个 sheaf 的"互相交集" 分析 (Mutual IoU = 20 个 sheaf 的总交集 / 总并集)：

| Task | Method | \|E_∩\| | \|E_∪\| | Mutual IoU | 平均 Acc |
|------|--------|--------|--------|------------|----------|
| IOI | Random Init | 20 | 6560 | 0.30% | 99.95% |
| IOI | **OASR** | 11 | 7382 | **0.15%** | 99.59% |
| BLiMP | Random Init | 50 | 4858 | 1.03% | 97.26% |
| BLiMP | **OASR** | 37 | 5289 | **0.70%** | 96.11% |
| ANA | Random Init | 26 | 4531 | 0.57% | 96.40% |
| ANA | **OASR** | 10 | 4890 | **0.20%** | 95.00% |

### 关键发现
- "多 sheaf 之间的公共交集" 随发现数量增加趋近于 0：从 2 个 sheaf 的 IoU ≈ 4–11% 到 20 个 sheaf 的 Mutual IoU < 1%，说明任务能力不是被任何"essential core" 携带的。
- 还发现了一个仅 3 条边的极端稀疏 IOI sheaf；逐边消融该 3 条边——去掉任意一条仍能用 OASR 找到另一组高质量 sheaf，证伪"必要核心边" 的弱化版假设。
- OASR 相比"重新随机初始化跑 DiscoGP" 在所有任务上都拿到更低 Mutual IoU 且 acc 几乎不掉，说明排斥项不是平凡的随机扰动。
- 层间分析 (Fig. 2) 显示两个 sheaf 在 mid-layer MLP 入边分布上差异最大，并非表层重参数化。

## 亮点与洞察
- **范式级冲击**：作者直接挑战了机理可解释性社区做了几年的核心目标——"找到 the circuit"。如果一个任务有无数个 faithful 解，那么 minimality-based 评测 (MIB)、ground-truth 比对 (Tracr) 的合理性都需要重写。这是"问题定义错了"级别的发现。
- **OASR 思想可泛化**：把"多样化检索"用排斥项写进可微目标，这一手法适用于任何"想枚举功能等价解" 的问题——比如稀疏特征字典、多解 NAS、对抗样本多模发现等。
- **理论与实证互证**：作者不是只发表实验现象，而是用 superposition 给出数学解释，把 backup head / hydra effect / 多电路三种现象在"分布式稠密电路" 框架下统一，提升了贡献的层次。
- **3 边 sheaf 的存在意义**：极端稀疏却没有任何一边不可或缺，说明 "minimal" 在 sparse 优化下找出的解可能恰恰是最不可解释的——任何一条边都能在其他 sheaf 中被替换。

## 局限与展望
- 实验主要在 GPT-2 small 上 (作者承认这是因为所有 CSD baseline 都只支持它)，更大模型 (Llama / Pythia) 的扩展实验仅在附录 H，未充分回应"是否 scale 后 anisotropy 反而回归"。
- 任务都是 short-context、单 step 的语言学诊断 (IOI/BLiMP/Docstring)，对长 reasoning、code、math 等真实任务的电路非唯一性尚无验证。
- OASR 需要循环训练 $K$ 次，计算成本随 $K$ 线性增长；20 个 sheaf 的设置已经很贵，对更大模型可能不可行。
- 理论命题 Distributive Dense Circuit Hypothesis 给出存在性下界但没有给出量化构造，且依赖 mild assumption——这个 "mild" 在真实 transformer 是否成立有待检验。
- "电路非唯一 → 机理解释失效" 是否过于悲观？作者未给出"在多解情况下如何提取共识机制" 的建设性方案。

## 相关工作与启发
- **vs Wang et al. 2022a (IOI 原始电路)**：提出了 Backup Name-Mover Heads 概念，但解释为"消融触发的备用机制"；本文证明 backup 在正常推理中就活跃，redundancy 是默认状态而非异常。
- **vs ACDC / EAP / EP / DiscoGP**：本文沿用了 DiscoGP 的可微优化框架，但把"找唯一最优解" 改为"枚举多解"，反过来用这些方法验证了它们都受 anisotropy 假设之害。
- **vs McGrath et al. 2023 (Hydra effect)**：hydra 描述消融后替补激活，本文说明多电路同时活跃，因此 hydra 只是分布式电路在 ablation 视角下的一个切面。
- **vs Méloux et al. 2025**：他们形式化证明了简单模型中电路非唯一，本文把视角推广到预训练 LM 与真实任务，并给出 superposition 视角的统一解释。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 直接挑战可解释性社区的隐含假设，提出新假设 + 新算法 + 新评测协议三件套，是范式级冲击。
- 实验充分度: ⭐⭐⭐⭐ 9 任务 × 20 sheaf 的系统实验设计令人信服，但模型规模与任务复杂度上仍局限于 GPT-2 small。
- 写作质量: ⭐⭐⭐⭐⭐ 概念命名 (Functional Anisotropy / Distributive Dense Circuit) 清晰准确，逻辑层次极佳，附理论与实证相互印证。
- 价值: ⭐⭐⭐⭐⭐ 强制整个机理可解释性社区重新评估其工作的解释力，影响深远。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] Certified Circuits: Stability Guarantees for Mechanistic Circuits](certified_circuits_stability_guarantees_for_mechanistic_circuits.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)
- [\[ACL 2025\] Position-aware Automatic Circuit Discovery](../../ACL2025/interpretability/position-aware_automatic_circuit_discovery.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)

</div>

<!-- RELATED:END -->
