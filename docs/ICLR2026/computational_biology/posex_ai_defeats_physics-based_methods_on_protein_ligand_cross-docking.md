---
title: >-
  [论文解读] PoseX: AI Defeats Physics-based Methods on Protein Ligand Cross-Docking
description: >-
  [ICLR 2026][计算生物][蛋白-配体对接] PoseX 构建了一个同时覆盖 self-docking 和更贴近真实场景的 cross-docking 的开源对接基准，用 718 + 1312 条无训练泄漏的新晶体结构、23 个跨三大类的对接方法、一套精心设计的能量松弛后处理和一个实时榜单，系统证明了在 cross-docking 这个更难的现实任务上 AI 方法已经全面碾压传统物理对接软件。
tags:
  - "ICLR 2026"
  - "计算生物"
  - "蛋白-配体对接"
  - "cross-docking"
  - "benchmark"
  - "能量松弛"
  - "AI co-folding"
---

# PoseX: AI Defeats Physics-based Methods on Protein Ligand Cross-Docking

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=qqzxKudD4T](https://openreview.net/forum?id=qqzxKudD4T)  
**代码**: https://github.com/CataAI/PoseX （含数据集 https://huggingface.co/datasets/CataAI/PoseX）  
**领域**: 计算生物学 / 分子对接 Benchmark  
**关键词**: 蛋白-配体对接, cross-docking, benchmark, 能量松弛, AI co-folding

## 一句话总结
PoseX 构建了一个同时覆盖 self-docking 和更贴近真实场景的 cross-docking 的开源对接基准，用 718 + 1312 条无训练泄漏的新晶体结构、23 个跨三大类的对接方法、一套精心设计的能量松弛后处理和一个实时榜单，系统证明了在 cross-docking 这个更难的现实任务上 AI 方法已经全面碾压传统物理对接软件。

## 研究背景与动机
**领域现状**：蛋白-配体对接是药物发现的核心环节——预测小分子如何与靶蛋白结合，从而筛选候选药物。近年深度学习方法（AI docking 如 DiffDock、AI co-folding 如 AlphaFold3）迅猛发展，社区也陆续推出了 PoseBuster、PoseBench、PLINDER 等基准来评测这些方法。

**现有痛点**：作者指出现有基准有三个硬伤。其一，几乎所有基准只评测 **self-docking**（把配体对接回它自己原生的共晶构象），但这在真实药物研发里几乎不存在——药化人员设计新分子，要把它对接进的是由**别的已发表化合物**共晶得到的蛋白构象。其二，PLINDER 这类基准框架很重，需要数据划分和训练才能跑，门槛高、不易用。其三，比较的模型集合太窄，PoseBuster 只比了 5 个 AI + 2 个物理方法，PLINDER 只对比了 DiffDock。

**核心矛盾**：评测设定（self-docking）与真实应用（蛋白构象会因不同配体诱导而变化）之间存在系统性错位。在 self-docking 里蛋白构象天然"适配"配体，结果偏乐观、也无法暴露方法处理蛋白柔性的真实能力。

**本文目标**：拆成几个具体子问题——(1) 引入更现实的 cross-docking 评测；(2) 构造一个无训练泄漏的高质量数据集；(3) 在统一标准下横向比较尽可能多的方法；(4) 提供能修复 AI 预测结构不合理性的后处理；(5) 提供实时可复现的公共榜单。

**核心 idea**：用 **cross-docking 这个更难、更贴近真实的任务**重新评测全谱对接方法，并配上轻量的"评测而非训练"框架 + 物理松弛后处理，得到一个结论：在现实场景下 AI 方法已经打败了物理方法。

## 方法详解

### 整体框架
PoseX 不是一个对接模型，而是一套**对接评测基准与流水线**，整体可以理解为四块：**任务设定 → 数据集构建 → 方法池评测 → 松弛后处理 + 榜单**。

任务上，PoseX 同时支持两种设定：self-docking（配体对接回原生共晶构象，验证方法能否复现已知结合姿态）和 **cross-docking**（把来自**同一蛋白的其它复合物**的小分子，对接到该蛋白除原生共晶外的所有构象上）。后者要求方法在蛋白构象并非为当前配体优化的情况下仍能给出正确姿态，更能反映真实药研。

数据上，作者从 RCSB PDB 精选 2022 年至 2025 年 1 月发布的晶体结构（晚于所有被评测 AI 方法的训练截止时间，杜绝泄漏），整理出 PoseX-SD（718 条）和 PoseX-CD（1312 条，涵盖 109 个蛋白靶点共 371 个结构、362 个小分子）。

方法上，把 23 个对接方法分成三类一并评测：5 个**物理方法**（如 Schrödinger Glide、Discovery Studio、GNINA）、11 个 **AI docking 方法**（如 DiffDock、SurfDock、UMD V2）、7 个 **AI co-folding 方法**（如 AlphaFold3、Chai-1、Boltz-1x）。每个预测再经过一套基于 OpenMM 的能量松弛后处理，并把结果汇入实时公开榜单。这是一篇 benchmark 论文，没有单一可画的训练 pipeline，因此各模块以下逐一说明。

### 关键设计

**1. Cross-docking 任务设定：把评测搬到真实药研场景**

这一设计直击"self-docking 不现实"的痛点。Self-docking 里蛋白构象就是和当前配体共晶来的，方法只需在一个已经为配体优化好的口袋里复现姿态；而真实药研中，化学家拿到的蛋白构象往往是别的化合物共晶得到的，配体进来时蛋白可能发生诱导契合（induced fit）。Cross-docking 正是模拟这种情形：对同一蛋白，取它在不同复合物中出现的所有非原生构象，把对应的小分子互相对接。这样评测就把"蛋白构象不适配"这一现实难点显式纳入，迫使方法暴露对蛋白柔性的真实处理能力。实验也证明这个设定确实更难、更有区分度（见下文物理方法在 CD 上的崩塌）。

**2. 无泄漏的高质量数据集构建：保证横向比较公平**

AI 方法大多在 PDBBind v2020（含 16,379 个复合物）上训练，若评测集与训练集重叠，结果就被"记忆"污染。作者用发布时间作为天然切分——只收 2022 年至 2025 年 1 月的新晶体结构，确保晚于所有被评 AI 方法的训练截止，从源头杜绝数据泄漏。同时整套数据处理流水线开源（修复缺失链、加端基帽、加电荷等），可复现。作者还顺带量化了泄漏风险：经典的 Astex Diverse Set（85 个复合物）里有 43 个落在 PDBBind 训练集内，正说明在旧基准上"高分"可能有记忆成分，反衬 PoseX 时间切分的必要性。

**3. 物理松弛后处理模块：用力场修复 AI 预测的结构不合理性**

AI 方法的通病是预测姿态常有分子内/分子间的原子碰撞（clash），几何上不合理。作者基于 OpenMM 实现了一套自动化能量最小化（必要时配短时分子动力学）后处理：自动修复缺失链、给 N/C 端封帽、给蛋白和小分子加形式电荷、对骨架原子（CA、C、N、O）施加约束以避免结构漂移；支持 GAFF / OpenFF 小分子力场和 Gasteiger / MMFF94 部分电荷计算。它把不合理构象大幅缓解，显著提升 PB-Valid（PoseBuster 合规性）通过率。其意义在于揭示了一条范式：**AI 建模 + 物理后处理**结合能拿到最佳性能——AI 负责把姿态放到大致正确的位置，物理松弛负责把局部几何擦干净。

**4. 双轨评测 + 口袋相似度泛化分析：解释"谁强、为什么强"**

仅给一个总成功率不足以理解方法差异，作者进一步把评测拆成 **Pocket-Given**（给定结合口袋）和 **Blind-Docking**（不给口袋）两轨分别分析，发现规律截然不同：给口袋时 SurfDock、UMD V2 这类 AI docking 方法领先；盲对接时 AlphaFold3 等 AI co-folding 方法因能同时建模蛋白柔性而胜出。成功率定义为 top-1 预测满足 RMSD $< 2\text{Å}$（或叠加 PB-Valid）的比例，CD 任务因每个靶点对接数量不均，按靶点层面平均。此外作者用**口袋相似度**（与 2022 年前晶体口袋的最大 TM-score，口袋取配体 10Å 内残基）量化泛化能力，发现多数 AI 方法的口袋相似度与配体 RMSD 呈中等负相关（如 Protenix $r=-0.390$、Chai-1 $r=-0.389$），即口袋越像见过的、姿态越准——这把"高分里有多少是泛化、有多少是记忆"摊开来看。

## 实验关键数据

### 主实验
三个基准（Astex / PoseX-SD / PoseX-CD）上、RMSD < 2Å 且 PB-Valid、带松弛的对接成功率（取三次独立运行均值）：

| 基准 | 最佳 AI 方法 | 成功率 | 代表物理方法 | 成功率 |
|------|------|------|------|------|
| Astex | UMD V2 / SurfDock | 94.1% | Glide / Discovery Studio | 约 56–67%（被超 25%+） |
| PoseX-SD | SurfDock | 78.0% | GNINA | 64.4% |
| PoseX-CD | SurfDock | 77.0% | GNINA | 54.1% |

PoseX-SD 上 SurfDock(78.0%) 居首、UMD V2 次之；AI co-folding 的 AlphaFold3(60.5%)、Protenix(56.3%) 表现良好；早期方法 EquiBind、TankBind 低于 20%。PoseX-CD 上 SurfDock(77.0%)、UMD V2(69.2%) 领先，AlphaFold3(68.6%) 紧追。

### 难度对比：cross-docking 才真正拉开 AI 与物理的差距

| 任务 | 超过领先物理方法 GNINA 的 AI 方法数 |
|------|------|
| PoseX-SD | 仅 3 个 AI docking 方法 |
| PoseX-CD | 9 个（4 个 AI docking + 5 个 AI co-folding） |

物理方法在 CD 上集体崩塌：MOE 33.3%、Glide 38.4%、Discovery Studio 43.7%，远逊于 self-docking——这正是论文标题"AI defeats physics-based methods on cross-docking"的核心证据。

### 双轨分析（PoseX-CD）

| 轨道 | 冠军 | 成功率 | 说明 |
|------|------|------|------|
| Pocket-Given | SurfDock | 77.0% | AI docking 利用显式口袋信息领先，物理方法全被超 |
| Blind-Docking | AlphaFold3 | 68.8% | AI co-folding 不依赖口袋、靠建模蛋白柔性胜出 |

### 关键发现
- **AI 全面领先**：无论 self/cross-docking，最新 AI docking 和 AI co-folding 都稳超物理方法，cross-docking 上差距最大。
- **松弛贡献关键**：力场能量最小化能大幅缓解 AI 方法的原子碰撞，是真实应用拿高分的必要步骤；指向"AI 建模 + 物理后处理"的最优组合。
- **手性问题**：多数 AI co-folding（AlphaFold3、Chai-1）存在配体手性错误，唯独 Boltz-1x 用推理时物理启发势能修复幻觉，立体化学建模显著提升结构合理性。
- **口袋信息重要**：DiffDock-Pocket 在 SD/CD 上都稳超无口袋的 DiffDock，说明显式口袋建模值得用好，对 AI co-folding 尤甚。

## 亮点与洞察
- **用"时间切分"反制数据泄漏**：只取 AI 训练截止之后发布的新晶体结构，这个看似朴素的做法是公平横评的前提；作者还实测 Astex 里 43/85 落在 PDBBind 训练集，把旧基准的记忆风险摆上台面。
- **cross-docking 是真正的区分器**：物理方法从 self 到 cross 的断崖式下滑，比任何单点 SOTA 数字都更能说明"现实任务里 AI 已胜出"，选题本身就是洞察。
- **"AI 定位 + 物理擦净"范式可迁移**：把神经网络的全局姿态预测和力场的局部几何优化解耦，这套思路对任何结构生成任务（蛋白结构、复合物、晶体）都有借鉴价值。
- **双轨 + 口袋相似度分析**：不止给排名，还把"给不给口袋""见没见过类似口袋"两个混杂变量拆开，让读者看清每个方法的真实能力边界。

## 局限与展望
- 作为 benchmark，结论受限于所选 23 个方法和数据时间窗（2022–2025.1），未来新方法/新泄漏边界需要持续更新榜单维护。
- 成功率主以 RMSD < 2Å + PB-Valid 衡量，偏重几何姿态与结构合理性，未直接评测结合亲和力预测等下游更关心的指标。
- cross-docking 仍依赖"同一蛋白的非原生构象"这一定义，对完全无已知构象的全新靶点（true apo / 大幅构象变化）覆盖有限。
- 手性、碰撞等问题揭示了 AI co-folding 的系统短板，作者建议把 Boltz-1x 式的物理启发约束、co-folding 式柔性建模引入更多方法，是明确的后续方向。

## 相关工作与启发
- **vs PoseBuster / PoseBench**：它们只做 self-docking 评测、松弛粗糙或缺失、方法数少（各 7 个）；PoseX 增加 cross-docking、精设松弛、扩到 23 个方法并开源数据流水线。
- **vs PLINDER**：PLINDER 框架重、需数据划分与训练、只对比 DiffDock 一个模型；PoseX 走"只评测不训练"的轻量路线，门槛更低、覆盖更广，并提供实时榜单。
- **启发**：评测设定的现实性（cross- vs self-）往往比刷单点 SOTA 更能推动领域认知；以及神经网络与物理先验互补（建模归 AI、修几何归力场）可能是结构预测的稳健范式。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统化 cross-docking 大规模横评 + 时间切分防泄漏，benchmark 创新扎实。
- 实验充分度: ⭐⭐⭐⭐⭐ 23 方法 × 3 基准 × 双轨 × 松弛对照 × 三次独立运行，覆盖极全面。
- 写作质量: ⭐⭐⭐⭐ 问题—方案—证据链条清晰，结论提炼到位。
- 价值: ⭐⭐⭐⭐⭐ 开源数据 + 实时榜单 + 明确范式启示，对药物发现社区实用价值高。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] HeurekaBench: A Benchmarking Framework for AI Co-scientist](heurekabench_a_benchmarking_framework_for_ai_co-scientist.md)
- [\[ICLR 2026\] SigmaDock: Untwisting Molecular Docking with Fragment-Based SE(3) Diffusion](sigmadock_untwisting_molecular_docking_with_fragment-based_se3_diffusion.md)
- [\[ICLR 2026\] Pallatom-Ligand: an All-Atom Diffusion Model for Designing Ligand-Binding Proteins](pallatom-ligand_an_all-atom_diffusion_model_for_designing_ligand-binding_protein.md)
- [\[ICLR 2026\] SAIR: Enabling Deep Learning for Protein-Ligand Interactions with a Synthetic Structural Dataset](sair_enabling_deep_learning_for_protein-ligand_interactions_with_a_synthetic_str.md)
- [\[ICML 2026\] Protein Circuit Tracing via Cross-layer Transcoders](../../ICML2026/computational_biology/protein_circuit_tracing_via_cross-layer_transcoders.md)

</div>

<!-- RELATED:END -->
