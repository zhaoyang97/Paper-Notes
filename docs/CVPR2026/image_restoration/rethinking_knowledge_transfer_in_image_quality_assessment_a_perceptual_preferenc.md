---
title: >-
  [论文解读] Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective
description: >-
  [CVPR 2026][图像恢复][图像质量评估] 作者把 IQA 跨数据集迁移失败的根因归结为「感知偏好结构」错配（即不同数据集的条件分布 $P(Y|X)$ 不同），提出用特征-分数相关向量 PPR 量化这种偏好、用余弦相似度 PPC 衡量数据集间兼容性，再用贪心剔除策略 PreSTA 只挑出与目标域偏好一致的源样本——仅用 20% 源数据就反超全量基线。
tags:
  - "CVPR 2026"
  - "图像恢复"
  - "图像质量评估"
  - "知识迁移"
  - "感知偏好结构"
  - "条件分布对齐"
  - "数据高效"
---

# Rethinking Knowledge Transfer in Image Quality Assessment: A Perceptual Preference Structure Alignment Perspective

**会议**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Li_Rethinking_Knowledge_Transfer_in_Image_Quality_Assessment_A_Perceptual_Preference_CVPR_2026_paper.html)  
**代码**: https://github.com/Li-aobo/PreSTA  
**领域**: 图像质量评估 / 知识迁移 / 低层视觉  
**关键词**: 图像质量评估, 知识迁移, 感知偏好结构, 条件分布对齐, 数据高效

## 一句话总结
作者把 IQA 跨数据集迁移失败的根因归结为「感知偏好结构」错配（即不同数据集的条件分布 $P(Y|X)$ 不同），提出用特征-分数相关向量 PPR 量化这种偏好、用余弦相似度 PPC 衡量数据集间兼容性，再用贪心剔除策略 PreSTA 只挑出与目标域偏好一致的源样本——仅用 20% 源数据就反超全量基线。

## 研究背景与动机

**领域现状**：图像质量评估（IQA）想让算法的打分和人眼主观感受对齐，深度 BIQA 方法（meta-learning、超网络、Transformer、大预训练视觉模型）在单数据集上已经很强。但每出现一种新成像场景（新设备、新失真、新内容），重新采集主观标注的成本极高，于是「如何把已有标注数据集里的感知知识迁移到新场景」成了关键方向。

**现有痛点**：迁移在实践中异常困难。直接把源数据集训练的模型搬到目标数据集（cross-domain 如合成失真→真实失真，甚至 within-domain 如合成→合成），SRCC 都会大幅掉点；而把多个数据集联合训练，虽然整体鲁棒性提升，却很少给特定目标带来稳定收益，有时增加数据多样性反而**损害**目标性能。

**核心矛盾**：现有迁移方法几乎都在对齐**边缘分布**——特征对齐 / 域选择处理 $P(X)$，learn-to-rank / 重缩放处理 $P(Y)$。它们隐含假设条件分布 $P(Y|X)$ 在不同域间足够稳定。但 IQA 里这个假设不成立：人判断质量时注意力随情境变化（判模糊看高频细节、判噪声看平滑区、人像看脸部清晰度、文档看文字可读性），这些「感知线索及其相对重要性」构成了**感知偏好结构**，会随场景系统性变化。论文用 Grad-CAM 给出直接证据：在不同数据集上训练的模型对**同一张图**的注意力模式截然不同，说明它们学到的 $P(Y|X)$ 本质不同。

**本文目标**：（1）找到一种**训练无关、可解释**的方式量化每个数据集的感知偏好结构并衡量两个数据集是否兼容；（2）据此挑出真正与目标域偏好对齐的源样本，实现鲁棒且数据高效的迁移。

**切入角度**：既然问题出在 $P(Y|X)$ 而非 $P(X)/P(Y)$，那就不该再去对齐边缘分布或盲目堆数据，而应直接刻画「特征如何映射到质量分」这条映射本身，并在**样本级**上做对齐。

**核心 idea**：用「视觉特征各维与质量分的相关系数向量」当作数据集的感知偏好指纹（PPR），用它们的余弦相似度（PPC）当迁移兼容性的训练无关指标，再用贪心剔除挑出让源域 PPR 最贴近目标域 PPR 的子集——**对齐偏好结构比扩大数据规模更重要**。

## 方法详解

### 整体框架
PreSTA 的输入是一个源数据集 $D_s=\{(x_i,y_i)\}$（图像 + 主观质量分）和一个目标数据集 $D_t$，输出是一个子集 $D_s'\subseteq D_s$，使得只在 $D_s'$ 上训练的 IQA 模型能很好地泛化到 $D_t$。整条流水线是：先用 ImageNet 预训练 backbone 抽层次化感知特征，把每个数据集压成一个「感知偏好向量」PPR；再用余弦相似度算出源-目标的偏好一致性 PPC；然后以「最大化所选子集 PPR 与目标 PPR 的 PPC」为目标，跑一个贪心剔除算法逐样本删掉拖后腿的源样本；最后用保留下来的偏好对齐子集训练常规 IQA 回归模型。整个 PPR/PPC 计算与样本选择**完全不需要训练**，只在最后才训模型。

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["源/目标数据集<br/>(图像 + 质量分)"] --> B["感知偏好表示 PPR<br/>特征各维与质量分的相关向量"]
    B --> C["感知偏好一致性 PPC<br/>源·目标 PPR 余弦相似度"]
    C --> D["偏好结构对齐的样本选择<br/>贪心剔除最大化 PPC"]
    D -->|增量统计 O(Nd)→O(d)| D
    D --> E["偏好对齐子集 D_s'"]
    E -->|PreSTA-S 跨/同域 · PreSTA-J 联合| F["训练 IQA 回归模型"]
```

### 关键设计

**1. 感知偏好表示 PPR：把「特征如何决定质量分」压成一个相关向量**

痛点直说：要对齐 $P(Y|X)$，首先得有办法把这个「条件映射」量化出来，否则无从比较两个数据集是否兼容。PPR 的做法是，对数据集里所有样本抽取感知特征 $F\in\mathbb{R}^{N\times d}$ 和质量分 $y\in\mathbb{R}^{N}$，逐特征维计算它与质量分的皮尔逊相关系数，拼成一个 $d$ 维向量：

$$r=[\rho_1,\rho_2,\dots,\rho_d],\qquad \rho_k=\frac{\mathrm{Cov}(f_k,y)}{\sigma_{f_k}\cdot\sigma_y}$$

其中 $f_k$ 是 $F$ 的第 $k$ 列（所有样本在第 $k$ 个特征维上的取值）。$\rho_k$ 衡量「这个感知线索在该数据集里对质量判断有多重要」，整条向量就刻画了「观察者在这个场景下如何给不同感知信号加权」。特征侧用 ImageNet 预训练的 ResNet-50（layer1–layer4）或 Swin-B（stage1–stage4），各层特征图全局平均池化后拼成统一向量——这样从低层结构/纹理到高层语义的线索都被覆盖，且因为用的是冻结的通用 backbone，构成了一个**域无关**的特征空间，使不同数据集的 PPR 能在同一坐标系下直接比较。之所以有效，是因为它绕开了「训练一个模型才能知道偏好」的循环，直接用相关性统计读出偏好指纹，天然可解释（每一维对应一个具体感知通道的权重）。

**2. 感知偏好一致性 PPC：训练无关的源选择标尺**

有了 PPR，判断两个数据集兼不兼容就变成比较两个向量的方向。PPC 直接取源、目标 PPR 的余弦相似度：

$$\mathrm{PPC}(r_s,r_t)=\frac{r_s\cdot r_t}{\lVert r_s\rVert_2\,\lVert r_t\rVert_2}$$

PPC 高意味着两个数据集在「特征如何关联到质量分」上高度一致，经验上也对应更好的跨数据集迁移。要强调的是，作者明确 PPC **不是用来预测迁移的绝对 SRCC**——绝对分还受目标数据集难度、失真构成等因素影响；它衡量的是**相对兼容性**，即给定目标域、在多个候选源里挑哪个更可能迁得好。实验中 12 组（6 目标 × 2 backbone）对比里有 11 组「PPC 更高的源 → SRCC 更高」，验证了它作为训练无关源选择准则的可靠性。它的价值在于：不用真训一遍迁移就能事先筛掉不兼容的源，省掉大量试错。

**3. 偏好结构对齐的贪心样本选择：在样本级把源 PPR 拉向目标 PPR**

数据集级的 PPC 只能选「用哪个源」，但同一个源里不同样本对偏好的贡献也不同。这一步把源选择写成优化问题：找子集 $D_s'\subseteq D_s$ 使 $\max_{D_s'}\mathrm{PPC}(r_{s'},r_t)$。直接搜子集是组合爆炸，作者用**贪心剔除**：从全量源开始，每轮对每个剩余样本 $i$ 试算「删掉它之后的源 PPR 与目标 PPR 的 PPC」，选出删掉后最能提升对齐的样本 $i^*=\arg\max_i \mathrm{PPC}(r_s^{(-i)},r_t)$ 删除，重复直到触发停止条件。三个停止条件共同平衡「对齐质量」和「数据多样性」：最小保留比例 $\alpha_{\min}$（防过度剪枝，强制 $|D_s'|/|D_s|\ge\alpha_{\min}$，固定 20%）、充分相似阈值 $\tau_{\mathrm{sim}}$（PPC 够高就停，跨/同域取 0.9、联合迁移取 0.95）、最小提升阈值 $\epsilon$（增益可忽略就停，$\mathrm{PPC}_{i^*}\le\mathrm{PPC}_{\text{current}}+\epsilon$，取 $10^{-6}$）。它有效是因为：删掉那些「让源偏好偏离目标」的样本，等价于把训练集的感知判断结构主动校准到目标域，比对齐 $P(X)$ 更直击迁移失败的根因。对 targeted joint 设置，则对每个辅助源**独立**跑一遍选择，再把选出的子集与目标数据并起来 $D_{\text{enhanced}}=D_t\cup\bigcup_k D_s^{(k)'}$，避免不兼容源带来的负迁移。

**4. 增量统计更新：把贪心选择的代价从 $O(Nd)$ 压到 $O(d)$**

贪心剔除每轮要对每个候选样本试算一次删除后的 PPR，朴素实现要在当前子集上重算整套相关统计，代价高到不可用。作者维护均值 $\mu_f,\mu_y$、方差 $\sigma_f^2,\sigma_y^2$ 和协方差 $\mathrm{Cov}(f,y)$，当删掉第 $i$ 个样本（特征值 $f^{(i)}$、分数 $y^{(i)}$）时用闭式增量公式精确刷新：

$$\mu_f'=\frac{N\mu_f-f^{(i)}}{N-1},\qquad (\sigma_f')^2=\frac{N\sigma_f^2-(f^{(i)}-\mu_f)(f^{(i)}-\mu_f')}{N-1}$$

$$\mathrm{Cov}'(f,y)=\frac{N\,\mathrm{Cov}(f,y)-(f^{(i)}-\mu_f)(y^{(i)}-\mu_y')}{N-1}$$

每次删除后用更新值覆盖维护的统计量，再据此刷新各维相关系数、重算 PPR 与 PPC。这样评估单次「假设删除」的成本从 $O(Nd)$ 降到 $O(d)$，让贪心选择能在大规模数据集（如 KADID-10k / KonIQ-10k）上跑得动。这是把方法从「理论可行」变成「工程可用」的关键一步。

### 损失函数 / 训练策略
PPR/PPC 与样本选择全程无需训练。最后的 IQA 模型用 backbone（ResNet-50 / Swin-B）接一个回归头预测质量分，用 L1 损失优化；单数据集训练共享一个回归头，多数据集联合训练时每个数据集配一个独立回归头。训练用 Adam，学习率 $2\times10^{-5}$、权重衰减 $5\times10^{-4}$、batch 32、32 epoch，随机裁剪 $224\times224$ patch + 随机水平翻转；测试时每图取 5 个 patch 预测求平均。联合迁移设置按 80/20 划分训练/测试、重复 10 次取 SRCC/PLCC 中位数。

## 实验关键数据

数据集覆盖合成失真（LIVE、CSIQ、TID2013、KADID-10k）与真实失真（LIVEC、KonIQ-10k、BID、SPAQ），指标为 SRCC 和 PLCC（越高越好）。

### 主实验：PreSTA-S 跨域 / 同域迁移（Swin-B）

| 设置 | 源→目标 | Baseline SRCC | PreSTA-S SRCC | 用数据量 |
|------|---------|---------------|---------------|----------|
| 跨域 合成→真实 | KADID-10k→LIVEC | 0.589 | **0.744** | 20% |
| 跨域 合成→真实 | KADID-10k→BID | 0.739 | **0.833** | 20% |
| 跨域 合成→真实 | KADID-10k→KonIQ-10k | 0.682 | **0.774** | 20% |
| 跨域 真实→合成 | KonIQ-10k→LIVE | 0.812 | **0.849** | 20% |
| 跨域 真实→合成 | KonIQ-10k→TID2013 | 0.452 | **0.516** | 20% |
| 同域 合成→合成 | KADID-10k→CSIQ | 0.773 | **0.840** | 27.9% |
| 同域 真实→真实 | KonIQ-10k→SPAQ | 0.865 | **0.880** | 20% |

跨域（感知差距最大）提升最显著，且只用 20% 源数据就反超 100% 数据的基线；同域差距较小，PreSTA-S 在维持或略升性能的同时大幅提升数据效率（CSIQ/TID2013 只用 28%/38% 数据）。

### PreSTA-J 联合迁移（Table 3）

| 方法 | LIVEC SRCC/PLCC | 额外样本 | BID SRCC/PLCC | 额外样本 |
|------|-----------------|----------|----------------|----------|
| Baseline（仅目标） | 0.883 / 0.909 | 0 | 0.868 / 0.897 | 0 |
| + 跨域 KADID-10k | 0.876 / 0.902 | 10,125 | 0.858 / 0.886 | 10,125 |
| + 全量同域联合 | 0.899 / 0.915 | 21,198 | 0.883 / 0.908 | 21,198 |
| **PreSTA-J** | **0.905 / 0.919** | **2,793** | **0.885 / 0.898** | **5,343** |

PreSTA-J 在 LIVEC 上用仅 2.8k 额外样本就超过用 21k 样本的全量同域联合训练；而盲目加入不兼容的跨域 KADID-10k 反而把 LIVEC 从 0.883 拉低到 0.876（负迁移），印证「结构兼容比数据规模重要」。

### 关键发现
- **PPC 是有效源选择准则**：12 组 backbone×目标对比中 11 组「PPC 高 → SRCC 高」；唯一例外是 Swin-B 在 SPAQ 上，作者归因于目标特异因素。说明 PPC 抓住了相对兼容性的主信号。
- **对齐 > 堆数据**：PreSTA-J 加入不兼容源会触发负迁移（LIVEC 0.883→0.876），而选择性加入偏好对齐样本即便量小也稳定增益，直接支撑核心论点。
- **数据效率惊人**：跨/同域只用 20%~38% 源数据即达到或超过全量基线，揭示现有数据集里被错配样本「拖累」的未挖掘潜力。
- **可视化佐证**：把 KADID-10k 向 LIVEC 对齐后，其逐通道 PPR 分布显著贴近目标域，Grad-CAM 注意力也随之迁移——说明对齐不止改了统计分布，更重塑了模型的内部决策焦点。

## 亮点与洞察
- **问题诊断角度新**：把 IQA 迁移失败明确归到 $P(Y|X)$（感知偏好结构）而非大家长期纠结的 $P(X)/P(Y)$，并用 Grad-CAM「同图不同注意力」给出直观且有说服力的证据，是全文最「啊哈」之处。
- **训练无关的偏好指纹**：PPR 用「特征各维 vs 质量分的相关向量」把抽象的条件分布压成一个可比向量，PPC 用余弦相似度一步给出兼容性——既可解释又零训练成本，这套「相关向量当指纹」的思路可迁移到任何「输入-标量分」的域选择问题（如美学评分、视频质量、可迁移性预估）。
- **增量统计是落地关键**：把贪心剔除单步代价从 $O(Nd)$ 降到 $O(d)$ 的闭式更新，是让组合优化在万级数据集上可跑的工程巧思，值得在其他「逐样本删除/增加」的选择算法里复用。
- **数据效率的现实价值**：20% 数据反超全量，对标注昂贵的 IQA 极具吸引力——等于说现有数据集大量样本不仅没用还有害，挑对子集即可。

## 局限与展望
- **依赖目标域 PPR 估计**：PPR 需要目标域的图像+质量分来算相关，全新无标注场景下如何估计目标 PPR 是隐忧。作者在补充材料讨论了「少量标注子集估计目标 PPR」的稳定性，但正文未充分展开，实际部署时这一步的鲁棒性存疑。
- **PPR 只是线性相关**：用皮尔逊相关刻画「特征维→质量分」假设了线性单调关系，对感知中常见的非线性/交互效应可能刻画不足；偏好结构是否真能被一个 $d$ 维相关向量完整表达，是更深的开放问题。
- **贪心 + 阈值依赖**：贪心剔除是启发式、不保证全局最优；$\alpha_{\min}/\tau_{\mathrm{sim}}/\epsilon$ 三个阈值需按设置手调（如 τ 在联合迁移用 0.95），跨场景的自适应设定还需探索。
- **backbone 选择影响**：PPR 建立在冻结的 ImageNet 特征上，特征空间本身的偏置会传入偏好向量；SPAQ 上的反例提示当目标特异因素较强时 PPC 失准，方法上限受特征质量约束。

## 相关工作与启发
- **vs 域适应（FreqAlign / DGQA / 特征风格对齐）**：它们对齐边缘特征分布 $P(X)$（频域对齐、域选择、风格空间对齐），本文指出只对齐 $P(X)$ 无法解决 $P(Y|X)$ 错配，转而在样本级对齐条件映射，直击「同图不同注意力」的语义鸿沟。
- **vs 联合训练（UNIQUE / StairIQA / LIQE / Q-Align）**：它们用 pairwise ranking、混库迭代、多任务或大模型文本质量等级来调和多数据集的 $P(Y)$，但盲目联合常无稳定增益甚至负迁移；本文用 PPC 先筛兼容样本再联合，2.8k 样本即超 21k 的全量联合。
- **方法论启发**：「用训练无关的相关指纹做源/样本选择」是一种轻量的可迁移性度量范式，可迁到任何标注昂贵、需要从多源里挑兼容数据的回归/打分任务。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 把 IQA 迁移失败重新归因到 $P(Y|X)$ 并给出训练无关的量化-选择闭环，视角和工具都新。
- 实验充分度: ⭐⭐⭐⭐ 覆盖 8 数据集、跨域/同域/联合三设置且有可视化与消融性分析，但目标 PPR 在无标注场景的鲁棒性主要靠补充材料。
- 写作质量: ⭐⭐⭐⭐⭐ 从矛盾→证据→方法→实验逻辑清晰，公式与算法交代完整，Grad-CAM 证据有说服力。
- 价值: ⭐⭐⭐⭐⭐ 20% 数据反超全量、可解释、即插即用的源选择对标注昂贵的 IQA 实用价值很高，思路可外溢到其他打分任务。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] DPGF-Net: Dual-Prior Guided Fusion Network for Joint Assessment of Perceptual Quality and Semantic Consistency in AI-Generated Images](dpgf-net_dual-prior_guided_fusion_network_for_joint_assessment_of_perceptual_qua.md)
- [\[CVPR 2026\] RL-ScanIQA: Reinforcement-Learned Scanpaths for Blind 360deg Image Quality Assessment](rl-scaniqa_reinforcement-learned_scanpaths_for_blind_360deg_image_quality_assess.md)
- [\[CVPR 2026\] Life-IQA: Boosting Blind Image Quality Assessment through GCN-enhanced Layer Interaction and MoE-based Feature Decoupling](life-iqa_boosting_blind_image_quality_assessment_through_gcn-enhanced_layer_inte.md)
- [\[CVPR 2025\] Augmenting Perceptual Super-Resolution via Image Quality Predictors](../../CVPR2025/image_restoration/augmenting_perceptual_super-resolution_via_image_quality_predictors.md)
- [\[AAAI 2026\] Temporal Inconsistency Guidance for Super-resolution Video Quality Assessment](../../AAAI2026/image_restoration/temporal_inconsistency_guidance_for_super-resolution_video_quality_assessment.md)

</div>

<!-- RELATED:END -->
