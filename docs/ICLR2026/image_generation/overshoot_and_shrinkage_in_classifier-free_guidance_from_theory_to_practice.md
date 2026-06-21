---
title: >-
  [论文解读] Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice
description: >-
  [ICLR 2026][图像生成][Classifier-Free Guidance] 本文用统计物理的"动力学相变"框架重新分析 Classifier-Free Guidance（CFG），证明在足够高的维度下 CFG 其实能精确还原目标分布（"维度的祝福"），并精确刻画了低维下出现的均值过冲与方差收缩，进而提出把分数差做非线性幂律放大的 Power-Law CFG，在理论上同时缓解这两种伪影、在 DiT/EDM2/文生图等 SOTA 模型上一致提升画质与多样性。
tags:
  - "ICLR 2026"
  - "图像生成"
  - "Classifier-Free Guidance"
  - "扩散模型"
  - "高维统计物理"
  - "均值过冲"
  - "方差收缩"
---

# Overshoot and Shrinkage in Classifier-Free Guidance: From Theory to Practice

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=cNsoOr1hTH](https://openreview.net/forum?id=cNsoOr1hTH)  
**代码**: 无  
**领域**: 扩散模型 / 图像生成  
**关键词**: Classifier-Free Guidance, 扩散模型, 高维统计物理, 均值过冲, 方差收缩

## 一句话总结
本文用统计物理的"动力学相变"框架重新分析 Classifier-Free Guidance（CFG），证明在足够高的维度下 CFG 其实能精确还原目标分布（"维度的祝福"），并精确刻画了低维下出现的均值过冲与方差收缩，进而提出把分数差做非线性幂律放大的 Power-Law CFG，在理论上同时缓解这两种伪影、在 DiT/EDM2/文生图等 SOTA 模型上一致提升画质与多样性。

## 研究背景与动机
**领域现状**：扩散模型和流匹配是当下高维信号生成的事实标准，而条件生成几乎都依赖 CFG。CFG 不需要额外分类器，只要模型同时学会有条件和无条件去噪，推理时沿"有条件方向"外推即可：$S_t^{\text{CFG}}(\vec{x},c)=S_t(\vec{x},c)+\omega\big(S_t(\vec{x},c)-S_t(\vec{x})\big)$，其中 $\omega>0$ 是引导强度。

**现有痛点**：CFG 一旦 $\omega>0$，采样的就不再是真正的条件分布。实践和此前的理论（Chidambaram et al. 2024；Wu et al. 2024）都观察到两个伪影：**均值过冲**（样本被推向类别边界、过饱和、过对比）和**方差收缩**（分布比目标更尖锐、多样性下降）。换句话说 CFG 是"画质换多样性"的 trade-off。

**核心矛盾**：此前对 CFG 的理论分析几乎都局限在一维或有限维的高斯混合（GMM）上，得出的结论是"CFG 必然扭曲目标分布"。但这与实践经验相悖——CFG 几乎总是有益的。于是有三个悬而未决的问题：CFG 究竟能不能生成正确分布？过冲与收缩到底由什么决定？能否设计出可证明地缓解伪影、又保留 CFG 收益的新引导方式？

**切入角度**：作者借用统计物理里**扩散动力学相变**（Biroli & Mézard 2023；Biroli et al. 2024）的框架——反向扩散过程随时间会经历若干"相区"，存在一个**物种分化时间（speciation time）** $t_s$，在它之前轨迹尚未"决定"属于哪一类，之后类别已定。把 CFG 放进这个 $d\to\infty$ 的框架里分析，正是此前研究没做过的角度。

**核心 idea**：在高维下 CFG 只在 $t_s$ 之前起加速作用、之后自动失效，因此**渐近还原正确分布**；伪影只是有限维修正（量级 $1/\sqrt{d}$）。顺着这个理解，把分数差做**非线性幂律放大**就能压制有限维伪影，同时不破坏高维保证。

## 方法详解

### 整体框架
本文不是提出一个新的生成网络，而是一条"从理论到实践"的链条：先用高维统计物理框架**解释 CFG 为什么/何时正确**，再**定位低维伪影的来源**，最后据此**设计一个极简的非线性引导改进**。

具体地，作者把数据建模成两个等权、各向同性方差 $\sigma^2$、均值为 $\pm\vec{m}$ 的高斯混合（取 $|\vec{m}|=\sqrt{d}$ 保证两类可分），前向是 Ornstein-Uhlenbeck 过程 $d\vec{x}(t)=-\vec{x}(t)\,dt+\sqrt{2}\,d\vec{B}(t)$，反向由分数 $S_t(\vec{x})=\nabla\log P_t(\vec{x})$ 驱动。关键观察是 CFG 只作用在 $\vec{m}$ 方向上（所有与 $\vec{m}$ 正交的方向都与 $\omega$ 无关），于是可以把高维动力学投影到标量 $q(t)=\vec{x}\cdot\vec{m}/|\vec{m}|$ 上，化成一个一维有效势 $V^{\text{CFG}}(q,\tau)$ 的 Langevin 过程来分析。沿着分化时间 $t_s=\tfrac{1}{2}\log d$ 把反向过程切成"分化前 / 分化时刻 / 分化后"三段，逐段证明 CFG 的作用，得到三条主结果，再把第三条落地成 Power-Law CFG。

整篇方法是纯机制分析 + 一个 loss/分数项的改写，没有多阶段 pipeline，因此不配框架图，用公式把每段的力学含义讲清即可。

### 关键设计

**1. 把 CFG 嵌入分化相变框架，证明"维度的祝福"**

针对"CFG 到底能不能生成正确分布"这个根本疑问，作者沿分化时间 $t_s=\tfrac{1}{2}\log d$ 把反向过程拆成三步逐段论证（投影到 $q$ 后的有效势见下式）：

$$V^{\text{CFG}}=\underbrace{\tfrac{1}{2}q^2-2e^{-(t-t_s)}q}_{\text{条件势}}+\omega\underbrace{\big[-qe^{-(t-t_s)}+\ln\cosh\big(qe^{-(t-t_s)}\big)\big]}_{\text{CFG 诱导势}}.$$

**Step I（分化前）**：CFG 诱导势沿 $\vec{m}$ 方向额外加一个"推力"，把那些本来会偏向错误类别的轨迹纠正过来，**加速向目标类收敛**。**Step II（临近 $t_s$）**：当 $q$ 涨到 $\mathcal{O}(\sqrt{d})$ 量级时，CFG 附加项只剩指数级小的修正，早先因不同 $\omega$ 造成的位置差异被"遗忘"，引导轨迹与无引导轨迹**重新对齐**。**Step III（分化后，Regime II）**：由于 $1-\tanh(\vec{x}\cdot\vec{m}e^{-t}/\Gamma_t)\to0$，CFG 项有效消失，轨迹完全跟随无引导的条件演化。三步合起来说明：在无穷维和足够高维下，无论 $\omega$ 取多大，CFG 都还原正确的目标分布。这与此前"CFG 必然扭曲分布"的结论相反，是本文最反直觉的贡献，也是后续设计的指导原则。直观图景是：维度越高，每个类别像磁铁一样的"吸引力"随 $d$ 指数增长，远超固定的 CFG 项，于是 CFG 在后期被自然吸引力淹没、不再扭曲分布。

**2. 精确刻画有限维下的均值过冲与方差收缩**

针对"伪影从哪来"，作者证明它们本质是**有限维修正**而非 CFG 的固有缺陷。在有限维下，退出 Regime I 时轨迹**不再完全对齐**：Regime I 里 CFG 多加的推力会渗透到 Regime II，造成对目标分布的过冲，其相对幅度为 $\mathcal{O}(1/\sqrt{d})$——维度越低过冲越明显。同时 CFG 附加项让有效势 $V^{\text{CFG}}(q,t)$ 的二阶导（曲率）更大，势更"陡峭、更收束"，从而**收缩生成分布的方差**。这把实践中观察到的"过饱和 + 多样性下降"对应到了可量化的力学量上，并解释了为什么这些伪影随维度下降而加剧。

**3. Power-Law CFG：把分数差做非线性幂律放大**

针对"如何在保留收益的前提下压制伪影"，作者提出最小改动方案——把沿 $\vec{m}$ 的条件分数差自乘 $\alpha>0$ 次幂：

$$\vec{S}_t^{\text{PL}}(\vec{x},c)=\vec{S}_t(\vec{x},c)+\omega\big[\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\big]\,\big\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\big\|^{\alpha}.$$

它有两个互补效果：当分数差 $\delta S_t=\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\|$ 很小（信号弱、可能不可靠）时，引导被自然抑制；当信号强时引导被放大，加强向正确类的推动。作者在 App. E 把过冲与收缩的来源归结为曲率相关的项 $B(q):=de^{-(t_f-t)}/\Gamma_u(q)$（几乎处处 $0<B(q)<1$），并证明幂律通过 $B(q)^\alpha$ 直接调制它：$\alpha>0$ 抑制过冲、减小曲率增量（减弱收缩），$\alpha<0$ 反而加剧两者。关键是这一改动**不破坏高维保证**——在大维度极限下幂律 CFG 仍还原正确条件分布，非线性收益被限制在有限维区间。对流匹配也能直接套用（为与理论完全一致写成 $\phi_t(s)=(\tfrac{1-t}{t})^\alpha s^\alpha$，但 $\phi_t(s)=s^\alpha$ 效果几乎相同）。这里用欧氏范数同时捕捉分数差的方向与幅度，$\omega$ 充当重整化因子，省去了额外的分辨率调整。

**4. 统一的非线性引导族，把已有方法纳为特例**

作者进一步指出 Power-Law 只是一类更广的合法非线性引导中的一个，统一写成

$$S_t^{\text{CFG-NL}}(\vec{x},c)=S_t(\vec{x},c)+\big(S_t(\vec{x},c)-S_t(\vec{x})\big)\,\phi_t\big(\|\vec{S}_t(\vec{x},c)-\vec{S}_t(\vec{x})\|\big),$$

只要满足 $\lim_{s\to0}s\,\phi_t(s)=0$（保证条件/无条件分数趋同时引导平滑消失，避免病态引导）。常数 $\phi_t(s)=\omega$ 即标准 CFG；$\phi_t(s)=\omega\cdot\mathbb{I}_{[t_1,t_2)}(t)$ 是 limited-interval CFG；时变 $\phi_t(s)=\omega_t$ 恢复权重调度器。这些已有方法都**对分数差保持线性**，而 Power-Law 的关键创新正是首次以"非线性改写分数差"的方式，既理论自洽又实践有效，从而打开一个可被直接优化的更大设计空间。

### 损失函数 / 训练策略
本文不改训练目标，方法纯粹作用在推理时的分数/引导项上，因此 Power-Law CFG 可即插即用，只需调一个额外超参 $\alpha$；实验发现潜空间下固定 $\alpha=0.9$ 即在各模型上稳定有效，无需大规模搜参。

## 实验关键数据

### 主实验
在 EDM2-S、DiT/XL-2（ImageNet-1K 类别条件）以及两个文生图 MMDiT（扩散 + 流匹配）上，用 FID 衡量画质、Precision/Recall 衡量多样性，把 Power-Law 叠加到标准 CFG 及最强竞品（Limited、CADS）上：

| 模型 | 方法 | FID↓ | Precision↑ | Recall↑ |
|------|------|------|-----------|---------|
| EDM2-S (CC, IM-1K 512) | Standard CFG | 2.29 | 0.751 | 0.582 |
| EDM2-S | Power-law CFG | 1.93 | 0.780 | 0.631 |
| EDM2-S | Power-law + CADS | **1.52** | 0.770 | 0.622 |
| DiT/XL-2 (CC, IM-1K 256) | Standard CFG | 2.27 | 0.829 | 0.584 |
| DiT/XL-2 | Power-law + CADS | **1.63** | 0.754 | 0.639 |
| Diff. MMDiT (T2IM, CC12M) | Standard CFG | 8.58 | 0.661 | 0.569 |
| Diff. MMDiT | Power-law + CADS | **7.98** | 0.690 | 0.573 |
| FM MMDiT (T2IM, COCO) | Standard CFG | 5.20 | 0.629 | 0.594 |
| FM MMDiT | Power-law + CADS | **4.71** | 0.640 | 0.624 |

Power-Law 在大多数情形下同时改善画质与多样性，且与 CADS / Limited 叠加后进一步降 FID，取得各设置下最优或次优。

### 消融实验

| 配置 | 现象 | 说明 |
|------|------|------|
| $\alpha=0$ | 退化为标准 CFG | 基线 |
| $\alpha$ 增大 | FID 持续改善 | EDM2-S 512 上 $\alpha$ 越大越好 |
| $\alpha$ 增大 | 对 $\omega$ 更鲁棒 | 在更大 $\omega$ 区间 FID 仍稳定 |
| $\alpha=0.9$（潜空间） | 一致最优 | 各模型无需精调即稳定提升 |

### 关键发现
- **过冲/收缩可被 $\alpha$ 单调调控**：GMM 模拟与真实模型都呈现 $\|S_t(\vec{x},c)-S_t(\vec{x})\|^{1+\alpha}$ 的"驼峰"形状，$\alpha$ 改变曲线形状，正好提供理论指出的"缓解伪影所需的灵活度"。
- **鲁棒性是主要卖点**：增大 $\alpha$ 不仅降 FID，更让对引导强度 $\omega$ 的敏感度显著下降——标准 CFG 在 $\omega=5$ 已出现多样性塌缩，Power-Law 在 $\omega=10$ 仍稳定。
- **潜空间 vs 像素空间**：潜空间固定 $\alpha=0.9$ 即稳健；像素空间最优 $\alpha$ 波动更大，需同时调 $\alpha$ 与 $\omega$ 才有更强收益。

## 亮点与洞察
- **"维度的祝福"翻转了主流认知**：此前一维/有限维 GMM 分析都判 CFG"必然扭曲分布"，本文用相变框架证明高维下 CFG 渐近正确，把伪影定性为 $1/\sqrt{d}$ 的有限维修正——这是把统计物理工具引入 CFG 分析的漂亮范例。
- **极简改动 + 理论护栏**：Power-Law 只在分数差上乘一个 $\|\cdot\|^\alpha$，却能用 $B(q)^\alpha$ 解析地解释它如何同时压过冲、减收缩，且高维保证不丢——"改一行、可证明、可即插即用"。
- **统一框架可迁移**：把 CFG、limited-interval、权重调度器都写成 $\phi_t$ 的特例，提示一个"可被直接优化的非线性引导函数空间"，这个视角可迁移到任何基于分数差外推的引导方法上。

## 局限与展望
- 全部理论建立在**完美分数估计**假设上，因此能解释"如何缓解伪影"，却**无法解释为什么标准 CFG（带伪影）在实践中反而常常更好**——作者自己承认这是框架的根本局限，猜测实践收益与真实分数估计器的不完美有关。
- 分析以两高斯等权各向同性混合为骨架，虽给出多分量/异方差/流形数据的推广，但真实文生图的复杂分布与之差距仍大，$1/\sqrt{d}$ 等结论的定量适用性需谨慎看待。
- Power-Law 相对其他非线性策略的优劣（尤其像素空间）尚未充分探索，作者明确把"优化 $\phi_t$、研究分数近似误差对引导的影响"列为后续方向。

## 相关工作与启发
- **vs Chidambaram et al. 2024 / Wu et al. 2024**：他们在一维/多维 GMM 上证明 CFG 造成过冲与收缩并停留在"CFG 扭曲分布"的结论；本文做高维统计物理分析，证明 $d\to\infty$ 时 CFG 反而对齐目标分布，把伪影归为有限维修正——是对前作的"维度补全"。
- **vs limited-interval CFG (Kynkäänniemi et al. 2024) / 权重调度器 (Wang et al. 2024) / REG / CADS / APG / CFG++**：这些都对分数差保持线性（仅在不同时间/强度上做线性加权），本文证明它们都是统一非线性族 $\phi_t$ 的特例，而 Power-Law 首次以非线性方式改写分数差，并能与 CADS / Limited 叠加进一步提升。
- **vs 标准 CFG (Ho & Salimans 2022)**：标准 CFG 是"画质换多样性"的 trade-off；Power-Law 在保留画质的同时缓解多样性损失，并对 $\omega$ 更鲁棒。

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用相变框架翻转"CFG 必然扭曲分布"的主流认知，并给出可证明的极简改进
- 实验充分度: ⭐⭐⭐⭐ 覆盖 GMM + DiT/EDM2/两类 MMDiT，但缺开源代码、像素空间分析较弱
- 写作质量: ⭐⭐⭐⭐ 理论链条清晰、三步论证讲究，但统计物理记号对非该领域读者门槛偏高
- 价值: ⭐⭐⭐⭐⭐ 即插即用、理论自洽，对所有用 CFG 的扩散/流匹配模型都有直接借鉴价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Improving Classifier-Free Guidance in Masked Diffusion: Low-Dim Theoretical Insights with High-Dim Impact](improving_classifier-free_guidance_in_masked_diffusion_low-dim_theoretical_insig.md)
- [\[ICLR 2026\] Stage-wise Dynamics of Classifier-Free Guidance in Diffusion Models](stage-wise_dynamics_of_classifier-free_guidance_in_diffusion_models.md)
- [\[CVPR 2026\] C$^2$FG: Control Classifier-Free Guidance via Score Discrepancy Analysis](../../CVPR2026/image_generation/c2fg_control_classifier-free_guidance_via_score_discrepancy_analysis.md)
- [\[AAAI 2026\] Studying Classifier(-Free) Guidance From A Classifier-Centric Perspective](../../AAAI2026/image_generation/studying_classifier-free_guidance_from_a_classifier-centric_perspective.md)
- [\[AAAI 2026\] DICE: Distilling Classifier-Free Guidance into Text Embeddings](../../AAAI2026/image_generation/dice_distilling_classifier-free_guidance_into_text_embedding.md)

</div>

<!-- RELATED:END -->
