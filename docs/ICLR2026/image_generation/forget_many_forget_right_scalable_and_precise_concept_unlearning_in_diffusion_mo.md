---
title: >-
  [论文解读] Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models
description: >-
  [ICLR 2026][图像生成][concept unlearning] ScaPre 用一个无需训练、无需额外数据的闭式解，同时解决大规模概念遗忘中的"更新冲突"和"误伤相似概念"两大顽疾，能在 120 秒内稳定遗忘 50 个概念，比最强基线多遗忘 5 倍概念而不崩坏生成质量。 领域现状：文生图扩散模型（Stable D…
tags:
  - "ICLR 2026"
  - "图像生成"
  - "concept unlearning"
  - "扩散模型"
  - "closed-form editing"
  - "注意力机制"
  - "mutual information"
---

# Forget Many, Forget Right: Scalable and Precise Concept Unlearning in Diffusion Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=zt7IPzsXrT](https://openreview.net/forum?id=zt7IPzsXrT)  
**代码**: [https://github.com/kaiyuan02415/scapre](https://github.com/kaiyuan02415/scapre)  
**领域**: 图像生成 / 扩散模型概念遗忘  
**关键词**: concept unlearning, diffusion model, closed-form editing, cross-attention, mutual information  

## 一句话总结
ScaPre 用一个无需训练、无需额外数据的闭式解，同时解决大规模概念遗忘中的"更新冲突"和"误伤相似概念"两大顽疾，能在 120 秒内稳定遗忘 50 个概念，比最强基线多遗忘 5 倍概念而不崩坏生成质量。

## 研究背景与动机
**领域现状**：文生图扩散模型（Stable Diffusion、DALL·E、Imagen）能合成逼真图像，但也带来版权侵权、有害内容和敏感信息滥用的风险。机器遗忘（machine unlearning）应运而生，目标是从训练好的模型里精准抹掉某个物体/风格/身份，同时保留其余生成能力。单概念遗忘已较成熟，多概念遗忘（通常 10-20 个）也有 MACE（多 LoRA）、SPM（可组合适配器）、UCE/RECE（闭式编辑）等方法。

**现有痛点**：当把规模从十几个推到几十个概念时，所有现有方法集体失效，暴露三个持续性难题：(i) **更新冲突**——不同概念的权重更新互相干扰，导致部分目标删不掉，或殃及无关参数使生成质量崩塌；(ii) **缺乏精度机制**——遗忘"溢出"到背景或视觉相似的非目标概念（比如想删金毛却连带毁掉哈巴狗），可靠性差；(iii) **可扩展性瓶颈**——多数方法依赖额外数据、子模型或适配器，概念越多计算成本越爆炸。

**核心矛盾**：规模、精度、效率三者难以兼得——闭式方法（UCE/RECE）快但大规模下生成坍缩，微调方法（MACE）精度尚可但慢且要额外模块。

**本文目标**：一个统一、轻量的框架，在大规模（几十个概念）下同时做到稳定遗忘、精确隔离、高效计算，且不牺牲非目标生成质量。

**核心 idea**：**把遗忘建模为一个带"冲突感知正则"和"信息论解耦"的二次优化问题，让它退化成可一步求解的闭式解（Sylvester 方程）**——既保留闭式法的极致效率，又通过精心设计的正则项把大规模冲突和误伤压下去。

## 方法详解

### 整体框架
ScaPre（Scalable-Precise Concept Unlearning）直接编辑交叉注意力的 Key/Value 投影矩阵 $W$，整条管线分三步串联：先用**谱迹正则器**塑造一个稳定的优化空间、压制冲突方向，再用**Informax 解耦器**逐通道算出目标概念相关度、把更新限制在相关子空间里，二者合并成一个二次目标解出闭式中间解 $W^\star$（实现遗忘，红色路径），最后用**几何对齐**沿 Bures 测地线把 $W^\star$ 往预训练参考 $W_0$ 拉回一点做近端修正（保护全局结构，绿色路径）。整个过程训练无关、不需额外数据。

```mermaid
flowchart LR
    A["目标概念 C_E<br/>交叉注意力 K/V"] --> B["谱迹正则器 L_t<br/>稳定优化空间<br/>压制冲突方向"]
    B --> C["Informax 解耦器 α<br/>MI 算通道相关度<br/>限制更新到目标子空间"]
    C --> D["解 Sylvester 方程<br/>闭式中间解 W*"]
    D --> E["几何对齐 L_g<br/>Bures 近端修正<br/>拉回预训练 W_0"]
    E --> F["遗忘完成<br/>非目标质量不变"]
```

### 关键设计

**1. 谱迹正则器（Spectral Trace Regularizer）：用二阶统计动态塑造优化空间，把冲突方向"踩刹车"。** 大规模同时遗忘时，不同概念的更新会在某些共享方向上互相打架、扭曲优化地形。ScaPre 把正则项写成 $L_t(W)=\mathrm{tr}\big(W(\lambda I + S + R)W^\top\big)$。其中 $\lambda I$ 是闭式解里常规的数值稳定项；$S=\sum_k\sum_t c_{k,t}c_{k,t}^\top$ 聚合了所有目标概念上下文特征的二阶统计，它的大特征值方向正是大规模遗忘中最易产生冲突和噪声的方向，正则一压就抑制了不稳定更新；$R$ 则专门调节目标子空间内部的概念交互——把概念嵌入矩阵 $C_E$ 做 SVD 得 $C_E=U\,\mathrm{diag}(\sigma)\,V^\top$，大奇异值 $\sigma_i$ 代表多个概念强烈重叠的方向，于是用平滑门控 $\tilde\sigma_i=(1-\mathrm{sigmoid}(\sigma_i))\,\sigma_i$ 软性衰减大奇异值、几乎不动小奇异值，重构 $R=U\,\mathrm{diag}(\tilde\sigma)\,U^\top$，从而只压高冲突的重叠方向、保住独立概念的低冲突方向。

**2. 几何对齐（Geometry Alignment）：用 Bures 距离匹配协方差结构，比 ℓ2 更稳地守住预训练全局结构。** 现有方法常用 $\ell_2$ 范数惩罚权重差异，但它只逐元素地拉权重、保不住高阶特征相关性。ScaPre 改把 $W$ 的每一行看作协方差因子，定义协方差矩阵 $WW^\top$ 与预训练参考 $W_0W_0^\top$，用 Bures 距离对齐：$L_g(W)=\mathrm{tr}(WW^\top)+\mathrm{tr}(W_0W_0^\top)-2\,\mathrm{tr}\big[((WW^\top)^{1/2}W_0W_0^\top(WW^\top)^{1/2})^{1/2}\big]$。它匹配的是协方差结构而非元素差，从而保留高阶特征相关性、让无关特征在大规模遗忘中保持稳定，与谱迹正则形成"局部压冲突 + 全局防退化"的互补防线。

**3. Informax 解耦器（Informax Decoupler）：用互信息量化每个通道与目标概念的耦合度，把更新精确限制在相关参数上。** 不同权重对目标概念的贡献天差地别——有些强绑定目标，有些只支撑无关背景，一视同仁更新要么删不干净要么误伤。ScaPre 对每个通道 $i$ 把激活离散成 $z=\mathbb{1}\{a_i(s)>\tau_i\}$，输入标签 $y\in\{0,1\}$（目标概念输入为 1，中性输入为 0），从激活-标签对估出经验联合分布，算互信息 $MI_i=\sum_{z,y}p_i(z,y)\log\frac{p_i(z,y)}{p_i(z)p_i(y)}$，$MI_i$ 越大说明该通道越能预测输入是否含目标概念。多目标时取逐概念分数的最大值 $MI_i=\max_k MI_i^{(k)}$，归一化得解耦权重 $\alpha_i=MI_i/\max_j MI_j\in[0,1]$，从而把概念相关参数从无关参数中解耦出来、自适应地重加权更新。

**4. 统一闭式解（Sylvester 方程）：把三件套合成一个二次目标，一步解出来。** 令 $A=\lambda I+S+R$、$B=\mathrm{diag}(\alpha)$，整体目标为 $\min_W \mathrm{tr}(WAW^\top)+\beta L_g(W)+\mathrm{tr}(W^\top BW)-\mathrm{tr}(WV^*C_E^\top)$，其中 $V^*$ 是替换目标（完全遗忘时设为零）。先忽略非二次的几何对齐项，对 $W$ 求导置零得 Sylvester 方程 $BW+WA=V^*C_E^\top$，向量化后闭式解为 $\mathrm{vec}(W^\star)=(I_{d_{in}}\otimes B+A^\top\otimes I_{d_{out}})^{-1}\mathrm{vec}(V^*C_E^\top)$；几何对齐项因含嵌套矩阵平方根破坏二次性，单独用近端修正处理——把 $W^\star W^{\star\top}$ 沿 Bures 测地线向 $W_0W_0^\top$ 移一段得新协方差，再用正交 Procrustes 旋转映射回权重空间，既保住主遗忘方向又强化全局稳定。

## 实验关键数据

实验基于 Stable Diffusion v1.4 & v1.5，单张 RTX A6000；大规模遗忘用 Imagenette（10 类）和自建 ImageNet-Diversi50（50 类），精确遗忘用自建 ImageNet-Confuse5（5 组视觉相似概念），显式内容用 I2P，风格遗忘选 50 位艺术家，生成质量用 MS COCO-30K。

### 主实验表格（Imagenette，10 概念）

| 指标 | SD v1.5 | FMN | SPM | ESD | MACE | UCE | RECE | SP | **ScaPre** |
|---|---|---|---|---|---|---|---|---|---|
| Avg Acc (↓) | 89.9 | 71.9 | 47.4 | 38.7 | 78.5 | 8.5 | 4.9 | 9.6 | **0.8** |
| CLIPcoco (↑) | 31.43 | 30.62 | 30.81 | 30.14 | 31.02 | 29.45 | 29.27 | 29.25 | **30.43** |
| UQ (↑) | — | 37.35 | 49.89 | 47.84 | 35.07 | 37.23 | 32.60 | 31.78 | **64.09** |

ImageNet-Diversi50（50 概念）上 ScaPre 的 UQ 达 65.30，遥遥领先（ESD 56.35、SP 51.28），而 UCE/RECE 已生成坍缩（CLIP 跌到 22 左右、UQ 仅 22-25）。

### 消融 / 精确遗忘表格（ImageNet-Confuse5）

| 指标 | SD v1.5 | ESD | MACE | UCE | RECE | SP | **ScaPre** |
|---|---|---|---|---|---|---|---|
| Unlearn Acc (↓) | 83.9 | 55.6 | 76.4 | 2.9 | 3.1 | 55.0 | **5.8** |
| Preserve Acc (↑) | 86.6 | 57.7 | 78.6 | 5.6 | 5.5 | 57.1 | **76.3** |
| Overall Acc (↑) | 27.2 | 50.2 | 36.3 | 10.6 | 10.4 | 50.3 | **84.3** |
| UQ (↑) | 38.27 | 46.69 | 42.13 | 31.88 | 20.41 | 47.47 | **65.49** |

关键对比：UCE/RECE 虽把目标删得很干净（Unlearn Acc ~3），但 Preserve Acc 只剩 5 左右——把相似概念也一起毁了；ScaPre 在删干净目标（5.8）的同时保住相似概念（76.3），Overall Acc 84.3 是唯一真正做到"精确"的。

### 关键发现
- **规模 ×5**：在可接受生成质量内，ScaPre 能比最强基线多遗忘 5 倍概念；随概念数增加，其他方法遗忘性能持续退化或坍缩，ScaPre 始终稳定。
- **效率碾压**：50 个概念仅需 120 秒，GPU-hours 和峰值显存均处于最低区间；UCE/RECE 虽也快，但遗忘与生成质量远逊。
- **风格遗忘**：CLIPx（CLIPcoco − CLIPart）达 3.44，远超次优 MACE 的 2.72，取得遗忘与质量的最佳折中。

## 亮点与洞察
- **把"防冲突"和"防误伤"拆成两个正交机制**：谱迹正则（含 $S$ 二阶统计 + $R$ 奇异值门控）管优化空间稳定，Informax 解耦器管参数级精度，思路清晰且互补。
- **Bures 距离替代 ℓ2** 是个漂亮的洞察——遗忘真正要守住的是特征的协方差/相关结构而非逐元素权重，这解释了为什么 ℓ2 在大规模下守不住质量。
- **互信息做参数选择**很自然：MI 直接度量"通道激活能多大程度预测输入是不是目标概念"，比启发式掩码更有原则。
- **完全闭式 + 训练无关 + 无额外数据**，把效率推到极致，对真实部署友好。

## 局限与展望
- 只在 SD v1.4/v1.5（UNet 交叉注意力）上验证，对 DiT/SDXL/Flux 等新架构的迁移性未知。
- 闭式解依赖求解 $(I\otimes B+A^\top\otimes I)$ 的逆，维度极大时的数值/内存代价随模型规模放大如何，论文未深入。
- 几何对齐的近端修正是分步近似（先解二次再 Procrustes 投影），与联合最优解之间的差距缺乏理论刻画。
- 互信息估计依赖激活离散化阈值 $\tau_i$ 和正/中性样本划分，对阈值与采样质量的敏感性可进一步分析。

## 相关工作与启发
- **单概念遗忘**：微调类（FMN、SA、SalUn、AC）、权重编辑类（TIME、SPEED）、剪枝类（ConceptPrune、MS、SEMU）。
- **多概念遗忘**：MACE（多 LoRA 微调，扩到上百概念）、SPM（可组合适配器 + 潜空间锚定）、Sculpting Memory（动态掩码）、ESD（负向引导微调）、UCE（统一闭式）、RECE（迭代嵌入推导的高效闭式编辑）——ScaPre 在"闭式范式"上接续 UCE/RECE，但用冲突感知正则和信息解耦补上了它们大规模下坍缩、误伤的短板。
- **启发**：闭式编辑不是和精度/稳定对立的，关键在于把领域先验（冲突来自哪些方向、相关参数如何识别、全局结构如何度量）编码进二次目标的正则结构里——这套"先验入正则、保闭式性"的思路可迁移到其他模型编辑（知识擦除、风格控制）任务。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 谱迹正则的 $S+R$ 双项设计、Bures 距离做几何对齐、互信息解耦器，三个组件各有原创性，且统一进闭式解里很优雅；不过整体仍在 UCE/RECE 闭式编辑范式内延伸。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖大规模（10/50 概念）、精确（Confuse5）、风格、显式内容、效率、对抗鲁棒等多维度，自建 3 个 benchmark，基线齐全（8 个）；但仅限 SD v1.x，缺新架构验证。
- **写作质量**: ⭐⭐⭐⭐ 三大挑战 → 三组件对应清晰，图 2 pipeline 和公式推导完整，UQ 统一指标设计合理。
- **价值**: ⭐⭐⭐⭐ 把概念遗忘从"十几个"推到"几十个"且兼顾精度与效率（120 秒/50 概念），对生成模型安全部署有实际意义，代码开源。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)
- [\[ICML 2026\] Forget-It-All: Multi-Concept Machine Unlearning via Concept-Aware Neuron Masking](../../ICML2026/image_generation/forget-it-all_multi-concept_machine_unlearning_via_concept-aware_neuron_masking.md)
- [\[ICLR 2026\] Continual Unlearning for Text-to-Image Diffusion Models: A Regularization Perspective](continual_unlearning_for_text-to-image_diffusion_models_a_regularization_perspec.md)
- [\[ICLR 2026\] AEGIS: Adversarial Target-Guided Retention-Data-Free Robust Concept Erasure from Diffusion Models](aegis_adversarial_target-guided_retention-data-free_robust_concept_erasure_from_.md)
- [\[ECCV 2024\] Challenging Forgets: Unveiling the Worst-Case Forget Sets in Machine Unlearning](../../ECCV2024/image_generation/challenging_forgets_unveiling_the_worst-case_forget_sets_in_machine_unlearning.md)

</div>

<!-- RELATED:END -->
