---
title: >-
  [论文解读] Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression
description: >-
  [NeurIPS 2025 Spotlight][迁移学习] 提出两步式Transfer MNI方法，在高维过参数化线性回归中通过"保留目标信号+零空间迁移源知识"机制增强良性过拟合的泛化能力，刻画了模型偏移和协变量偏移下的非渐近excess risk，并发现了"免费午餐"协变量偏移区间。 问题背景 迁移学习通过利用源任务知…
tags:
  - "NeurIPS 2025 Spotlight"
  - "迁移学习"
  - "良性过拟合"
  - "高维线性回归"
  - "最小范数插值"
  - "协变量偏移"
  - "模型偏移"
---

# Transfer Learning for Benign Overfitting in High-Dimensional Linear Regression

**会议**: NeurIPS 2025 Spotlight  
**arXiv**: [2510.15337](https://arxiv.org/abs/2510.15337)  
**作者**: Yeichan Kim (Yonsei University), Ilmun Kim (KAIST), Seyoung Park (Yonsei University)  
**代码**: 未公开  
**领域**: 其他  
**关键词**: 迁移学习, 良性过拟合, 高维线性回归, 最小范数插值, 协变量偏移, 模型偏移  

## 一句话总结

提出两步式Transfer MNI方法，在高维过参数化线性回归中通过"保留目标信号+零空间迁移源知识"机制增强良性过拟合的泛化能力，刻画了模型偏移和协变量偏移下的非渐近excess risk，并发现了"免费午餐"协变量偏移区间。

## 研究背景与动机

### 问题背景
迁移学习通过利用源任务知识提升目标任务性能，在高维回归中已有大量成功案例（如LASSO及其变体、GLM、非参数回归）。与此同时，过参数化模型（如最小$\ell_2$范数插值器MNI）在$n < p$设定下展现出令人惊讶的良性过拟合（benign overfitting）特性——零训练误差却能良好泛化。

### 已有工作的不足
- 现有迁移学习方法依赖**显式正则化**（如LASSO），而良性过拟合依靠**隐式正则化**，两者交叉研究几乎空白
- Mallinar等(2021)研究了OOD设定但未使用目标样本训练；Wu等(2023)提出SGD方法但未考虑模型偏移，实验集中在欠参数化区间
- Song等(2024)提出pooled-MNI，但池化多源数据对分布偏移极其敏感
- 缺乏对MNI迁移学习中正迁移条件、最优迁移样本量、最大改进量的精确刻画

### 核心问题
迁移学习能否进一步增强过参数化插值器在高维线性回归中已经出色的样本外泛化能力？

## 方法详解

### 问题建模
考虑1个目标任务和$Q$个源任务的过参数化线性回归（$n_q \leq p$）：

$$\mathbf{y}^{(q)} = \mathbf{X}^{(q)} \boldsymbol{\beta}^{(q)} + \boldsymbol{\epsilon}^{(q)}, \quad q \in [Q]_0$$

其中$q=0$为目标任务。分布偏移由两部分刻画：
- **模型偏移**：模型对比向量 $\boldsymbol{\delta}^{(q)} = \boldsymbol{\beta}^{(q)} - \boldsymbol{\beta}^{(0)}$
- **协变量偏移**：协方差矩阵 $\boldsymbol{\Sigma}^{(q)}$ 的结构差异（不要求同时对角化）

### Transfer MNI（TM）：两步式迁移
1. **预训练**：在第$q$个源数据上训练源MNI $\hat{\boldsymbol{\beta}}_M^{(q)}$
2. **微调**：在插值目标数据的约束下，最小化与源MNI的欧氏距离：

$$\hat{\boldsymbol{\beta}}_{TM}^{(q)} = \arg\min_{\boldsymbol{\beta}} \left\{ \|\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}_M^{(q)}\| : \mathbf{X}^{(0)}\boldsymbol{\beta} = \mathbf{y}^{(0)} \right\}$$

### "保留+迁移"机制
TM估计具有优雅的分解结构：

$$\hat{\boldsymbol{\beta}}_{TM}^{(q)} = \underbrace{\hat{\boldsymbol{\beta}}_M^{(0)}}_{\text{目标信号}} + \underbrace{(\mathbf{I}_p - \mathbf{H}^{(0)}) \hat{\boldsymbol{\beta}}_M^{(q)}}_{\text{零空间知识迁移}}$$

- 在目标行空间$\mathcal{S}_0$中**保留**目标MNI学到的信号（良性过拟合保证预测精度）
- 仅在目标零空间$\mathcal{S}_0^\perp$中**迁移**源信息（目标样本在此空间无信息量）
- 核心权衡：知识迁移必然带来**方差膨胀** $\mathcal{V}_\uparrow^{(q)} > 0$，但若偏差减少足以抵消，则实现正迁移

### Theorem 1：各向同性协方差下的精确分析
在$\boldsymbol{\Sigma}^{(0)} = \boldsymbol{\Sigma}^{(q)} = \mathbf{I}_p$、高斯设计下，期望偏差和方差有精确闭式解。定义偏移信噪比SSR和信噪比SNR：

$$\text{SSR}_q = \frac{\|\boldsymbol{\delta}^{(q)}\|^2}{\|\boldsymbol{\beta}^{(0)}\|^2}, \quad \text{SNR}_q = \frac{\|\boldsymbol{\beta}^{(0)}\|^2}{\sigma_q^2}$$

**正迁移条件**（Corollary 1，充要条件）：

$$\text{SSR}_q < 1 \quad \text{且} \quad \text{SNR}_q(1 - \text{SSR}_q) > \frac{p}{p - (n_q + 1)}$$

**最优迁移样本量**：$n_q^* = p - 1 - \sqrt{p(p-1)/[\text{SNR}_q(1 - \text{SSR}_q)]}$，超过$n_q^*$后继续增加源样本反而降低迁移效果（严格凹性）。

### Theorem 2：良性协变量下的非渐近分析
在一般sub-Gaussian协变量和有效秩条件（Assumption 2）下，给出TM偏差和方差膨胀的高概率上界。偏差上界取决于模型对比$\|\boldsymbol{\delta}^{(q)}\|^2$和有效秩比$r_0/n_q$，方差膨胀是良性项$\Upsilon_q$和$\psi_0$的乘积。

### 免费午餐协变量偏移（Corollary 2）
当源协方差特征值均匀放大$\boldsymbol{\Lambda}^{(q)} = \alpha \boldsymbol{\Lambda}^{(0)}$（$\alpha > 1$）时：
- 偏差上界保持不变（与$\alpha$无关）
- 方差膨胀缩小$\alpha$倍

即在不增加偏差代价的前提下获得方差减少的"免费午餐"。仅要求前$\tau^*$个高信号特征向量对齐即可。

### WTM：信息源加权集成
1. 用K-fold交叉验证（$K=5$）检测信息源：比较各TM的CV loss与目标MNI的CV loss
2. 以CV loss的倒数为自适应权重，加权组合所有被检测为正迁移的TM估计
3. WTM自动过滤负迁移源，聚合多个正迁移源

## 实验关键数据

### 实验1：良性过拟合设定（3个源，$n_0=25$，$n_q=75$，$S=500$）

| 方法 | 纯模型偏移 SSR=(0,0.3,0.6) | +协变量偏移 SSR=0.3 | +免费午餐 $\alpha=8$ |
|------|---------------------------|--------------------|--------------------|
| Target MNI（基线） | 随$p$缓慢下降 | 随$p$缓慢下降 | 随$p$缓慢下降 |
| Pooled-MNI | 全面崩溃（对偏移极敏感） | 全面崩溃 | 全面崩溃 |
| TM（各单源） | 即使SSR=0.6仍优于基线 | TM(3)出现负迁移 | TM(3)恢复至基线水平 |
| WTM（集成） | **一致最优**，超越所有单TM | 自动过滤负迁移源，**一致最优** | 所有TM加速收敛，**WTM最优** |

关键发现：Pooled-MNI在分布偏移下表现灾难性；WTM通过CV检测有效识别并排除负迁移源。

### 实验2：无害插值设定（2个源，各向同性$\mathbf{I}_p$，$S=10$，SSR=0.4）

| 方法 | 无协变量偏移 | 免费午餐 $\alpha=8$（原$n_2^*$） | 免费午餐 $\alpha=8$（调整$n_2^*$） |
|------|------------|-------------------------------|--------------------------------|
| Target MNI | excess risk收敛至10 | excess risk收敛至10 | excess risk收敛至10 |
| Pooled-MNI | 最终超越基线但收敛慢 | 表现改善 | 表现改善 |
| SGD迁移 | 落后于TM显著 | 落后于TM | 落后于TM |
| TM（最优$n_2^*$） | **显著优于**基线和竞争方法 | 进一步改善 | 利用$\text{SNR}_\alpha$迁移更多样本，**最佳** |
| WTM | **最优**（超越单TM） | 进一步改善 | **最优** |

关键发现：在免费午餐偏移下可使用修正的$\text{SNR}_\alpha = \alpha\|\boldsymbol{\beta}^{(0)}\|^2/\sigma^2$来增加最优迁移样本量，进一步提升性能。实验中每个设定进行50次独立重复，绘制excess risk对$p \in \{300, 400, \ldots, 1000\}$的曲线。

## 亮点

- **"保留+迁移"机制**：TM在目标行空间保留MNI信号、仅在零空间迁移源知识，理论优雅且实践鲁棒
- **充要正迁移条件**：首次给出MNI迁移学习中正迁移的精确判据（SSR<1且SNR足够大）和最优迁移样本量闭式解
- **免费午餐协变量偏移**：发现不要求源-目标特征向量完全对齐，仅需前$\tau^*$个高信号方向对齐即可获得方差减少的免费收益
- **自适应集成WTM**：基于CV的informative source检测+逆CV loss加权，在所有设定下一致最优
- **对比pooled-MNI的显著优势**：TM的late-fusion架构对分布偏移天然鲁棒，而pooled-MNI在偏移下崩溃

## 局限与展望

- **WTM理论保证缺失**：CV检测informative source的一致性尚未证明（$\mathcal{I} = \hat{\mathcal{I}}$的高概率保证为开放问题）
- **方差膨胀上界较松**：非同时对角化时，上界含$(\lambda_p^{(q)})^{-1}$项可能较大，需更精细的协方差结构分析
- **仅限线性回归**：最小范数插值器的分析假设线性模型，对深度网络的良性过拟合迁移仅提供NTK层面的初步讨论
- **各向同性分析的局限**：Theorem 1和Corollary 1的精确结果仅适用于各向同性协方差+高斯设计
- **源任务相关性未利用**：多源之间的相关性未被建模，WTM对各源独立训练后简单加权
- **RKHS扩展待验证**：文中讨论了minimum-RKHS-norm interpolator的Transfer MNI扩展，但理论分析和实验均未完成

## 与相关工作的对比

- **Bartlett et al. (2020)**：建立单任务MNI良性过拟合的非渐近理论基础（effective rank条件），本文继承并扩展到迁移学习
- **Song et al. (2024) Pooled-MNI**：将所有数据池化训练单个MNI，对分布偏移极度敏感；本文TM的late-fusion架构显著更鲁棒
- **Mallinar et al. (2021)**：研究OOD设定但不使用目标数据训练；本文利用目标数据微调
- **Wu et al. (2023) SGD迁移**：SGD预训练+微调，但未考虑模型偏移，实验集中于欠参数化区间；本文专注过参数化
- **Tahir et al. (2024)**：用余弦相似度量化模型偏移，但依赖显式正则化；本文在隐式正则化框架下给出精确分析
- **Tian & Feng (2024)**：CV检测源可迁移性用于GLM（显式正则化），本文将此思路扩展到良性过拟合设定

## 评分

- 新颖性: ⭐⭐⭐⭐ — 首次系统研究MNI迁移学习，机制分解优雅，免费午餐偏移概念新颖
- 实验充分度: ⭐⭐⭐⭐ — 涵盖良性过拟合和无害插值两类设定，多种偏移组合，50次独立重复
- 写作质量: ⭐⭐⭐⭐⭐ — 理论体系完整，从各向同性到一般协变量逐层推进，行文清晰
- 价值: ⭐⭐⭐⭐ — 填补了过参数化迁移学习的理论空白，但限于线性模型削弱了实际影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] A High-Dimensional Statistical Method for Optimizing Transfer Quantities in Multi-Source Transfer Learning](a_highdimensional_statistical_method_for_optimizing_transfer.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](conformal_online_learning_of_deep_koopman_linear_embeddings.md)
- [\[ICML 2025\] Heavy-Tailed Linear Bandits: Huber Regression with One-Pass Update](../../ICML2025/learning_theory/heavy-tailed_linear_bandits_huber_regression_with_one-pass_update.md)
- [\[ICML 2026\] Asymptotic Optimality of the High-Dimensional Gaussian Mechanism and Improved Low-Dimensional Mechanisms for Differential Privacy](../../ICML2026/learning_theory/asymptotic_optimality_of_the_high-dimensional_gaussian_mechanism_and_improved_lo.md)
- [\[NeurIPS 2025\] Infrequent Exploration in Linear Bandits](infrequent_exploration_in_linear_bandits.md)

</div>

<!-- RELATED:END -->
