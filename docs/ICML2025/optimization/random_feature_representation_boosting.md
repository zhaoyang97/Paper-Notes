---
title: >-
  [论文解读] Random Feature Representation Boosting
description: >-
  [ICML2025][优化/理论][random features] 提出 RFRBoost，利用梯度表示提升（gradient representation boosting）理论构建深层残差随机特征神经网络，在 MSE 损失下获得封闭解，在一般损失下化归为二次约束最小二乘问题，在表格数据上显著超越单层 RFNN 与端到端训练的 MLP ResNet。
tags:
  - "ICML2025"
  - "优化/理论"
  - "random features"
  - "gradient boosting"
  - "residual neural networks"
  - "tabular data"
  - "convex optimization"
---

# Random Feature Representation Boosting

**会议**: ICML2025  
**arXiv**: [2501.18283](https://arxiv.org/abs/2501.18283)  
**作者**: Nikita Zozoulenko, Thomas Cass, Lukas Gonon
**代码**: 待确认  
**领域**: 优化  
**关键词**: random features, gradient boosting, residual neural networks, tabular data, convex optimization

## 一句话总结

提出 RFRBoost，利用梯度表示提升（gradient representation boosting）理论构建深层残差随机特征神经网络，在 MSE 损失下获得封闭解，在一般损失下化归为二次约束最小二乘问题，在表格数据上显著超越单层 RFNN 与端到端训练的 MLP ResNet。

## 研究背景与动机

随机特征神经网络（RFNN）只训练线性输出层、隐藏层参数随机固定，具备凸优化、快速训练、可证泛化保证等优势。然而现有深层 RFNN 理论仅限于 Fourier 激活函数，且将随机层简单堆叠为 ResNet 结构时面临关键困难：

**残差块尺度问题**：若残差块 $g_t$ 过小，初始表示 $\Phi_0$ 主导，新增随机特征无效；若过大，前层信息丢失。

**方向问题**：理想情况下每个残差块应近似损失函数关于网络表示的负泛函梯度，但随机层无法保证具备此性质。

**训练机制缺失**：端到端训练可通过 SGD+反向传播调整所有层权重，RFNN 隐藏层固定后缺少类似机制。

这些挑战使得"如何在保留随机特征计算优势的前提下构建有效的深层残差 RFNN"成为核心问题。

## 核心思想

RFRBoost 的核心洞察：**将残差块限制为 $g_t = A_t f_t$ 的形式**（$f_t$ 是随机特征、$A_t$ 是可学习的线性映射），然后利用梯度表示提升理论求解最优 $A_t$。这样既保留了 RFNN 的凸优化特性，又通过 boosting 框架赋予了深层结构理论保证。

ResNet 表示递归定义为：

$$\Phi_t(x) = \Phi_{t-1}(x) + \eta A_t f_t(x, \Phi_{t-1}(x))$$

其中 $f_t$ 可同时依赖原始输入 $x$ 和前一层输出 $\Phi_{t-1}(x)$，$\eta$ 为学习率。

## 方法详解

### 1. 随机特征层定义

随机特征层生成固定的特征向量 $f_t \in \mathbb{R}^p$，初始化后不再训练。实验中具体选择：

$$f_t(x) = \sigma(\text{concat}(B_t \Phi_{t-1}(x),\; C_t x))$$

其中 $B_t, C_t$ 使用 SWIM 随机特征初始化（非 i.i.d.），$\sigma$ 为激活函数。这种设计让随机特征同时利用原始数据和前层表示，增强表达力。

### 2. Exact-Greedy 策略（MSE 损失）

对于 MSE 损失 $l(x,y) = \frac{1}{2}\|x - y\|^2$，每层贪心求解联合优化：

$$W_t, g_t = \arg\min_{W, g} \hat{\mathcal{R}}(W, \Phi_{t-1} + g)$$

将 $g_t = A_t f_t$ 代入后转化为"夹心最小二乘问题"（sandwiched least squares）：

$$A_t = \arg\min_A \frac{1}{n} \sum_{i=1}^n \|r_i - W_{t-1}^\top A f_{t,i}\|^2 + \lambda\|A\|_F^2$$

其中 $r_i = y_i - W_{t-1}^\top \Phi_{t-1}(x_i)$ 为残差。**关键贡献**：Theorem 3.1 证明该问题在标量、对角、稠密三种 $A_t$ 形式下均存在封闭解：

| $A_t$ 类型 | 封闭解形式 | 说明 |
|:---:|:---:|:---|
| 标量 | $A_{\text{scalar}} = \frac{\langle R, ZW \rangle_F}{\|ZW\|_F^2 + n\lambda}$ | 全局统一缩放 |
| 对角 | $(WW^\top \odot Z^\top Z + \lambda I)^{-1} \text{diag}(WR^\top Z)$ | 逐维独立学习率 |
| 稠密 | 基于 $WW^\top$ 和 $Z^\top Z$ 的谱分解 | 完整线性映射，学习泛函梯度方向 |

求解 $A_t$ 后更新 $\Phi_t = \Phi_{t-1} + \eta A_t f_t$，再通过最小二乘更新顶层线性预测器 $W_t$。

### 3. Gradient-Greedy 策略（一般损失）

对交叉熵等一般可微损失，采用泛函 Taylor 一阶展开近似风险：

$$\mathcal{R}(W, \Phi + g) \approx \mathcal{R}(W, \Phi) + \langle g, \nabla_2 \mathcal{R}(W, \Phi) \rangle_{L_2^D(\mu)}$$

论文指出直接最小化内积的问题：$g$ 可能通过增大自身范数来减小内积，而非对齐泛函梯度方向。解决方案是**施加单位范数约束** $\|g\|_{L_2^D(\hat{\mu})} = 1$。

**Theorem 3.2** 证明，约束优化问题等价于二次约束最小二乘：

$$\min_A \frac{1}{n}\sum_{i=1}^n \|\nabla_2\hat{\mathcal{R}}(W,\Phi)(x_i) - A f(x_i)\|^2 \quad \text{s.t.} \quad \frac{1}{n}\sum_{i=1}^n \|Af(x_i)\|^2 = 1$$

且存在解析解 $A = \frac{\sqrt{n}}{\|G\|_F} G^\top F (F^\top F)^{-1}$（$F$ 满秩时）。

实际训练三步骤：
1. 生成随机特征 $f_t$，计算泛函梯度矩阵 $G$，用正规化最小二乘拟合 $A_t$
2. 线搜索求最优步长 $\alpha_t$（MSE 用 Theorem 3.1，交叉熵用 Newton 法）
3. 更新 $\Phi_t = \Phi_{t-1} + \eta \alpha_t A_t f_t$，凸优化更新 $W_t$

### 4. 理论保证

基于 Suggala et al. (2020) 的广义提升框架，在 $(\beta, \epsilon)$-弱学习条件下给出遗憾界（regret bound），即超额风险随层数 $T$ 和样本量 $\tilde{n}$ 衰减的速率。该理论保证是端到端训练 ResNet 所不具备的。

## 实验关键数据

### 实验设置

- **数据集**：来自 OpenML 的 91 个表格数据任务（回归 + 分类），涵盖小到中等规模
- **基线方法**：单层 RFNN、端到端训练 MLP ResNet
- **评估**：标准 train/val/test 划分，MSE（回归）/Accuracy（分类）

### 主要结果

| 对比维度 | 结果 |
|:---|:---|
| RFRBoost vs 单层 RFNN | 显著优于，证明深层残差结构有效提升随机特征表达力 |
| RFRBoost vs MLP ResNet (SGD) | 在小到中等规模数据上显著优于端到端训练模型 |
| 计算效率 | 无需反向传播，训练速度远快于梯度下降方法 |
| Gradient-Greedy vs Exact-Greedy | 带单位范数约束的 Gradient-Greedy 策略优于 Exact-Greedy |

### 关键实验发现

1. **深层有效性**：随着层数 $T$ 增加，RFRBoost 性能持续提升，而简单堆叠随机层的 naive 方法性能退化
2. **范数约束的重要性**：Gradient-Greedy 策略中施加 $\|g\|=1$ 约束是性能优于 Exact-Greedy 的关键——这一发现纠正了先前文献中 gradient-greedy 策略表现不佳的结论
3. **稠密 $A_t$ 最优**：三种 $A_t$ 类型中，稠密矩阵（学习完整泛函梯度方向）效果最好
4. **计算优势**：所有层的优化问题都是凸的（最小二乘或二次约束最小二乘），无需超参搜索随机特征的尺度

## 亮点与洞察

- 将 boosting 从"提升预测"拓展到"提升表示"并与随机特征结合，既优雅又实用
- "凸优化 + 深层结构"的组合在特定场景下可行且有理论保证，挑战了"深度必须非凸"的刻板印象
- 对 gradient-greedy 策略失败原因的分析（范数约束缺失）有启发性，纠正了先前文献的消极结论

## 局限与展望

1. **适用范围**：论文明确定位于"小到中等规模"表格数据，在大规模数据/图像/文本等高维任务上未验证，预期难以与深度学习竞争
2. **线性预测头**：顶层限制为线性映射 $W^\top \Phi$，表达力受限
3. **随机特征质量依赖**：性能依赖随机特征的质量（如 SWIM 初始化），不同初始化策略对不同数据集的影响未充分探讨
4. **与 GBDT 对比缺失**：表格数据上 XGBoost/LightGBM/CatBoost 是最强基线，论文未与之直接对比，仅对比了神经网络类方法
5. **弱学习条件的实际验证**：理论保证依赖 $(\beta, \epsilon)$-弱学习条件，但论文未验证该条件在实际数据集上是否成立
6. **仅限监督学习**：方法框架未扩展到无监督/自监督/半监督场景

## 相关工作与启发

- **经典梯度提升**：XGBoost、LightGBM、CatBoost 用决策树做弱学习器提升标签空间
- **神经网络提升**：AdaNet（Cortes et al., 2017）、GrowNet（Badirli et al., 2020）在标签空间提升
- **梯度表示提升**：Nitanda & Suzuki (2018, 2020)、Suggala et al. (2020) 在表示空间提升——RFRBoost 直接继承此框架
- **随机特征模型**：RFNN、Reservoir Computing、随机 Fourier 特征、SWIM（Bolager et al., 2023）

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将梯度表示提升理论与随机特征 ResNet 结合是新颖且自然的，sandwiched least squares 封闭解有技术贡献
- 实验充分度: ⭐⭐⭐ — 91 个 OpenML 数据集规模不小，但缺少与 GBDT 的对比削弱了说服力
- 写作质量: ⭐⭐⭐⭐ — 理论推导清晰，算法伪代码完整，两种策略的对比分析有深度
- 价值: ⭐⭐⭐ — 在随机特征/表格数据的小众领域有扎实贡献，但适用范围限制了影响力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Global Convergence and Rich Feature Learning in $L$-Layer Infinite-Width Neural Networks under $\mu$P Parametrization](global_convergence_and_rich_feature_learning_in_l-layer_infinite-width_neural_ne.md)
- [\[ICML 2025\] Provable Benefit of Random Permutations over Uniform Sampling in Stochastic Coordinate Descent](provable_benefit_of_random_permutations_over_uniform_sampling_in_stochastic_coor.md)
- [\[NeurIPS 2025\] From Linear to Nonlinear: Provable Weak-to-Strong Generalization through Feature Learning](../../NeurIPS2025/optimization/from_linear_to_nonlinear_provable_weak-to-strong_generalization_through_feature_.md)
- [\[ICCV 2025\] Zeroth-Order Fine-Tuning of LLMs in Random Subspaces](../../ICCV2025/optimization/zeroth-order_fine-tuning_of_llms_in_random_subspaces.md)
- [\[ICLR 2026\] Rapid Training of Hamiltonian Graph Networks using Random Features](../../ICLR2026/optimization/rapid_training_of_hamiltonian_graph_networks_using_random_features.md)

</div>

<!-- RELATED:END -->
