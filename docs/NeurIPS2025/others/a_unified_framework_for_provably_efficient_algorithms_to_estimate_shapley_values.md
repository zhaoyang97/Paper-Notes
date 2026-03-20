# A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values

**会议**: NEURIPS2025  
**arXiv**: [2506.05216](https://arxiv.org/abs/2506.05216)  
**代码**: 待确认  
**领域**: others  
**关键词**: Shapley Values, KernelSHAP, LeverageSHAP, Explainable AI, Randomized Numerical Linear Algebra  

## 一句话总结

提出统一框架将 KernelSHAP、LeverageSHAP 等 Shapley 值估计器纳入随机草图（sketching）视角，首次为 KernelSHAP 提供非渐近理论保证，并通过算法改进（Poisson 近似等）将方法扩展到 CIFAR-10 等高维数据集。

## 研究背景与动机

1. **可解释 AI 核心地位**：Shapley 值已成为解释黑盒模型特征贡献的主导方法，在金融、医疗、法律等安全敏感场景中至关重要。
2. **精确计算不可行**：计算精确 Shapley 值需要对 $2^d$ 个子集求值，随特征维度 $d$ 指数增长，对任意模型不可行。
3. **KernelSHAP 缺乏理论保证**：KernelSHAP 是 SHAP 库中使用最广泛的估计器，但其理论收敛性一直缺乏严格证明。已有理论保证的方法（unbiased KernelSHAP、LeverageSHAP）在实践中表现反而不如 KernelSHAP。
4. **高维扩展瓶颈**：现有有理论保证的方法面临组合数溢出和大支撑二项式采样两大瓶颈，无法扩展到 $d > 100$ 的真实场景。
5. **估计器间缺乏统一比较**：不同估计器（回归型 vs 矩阵向量乘型）和不同采样策略（有放回 vs 无放回）的理论性能权衡尚不清楚。
6. **核心科学问题**：能否建立一个统一框架，同时涵盖 KernelSHAP 和相关估计器，并为所有方法提供可证明的样本复杂度保证？

## 方法详解

### 整体框架

```
Shapley 值 → 约束回归问题 (Eq.1.2)
        ↓  变量替换 (Theorem 2.1)
  无约束问题: φ* = Q·U^T·b_λ + α·1
        ↓  随机草图 S
  ┌─────────────────┬──────────────────────┐
  │ 回归估计器 φ^R   │ 矩阵向量乘估计器 φ^M  │
  │ (KernelSHAP等)  │ (unbiased KernelSHAP) │
  └─────────────────┴──────────────────────┘
        ↓  采样分布选择
  ┌────────┬────────────┬──────────────┐
  │ Kernel │ Leverage   │ Modified ℓ₂  │
  │ 权重   │ Score/ℓ₂²  │ (几何平均)    │
  └────────┴────────────┴──────────────┘
```

### 三个关键设计

**设计一：约束消除与变量替换 (Theorem 2.1)**

- 引入正交矩阵 $Q$（列构成 $\mathbf{1}$ 正交补空间的标准正交基），将约束回归问题转化为无约束形式
- 定义 $U = \sqrt{d/(d-1)} \cdot Z'Q$，其中 $U^T U = I$（列正交性）
- 关键自由度：参数 $\lambda$ 可任意选择，$b_\lambda = \sqrt{d/(d-1)}(b - \lambda Z'\mathbf{1})$
- 统一表达：$\phi^* = Q \cdot U^T b_\lambda + \alpha \cdot \mathbf{1}$，其中 $\alpha = (v([d]) - v(\emptyset))/d$

**设计二：两类估计器的草图化构造**

- **回归估计器** $\phi_\lambda^R$：求解草图化最小二乘 $\min_x \|S(Ux - b_\lambda)\|^2$，计算复杂度 $O(md^2 + mT_v)$。KernelSHAP 对应 $\lambda = \alpha$ 的回归估计器 + Kernel 权重采样
- **矩阵向量乘估计器** $\phi_\lambda^M$：直接近似 $U^T b_\lambda$ 为 $U^T S^T S b_\lambda$，无偏估计，复杂度 $O(md + mT_v)$，但精度通常低于回归估计器

**设计三：三种采样分布的统一参数化**

通过加权几何平均 $p_S^\tau \propto (k(S))^\tau (\|u_S\|^2)^{1-\tau}$ 统一三种分布：

- $\tau = 1$：Kernel 权重采样（KernelSHAP 所用）
- $\tau = 0$：Leverage score / $\ell_2^2$ 采样（LeverageSHAP 所用）
- $\tau = 1/2$：Modified $\ell_2$ 采样（本文提出）

### 核心理论结果 (Theorem 2.2)

给出统一的样本复杂度界，使 $\Pr[\|\phi^* - \hat\phi\| < \varepsilon] > 1 - \delta$：

- 回归估计器：$m = O\left(\frac{\gamma(P_U b_\lambda)}{\delta \varepsilon^2} + \eta \log(d/\delta)\right)$
- 矩阵向量乘估计器：$m = O\left(\frac{\gamma(b_\lambda)}{\delta \varepsilon^2}\right)$

其中 $\eta$ 和 $\gamma$ 取决于采样分布的选择。Modified $\ell_2$ 采样的界永远不差于 Leverage score，最优情况下好 $\sqrt{d}$ 倍；Kernel 权重的界最优情况下好 $d/\log d$ 倍。

## 实验

### 主要实验设置

| 设置 | 详情 |
|------|------|
| 表格数据 | 8 个经典数据集（$d$ 最大 101），XGBoost 模型 |
| 图像数据 | MNIST ($d=784$), CIFAR-10 ($d=3072$) |
| 精确值 | TreeExplainer 计算真实 Shapley 值 |
| 评估指标 | 归一化 MSE, AUC (insertion/deletion), Spearman 秩相关 |
| 随机种子 | 100 个种子 (0-99) |
| 样本量 | $m = 10^3$ 到 $10^6$ |

### 关键发现

| 对比维度 | 结论 |
|----------|------|
| 回归 vs 矩阵向量乘 | 回归估计器在几乎所有数据集上精度更高，尤其在高维场景差距更大 |
| 三种采样分布 | $\ell_2^2$（Leverage score）略优于 Modified $\ell_2$，显著优于 Kernel 权重 |
| 有/无放回 | 矩阵向量乘估计器差异不大；回归估计器中有放回在部分数据集更优 |
| $\lambda = 0$ vs $\lambda = \alpha$ | $\lambda = \alpha$ 的矩阵向量乘估计器一致优于 $\lambda = 0$（即 unbiased KernelSHAP） |
| 高维表现 | 本文方法在 MNIST/CIFAR-10 上均显著优于 SHAP 库的 KernelSHAP 实现 |
| 忠实度 | MNIST 100k 样本后各方法趋同，但 KernelSHAP 秩相关略高；CIFAR-10 上本文方法秩相关提升明显 |

## 亮点

1. **首次为 KernelSHAP 提供非渐近理论保证**，解决了该领域长期悬而未决的问题
2. **统一框架优雅简洁**：通过变量替换 + 草图化将多个估计器纳入同一分析体系，参数 $\lambda$ 的自由度揭示了不同方法间的隐含联系
3. **Modified $\ell_2$ 采样**具备"绝不更差"的理论保证，是一个稳健的默认选择
4. **高维算法创新实用**：Poisson 近似绕过组合数溢出，首次将有理论保证的方法推到 $d = 3072$（CIFAR-10）

## 局限性

1. 理论界依赖于 $\|b_\lambda\|$ 等在实际中无法高效计算的量，用户无法直接实例化这些界
2. 无放回采样的多种变体未给出明确的选择准则
3. Kernel 权重和 Modified $\ell_2$ 能否在实际神经网络上真正优于 Leverage score 仍是开放问题
4. 实验仅使用 XGBoost/Decision Tree 模型验证，未在深度神经网络模型上评估
5. 收敛速率为 $\sim 1/\sqrt{m}$，对极高精度需求场景样本量仍然很大

## 相关工作

| 方法 | 类型 | 理论保证 | 本文定位 |
|------|------|----------|----------|
| KernelSHAP [LL17] | 回归 + Kernel 权重 | ❌ 无（本文首次提供） | 框架内特例 ($\lambda=\alpha$, 回归) |
| Unbiased KernelSHAP [CL20] | 矩阵向量乘 + Kernel 权重 | 渐近方差分析 | 框架内特例 ($\lambda=0$, MV) |
| LeverageSHAP [MW25] | 回归 + Leverage score | ✅ 非渐近界 | 框架内特例 ($\lambda=\alpha$, 回归) |
| SimSHAP [Zha+24] | 参数化方法 | 部分 | 高维替代方案 |
| LIME [RSG16] | 局部线性近似 | — | XAI 基线方法 |

## 评分

- 新颖性: ⭐⭐⭐⭐ — 统一框架视角新颖，首次理论覆盖 KernelSHAP
- 实验充分度: ⭐⭐⭐⭐ — 8 个表格数据集 + 2 个图像数据集，100 种子，多指标对比
- 写作质量: ⭐⭐⭐⭐ — 符号系统清晰，理论与实验组织良好
- 价值: ⭐⭐⭐⭐ — 为 XAI 社区最常用工具提供理论依据，具有实际影响力
