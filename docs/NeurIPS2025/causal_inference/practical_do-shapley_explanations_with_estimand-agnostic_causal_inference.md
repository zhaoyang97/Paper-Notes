# Practical do-Shapley Explanations with Estimand-Agnostic Causal Inference

**会议**: NeurIPS 2025  
**arXiv**: [2509.20211](https://arxiv.org/abs/2509.20211)  
**代码**: 待确认  
**领域**: 因果推断 / 可解释性  
**关键词**: Shapley值, 因果推断, do-SHAP, 结构因果模型, 可辨识性

## 一句话总结
提出 Estimand-Agnostic（EA）方法和 Frontier-Reducibility Algorithm（FRA）来高效计算因果 Shapley 值（do-SV），通过训练单个 SCM 学习观测分布即可回答任意可辨识的因果查询，并通过联盟约减将计算量降低约 90%。

## 研究背景与动机

1. **领域现状**：do-SHAP 将 Shapley 值与因果推断结合，用因果干预值 $\nu(S) = E[Y|\text{do}(X_S = x_S)]$ 替代条件期望，消除了传统 SHAP 的虚假相关问题。但计算 do-SV 需要评估 $2^{|X|}$ 个不同的因果查询。
2. **现有痛点**：现有 Estimand-Based（EB）方法需要为每个联盟 $S$ 手动指定因果 estimand（如 backdoor/frontdoor 准则），然后分别建模——极不实用。对 $K=10$ 个特征，需要处理 1024 个不同的因果查询，每个可能需要不同的 estimand。
3. **核心矛盾**：do-SV 的理论优越性（无虚假相关）与计算的不实用性之间的矛盾——联盟数指数增长，且每个联盟需要独立的因果推断。
4. **本文要解决什么？** (a) 消除对每个联盟手动指定 estimand 的需要；(b) 减少需要计算的联盟数量。
5. **切入角度**：直接学习数据生成过程（SCM），一旦 SCM 拟合好，任何可辨识的因果查询都可以通过模拟 do 操作来回答，无需预先推导 estimand。同时利用图结构发现冗余联盟。
6. **核心 idea 一句话**：用单个 SCM 拟合观测分布实现estimand-agnostic 因果推断 + Frontier-Reducibility 算法约减联盟数 → 实用化 do-SHAP。

## 方法详解

### 整体框架
输入：因果图 $G$、观测数据 → **Step 1**: 训练 SCM（可选架构：线性/DCN/DCG/CNF）拟合 $\mathcal{P}(V)$ → **Step 2**: Frontier-Reducibility 算法将 $2^K$ 个联盟约减为不可约子集 → **Step 3**: 对每个不可约联盟，通过 SCM 模拟 do 操作计算 $\nu(S)$，缓存结果 → **Step 4**: 聚合 Shapley 值

### 关键设计

1. **Estimand-Agnostic (EA) 因果查询**:
   - 做什么：训练单个 SCM 来回答任意因果查询，无需为每个联盟推导 estimand
   - 核心思路：SCM 学习每个变量的生成机制 $V_i = f_i(\text{Pa}_i, U_i)$。做 do 干预时直接将目标变量 $X_S$ 固定为观测值，从噪声先验中采样 $U$，按拓扑序逐步生成非干预变量。最终 $E[Y|\text{do}(x_S)]$ 通过 Monte Carlo 估计
   - 设计动机：EB 方法需要为每个可辨识查询推导单独的统计 estimand，在联盟数指数增长时不可行。EA 方法训练一次 SCM 即可回答所有查询

2. **Frontier-Reducibility Algorithm (FRA)**:
   - 做什么：识别产生相同因果效应的联盟，减少重复计算
   - 核心思路：对联盟 $S$，检查是否存在可约子集 $S' \subset S$ 使得 $\nu(S) = \nu(S')$。约减条件：$S$ 中后面的变量（在拓扑序上）阻断了前面变量到 $Y$ 的所有路径。用"frontier"概念——$S$ 中距离 $Y$ 最近的变量层形成不可约核
   - 设计动机：在稀疏因果图中，很多联盟是冗余的（~90% 可约减）。FRA 的计算开销极小（$K=100$ 时仅 8 秒），但可大幅减少因果查询次数

3. **多种 SCM 架构支持**:
   - 做什么：支持线性模型、深度因果网络（DCN）、连续归一化流（CNF）等
   - 核心思路：线性 SCM 用于验证理论正确性，DCN/DCG 用于非线性场景，CNF 用于处理连续变量和复杂分布
   - 设计动机：不同数据场景需要不同的 SCM 表达能力，框架需要灵活

### 损失函数 / 训练策略
- SCM 训练：最大化观测数据的似然 $\mathcal{P}(V)$
- 支持 Markovian（无隐变量）和 Semi-Markovian（有隐变量）因果图
- 半 Markovian 情况需要更复杂的 SCM 来建模隐变量

## 实验关键数据

### 主实验

| 数据集 | 设置 | EA 方法误差 | EB 方法误差 | FRA 加速 |
|--------|------|-----------|-----------|---------|
| 合成 (Markovian) | 线性 SCM | ~0.01 MSE | — | ~90% 联盟约减 |
| 合成 (Semi-Markovian) | DCN | ~0.03 MSE | ~0.02 MSE | ~85% 约减 |
| Adult Income | DCG | 可计算 | 需手动推导 | 显著加速 |
| Earthquake | CNF | 可计算 | — | — |

### 效率对比

| 方法 | $K=100$ 时间 |
|------|-------------|
| do-SHAP (无缓存) | 26m01s |
| do-SHAP + FRA 缓存 | 21m14s |
| FRA 约减计算 | ~8s |

### 关键发现
- EA 方法在 Markovian 设置下精度与 EB 方法相当，但无需为每个联盟推导 estimand
- FRA 在稀疏图上约减效果最好（~90%），在所有变量都是 $Y$ 的父节点时无效
- SCM 的训练误差会传播到 do-SV 估计，但在足够数据下影响可控
- 与 observational SHAP 对比，do-SHAP 能正确识别虚假相关变量（如 collider bias）

## 亮点与洞察
- **Estimand-Agnostic 范式优雅**：把"推导 estimand"这个因果推断中最难的步骤完全绕过，只需拟合 SCM。这在联盟数指数增长时是唯一可行的方案
- **FRA 的图论思想巧妙**：利用因果图的结构信息（拓扑序、路径阻断）发现计算冗余，成本极低但收益巨大
- **do-SHAP 的实用化扫清了障碍**：从"理论上正确但不可计算"到"可以在真实数据上运行"

## 局限性 / 可改进方向
- 需要已知的因果图——实际中因果发现算法可能有误，误差会传播
- EA 方法没有 doubly-robust 保证（EB 的 estimand 方法可以有）
- SCM 训练质量直接影响 do-SV 估计精度
- 大规模变量（$K \gg 100$）时仍面临组合爆炸（FRA 只是常数因子改善）

## 相关工作与启发
- **vs Observational SHAP**: do-SHAP 消除虚假相关但计算更贵，本文大幅降低了计算门槛
- **vs Causal SHAP (Heskes)**: Heskes 的 Asymmetric SHAP 也考虑因果结构但不做 do 干预
- **vs 因果推断文献**: EA 方法连接了 SCM 学习社区和可解释 AI 社区

## 评分
- 新颖性: ⭐⭐⭐⭐ EA + FRA 的组合使 do-SHAP 首次实用化
- 实验充分度: ⭐⭐⭐⭐ 合成 + 真实数据集，多种 SCM 架构，但规模受限
- 写作质量: ⭐⭐⭐⭐ 问题动机清晰，方法描述系统
- 价值: ⭐⭐⭐⭐⭐ 将 do-SHAP 从理论概念推向实际可部署的因果解释工具

### 补充技术细节
- SCM 训练使用 30 个不同随机种子验证估计稳定性
- Feature Importance 计算：$FI_X = \frac{1}{N}\sum_{i\in[N]}|\phi_X^{(i)}|/\sum_{X'\in X}|\phi_{X'}^{(i)}|$
- Semi-Markovian 实验（含隐变量 $U_{X,B}$）同样验证了 EA 方法的有效性
- 两个真实数据集：CDC 糖尿病健康指标 + 共享单车使用预测
- FRA 缓存的 Frontier 检查复杂度随联盟大小线性增长而非指数增长
- 价值: ⭐⭐⭐⭐ 为因果可解释 AI 提供了实用工具