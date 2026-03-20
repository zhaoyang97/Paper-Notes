# Bayesian Influence Functions for Hessian-Free Data Attribution

**会议**: ICLR 2026  
**arXiv**: [2509.26544](https://arxiv.org/abs/2509.26544)  
**代码**: 无  
**领域**: 人体理解 / 数据归因与可解释性  
**关键词**: influence functions, training data attribution, Bayesian inference, SGMCMC, Hessian-free, data valuation  

## 一句话总结
提出 Local Bayesian Influence Function (BIF)，用 SGLD 采样估计的协方差替代经典影响函数中不可行的 Hessian 逆运算，实现了对数十亿参数模型的无架构限制数据归因，在重训练实验中达到 SOTA。

## 研究背景与动机
1. **领域现状**：训练数据归因（Training Data Attribution, TDA）研究训练数据如何塑造模型行为，是 AI 可解释性和安全性的基础问题。经典影响函数（Influence Functions, IF）通过 Hessian 逆来度量数据点的影响。
2. **现有痛点**：(a) 深度神经网络的 Hessian 通常是退化的（非可逆），经典 IF 的理论前提不成立；(b) 对大模型直接计算 Hessian 不可行，EK-FAC 等近似方法引入结构性偏差且仅支持 Linear/Conv2D 层；(c) Per-token 级别的细粒度归因在经典方法中需要逐 token 串行计算，不可扩展。
3. **核心矛盾**：需要一个既理论合理（不依赖 Hessian 可逆性）又计算可行（能扩展到数十亿参数）的数据归因方法。
4. **本文要解决什么？** 能否用贝叶斯推断框架彻底绕开 Hessian，同时保持或超越经典 IF 的归因质量？
5. **切入角度**：将经典 IF 的 Hessian 逆替换为局部后验分布上的协方差估计，利用 SGLD 采样实现。
6. **核心 idea 一句话**：$\text{BIF}(z_i, \phi) = -\text{Cov}_\gamma(\ell_i(\boldsymbol{w}), \phi(\boldsymbol{w}))$——影响就是训练损失与目标可观测量在局部后验下的负协方差。

## 方法详解

### 整体框架
将经典影响函数从点估计问题重新构建为分布性问题：不在单一最优参数 $\boldsymbol{w}^*$ 上计算梯度和 Hessian，而是在以 $\boldsymbol{w}^*$ 为中心的局部后验分布上估计协方差。

### 关键设计

1. **从 IF 到 BIF 的理论推导**：
   - 经典 IF：$\text{IF}(z_i, \phi) = -\nabla\phi(\boldsymbol{w}^*)^\top \boldsymbol{H}^{-1} \nabla\ell_i(\boldsymbol{w}^*)$（需要 Hessian 逆）
   - 贝叶斯 IF：$\text{BIF}(z_i, \phi) = -\text{Cov}(\ell_i(\boldsymbol{w}), \phi(\boldsymbol{w}))$（统计物理标准结果）
   - 关键等价性：在非退化模型中，BIF 的首阶 Taylor 展开等价于经典 IF（Appendix A）

2. **局部化机制（Local BIF）**：
   - 引入以 $\boldsymbol{w}^*$ 为中心的各向同性高斯先验：$p_\gamma \propto \exp(-\sum \ell_i - \frac{\gamma}{2}\|\boldsymbol{w}-\boldsymbol{w}^*\|^2)$
   - 精度参数 $\gamma$ 控制局部性，类似于经典 IF 中的 Hessian dampening $(\boldsymbol{H} + \gamma\boldsymbol{I})$
   - Local BIF 是 dampened IF 的高阶推广

3. **SGLD 采样估计**：
   - 使用随机梯度 Langevin 动力学从局部后验采样
   - 多条独立 SGLD 链提升覆盖率，总共 $N_{\text{draws}} = C \times T$ 个采样点
   - 每步只需前向传播计算训练集和查询集的损失值——无需反向传播计算梯度用于归因

4. **Per-token 归因**：
   - 自回归模型中损失按 token 分解：$\ell_i = \sum_s \ell_{i,s}$
   - BIF 天然支持 token 级协方差：$\text{BIF}(z_{i,s}, z_{j,s'}) = -\text{Cov}_\gamma(\ell_{i,s}, \ell_{j,s'})$
   - 一次采样即可并行计算整个 token-token 影响矩阵，经典方法需逐 token 串行

5. **归一化 BIF（Posterior Correlation）**：
   - 原始协方差受高方差数据点主导
   - 归一化为 Pearson 相关系数，值在 [-1, 1] 之间，更稳定和可比较

## 实验关键数据

### 表1：BIF vs EK-FAC 复杂度对比

| 维度 | Local BIF | EK-FAC |
|------|-----------|--------|
| 时间复杂度 | $O(N_{\text{draws}}(n+q)d)$，无 fit 阶段 | Fit: $O(N_{\text{fit}}d + \sum d_\ell^3)$; Score: $O(nqd)$ |
| 额外存储 | $O(N_{\text{draws}}(n+q))$ 损失轨迹 | $O(\sum(d_{\text{in},l}^2 + d_{\text{out},l}^2))$ 因子 |
| 误差来源 | 有限采样 + SGLD 偏差 | 有限采样 + 结构偏差（Kronecker/Fisher） |
| 架构支持 | 任意可微模型 | 仅 Linear 和 Conv2D 层 |

### 表2：CIFAR-10 重训练实验 LDS 得分

| 数据规模 $\alpha$ | BIF | EK-FAC | TRAK | GradSim |
|-------------------|-----|--------|------|---------|
| 小数据集（高方差） | **略优于 EK-FAC** | SOTA 基线 | 明显低于 BIF/EK-FAC | 最低 |
| 大数据集 | 与 EK-FAC 相当（误差范围内） | SOTA | 低于两者 | 最低 |
| Pythia-14M 微调 | 低于 EK-FAC | 更优 | — | — |

> 在 ResNet-9/CIFAR-10 上，BIF 和 EK-FAC 的 LDS 得分均显著优于 TRAK 和 GradSim。BIF 在小数据高方差 regime 下略优。

### 扩展性分析（Pythia 模型族）
- 在 Pythia-2.8B 上，BIF 评估时间比 EK-FAC 快约 **2 个数量级**
- EK-FAC 有巨大的前期 fit 成本（存储 Kronecker 因子 + 特征分解），BIF 无此开销
- GPU 内存使用两者相当（4×A100 节点）
- BIF 的优势随模型规模增大而更加明显

## 亮点与洞察
- **概念优雅**：将 Hessian 逆问题转化为协方差估计问题，一个公式 $-\text{Cov}(\ell_i, \phi)$ 统一了整个方法——理论简洁、实现直观。
- **Per-token 归因的实用突破**：经典方法逐 token 计算不可行，BIF 的批量协方差天然并行，使 LLM 的 token 级归因成为现实。
- **语义质量**：Figure 2 展示 Pythia-2.8B 上 token 间的后验相关性能捕捉翻译、同义词、数字-拼写等语义关系，定性效果出色。
- **统计物理视角**：BIF 与 susceptibilities、local learning coefficient 等 SLT 量有深层联系，是 developmental interpretability 研究议程的重要组成。
- **通用性强**：不依赖特定层类型，适用于任何可微架构（含 attention、normalization 等 EK-FAC 不支持的层）。

## 局限性
1. **采样质量依赖**：BIF 精度取决于 SGLD 对局部后验的采样质量，DNN 的奇异损失景观使标准 SGLD 收敛保证可能不成立。
2. **超参数敏感性**：步长 $\epsilon$、局部化强度 $\gamma$、逆温度 $\beta$ 的最优选择尚无理论指导，特别是在语言模型设置下。
3. **序列级归因不如 EK-FAC**：在 Pythia-14M 微调场景中，BIF 的 LDS 低于 EK-FAC，反映 LLM 后验采样的困难。
4. **计算成本扩展**：虽然无 fit 阶段，但每个 SGLD draw 需遍历全部训练集和查询集做前向传播，当归因数据量极大时开销仍然可观。
5. **无理论收敛速率**：BIF 的采样误差如何随 $N_{\text{draws}}$ 衰减缺乏严格理论分析。

## 相关工作
- **经典影响函数**：Cook (1977) 提出，Koh & Liang (2020) 引入深度学习，Grosse et al. (2023) 提出 EK-FAC 近似（当前 SOTA）。
- **梯度相似性方法**：TRAK (Park et al., 2023b) 用表示空间相似度近似归因；GradSim 直接用梯度内积。
- **分布性 TDA**：Mlodozeniec et al. (2025) 提出 d-TDA 框架，BIF 可视为其 mean-shift 特例。
- **贝叶斯无穷小 Jackknife**：Giordano & Broderick (2024) 在贝叶斯模型分析中提出类似思想，但未局部化或扩展至大规模 LLM。
- **奇异学习理论**：Lau et al. (2025) 提出 SGMCMC 估计 local learning coefficient 的方法，本文使用相同的局部化机制。

## 评分
- **创新性**: ★★★★★ — 用协方差替代 Hessian 逆的理论洞察优雅深刻，local BIF 的定义自然且具有广泛适用性。
- **实用性**: ★★★★☆ — 对大模型架构无限制，per-token 归因实用价值高，但超参数调优仍需经验。
- **理论深度**: ★★★★☆ — BIF 与经典 IF 的渐近等价性证明严谨，但采样收敛性和超参选择缺少理论保证。
- **实验充分性**: ★★★★☆ — 覆盖视觉（CIFAR-10, ImageNet）和语言（Pythia 系列）模型，含定性和定量评估，但重训练实验规模较小。
- **表达清晰度**: ★★★★★ — 从 IF 到 BIF 的推导层次清晰，Figure 1-2 的可视化直观有力，方法与基线的对比表格（Table 1）信息量大。
