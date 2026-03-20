# Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention

**会议**: NeurIPS 2025  
**arXiv**: [2502.01473](https://arxiv.org/abs/2502.01473)  
**代码**: [https://github.com/Arya-Honarpisheh/gen_err_sel_ssm](https://github.com/Arya-Honarpisheh/gen_err_sel_ssm)  
**领域**: 理论 / SSM  
**关键词**: Mamba, 状态空间模型, 泛化界, 覆盖数, 谱横断面  

## 一句话总结
将选择性SSM（Mamba）展开为注意力形式，利用覆盖数技术推导出受连续时间状态矩阵谱横断面$s_{\mathbf{A}}$控制的泛化界——$s_{\mathbf{A}}<0$时泛化界与序列长度无关，$s_{\mathbf{A}}\geq0$时指数增长，并证明这种依赖不可消除。

## 研究背景与动机

1. **领域现状**：Mamba等选择性SSM在多种序列任务上与Transformer竞争，但缺乏理论泛化分析。
2. **现有痛点**：LTI SSM 的泛化理论依赖控制论工具（脉冲响应 ℓ1 范数、传递函数 H2 范数），但选择性 SSM 的非线性输入依赖动力学使这些工具不适用。Transformer 的覆盖数理论已较成熟，但无法直接应用于 SSM 的递归结构。
3. **核心矛盾**：选择性 SSM 既有 RNN 的递归结构（需要控制状态矩阵增长），又有注意力的输入依赖投影（$W_B, W_C$ 类似 key-query），需要一个统一的分析框架
4. **切入角度**：将选择性 SSM 递归展开为类注意力形式，构建两层覆盖——状态矩阵用 RNN 工具覆盖，输入投影用 Transformer 工具覆盖
5. **核心idea一句话**：连续时间状态矩阵的频谱横断面 $s_A$ 决定了泛化界是否与序列长度无关

## 方法详解

### 整体框架
通过 covering number 技术推导 Rademacher 复杂度上界，进而得到泛化误差界。关键在于如何为选择性 SSM 的参数空间构造有效的 ε-cover。

### 关键设计

1. **SSM→注意力展开**：
   - 做什么：将 Mamba 的递归计算 $y[t'] = C[t'] \sum_{t=0}^{t'-1} A^t \Delta[t'-1-t] B[t'-1-t] u[t'-1-t]$ 展开为类注意力形式
   - 核心思路：$W_C$ 对应 Query 投影，$W_B$ 对应 Key 投影，$u$ 本身作为 Value。$$z = w^\top \sum_{t=0}^{T-1} \underbrace{(I_d \otimes u[T]^\top W_C^\top)}_{\text{Query}} \underbrace{(I_d \otimes W_B u[T-1-t])}_{\text{Key}} \underbrace{u[T-1-t]}_{\text{Value}}$$
   - 设计动机：这种展开使得 $W_B, W_C$ 可以复用 Transformer 泛化理论中的线性函数类覆盖技术

2. **两层覆盖构造（核心技术贡献）**：
   - 第一层（状态矩阵 $A_c$）：用 Gelfand 公式控制 $\|A^t\|_2 \leq \rho_A^t$，其中 $\rho_A = (1+e^{p-\mathfrak{B}_q\mathfrak{B}_u})^{s_A+\eta}$。当 $s_A < 0$ 时 $\rho_A < 1$，几何级数收敛保证长度无关
   - 第二层（输入投影 $W_B, W_C, q, w$）：作为有界 $\|\cdot\|_{1,1}$ 范数的线性函数类，直接应用 Transformer 理论中的覆盖引理
   - Cartesian 乘积组合各参数的覆盖，最优化分配覆盖半径

3. **主定理 (Thm 3.3)**：
   - 泛化界中容量项 $\mathcal{C}_{\mathcal{F}_{SSM}} = \tilde{O}(\mathfrak{M}_\Delta \mathfrak{B}_w \mathfrak{B}_u^3 \mathfrak{B}_B \mathfrak{B}_C \mathfrak{B}_A S_2 (\cdot)^{3/2})$
   - 关键量 $S_2 = \frac{\rho_A(1-\rho_A^T)}{(1-\rho_A)^2} - \frac{T\rho_A^T}{\rho_A - 1}$
   - 当 $s_A < 0$: $\rho_A < 1$, $S_2$ 有界，泛化界**与序列长度 T 无关**
   - 当 $s_A > 0$: $\rho_A > 1$, $S_2 \sim T\rho_A^T$，泛化界**指数增长**

4. **下界 (Thm 4.1)**：
   - $s_A > 0$ 时 Rademacher 复杂度下界 $\geq \mathfrak{B}_w \frac{(1+s_A)^T - 1}{s_A}\sqrt{\frac{2}{\pi m}}$
   - 证明 T 依赖性不可通过更紧的上界消除——这是本质性的

### 与其他架构的泛化界对比

| 模型 | T 依赖 | d 依赖 | $\mathfrak{B}_u$ 依赖 |
|------|--------|--------|---------------------|
| Selective SSM ($s_A<0$) | **1** | $d^{1/2}$ | $\mathfrak{B}_u^4$ |
| Selective SSM ($s_A\geq 0$) | $T\rho_A^T$ | $d^{1/2}$ | $\mathfrak{B}_u^4$ |
| Linear Attention | T | 1 | $\mathfrak{B}_u^3$ |
| Softmax Attention | **1** | 1 | $\mathfrak{B}_u^3$ |
| Vanilla RNN ($\mathfrak{l}_x\|A\|_2<1$) | **1** | d | $\mathfrak{B}_u$ |

## 实验关键数据

### 实验1：不稳定初始化 ($s_A = 0.1$)

| 任务 | 短序列(T小) | 长序列(T大) |
|------|------------|------------|
| Majority | 训练成功，$s_A$ 被驱向0 | 训练失败，loss指数增长 |
| IMDb | 类似 | 类似 |
| ListOps | 类似 | 类似 |

关键观察：每当训练成功降低 loss 时，$s_A$ 都被驱向负值——模型自发学会稳定化。

### 实验2：稳定初始化 ($s_A = 0$)
- Majority：泛化 gap 跨 T=50 到 T=500 保持稳定，验证长度无关性
- IMDb：gap 在 T>300 后稳定（平均评论长度~300）
- ListOps：一致的泛化 gap

### 关键发现
- **训练隐含驱动稳定化**：即便初始化为不稳定，成功训练总伴随 $s_A \to 0^-$
- **长序列+不稳定 = 训练灾难**：不稳定初始化在长序列上训练失败不是偶然，而是泛化界指数爆炸的必然结果
- **稳定 SSM 的泛化优势**：与 softmax attention 一样实现长度无关泛化，优于 linear attention（线性依赖T）

## 亮点与洞察
- **SSM-Attention 桥接优雅**：将 SSM 展开为注意力形式不仅是技术手段，更揭示了两类模型本质相似性
- **上下界匹配性**：上界和下界都在 $s_A > 0$ 时给出指数增长，差仅 O(T) 因子，界的质量高
- **实践指导明确**：$s_A$ 必须保持负值 → 直接指导 Mamba 初始化和正则化策略
- **Gelfand 公式的巧妙使用**：用频谱半径替代算子范数，比 RNN 理论中的 $\|A\|_2$ 更紧

## 局限性 / 可改进方向
- 目前仅分析单层 SSM block，多层/深度 SSM 的泛化分析是自然扩展
- 上下界之间仍有 $O(T)$ gap，可能通过更精细的覆盖构造缩小
- 实验在相对简单的任务上验证（合成 Majority、IMDb 二分类），更复杂的生成任务需验证
- 未考虑 Mamba-2 的简化 SSM 结构（多头注意力视角），可能有更紧的界

## 相关工作与启发
- **vs Rácz et al. (2024)**：他们的 LTI SSM 界依赖脉冲响应范数，无法推广到非线性选择性 SSM
- **vs Trauger & Tewari (2023)**：他们的 Transformer 长度无关界是本文 $W_B, W_C$ 覆盖的理论基础
- **vs RNN 泛化理论**：RNN 通过有界激活函数避免指数增长，但选择性 SSM 没有这种机制——稳定性是唯一出路
- 对 Mamba 模型的预训练策略有直接启示：初始化 $A_c$ 时应确保 $s_A < 0$

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ SSM 泛化理论的开创性工作，SSM-Attention 桥接分析新颖
- 实验充分度: ⭐⭐⭐⭐ 三个任务验证理论预测，但任务复杂度有限
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨，直觉解释清晰，证明思路透明
- 价值: ⭐⭐⭐⭐⭐ 为 SSM 设计提供了理论指导（保持稳定性），填补了重要理论空白
