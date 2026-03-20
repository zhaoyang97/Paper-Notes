# Group Representational Position Encoding (GRAPE)

**会议**: ICLR 2026  
**arXiv**: [2512.07805](https://arxiv.org/abs/2512.07805)  
**代码**: [github.com/model-architectures/GRAPE](https://github.com/model-architectures/GRAPE)  
**领域**: llm_efficiency  
**关键词**: 位置编码, 群论, RoPE, ALiBi, Lie群, 旋转编码, 长上下文  

## 一句话总结

提出 GRAPE 框架，基于群作用（group actions）统一了 Transformer 中乘法型（RoPE）和加法型（ALiBi/FoX）两大位置编码家族，证明 RoPE 和 ALiBi 是其精确特例，并提出路径积分加法变体 GRAPE-AP 在下游任务上超越现有方法。

## 研究背景与动机

1. **位置编码碎片化**：现有方法包括绝对编码（sinusoidal/learned）、相对编码（RoPE）、线性偏置（ALiBi）和遗忘机制（FoX），各自独立设计，缺乏统一理论框架
2. **RoPE 的局限性**：RoPE 固定坐标平面和对数均匀频谱，无法实现跨子空间的特征耦合（cross-subspace coupling）和上下文相关的相位弯曲
3. **绝对编码破坏平移等变性**：基于表的相对编码引入窗口依赖的额外开销
4. **缺乏理论保证**：现有方法分散了稳定性、单调距离惩罚、表达力等关键性质，需要统一框架将这些性质整合
5. **长上下文建模需求**：长序列模型需要原理性的位置几何设计空间

## 方法详解

### 整体框架

GRAPE 基于 Lie 群理论，将位置编码统一为群作用 $\mathbf{G}(n) = \exp(n\omega\mathbf{L})$，分为两大家族：

- **Multiplicative GRAPE (GRAPE-M)**：特殊正交群 $\mathrm{SO}(d)$ 中的保范旋转
- **Additive GRAPE (GRAPE-A)**：一般线性群 $\mathrm{GL}$ 中的幂么（unipotent）作用，产生线性偏置

### Multiplicative GRAPE

**核心构造**：用秩-2 反对称生成元 $\mathbf{L} = \mathbf{ab}^\top - \mathbf{ba}^\top \in \mathfrak{so}(d)$ 构造旋转：

$$\mathbf{G}(n) = \exp(n\omega\mathbf{L}) \in \mathrm{SO}(d)$$

**关键性质**：
- **精确相对律**：$\mathbf{G}(n+m) = \mathbf{G}(n)\mathbf{G}(m)$，注意力分数仅依赖偏移量 $j-i$
- **保范性**：$\mathbf{G}(n)^\top\mathbf{G}(n) = \mathbf{I}$
- **Rodrigues 闭式公式**：$\exp(\mathbf{L}) = \mathbf{I} + \frac{\sin s}{s}\mathbf{L} + \frac{1-\cos s}{s^2}\mathbf{L}^2$，$O(d)$ 复杂度，无需显式矩阵化

**多子空间 GRAPE-M**：$d/2$ 个秩-2 生成元分别作用于正交 2D 子空间。当子空间为标准坐标对且频率为对数均匀谱时，精确恢复 RoPE。可学习正交基和非交换混合进一步扩展表达力。

### Additive GRAPE

**核心构造**：通过齐次坐标提升到 $\mathrm{GL}(d+k)$，使用幂零（nilpotent）生成元 $\mathbf{A}$（$\mathbf{A}^2=\mathbf{0}$），产生幂么作用：

$$\mathbf{G}_\mathrm{add}(n) = \exp(n\omega\mathbf{A}) = \mathbf{I} + n\omega\mathbf{A}$$

**精确恢复 ALiBi**：在 $\mathrm{GL}(d+2)$ 中用秩-1 幂零生成元，logit = $\mathbf{q}_i^\top\mathbf{k}_j + (j-i)\beta_h$

**内容门控变体 (GRAPE-A-QK)**：用 softplus 门控的 query/key 依赖斜率：

$$\text{logit} = \mathbf{q}_i^\top\mathbf{k}_j + (j-i)\omega[\text{softplus}(\mathbf{v}^\top\mathbf{q}_i/\sqrt{d}) + \text{softplus}(\mathbf{u}^\top\mathbf{k}_j/\sqrt{d})]$$

**精确恢复 FoX**：逐 token 的遗忘标量 $f_t$ 对应 $\omega_t = \log f_t$，累积偏置与 FoX 的遗忘偏置 $D_{ij}$ 一致。

### Path-Integral Additive GRAPE (GRAPE-AP)

在 GRAPE-A 基础上引入路径积分偏置，每一步的边势函数为：

$$\psi_h(t,\ell) = \alpha_h \cdot g\left(\frac{1}{d}\langle\mathbf{p}_{t,h},\, \mathbf{R}_\ell\mathbf{p}_{\ell,h}\rangle\right) \leq 0$$

路径积分偏置 $b_h(t,j) = \sum_{\ell=j+1}^{t}\psi_h(t,\ell)$，可与乘法型 GRAPE 组合使用，支持因果约束和流式推理。

## 实验

### 实验设置

- 基于 nanoGPT / Llama 架构，仅替换位置编码
- 数据集：FineWeb-Edu 100B（取 50B token 训练）
- 模型规模：Medium (350M, 24层8头) / Large (770M, 36层10头)
- 上下文长度 4096，batch size 480
- Baseline：RoPE, ALiBi, FoX

### 主实验 (Medium 350M, 0-shot, 7任务平均)

| 方法 | ARC-E | ARC-C | HellaSwag | PIQA | SciQ | **Avg.** |
|------|-------|-------|-----------|------|------|----------|
| RoPE | 56.36 | 30.38 | 44.65 | 68.77 | 74.40 | 51.73 |
| ALiBi | 58.21 | 29.78 | 45.38 | 70.08 | 78.50 | 52.87 |
| FoX | 58.38 | 30.89 | 45.80 | 69.37 | 78.40 | 52.96 |
| GRAPE-A-QK | 57.95 | **32.00** | 45.77 | 69.37 | 79.00 | 53.00 |
| **GRAPE-AP** | **59.26** | 31.31 | 45.42 | 68.17 | **79.70** | **53.25** |
| GRAPE-AP+KV-shift | 57.32 | 30.55 | **46.18** | 69.10 | 79.60 | **53.46** |

### 主实验 (Large 770M, 0-shot, 7任务平均)

| 方法 | ARC-E | ARC-C | HellaSwag | PIQA | SciQ | **Avg.** |
|------|-------|-------|-----------|------|------|----------|
| RoPE | 62.63 | 32.76 | 51.01 | 71.33 | 80.50 | 55.76 |
| ALiBi | 62.67 | 34.39 | 51.33 | 71.11 | 82.70 | 56.44 |
| FoX | 61.07 | 33.11 | 51.85 | 71.27 | 83.70 | 56.30 |
| **GRAPE-AP** | **63.89** | 34.22 | 51.52 | **71.98** | **84.40** | **56.91** |
| FoX+KV-shift | 63.55 | 33.96 | **52.72** | 71.71 | 83.20 | 57.09 |
| GRAPE-AP+KV-shift | 63.72 | 33.11 | 52.29 | 71.65 | 83.50 | 56.86 |

### 关键发现

1. **GRAPE-AP 在无 KV-shift 条件下全面最优**：350M Avg. 53.25 > FoX 52.96 > RoPE 51.73；770M Avg. 56.91 > ALiBi 56.44
2. **训练稳定性优势**：RoPE 在 770M 训练中出现不稳定（loss spike），GRAPE 保持稳定改善
3. **乘法型 GRAPE-M 与 RoPE 持平**：验证了理论等价性，GRAPE-M 本身未显著超越 RoPE
4. **加法型是核心增益来源**：GRAPE-A 和 GRAPE-AP 系列一致优于纯乘法方法
5. **KV-shift 与 GRAPE-AP 互补**：加入 KV-shift 后 350M 进一步提升至 53.46

## 亮点

- **优雅的理论统一**：用 Lie 群框架将看似不相关的 RoPE、ALiBi、FoX 统一为同一数学对象的特例，给出严格证明
- **实用性强**：Rodrigues 闭式公式使得计算复杂度与 RoPE 一致（$O(d)$），流式推理/KV-cache 完全兼容
- **设计空间可扩展**：框架自然给出可学习正交基、内容门控斜率、路径积分偏置等扩展方向
- **数学表述严谨**：群论视角为位置编码提供了清晰的几何直觉（旋转平面、幂么平移）

## 局限性

- **实验规模有限**：仅在 350M/770M 模型上验证，缺少 >1B 大模型实验；训练仅 50B token
- **GRAPE-M 未显著超越 RoPE**：乘法型的理论优势（可学子空间、非交换混合）在实验中未体现明显增益
- **长上下文评估缺失**：训练仅用 4096 上下文，未测试长上下文外推能力（这恰是 ALiBi/RoPE 的关键差异场景）
- **路径积分 GRAPE-AP 计算开销未充分分析**：边势函数需要逐步计算内积，实际推理延迟未报告
- **下游任务覆盖有限**：仅做 0-shot LM evaluation，缺少生成质量、微调后的评估

## 相关工作

- **RoPE** (Su et al., 2021): GRAPE-M 的精确特例（标准坐标对 + 对数均匀谱）
- **ALiBi** (Press et al., 2021): GRAPE-A 在 $\mathrm{GL}(d+2)$ 中的精确特例
- **Forgetting Transformer (FoX)** (Lin et al., 2025): 证明为 GRAPE-A 的路径依赖形式
- **PaTH Attention** (Yang et al., 2025): 论文分析其为收缩性的、近奇异的，可能损害长上下文建模
- **NoPE / 无位置编码**: 未在框架中讨论

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 群论统一视角非常优雅，RoPE/ALiBi/FoX 的精确恢复证明是亮点
- 实验充分度: ⭐⭐⭐ — 模型规模偏小，缺少长上下文和大模型验证
- 写作质量: ⭐⭐⭐⭐ — 数学推导清晰严谨，但符号较多，门槛偏高
- 综合价值: ⭐⭐⭐⭐ — 理论贡献显著，为位置编码设计提供了统一原则性框架
