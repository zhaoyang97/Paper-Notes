# Beyond Higher Rank: Token-wise Input-Output Projections for Efficient Low-Rank Adaptation

**会议**: NeurIPS 2025  
**arXiv**: [2510.23123](https://arxiv.org/abs/2510.23123)  
**代码**: https://github.com/Leopold1423/toplora-neurips25  
**领域**: 参数高效微调  
**关键词**: LoRA, 低秩适配, Token级自适应, 输入输出投影, PEFT

## 一句话总结
TopLoRA 从输入-输出投影角度分析 LoRA 的表达能力，发现所有 token 共享同一投影矩阵是关键瓶颈，提出通过可学习的 token 级对角矩阵 $\Sigma_X$ 动态调整 LoRA 权重（$\Delta W_X = B\Sigma_X A$），在不增加秩的前提下实现细粒度适配，跨任务一致优于 LoRA 2-3%。

## 研究背景与动机

1. **领域现状**：LoRA 通过低秩矩阵 $\Delta W = BA$ 参数高效微调大模型。现有改进主要关注提高秩——HiRA（Hadamard 乘积）、MELoRA（mini-ensemble 堆叠）、MoELoRA（专家混合）。
2. **现有痛点**：LoRA 中所有 token 共享同一个 $\Delta W = BA$，即同一个输入-输出投影矩阵 $P = R_B L_A$。但不同 token 语义差异大，同一投影方向对不同 token 可能代表完全不同的信息，需要不同的处理方式。
3. **核心矛盾**：提高秩增加输入/输出空间维度但参数量线性增长；共享投影的表达能力瓶颈不随秩增加而消除。即使秩很高，所有 token 仍共享同一映射关系。
4. **本文要解决什么？** 在不增加 LoRA 秩的前提下，为每个 token 学习不同的输入-输出投影。
5. **切入角度**：将 LoRA 通过 QR/LQ 分解为三个组件——输入空间 $Q_A$、输出空间 $Q_B$、投影矩阵 $P = R_B L_A$。秩决定空间维度，$P$ 决定映射关系。投影应该 token-specific。
6. **核心 idea 一句话**：用轻量投影网络从 token $X$ 生成对角矩阵 $\Sigma_X$，修改投影 $P \to P_X = R_B \Sigma_X L_A$，实现 token 级自适应的输入-输出映射。

## 方法详解

### 整体框架
原始 LoRA：$Y = (W + BA)X$。TopLoRA：$Y = (W + B\Sigma_X A)X$，其中 $\Sigma_X = \text{Diag}(\text{Exp}(\text{RMSNorm}(\Theta X)))$ 是从输入 token $X$ 动态生成的 $r \times r$ 对角矩阵，$\Theta \in \mathbb{R}^{r \times n}$ 是可学习的投影参数。

### 关键设计

1. **Token-wise 对角矩阵生成**:
   - 做什么：为每个输入 token 生成不同的 $r$ 维对角缩放矩阵
   - 核心思路：$\Sigma_X = \text{Diag}(\text{Exp}(\text{RMSNorm}(\Theta X)))$。$\Theta X$ 将 token 投影到 $r$ 维空间 → RMSNorm 归一化消除量级影响 → Exp 转为正值缩放因子
   - 设计动机：RMSNorm 防止 $\Sigma_X$ 受 token 量级和 $\Theta$ 大小影响，增大不同 token 的 $\Sigma_X$ 差异；Exp 防止近零值导致信息丢失，确保即使微小的归一化差异也能被放大

2. **与标准 LoRA 的关系**:
   - 做什么：TopLoRA 输出可分解为 LoRA 基础项 + token 自适应修正项
   - 核心思路：$\Delta W_X X = BAX + B(\Sigma_X - I)AX$。第一项 $BAX$ 是标准 LoRA 输出（全局模式），第二项 $B(\Sigma_X - I)AX$ 是 token 特定的修正
   - 设计动机：当 $\Sigma_X = I$ 时退化为标准 LoRA，保持了与 LoRA 的兼容性

3. **Kaiming 初始化**:
   - 做什么：$\Theta$ 使用 Kaiming 初始化而非零初始化
   - 核心思路：与 $A$ 矩阵保持一致的初始化策略，确保使用统一学习率时训练稳定
   - 设计动机：零初始化会导致初始 $\Sigma_X$ 全为 1（退化为 LoRA），Kaiming 初始化提供有意义的初始差异

### 损失函数 / 训练策略
- 直接使用下游任务损失，与 LoRA 训练流程完全一致
- AdamW 优化器，LoRA dropout 0.05
- 参数量分析：额外引入 $\Theta \in \mathbb{R}^{r \times n}$，约为 LoRA 参数量的 0.5×

## 实验关键数据

### 主实验

| 任务 | 模型 | LoRA-r8 | LoRA-r32 | TopLoRA-r8 | 提升 |
|------|------|---------|----------|-----------|------|
| GLUE (NLU) | RoBERTa-Base | 82.55% | 84.19% | **84.14%** | +1.59% |
| GLUE (NLU) | RoBERTa-Large | 87.06% | 87.75% | **87.64%** | +0.58% |
| 数学推理 | Gemma-7B | 71.44% | 72.22% | **73.11%** | +1.67% |
| 数学推理 | LLaMA-3-8B | 64.06% | 64.48% | **66.36%** | +2.30% |
| 常识推理 | Gemma-7B | 80.22% | 80.43% | **81.10%** | +0.88% |
| 图文理解 | BLIP-2 COCO | B4=40.7 | — | B4=**43.3** | +2.6 |

### 消融实验

| 变体 | GLUE Avg | 说明 |
|------|---------|------|
| TopLoRA (full) | **84.14%** | 完整模型 |
| w/o RMSNorm | 83.72% | RMSNorm 增大 token 间差异 |
| w/o Exp | 83.85% | Exp 防止信息丢失 |
| 零初始化 $\Theta$ | 83.51% | 初始退化为 LoRA |
| 固定 $\Sigma_X = I$ (=LoRA) | 82.55% | 无 token 自适应 |

### 关键发现
- TopLoRA-r8 在 GLUE 上超过 LoRA-r32（84.14 vs 84.19），使用约 1.5× LoRA-r8 的参数量 vs 4× 参数量
- 跨模型（RoBERTa/Gemma/LLaMA/BLIP-2）和跨任务（NLU/NLG/Vision-Language）一致有效
- 提升在数学推理和视觉语言任务上更显著（2-3%），在 NLU 上也有 1%+ 改善
- 优于 DoRA、MELoRA、HydraLoRA 等 LoRA 变体
- TopLoRA 可作为插件应用于 MoELoRA 框架中进一步提升

## 亮点与洞察
- **输入-输出投影的新视角**：将 LoRA 分解为输入空间 + 输出空间 + 投影矩阵三个组件，发现投影矩阵的 token 共享是被忽视的瓶颈。这个分析框架本身很有启发
- **方法极其简单高效**：只增加一个线性投影 + RMSNorm + Exp，几乎没有架构复杂度，但跨任务一致有效
- **不增加秩但增加表达能力**：与"提高秩 = 提高表达力"的主流思路形成对比，说明秩不是唯一的表达能力维度

## 局限性 / 可改进方向
- 在已经使用很高秩的场景下优势减小
- 投影网络 $\Theta$ 的额外参数和计算开销（虽小但非零）
- 对角矩阵限制了 $\Sigma_X$ 的表达力——全矩阵 $\Sigma_X$ 可能更强但参数量太大
- 未探索 $\Sigma_X$ 在不同层的行为差异

## 相关工作与启发
- **vs LoRA**: LoRA 共享投影，TopLoRA token 级自适应；TopLoRA-r8 ≈ LoRA-r32 效果
- **vs DoRA**: DoRA 分解为方向 + 大小，TopLoRA 分解为空间 + 投影，角度不同
- **vs HiRA/MELoRA**: 这些方法提高秩，TopLoRA 不改变秩而优化投影，两者正交可组合
- **vs MoELoRA**: MoELoRA 用多个 LoRA 专家实现 token 自适应但参数多，TopLoRA 更参数高效

## 评分
- 新颖性: ⭐⭐⭐⭐ 输入-输出投影分析视角新颖，方法简洁优雅
- 实验充分度: ⭐⭐⭐⭐⭐ 三类任务 × 多个模型 × 充分消融 × 与多个 baseline 对比
- 写作质量: ⭐⭐⭐⭐ 动机推导清晰，公式推导严谨
- 价值: ⭐⭐⭐⭐ 对 LoRA 家族方法的理论理解和实践改进都有贡献
- 价值: ⭐⭐⭐⭐ 为 LoRA 改进提供了新的维度（投影多样性 vs 秩），实用性强