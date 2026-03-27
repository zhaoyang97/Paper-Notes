# DiaBlo: Diagonal Blocks Are Sufficient For Finetuning

**会议**: ICLR2026  
**arXiv**: [2506.03230](https://arxiv.org/abs/2506.03230)  
**代码**: 待确认  
**领域**: 模型压缩  
**关键词**: PEFT, diagonal blocks, LoRA alternative, LLM fine-tuning, parameter efficiency

## 一句话总结
提出 DiaBlo，仅微调权重矩阵的对角块作为参数高效微调方法：避免了 LoRA 低秩矩阵乘积的优化难题，zero 初始化即可稳定收敛，GPU 友好的 batched 矩阵乘法实现，理论证明在参数预算相同时表达力严格优于 LoRA，在常识推理/算术推理/代码生成/安全对齐上全面优于 LoRA 及其变体。

## 研究背景与动机

1. **领域现状**：LoRA 及其变体（DoRA、PiSSA、MiLoRA、LoRA-GA）是主流 PEFT 方法，通过低秩矩阵乘积 $\mathbf{AB}$ 适配预训练权重。
2. **现有痛点**：(a) LoRA 的 $\mathbf{AB}$ 乘积是非凸优化，梯度依赖于 $\mathbf{A}$ 和 $\mathbf{B}$ 的值→对初始化敏感、收敛不稳定；(b) 各种变体的核心是设计更好的初始化/优化策略→增加算法复杂度；(c) 稀疏方法多为非结构化→硬件不友好。
3. **核心洞察**：权重矩阵的对角块更新等价于全量微调在对应子空间的行为——梯度 $\mathbf{g}_{\mathbf{D}_i} = \mathbf{g}_{\mathbf{W}_{ii}}$，不经过矩阵乘积，零初始化不会梯度消失。
4. **核心idea一句话**：不做低秩分解，直接更新 $N$ 个对角块 $\mathbf{D}_i \in \mathbb{R}^{d_1 \times d_2}$，用 torch.einsum 做 batched matmul。

## 方法详解

### 关键设计

1. **对角块适配**:
   - 权重分为 $N \times N$ 块矩阵，只更新对角块 $\mathbf{W}_{11}, \ldots, \mathbf{W}_{NN}$
   - 存储为张量 $\mathcal{D} \in \mathbb{R}^{N \times d_1 \times d_2}$，前向/反向都用 batched matmul
   - 初始化：全零（LoRA 需要精心设计初始化）

2. **理论保证**:
   - 线性问题：DiaBlo 收敛到全量微调的全局最优，且表达力严格优于同参数量 LoRA
   - 非线性问题：在激活和梯度低秩条件下，收敛到全量微调的驻点
   - 梯度稳定性：$\mathbf{g}_{\mathbf{D}_i} = \mathbf{X}_i^\top \mathbf{g}_{\mathbf{Y}_i}$——不经过 $\mathbf{A}, \mathbf{B}$，无梯度消失/不稳定

3. **实现效率**:
   - 参数量与 LoRA rank $r$ 的对应：$N$ 个块 × $d_1 d_2$ 参数，$N=32-128$ 常用
   - PyTorch 一行实现：`torch.einsum("bNd1,Nd1d2->bNd2", X, D)`

## 实验关键数据

### 主实验（LLaMA2-7B）

| 方法 | 常识推理 Avg | 算术推理 Avg | 代码生成 | 安全对齐 |
|------|------------|------------|---------|---------|
| LoRA | 基线 | 基线 | 基线 | 基线 |
| DoRA | +小 | +小 | ~ | ~ |
| PiSSA | +中 | +中 | ~ | ~ |
| **DiaBlo** | **最优** | **最优** | **最优** | **最优** |

### 消融（量化设置下的鲁棒性）

| 量化精度 | LoRA 算术 | DiaBlo 算术 |
|---------|----------|------------|
| FP16 | 中等 | 高 |
| 4-bit | 显著下降 | 保持高 |
| 2-bit | 严重退化 | **仍然稳定** |

### 关键发现
- DiaBlo 在所有任务/精度设置下一致优于 LoRA 系列
- 2-bit 量化下优势最显著——LoRA 严重退化而 DiaBlo 保持稳定
- 梯度方差始终低于 LoRA（更稳定的训练）

## 亮点与洞察
- **极度简单但出人意料地有效**：zero init + 对角块更新→无需任何 trick
- **理论严格**：线性问题下严格优于 LoRA（不是近似优于）
- **量化友好**：对角块结构在低比特下比低秩乘积更鲁棒
- **LoRA 的优化困难是根本性的**：低秩矩阵乘积本身就是非凸难题，DiaBlo 完全绕过

## 局限性 / 可改进方向
- 对角块假设不考虑跨块信息——可能在需要全秩更新的任务上受限
- $N$ 的选择需要匹配硬件和参数预算
- 未与 adapter-based 方法系统对比

## 相关工作与启发
- **vs LoRA**：LoRA 用 $\mathbf{AB}$ 低秩近似 $\Delta \mathbf{W}$。DiaBlo 用结构化稀疏（对角块）—— 更稳定、更有表达力
- **vs S²FT**：也是结构化稀疏微调。DiaBlo 的对角块更规整→GPU 效率更高
- **vs QLoRA**：量化+LoRA。DiaBlo+量化的组合可能更好（2-bit 下优势明显）

## 评分
- 新颖性: ⭐⭐⭐⭐ 思路极简但有效，理论支撑充分
- 实验充分度: ⭐⭐⭐⭐⭐ 多任务、多精度、多模型全面验证
- 写作质量: ⭐⭐⭐⭐ 理论和实验叙述清晰，图表直观
- 价值: ⭐⭐⭐⭐⭐ 可能替代 LoRA 成为新的 PEFT 默认选择
