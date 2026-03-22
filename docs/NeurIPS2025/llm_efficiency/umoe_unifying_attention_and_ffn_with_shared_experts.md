# UMoE: Unifying Attention and FFN with Shared Experts

**会议**: NeurIPS 2025  
**arXiv**: [2505.07260](https://arxiv.org/abs/2505.07260)  
**代码**: [github.com/ysngki/UMoE](https://github.com/ysngki/UMoE)  
**领域**: LLM效率  
**关键词**: MoE统一架构, 前混合注意力, 专家共享, 注意力-FFN融合, 参数效率

## 一句话总结

通过重新表述多头注意力机制，揭示其与 FFN 共有的"两层矩阵乘法"结构，据此提出 UMoE 统一架构——在注意力和 FFN 层使用相同设计的专家并支持参数共享，在 Base(134M) 和 Large(1.1B) 模型上均优于现有 FFN-MoE 和 Attention-MoE 基线。

## 研究背景与动机

1. **领域现状**：稀疏 MoE 是扩展 LLM 容量的主流范式。目前主要有两条路线：FFN-MoE（Switch Transformer、DeepSeek-MoE）将 FFN 层替换为 MoE，以及 Attention-MoE（MoA、SwitchHead）将注意力层替换为 MoE。两者各自发展，使用不同的专家设计。
2. **现有痛点**：Attention-MoE 的性能普遍不如 FFN-MoE——在相同参数和计算预算下，MoA 和 SwitchHead 都不如 fine-grained FFN-MoE。性能差距来自两个原因：(a) 注意力层和 FFN 层结构不同，导致专家设计也不同；(b) 注意力 MoE 引入稀疏性时需要牺牲原始注意力的表达能力（如 MoA 必须共享 K/V 投影，限制了每个头的独立性）。
3. **核心矛盾**：注意力层看似涉及多重投影和 softmax 非线性，与 FFN 的"两层矩阵乘法"结构截然不同。能否找到某种等价表述让两者统一？
4. **本文要解决什么**：(a) 能否重构注意力使其揭示 FFN-like 的内在结构？(b) 能否用相同的专家设计同时服务于注意力和 FFN 层，实现参数共享？
5. **切入角度**：将多头注意力的 $W_o$ 按头分解后，改变矩阵乘法顺序——先做 token mixing（加权聚合），再做 $W_v W_o$ 投影。这样注意力的"投影部分"就变成了与 FFN 相同的两层结构。
6. **核心 idea**：注意力 = token mixing + FFN-like 专家处理。FFN = 自注意力（恒等 attention matrix）+ FFN-like 专家处理。两者只差一个 token mixing 操作，专家可以共享。

## 方法详解

### 整体框架

UMoE 将 Transformer 层抽象为三个基本组件：**专家**（两层 FFN）、**token mixing 操作**（加权求和）、**路由器**（top-k 选择）。每层包含一个注意力 MoE 模块和一个 FFN MoE 模块，两者共享同一组专家参数。注意力和 FFN 的唯一区别在于：注意力模块先对 token 做加权聚合再送入专家，FFN 模块直接将 token 送入专家。

### 关键设计

1. **前混合注意力（Pre-Mixing Attention）**
   - 做什么：将标准多头注意力等价重写为 $y = \sum_{i=1}^{h}(a_i X)(W_v^i W_o^i)$
   - 核心思路：传统写法是 $o_i = a_i X W_v^i$（先投影再聚合再输出投影）。利用矩阵乘法结合律重排为"先聚合再投影"：$(a_i X)$ 得到上下文化表示，然后 $(W_v^i W_o^i)$ 两层投影等价于一个无激活的 FFN
   - 设计动机：这种重构暴露出注意力中隐藏的 FFN 结构。一旦在两层投影间加入非线性激活，就得到标准 FFN 专家，自然与 FFN-MoE 的专家设计统一
   - 与传统写法的区别：数学上完全等价，但提供了全新的视角——每个注意力头就是一个"对上下文化输入做 FFN 处理"的专家

2. **统一 MoE 架构**
   - 做什么：将注意力和 FFN 层的专家统一为相同结构的两层 MLP（中间维度 $d_v$），通过 top-k 路由器选择专家
   - 核心思路：注意力 MoE 的输出为 $y = \sum_{i \in \mathcal{T}} p_i E_i(a_i X)$，FFN MoE 的输出为 $y = \sum_{i \in \mathcal{T}} p_i E_i(x)$。两者使用完全相同的专家 $E_i$，差别仅在输入——注意力用上下文化输入 $a_i X$，FFN 用原始 token $x$
   - 设计动机：FFN-MoE 可以被视为注意力 MoE 的特例（attention matrix 退化为恒等矩阵，每个 token 只关注自己）。统一后可以在两个模块间共享专家参数，提升参数效率

3. **低秩专家查询投影**
   - 做什么：为每个专家生成独立的 query 向量
   - 核心思路：$q_i = x W_q + x W_a^i W_b^i$，其中第一项跨专家共享，第二项通过低秩矩阵 $W_a^i \in \mathbb{R}^{d \times r}, W_b^i \in \mathbb{R}^{r \times d_k}$ 为每个专家提供特化的 query
   - 设计动机：每个专家需要不同的 attention pattern，但全秩 query 投影参数量过大。低秩分解在保持专家特化的同时控制参数数量，使总参数与现有 MoE 模型可比

4. **参数共享策略**
   - 做什么：注意力和 FFN 模块使用同一组固定专家（fixed experts），但保留独立的路由器
   - 核心思路：实验发现共享固定专家 + 独立路由器是最优配置（PPL 22.82 vs 完全不共享 23.02）
   - 设计动机：共享专家无需增加总参数即可让注意力层也享受 MoE 的扩展优势；独立路由器让两个模块根据各自需求选择不同专家子集

### 损失函数 / 训练策略

- 语言建模交叉熵损失 + Switch Transformer 的负载均衡辅助损失
- Decoder-only Transformer + RoPE，专家实现为两层 MLP（带非线性激活）
- 数据集：FineWeb-Edu 100B（主要）和 Wikitext-103（对比），LLaMA tokenizer（32K）
- Base 模型 12 层 / 768 维 / 134M 参数；Large 模型 24 层 / 2048 维 / 1.1B 参数

## 实验关键数据

### 主实验：PPL 对比（FineWeb-Edu 50B tokens）

| 模型 | 总参数 | FineWeb PPL↓ | Wikitext PPL↓ | MACs |
|------|--------|-------------|-------------|------|
| Dense (Base) | 134M | 25.79 | 30.41 | 525G |
| FFN-MoE | 535M | 21.19 | 27.94 | 530G |
| MoA | 525M | 22.28 | 27.57 | 486G |
| SwitchHead | 533M | 22.91 | 29.47 | 542G |
| UMoE-Att | 547M | 20.81 | 27.45 | 611G |
| **UMoE** | **540M** | **20.44** | **26.67** | 616G |

Large 模型（1.1B dense → 3.6B MoE）：UMoE PPL 15.95 vs FFN-MoE 16.09 vs MoA 16.72，同样最优。

### 消融实验与分析

| 实验 | 关键结论 |
|------|---------|
| 参数共享策略 | 共享固定专家 + 独立路由器最优（22.82）；完全不共享 23.02 |
| 专家分配（总20个） | 全部给注意力 PPL 21.75 > 16:4 分配 22.50 > 4:16 分配 22.82 |
| 激活函数 | 去掉非线性降 1.2-1.6 PPL，但不至于崩溃（token mixing 保留了非线性） |
| Pre-mixing vs Post-mixing | Pre-mixing 显著优于 Post-mixing（上下文化输入更有利于精确检索） |

### 关键发现

- **注意力专家比 FFN 专家更有价值**：当所有专家都分配给注意力层时达到最佳 PPL（21.75），说明注意力层的表达能力更强，FFN 确实是注意力的"特例"
- **计算开销可忽略**：Base 模型约 1.17× 计算开销，Large 模型仅 1.03×——因为专家计算随维度平方增长而 token mixing 仅线性增长
- **专家展现出双重特化**：共享专家在注意力和 FFN 层发展出不同的特化模式（如专家64：注意力中处理标点，FFN 中处理程度副词），说明共享参数可以高效支持多功能
- **零样本性能一致领先**：Base 模型 UMoE 40.06% vs FFN-MoE 39.55%；Large 模型 UMoE 47.58% vs FFN-MoE 47.12%

## 亮点与洞察

- **统一视角的理论优雅**：将多头注意力等价重写为"token mixing + 两层 FFN"，揭示了注意力和 FFN 的本质统一性。这不仅是工程上的简化，更是对 Transformer 内部机制的深刻理解
- **FFN 是注意力的退化形式**：这个洞察非常有力——当 attention matrix 为单位阵时，注意力层退化为 FFN 层。消融实验也证实了注意力专家比 FFN 专家更有价值
- **KV 缓存友好**：前混合注意力只需缓存每个 token 的一对 key + hidden state（而非多头各有一对 K/V），天然适配低缓存推理场景

## 局限性 / 可改进方向

- 前混合注意力不兼容 GQA（已经只有单对 K/V），但可以结合 MLA（Multi-head Latent Attention）做进一步压缩
- token mixing 在小模型上引入约 1.17× 计算开销，虽然大模型上摊薄到 1.03×，但对资源敏感场景仍需考虑
- 专家实现使用无门控的两层 MLP，未用 SwiGLU 等更强变体；作者也承认换用 SwiGLU 可能进一步提升
- 实验最大规模 1.1B dense / 3.8B MoE，7B+ 级别的验证缺失
- 未探索更高效的 token mixing 替代方案（如线性注意力），这是论文明确指出的未来方向

## 相关工作与启发

- **vs MoA**：MoA 将整个注意力头视为专家，用共享 K/V + 独立 Q/O 实现稀疏化。但这限制了注意力的表达——不同头必须看同样的 K/V。UMoE 直接在 hidden state 上做 token mixing，每个专家有自己的低秩 query，保留了更多灵活性。Large 模型上 UMoE PPL 15.95 vs MoA 16.72
- **vs SwitchHead**：SwitchHead 把注意力内的各个投影矩阵视为专家，但需要对 Q/K/V/O 分别建 MoE。UMoE 将 $W_v W_o$ 合并后直接作为 FFN 专家，更简洁且性能更好
- **vs DeepSeek-MoE**：UMoE 的 fine-grained 专家设计受 DeepSeek-MoE 启发，但将其从 FFN 扩展到了注意力层，并实现跨模块共享

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 注意力-FFN 统一视角非常有洞察力，"FFN 是注意力的退化"这个发现令人信服
- 实验充分度: ⭐⭐⭐⭐ 多尺度对比、消融、专家分析、零样本评估都有，但缺少 7B+ 级别验证
- 写作质量: ⭐⭐⭐⭐⭐ 从重构到统一的论证链非常清晰，伪代码和图示直观
- 价值: ⭐⭐⭐⭐ 对 MoE 架构设计有重要影响——未来可能不再需要分别设计 Attention-MoE 和 FFN-MoE
