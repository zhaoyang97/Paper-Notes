# How Does Sequence Modeling Architecture Influence Base Capabilities of Pre-trained Language Models?

**会议**: NeurIPS 2025  
**arXiv**: [2505.18522](https://arxiv.org/abs/2505.18522)  
**代码**: [GitHub](https://github.com/luxinxyz/TSA)  
**领域**: LLM 架构 / 序列建模  
**关键词**: sequence modeling, base capabilities, Transformer, Mamba, RWKV, attention, Top-1 selection

## 一句话总结
通过"限定领域预训练 + OOD 测试"的评估框架揭示 Mamba/RWKV 等 stateful 架构存在基础能力退化，并归纳出关键设计原则——"全序列任意选择能力"（full-sequence visibility + real relation calculation + non-uniform distribution），用极简的 Top-1 Element/Chunk Selection 架构验证该原则可恢复至接近 Transformer 的基础能力。

## 研究背景与动机

1. **领域现状**：Mamba、RWKV、Gated DeltaNet 等 stateful 序列建模架构以线性复杂度替代 Transformer 的自注意力机制，在语言建模和 few-shot 学习中表现与 Transformer 相当或更优，同时效率更高。
2. **现有痛点**：已有研究发现这些架构在检索、复制、关联记忆等专项能力上存在缺陷，但对"基础能力"（OOD 语言建模泛化）的影响尚不清楚——因为常用的混合领域预训练设置让所有架构看起来都差不多。
3. **核心矛盾**：混合领域预训练本质上是 in-distribution 评估，无法暴露架构差异。这导致一个错觉：Mamba 和 Transformer 基础能力相当。但换到 OOD 场景，差异可能显著。
4. **本文要解决什么**：(1) 设计能揭示架构基础能力差异的评估方法；(2) 找出导致 stateful 架构基础能力退化的关键因素；(3) 提出避免退化的架构设计原则。
5. **切入角度**：限定领域预训练（只用 cc+c4 训练）+ 跨领域测试（在 arxiv/github/stack 上测 OOD performance），在训练早期即可暴露架构差异。
6. **核心 idea 一句话**：序列建模架构必须具备"全序列任意选择能力"（能看到全部序列、能计算真实关系、分布非均匀）才能保持基础能力不退化。

## 方法详解

### 整体框架
分三步：(1) 提出限定领域预训练 + OOD 测试评估框架，揭示基础能力差异；(2) 系统消融 Mamba 家族架构和通用序列建模因素，识别关键因素；(3) 将关键因素归纳为设计原则，并用极简 Top-1 Selection 架构验证。

### 关键设计

1. **限定领域预训练评估框架**：
   - 做什么：用受限领域数据训练，在未见领域上评估 OOD 泛化
   - 核心思路：只用 SlimPajama 的 cc+c4 领域训练，在 arxiv/github/stack 上测试。绘制"训练 loss vs OOD test loss"散点图，同一训练 loss 下不同架构的 OOD test loss 差异即为基础能力差异
   - 设计动机：混合领域训练让 test 变成 in-distribution，掩盖差异；限定领域训练让 test 成为 OOD，暴露架构本身的泛化能力

2. **架构因素分析（非决定性因素）**：
   - 做什么：消融 Mamba 的 data-dependent decay、convolution、GroupNorm 和位置编码
   - 核心发现：这些因素只影响收敛速度，不影响基础能力。即便去掉 data-dependent decay 和 conv，OOD 性能不降反略升
   - 设计动机：排除干扰因素，找到真正关键的架构要素

3. **架构因素分析（决定性因素）**：
   - **Full-Sequence Visibility**：滑动窗口越大，基础能力越好；window=256 导致显著退化
   - **Real Relation Calculation**：将 key 替换为随机常量（不计算真实 QK 关系），基础能力大幅退化
   - **Non-Uniform Distribution**：降低 softmax 温度使注意力分布更尖锐（更非均匀），基础能力更好

4. **Top-1 Element/Chunk Selection 架构验证**：
   - 做什么：设计满足三要素的极简架构来验证原则
   - 核心思路：Top-1 Element Selection 直接选注意力分布中概率最高的元素作为输出（用 straight-through trick 训练）。Top-1 Chunk Selection 是其实用化版本——将序列分 chunk，每个 chunk 内选 top-1
   - 设计动机：如果一个如此极端简化的架构（只保留 top-1 选择）仍然能达到 Transformer 的基础能力水平，就强有力地验证了"全序列任意选择能力"是关键

### 训练策略
- 110M 和 1.3B 两种规模，100B tokens
- 序列长度 2K（短序列）和 100K（长序列）
- Chunk size=128

## 实验关键数据

### OOD 语言建模（110M, 100B tokens）
| 架构 | Mixed Domain Test | OOD Test | 说明 |
|------|------------------|----------|------|
| Transformer++ | 最优 | 最优 | 基准 |
| Mamba-1 | ≈Transformer++ | 显著退化 | 混合域掩盖差异 |
| Mamba-2 | ≈Transformer++ | 显著退化 | 同上 |
| RWKV-6/7 | ≈Transformer++ | 中度退化 | 同上 |
| Top-1 Element Selection | 略差 | ≈Transformer++ | 验证原则 |
| Top-1 Chunk Selection | ≈Transformer++ | ≈Transformer++ | 实用版本 |

### 消融：架构因素影响
| 因素 | 对基础能力的影响 |
|------|----------------|
| Data-dependent decay | 无影响（仅加速收敛）|
| Convolution | 无影响（仅加速收敛）|
| Position encoding | 无影响（ALiBi 除外）|
| Full-sequence visibility | **关键** — 窗口越大越好 |
| Real relation calc | **关键** — 去掉后严重退化 |
| Non-uniform distribution | **关键** — 越尖锐越好 |

### 关键发现
- 混合领域预训练 + 标准 test 或 few-shot eval 都无法区分架构差异——这是现有 benchmark 的根本不足
- Mamba 的成功组件（data-dependent decay、conv）对效率有帮助但对基础能力无正面贡献
- Top-1 Element Selection（极端非均匀 + 全序列 + 真实关系）以极简架构达到了 Transformer 级别的 OOD 泛化

## 亮点与洞察
- **评估方法的创新**：限定领域预训练 + OOD 测试是一个简单但有效的架构分析工具，今后所有新架构的评估都应纳入此维度
- **反直觉发现**：Mamba 的标志性设计（data-dependent decay、conv）对基础能力无贡献——真正重要的是注意力机制的三个基本属性
- **极简验证**：用 Top-1 选择（几乎是最简单的"全序列任意选择"实现）就能恢复基础能力，说明该原则不是充分条件而是必要条件

## 局限性 / 可改进方向
- **只评估了语言建模**：其他基础能力（如推理、代码生成）是否也遵循相同原则未验证
- **Top-1 Selection 效率低于 Mamba**：Top-1 Chunk Selection 虽然比 full attention 快，但仍不如 Mamba 高效
- **未探索混合架构**：hybrid attention-SSM 架构（如 Jamba）是否能兼得两者优势未讨论
- **改进方向**：(1) 在保证"全序列任意选择"原则的前提下探索效率优化（如稀疏选择+线性 recurrence 混合）；(2) 将该原则验证扩展到更大规模（7B+）

## 相关工作与启发
- **vs Mamba/RWKV**：这些架构追求效率但牺牲了全序列任意选择能力，本文证明这是基础能力退化的根因
- **vs Linear Attention (Based)**：线性注意力也缺乏非均匀分布和真实关系计算，基础能力同样退化
- **vs Gated DeltaNet**：Delta rule 试图增强表达力但仍受限于 stateful 框架，未完全恢复基础能力
- **启发**：这项工作为"效率 vs 能力"的 trade-off 提供了清晰的理论指导——未来架构设计应以全序列任意选择为底线，在此基础上追求效率

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次系统性揭示 stateful 架构的基础能力退化，并提炼出可操作的设计原则
- 实验充分度: ⭐⭐⭐⭐⭐ 覆盖 11 种架构、110M-1.3B 两种规模、系统消融、Top-1 验证
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑链完整（发现问题→分析因素→提出原则→验证原则），图表清晰
- 价值: ⭐⭐⭐⭐⭐ 对序列建模架构设计有根本性指导意义，评估方法论也有独立价值
