# DreamOn: Diffusion Language Models For Code Infilling Beyond Fixed-size Canvas

**会议**: ICLR 2026  
**arXiv**: [2602.01326](https://arxiv.org/abs/2602.01326)  
**代码**: https://github.com/DreamLM/DreamOn  
**领域**: 文本生成  
**关键词**: diffusion language model, code infilling, variable-length generation, discrete diffusion, DLM

## 一句话总结
DreamOn 通过引入 [expand] 和 [delete] 两个特殊状态解决了扩散语言模型（DLM）的固定长度生成限制，无需架构修改即可实现变长代码填充，在 HumanEval-Infilling 上比扩散基线平均提升 26.4%，达到与 SOTA 自回归模型持平的水平。

## 研究背景与动机

1. **领域现状**：扩散语言模型（DLM，如 LLaDA、Dream、DiffuCoder）通过迭代去噪实现灵活的任意顺序生成，天然适合代码填充（infilling）——在给定前缀和后缀之间生成缺失代码。自回归模型做填充需要笨拙的 FIM（Fill-in-the-Middle）hack，破坏自然上下文结构。

2. **现有痛点**：DLM 的致命瓶颈——**必须预先指定固定长度的 mask 序列**。输入输出必须等长，模型无法动态决定生成长度。当预设 mask 长度与真实补全长度不匹配时：mask 太少→补全不完整；mask 太多→生成多余错误代码。实测平均性能下降 38%！

3. **核心矛盾**：DLM 的双向注意力天然适合 infilling，但固定长度约束完全抵消了这一优势。如何在保持 DLM 架构不变的前提下让它学会动态调整输出长度？

4. **本文要解决什么？** 让 DLM 在生成过程中自主决定扩展或收缩序列长度。

5. **切入角度**：在扩散过程中引入两个特殊状态作为长度控制信号——预测 [expand] 表示"这里需要更多空间"，预测 [delete] 表示"这里多余了"。

6. **核心 idea 一句话**：将长度控制编码为扩散词表中的两个特殊 token（[expand]→分裂为两个 [mask]，[delete]→删除该位置），通过数据增强训练模型预测这些状态，实现零架构改动的变长生成。

## 方法详解

### 整体框架
在标准 masked diffusion 基础上，引入增强扩散过程：训练时将原始序列 $\mathbf{x}_0$ 通过 span merging 和 delete insertion 变为增强序列 $\mathbf{z}_0$（包含 [expand] 和 [delete]），然后做标准 masked diffusion 训练。推理时模型在 denoise 过程中自主预测 [expand]（当前位置扩展为两个 [mask]）或 [delete]（删除当前位置），从而动态调整输出长度。

### 关键设计

1. **[expand] 和 [delete] 特殊状态**:
   - 做什么：作为扩散词表的额外 token，编码长度控制语义
   - 核心规则：预测 [expand] → 将该位置替换为两个 [mask]（序列变长）；预测 [delete] → 删除该位置（序列变短）
   - 设计动机：这两个操作是图灵完备的长度调整原语——任意长度的序列都可以通过多次 expand 和 delete 达到目标长度

2. **增强数据构造（训练时）**:
   - 做什么：从原始序列 $\mathbf{x}_0$ 构造包含 [expand] 和 [delete] 的增强序列 $\mathbf{z}_0$
   - 核心思路：[expand] 通过 span merging 产生——将连续的 [mask] token 合并为一个 [expand]；[delete] 通过在末尾追加 0-64 个随机 delete token 产生
   - 设计动机：merge scheduling 控制 [expand] 比例（静态 + 动态逆调度器 1:1 混合），避免特殊 token 过多影响原始性能

3. **加权训练损失**:
   - 做什么：平衡 [delete] 的过度贡献
   - 核心思路：[delete] 每个对应一个 [mask]，[expand] 多个 [mask] 合并为一个，导致 [delete] 在损失中占比过大。引入权重 $w_n$ 使 [delete] 的总贡献等价于单个 [mask]
   - 设计动机：校准损失分布，防止模型过度学习 delete 而忽略 expand

4. **Broadcasting Deletion（推理优化）**:
   - 做什么：检测到 [delete] 时，同时删除其右侧所有连续 [mask] token
   - 设计动机：避免模型通过大量前向传递逐个 delete 来调整长度，显著加速推理

### 损失函数 / 训练策略
- 标准 masked diffusion 的加权交叉熵损失，扩展词表包含 [expand] 和 [delete]
- 基于 DreamCoder-7B/DiffuCoder-7B 微调，仅 110K Python 代码对，10 个 epoch，8×H800 约 5 小时
- **训练计算量仅为预训练的 0.15%**——极其轻量

## 实验关键数据

### 主实验

| 模型 | HE-Single (Pass@1) | HE-Multi (Pass@1) | SantaCoder (EM) |
|------|--------------------|--------------------|-----------------|
| Qwen2.5-Coder-7B (AR) | 92.6 | 58.7 | 79.8 |
| Seed-Coder-8B (AR) | 89.7 | 59.3 | 77.2 |
| DreamCoder-7B (DLM) | 55.5 | 43.2 | 59.3 |
| **DreamCoder + DreamOn** | **92.1** (+36.6) | **63.8** (+20.6) | **79.0** (+19.7) |
| DiffuCoder + DreamOn | 92.2 (+38.5) | 63.1 (+18.1) | 77.4 (+19.4) |

DreamOn 使 DLM 在 multi-line infilling 上甚至**超越** SOTA 自回归模型！

### 消融实验

| 配置 | Single-line Avg | Multi-line Avg | 说明 |
|------|----------------|----------------|------|
| DreamCoder-7B 基线 | 55.3 | 26.0 | 固定 mask=64 |
| + DreamOn (完整) | **90.8** | **57.1** | 两个状态都有 |
| w/o Delete | 67.4 | 39.2 | 无法缩短，mask 多时退化 |
| w/o Expand | 73.4 | 43.2 | 无法扩展，mask 少时退化 |
| Oracle（真实长度） | 91.6 | 69.0 | 上界参考 |

### 关键发现
- **DreamOn 几乎达到 Oracle 水平**（单行：90.8 vs 91.6），证明长度适配非常有效
- **Expand 和 Delete 互补且缺一不可**：去掉 Delete 在长 mask 时退化严重；去掉 Expand 在短 mask 时退化严重
- **对初始 mask 长度高度鲁棒**：初始 mask 从 4 到 64 变化时，DreamOn 性能几乎不变（88.7-92.1），而基线从 24.9 变到 55.5
- **仅 0.15% 预训练计算量**的微调就实现了巨大提升，证明方法的轻量性
- Broadcasting Deletion 显著减少推理步数，不损失精度

## 亮点与洞察
- **优雅的最小化设计**：仅在词表中添加两个特殊 token + 数据增强，零架构改动。这种"用词表操作编码结构操作"的思路非常巧妙
- **DLM infilling 首次达到 AR 水平**：证明了 DLM 的天然 infilling 优势一旦解决长度问题就能完全释放
- **可迁移性**：DreamOn 在 Dream-7B、DiffuCoder-7B、DreamCoder-7B 三个不同 DLM 上都有效，说明是 model-agnostic 的增强
- **Broadcasting Deletion 的洞察**：推理时如果模型预测 delete，其后续全 mask 区域大概率也是多余的——这个简单启发式大幅加速推理

## 局限性 / 可改进方向
- 每次 [expand] 只能扩展为 2 个 [mask]，大幅度长度扩展需要多步前向传递
- 目前仅在代码 infilling 验证，自然语言 infilling（如文本编辑、故事续写）的效果待探索
- 最大扩展长度 $L_{max}=128$ 是人为限制，更长的代码填充可能受限
- 训练数据仅 110K Python 代码，多语言代码 infilling 的泛化性待验证

## 相关工作与启发
- **vs 自回归 FIM（Qwen2.5-Coder 等）**：FIM 需要在训练时重排序列（把 middle 挪到末尾），推理时需要特殊 prompt。DLM + DreamOn 天然支持双向上下文的 infilling，更自然
- **vs 固定长度 DLM（LLaDA、Dream）**：DreamOn 解决了 DLM 最关键的实用性瓶颈，使 DLM 首次在 infilling 上具有竞争力
- **vs Edit-based 方法**：一些方法通过显式的 insert/delete 操作做文本编辑，DreamOn 将这些操作融入扩散过程中，更优雅且端到端可训练

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 用两个特殊 token 解决 DLM 的根本性限制，思路简洁有力
- 实验充分度: ⭐⭐⭐⭐ 三个 DLM 基线 + 两个 benchmark + 详细消融，但仅限代码任务
- 写作质量: ⭐⭐⭐⭐⭐ motivation 图例直观，算法描述清晰，消融分析透彻
- 价值: ⭐⭐⭐⭐⭐ 解决了 DLM 领域的根本性瓶颈，使 DLM 在 infilling 场景上首次可与 AR 模型竞争
