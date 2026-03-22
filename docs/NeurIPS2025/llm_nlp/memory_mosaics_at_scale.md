# Memory Mosaics at Scale

**会议**: NeurIPS 2025  
**arXiv**: [2507.03285](https://arxiv.org/abs/2507.03285)  
**代码**: [https://github.com/facebookresearch/MemoryMosaics](https://github.com/facebookresearch/MemoryMosaics)  
**作者**: Jianyu Zhang, Léon Bottou (NYU & FAIR, Meta)
**领域**: LLM 架构设计, 关联存储  
**关键词**: 内存马赛克, 高斯核回归, 上下文学习, 组合性, 大规模扩展

## 一句话总结
Memory Mosaics v2 将关联存储网络扩展至 10B 参数、1T token 训练规模，在新任务学习和上下文学习上显著超越同规模甚至 8T token 训练的 Transformer。

## 研究背景与动机
- 组合能力和上下文学习 (ICL) 一直是机器学习的核心追求，但现有 Transformer 在这两方面的机制仍不透明
- 早期工作通过统计独立性 (ICA)、多环境优化 (IRM/MAML) 等方式尝试，效果有限
- Memory Mosaics (Zhang et al., 2025) 用简单的 key-value 关联存储（无位置编码）替代注意力，在 GPT-2 规模和合成数据上展现了优越的 ICL 能力
- **核心问题**：这些优势能否在大规模真实数据上保持？本文将 Memory Mosaics 扩展到 LLaMA-8B 规模回答这一问题

## 方法详解

### 关联存储基础
- 关联存储是一个存储 key-value 对 $\{(k_1,v_1)\dots(k_n,v_n)\}$ 并根据查询 key 检索 value 的装置
- 存储集合具有**置换不变性**，可视为估计条件概率 $P(V|K)$，检索即计算条件期望
- 通过高斯核回归实现检索：$f(k) = \sum_i \frac{e^{-\beta\|k-k_i\|^2}}{\sum_j e^{-\beta\|k-k_j\|^2}} v_i$
- 当所有 key 向量的 L2 范数相同时，退化为标准 softmax attention（内积形式）

### 与 Transformer 注意力的关键差异
1. **L2 归一化 key + 显式带宽 $\beta$**：控制核回归的偏差-方差权衡
2. **对称 key-query 公式**：key 和 query 使用相同的提取器，无需分别学习 $W_q, W_k$
3. **无位置编码**：key 表示近期过去、value 表示近期未来，单层即可实现感应头 (induction head)

### Memory Mosaics v2 三项架构改进

#### 1. 自适应高斯核带宽
- 原版使用固定 $\beta$，但最优带宽依赖于样本数量 $n$ (偏差-方差权衡)
- v2 采用可学习的自适应带宽：$\beta = \beta_1 n^\alpha + \beta_0$
- 其中 $\beta_0 \geq 0, \beta_1 > 0, 0 < \alpha < 1$ 均为可学习参数
- 直觉：记忆中的 key-value 对越多，带宽越小（$1/\sqrt{\beta}$ 越小），估计越精细

#### 2. 门控时变 key 特征提取器
- 原版使用固定权重的泄漏平均：$\bar{k}_T = \tilde{k}_T + \lambda \bar{k}_{T-1}$，$\lambda$ 固定
- 问题："tom-and-jerry"和"tom---and---jerry"语义相同但得到不同 key 特征
- v2 引入输入依赖的门控机制：
  - $g_T = e^{W_g x_T}$（指数门控，控制当前输入的贡献）
  - $\lambda_T = e^{-|W_\lambda x_T|}$（时变遗忘因子，语义驱动）
  - $\bar{k}_T = g_T \tilde{k}_T + \lambda_T \bar{k}_{T-1}$
- 受 RWKV、Mamba、xLSTM 等循环架构启发，但**仅用于构造 key，关联存储仍保留所有 key-value 对**

#### 3. 三层记忆设计
- **短期记忆**：只存储位置 $t$ 附近 $h=256$ 步内的 key-value 对，处理位置敏感信号
- **长期记忆**：跳过近距 token，只存储位置 $t-m$ 之前的 key-value 对，处理位置不变信号
  - 训练时随机采样 $m \in [64, 256]$，推理时固定 $m=64$
  - 设置 $m < h$ 使长短期记忆有重叠，形成软边界
- **持久记忆**：两层 SwiGLU FFN，存储训练数据中的全局知识（等价于大容量关联存储）
- 多个长/短期记忆的输出拼接后经线性投影 $W_o$ 融合

### 训练配置
| 配置 | Small (LLaMA-1.5B 级) | Large (LLaMA-8B 级) |
|------|---------|---------|
| 层数 | 24 | 32 |
| 隐藏维度 | 2048 | 4096 |
| 注意力头数 | 16 | 32 |
| 训练 token | 200B | 1T |
| 训练上下文 | 4096 → 微调至 32768 | 4096 → 微调至 32768 |

## 三维度评估体系

### 维度一：持久知识存储与检索
- 评估持久记忆（FFN）存储训练数据知识的能力
- 19 个常用语言基准（ARC、PIQA、BoolQ、HellaSwag、MMLU 等）
- **结果**：MM v2 和 Transformer 表现接近（52.2% vs 52.2%），符合预期（共享持久记忆架构）
- **验证方法**：移除长期记忆后 13 个基准几乎不受影响（56.6% vs 56.8%），证明这些任务仅依赖持久知识

### 维度二：新知识存储与检索
- 评估模型在推理时存储和检索新信息的能力
- 使用 Ruler 基准的"多无关文档问答"任务（拼接多篇文章 + 问题）
- 比"大海捞针"难得多——信息熵高，不是简单的精确匹配

| 模型 | 训练 ctx | 4k | 8k | 16k | 32k | 64k |
|------|---------|-----|-----|------|------|------|
| Transformer large | 32k | 51.2 | 48.8 | 44.7 | 41.1 | × |
| MM v2 large | 32k | **58.9** | **55.5** | **54.9** | **53.4** | **46.4** |

- MM v2 在 32k 任务长度上超越 Transformer **12.3%**
- MM v2 训练在 4k 可无微调外推至 32k（Transformer 在 4k→8k 即失败）
- RNN/SSM/滑动窗口等压缩记忆方法在此任务上**结构性失败**——无法在读到问题前存储所有文章

### 维度三：上下文学习 (ICL)
- 使用经典多类分类任务：Banking77 (77类)、Tacred (41类)、GoEmotion (28类)
- 设置语义标签版本和匿名标签版本，后者更能测试真正的新任务学习能力
- **核心发现**：
  - MM v2 随 shot 数增加**持续提升**分类准确率
  - Transformer 反常——shot 数越多性能反而下降
  - MM v2 超越 Transformer **10% 以上**
- 将 Transformer 加上长短期注意力分离机制也无法复制此优势，说明 MM v2 不是简单的 Transformer 变体

## 扩展数据对比：1T MM v2 vs 8T Transformer

| 对比维度 | Transformer 1T | Transformer 8T | MM v2 1T |
|---------|--------------|--------------|----------|
| 新知识存储 (32k) | 41.1% | 46.9% | **53.4%** |
| 语义标签 ICL | 较低 | 接近 MM v2 | **最佳** |
| 匿名标签 ICL | 低 | 更低（退化） | **显著最佳** |

- 8× 训练数据的 Transformer 仍落后 MM v2 (1T) 约 **6.5%**（新知识存储）
- 匿名标签 ICL 上，更多训练数据反而令 Transformer 性能退化——"更多数据"策略彻底失效

## 微调效率
- MM v2 仅用 **1 个 mini-batch** 微调即可获得 22% 的准确率提升
- 2 个 mini-batch 即可达到最优性能
- Transformer 用 800 个 mini-batch 微调仍不如 MM v2 的 1 个 mini-batch

## 计算开销
| 模型 | 参数量 | FLOPs/token |
|------|-------|------------|
| Transformer large | 8.8B | 16.7B |
| MM v2 large | 9.9B | 18.9B |
| MM v2 (去除长期记忆) | 8.3B | 15.6B |

- MM v2 参数和计算略高，但去除长期记忆后反而更轻量
- 10%+ 的性能优势远超 13% 的计算开销增长

## 亮点与洞察
1. **可解释的记忆机制**：显式 key-value 存储使注意力分配透明，attention score 在远距 token 上位置不变（vs Transformer 的位置依赖曲线）
2. **挑战 scaling law 信仰**：8T token 训练的 Transformer 在新任务学习上仍不如 1T token 的 MM v2，说明架构创新比堆数据更重要
3. **上下文长度外推**：无位置编码 + 自适应带宽使 4k 训练的模型可直接外推到 32k，无需微调
4. **极速微调**：1 个 mini-batch 即可适配新领域，实用价值极高
5. **三层记忆分工明确**：短期处理局部模式、长期处理远程检索、持久存储全局知识

## 局限性
- FLOPs/token 略高于同规模 Transformer（18.9B vs 16.7B），长上下文时开销更大
- 长期记忆需存储所有历史 key-value 对，极长上下文时内存占用大
- 尚未探索模糊哈希 (fuzzy hashing) 和层次化记忆等长上下文优化技术
- 仅在 10B 规模验证，是否在 70B/400B 级别仍保持优势尚不清楚

## 相关工作
- Bietti et al. (2023) 分析了 Transformer 感应头需要位置编码和非对称 QK，MM 用对称 key 单层实现
- Olsson et al. (2022) 发现感应头机制是 ICL 的核心，MM 显式构造此机制
- RWKV、Mamba、xLSTM 等记忆压缩方法在多文档 QA 上结构性失败，因为无法完整保留所有信息

## 评分
⭐⭐⭐⭐ (4/5)
