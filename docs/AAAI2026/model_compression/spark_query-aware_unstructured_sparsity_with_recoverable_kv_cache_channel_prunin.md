# SparK: Query-Aware Unstructured Sparsity with Recoverable KV Cache Channel Pruning

**会议**: AAAI 2026  
**arXiv**: [2508.15212](https://arxiv.org/abs/2508.15212)  
**代码**: 待公开  
**领域**: 模型压缩  
**关键词**: KV Cache压缩, 通道剪枝, 非结构化稀疏, 长上下文推理, 注意力机制

## 一句话总结
提出SparK——一种training-free的KV cache通道级非结构化剪枝方法，通过query-aware的saliency评估选择关键通道+recovery机制恢复被剪枝通道的贡献，在80%剪枝率下性能损失<5%，与token eviction方法正交互补，可额外减少30%+ KV cache存储。

## 研究背景与动机
1. **领域现状**：LLM长上下文推理的主要瓶颈是KV cache——100K token在LLaMA3.1-8B中需要>50GB存储，且attention计算量随序列长度二次增长。现有压缩方法主要在三个维度操作：时间轴（token eviction/merging）、空间轴（layer/head sharing）、通道轴（低秩分解/结构化剪枝）。
2. **现有痛点**：
   - **Token eviction**（SnapKV/PyramidKV等）只减少序列长度S，不触及head dimension D
   - **结构化通道剪枝**（ThinK）对所有token应用相同的剪枝mask，假设通道重要性全局一致——但实际观察显示通道重要性是token-dependent的
   - 结构化剪枝在高剪枝率（≥70%）下性能严重退化（ThinK在80%剪枝时性能损失47.6%）
3. **核心矛盾**：通道重要性高度token-specific（CV>1.1），但现有方法用固定mask做structuredpruning，无法捕捉这种动态性
4. **本文要解决什么？** 在KV cache的通道维度做fine-grained、token-aware的非结构化稀疏，同时保持/恢复被剪枝通道的信息
5. **切入角度**：两个关键观察——(1) 通道重要性在不同token间差异巨大，非结构化剪枝远优于结构化；(2) 用小常数替代被剪枝通道（而非直接置零）能大幅减少信息损失
6. **核心idea一句话**：对KV cache做token-wise的非结构化通道剪枝（保留最重要通道）+轻量recovery函数恢复被剪通道贡献

## 方法详解

### 整体框架
SparK在推理的两个阶段工作：
- **Prefill阶段**：计算每个token每个通道的saliency score → 非结构化剪枝，只保留top-T通道 → 存储被剪通道的分布统计信息
- **Decode阶段**：对每个新query，用recovery函数 $\mathcal{F}$ 从缓存的分布中重建被剪通道的值 → 执行标准full attention

### 关键设计

1. **非结构化通道剪枝（Saliency-based Selection）**:
   - 做什么：为每个head的每个token选择最重要的T个通道
   - 核心思路：将通道剪枝形式化为最大化保留通道saliency的选择问题。目标是最小化剪枝后attention score的Frobenius norm误差：$\min_{\mathcal{S}_{i,t}} \|\mathbf{q}_{i,t}\mathbf{k}_{i,t}^\top - (\mathbf{q}_{i,t}\mathcal{S}_{i,t})(\mathbf{k}_{i,t}\mathcal{S}_{i,t})^\top\|_F$。展开后发现当通道间近似不相关时，可简化为独立的通道贡献求和。定义saliency score $w_{i,t}^j = \|q_{i,t}^j\|_2 \|k_{i,t}^j\|_2$，贪心选择top-T个最高分通道。实际中用observation window内的mean query代替per-token query以降低计算量
   - 设计动机：非结构化→每个token有自己的剪枝mask，完美适配token-dependent的通道重要性。saliency度量同时考虑query和key的通道强度，比只看key的方法更准确

2. **Recovery机制**:
   - 做什么：在attention计算时恢复被剪枝通道的贡献
   - 核心思路：不直接丢弃被剪通道，而是用recovery函数 $\mathcal{F}$ 从被剪通道的统计信息（分布参数）中采样近似值。保存被剪通道的均值和方差，在decode时用这些统计量重建通道值并参与attention score计算
   - 设计动机：观察到即使用粗糙的常数（0.01）替代被剪通道也能大幅减少性能损失（平均减少32.4%的accuracy loss）。用分布采样代替常数可以更好地保持attention score的分布特性

3. **Ratio-free变体**:
   - **SparK-g（Group-based）**：将通道分组，每组内独立选top通道，不需要预设全局剪枝率
   - **SparK-p（Top-p）**：类似top-p采样，累积saliency达到阈值p后停止保留通道

### 损失函数 / 训练策略
- **完全Training-free**：无需修改模型参数，即插即用
- 与现有方法**正交互补**：可以叠加在SnapKV/PyramidKV（时间轴）之上进一步压缩
- 也可以和KV cache量化方法并用

## 实验关键数据

### 主实验
LLaMA3.1-8B上LongBench结果（KV-size 128，与SnapKV结合）：

| 方法 | NrtvQA | HotpotQA | SAMSum | TriviaQA | Avg. |
|------|--------|----------|--------|----------|------|
| Vanilla (full KV) | 22.48 | 48.49 | 42.28 | 88.35 | 44.27 |
| SnapKV | 15.29 | 39.92 | 36.64 | 68.64 | 32.38 |
| +ThinK (50%) | 13.60 | 36.24 | 32.80 | 50.73 | 28.20 |
| +ThinK (80%) | 7.60 | 19.98 | 11.22 | 21.36 | **16.97** |
| **+SparK (50%)** | 13.52 | 38.77 | 36.66 | 68.95 | **32.04** |
| **+SparK (80%)** | 13.82 | **40.84** | 35.41 | 57.20 | **31.16** |

SparK在80%剪枝下Avg=31.16 vs ThinK的16.97，差距巨大！

### 消融实验

| 配置 | 效果 | 说明 |
|------|------|------|
| Structured pruning (50%) | -4.2% | 全局固定mask |
| Unstructured pruning (50%) | -1.2% | token-wise mask |
| Unstructured + 常数替代 (80%) | 大幅改善 | SAMSum: 55.7%→12.2% loss |
| Unstructured + Recovery (80%) | <5% degradation | 分布采样恢复最佳 |
| ThinK (80%) | -47.6% | 结构化剪枝在高ratio下崩溃 |
| SparK (80%) | <5% degradation | 非结构化+recovery很鲁棒 |

### 关键发现
- **通道重要性高度token-dependent**：CV均值>1.1，说明同一通道在不同token上的重要性差异巨大——这直接否定了结构化剪枝的核心假设
- **非结构化 vs 结构化的差距随剪枝率增大而剧增**：50%时差3%；80%时差27.4%
- **Recovery机制是高剪枝率下的关键**：80%剪枝下，无recovery性能崩溃，有recovery性能损失<5%
- **与token eviction正交互补**：叠加在SnapKV上可额外减少30%+存储，同时性能几乎不降

## 亮点与洞察
- **"通道轴是被忽视的维度"**这个insight很有价值：绝大多数KV cache压缩工作只关注序列长度S（token eviction），SparK开辟了正交的通道维度压缩方向，可以和现有方法叠加
- **Recovery机制设计精巧**：不只是简单剪枝，而是剪+恢复。观察到常数替代就能大幅减少信息损失是一个反直觉但有力的发现，启发了distribution-based recovery
- **从问题形式化到贪心解的推导很清晰**：将通道剪枝形式化为max-saliency选择问题，利用通道间近似不相关的性质简化为可贪心求解的线性问题

## 局限性 / 可改进方向
- **Observation window的选择**：用window内mean query近似all queries会引入误差，尤其在长上下文中不同位置的query pattern差异大
- **Recovery函数的设计空间还没充分探索**：目前用分布采样，但更复杂的recovery（如小网络预测）可能效果更好
- **Value cache的剪枝只用了简单的norm-based heuristic**：Key cache有详细的理论推导，Value cache的剪枝策略可以进一步优化
- **只在LongBench和RULER上评估**：缺少对reasoning task（math/code）的评估

## 相关工作与启发
- **vs ThinK**: 同样做通道剪枝，但ThinK是structured（所有token共享mask），SparK是unstructured（每个token独立mask）。在80%剪枝下SparK的Avg=31.16远超ThinK的16.97
- **vs SnapKV/PyramidKV**: 它们做时间轴压缩（减序列长度），SparK做通道轴压缩。两者正交，可叠加使用
- **vs KV量化（KIVI等）**: 量化减少每个数值的bit数，SparK减少参与计算的channel数。两者也可互补

## 评分
- 新颖性: ⭐⭐⭐⭐ 非结构化通道剪枝+recovery的组合在KV cache压缩中是新的，开辟了channel维度的压缩方向
- 实验充分度: ⭐⭐⭐⭐ 多个benchmark、多种基线、多模型、详细消融和可视化分析
- 写作质量: ⭐⭐⭐⭐ 数学推导严谨，观察-动机-方法的逻辑链清晰
- 价值: ⭐⭐⭐⭐⭐ 高度实用——training-free、plug-and-play、与现有方法正交互补，直接可用于长上下文LLM推理加速
