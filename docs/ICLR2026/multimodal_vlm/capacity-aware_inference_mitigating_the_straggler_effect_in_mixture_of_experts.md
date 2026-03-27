# Capacity-Aware Inference: Mitigating the Straggler Effect in Mixture of Experts

**会议**: ICLR2026  
**arXiv**: [2503.05066](https://arxiv.org/abs/2503.05066)  
**代码**: [https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE](https://github.com/CASE-Lab-UMD/Capacity-Aware-MoE)  
**领域**: LLM效率  
**关键词**: Mixture of Experts, inference efficiency, straggler effect, token drop, expert parallelism

## 一句话总结
针对 MoE 推理时因 token 分配不均导致的 Straggler Effect（最重负载专家决定整体延迟），提出 Capacity-Aware Token Drop（丢弃过载专家的低分 token）和 Expanded Drop（将溢出 token 重路由到本地低负载专家），在 Mixtral-8×7B 上实现 1.85× 加速且性能提升 0.2%。

## 研究背景与动机

1. **领域现状**：MoE 是扩展 LLM 的关键架构——通过稀疏激活多个专家平衡性能与效率。在 expert parallelism 下，专家分布在多个 GPU 上并行计算。
2. **现有痛点**：训练时的负载均衡损失（auxiliary balance loss）无法保证推理时的均衡。实测表明推理时最重负载专家可承担超过平均值 7 倍的 token——低负载专家算完后只能等待高负载专家同步，造成严重延迟。
3. **核心矛盾**：这就是 **Straggler Effect**——MoE 层的延迟由最重负载专家决定（$L \propto \max(\{N_i\})$），而非平均负载。现有方案（如 DeepSeek-V3 复制高负载专家）需要额外GPU资源。
4. **本文要解决什么？** 在推理时不增加 GPU 资源的前提下，通过智能 token 调度缓解 Straggler Effect，提高推理速度。
5. **切入角度**：两个互补策略——对高负载专家设容量上限丢弃低重要性 token；对低负载专家扩展候选集接收溢出 token。
6. **核心idea一句话**：用 gating score 作为重要性指标限制高负载专家的 token 数，同时将溢出 token 重路由到同一 GPU 上的低负载专家，实现负载均衡和速度提升。

## 方法详解

### 整体框架
MoE 推理时，router 为每个 token 选择 top-k 专家。Capacity-Aware Inference 在 router 之后加入两步处理：(1) Token Drop 阶段——检查每个专家的负载是否超过容量限制 $C = \gamma \bar{N}$，超过的用 gating score 排序丢弃低分 token；(2) Expanded Drop 阶段——被丢弃的 token 扩展候选集到同一 GPU 上的其他低负载专家，利用空闲容量。整个过程在 All-to-All 通信之前完成，零额外通信开销。

### 关键设计

1. **Capacity-Aware Token Drop（处理高负载专家）**:
   - 做什么：对超过容量限制的专家，丢弃多余的低重要性 token
   - 核心思路：定义容量因子 $\gamma$，每个专家最多处理 $C = \gamma \bar{N}$ 个 token（$\bar{N} = tk/n$ 为期望均值）。当专家 $j$ 的负载 $N_j > C$ 时，用评分函数 $\mathcal{S}$ 对该专家的所有 token 评分，保留 top-$C$ 个，丢弃其余。评分函数比较了 Order/Reverse Order/Random/Score 四种，Score（gating 分数）远优于其他
   - 设计动机：虽然丢弃 token 似乎会损害性能，但过载专家中绝大多数 token 是冗余的——丢弃 12% 的溢出 token 就能带来 85% 的加速（Mixtral）。gating score 自然反映 token-expert 匹配度，用它选择保留哪些 token 最合理

2. **Capacity-Aware Expanded Drop（利用低负载专家）**:
   - 做什么：将被 Token Drop 丢弃的 token 重路由到同一 GPU 上的低负载专家
   - 核心思路：对每个 token，除了原始 top-k 专家外，还将同一 GPU 上的 $m$ 个本地专家加入候选集（共 $k+m$ 个候选）。Token Drop 后被原始专家拒绝的 token 可以被本地低负载专家接收。由于在同一 GPU 上，不需要额外的跨设备通信
   - 设计动机：Token Drop 后低负载专家仍在等待同步，其空闲算力被浪费。Expanded Drop 利用这部分空闲容量处理被丢弃的 token。gating score 在 top-k 之后衰减平缓（Figure 8），说明被重路由到排名稍后的专家不会显著影响输出质量

3. **Device-Level Capacity（高级变体）**:
   - 做什么：在设备级别而非专家级别施加容量约束
   - 核心思路：当单个 GPU 上部署多个专家时，约束 $N_1 + N_2 + \cdots + N_{n_l} \leq n_l \cdot \gamma \bar{N}$，允许同一 GPU 上的专家之间负载转移
   - 设计动机：专家级约束可能过于严格——某个专家超限但同 GPU 其他专家很空闲时，仍会不必要地丢弃 token

### 损失函数 / 训练策略
本方法是纯推理时技术，**不需要重新训练**。直接在已训练好的 MoE 模型上应用，零训练成本。

## 实验关键数据

### 主实验（Expanded Drop vs Token Drop vs Expert Drop vs Baseline）

| 模型 | 方法 | 平均性能 | vs Baseline |
|------|------|---------|-------------|
| **Mixtral-8×7B-Instruct** | Baseline | 74.3 | - |
| | Token Drop ($\gamma$=1.5) | 73.8 | -0.5% |
| | Expanded Drop ($\gamma$=1.5) | **74.5** | **+0.2%** |
| | Expert Drop | 72.2 | -2.1% |
| **OLMoE-Instruct** | Baseline | 63.5 | - |
| | Token Drop ($\gamma$=2.0) | 62.3 | -1.2% |
| | Expanded Drop ($\gamma$=2.0) | **63.2** | -0.3% |
| | Expert Drop | 60.5 | -3.0% |
| **DeepSeek-V2-Lite-Chat** | Baseline | 69.3 | - |
| | Token Drop ($\gamma$=2.0) | 68.2 | -1.1% |
| | Expanded Drop ($\gamma$=2.0) | **68.9** | -0.4% |

### 消融实验（Token Drop 评分函数比较，OLMoE，$\gamma$=1.0）

| 评分函数 | OBQA | PIQA | MMLU | 平均 |
|---------|------|------|------|------|
| Order | 36.0 | 60.2 | 36.9 | 51.8 |
| Reverse Order | 36.2 | 59.5 | 38.7 | 52.0 |
| Random | 34.0 | 63.1 | 35.7 | 53.1 |
| **Score** | **41.6** | **76.0** | **47.8** | **61.1** |

### 关键发现
- **Score 评分远优于其他方法**：在 $\gamma$=1.0 时平均性能 61.1 vs Random 53.1（+8%），gating score 是 token 重要性的有效指标
- **低负载专家至关重要**：Expert Drop（跳过 10% 最轻专家）仅移除 2% token 却导致 3% 性能下降，而 Token Drop 移除 12% token 仅降 0.9%——说明每个专家即使负载低也承载独特知识
- **Expanded Drop 性能可超 baseline**：在 Mixtral 上 Expanded Drop 比无约束 baseline 还高 0.2%，说明重路由 token 到更多专家反而提升了表征质量
- **加速效果受 GPU-专家比影响**：每 GPU 1-2 个专家时加速最大（Mixtral 1.85×），每 GPU 8 个专家时效果减弱（因为聚合负载稀释了单个专家的瓶颈效应）
- **多模态模型中图像 token 可激进压缩**：对视觉 MoE 模型可以用 $\gamma$=0.5 仍保持性能，说明图像 token 在专家中有大量冗余

## 亮点与洞察
- **推理时免训练的负载均衡**：不需要重训模型，直接在推理时通过容量约束和重路由实现负载均衡——这对已部署的 MoE 模型（如 Mixtral、DeepSeek）有直接应用价值
- **Expanded Drop 的本地性设计**：只在同一 GPU 上扩展候选专家，完全避免跨设备通信开销。这是一个简单但关键的工程洞察——利用等待同步的空闲时间做有用计算
- **gating score 尾部平坦的发现**：Figure 8 展示 top-k 之后的专家 gating score 衰减平缓，为重路由提供了理论支撑——被路由到"次优"专家的 token 其实匹配度也不差
- **丢弃少量 token 的巨大加速**：在 Mixtral 上丢弃 12% 溢出 token 获得 85% 加速，说明 Straggler Effect 的长尾分布特性使得少量干预即可获得巨大收益

## 局限性 / 可改进方向
- **未考虑 token 丢弃对生成质量的影响**：评估仅在分类/选择题 benchmark 上进行，未测试开放式文本生成时丢弃 token 是否导致输出连贯性问题
- **静态容量因子**：$\gamma$ 是全局固定的。不同层、不同输入可能需要不同的容量策略——自适应 $\gamma$ 可能效果更好
- **仅测试推理场景**：在训练时的 Token Drop 和推理时的区别未深入探讨，且与训练时辅助损失的交互未研究
- **KV cache 影响未讨论**：token 在某层被丢弃后，后续层如何处理其缺失的信息传递（残差连接）需要更多分析

## 相关工作与启发
- **vs DeepSeek-V3 的专家复制**：DeepSeek-V3 通过在多个设备上复制高负载专家来缓解不均衡，需要额外 GPU 资源。本文方法零额外硬件开销，更加实用
- **vs Switch-Transformer Token Drop**：Switch-Transformer 在训练时用 Order 策略做 Token Drop。本文证明 Score 策略远优于 Order（+9%），且首次将 Token Drop 系统性地应用于推理
- **vs Expert Pruning**：专家剪枝（跳过低负载专家）虽然也减少计算，但性能下降严重。本文对比清楚表明低负载专家不可删除

## 评分
- 新颖性: ⭐⭐⭐⭐ Straggler Effect 的明确定义和系统分析是贡献，Expanded Drop 利用空闲容量的思路巧妙
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 MoE 模型 + 多模态实验 + 评分函数消融 + 效率分析 + 设备级变体
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰，公式推导完整，图表丰富
- 价值: ⭐⭐⭐⭐⭐ 对已部署 MoE 模型的推理优化有直接实用价值，代码已开源
