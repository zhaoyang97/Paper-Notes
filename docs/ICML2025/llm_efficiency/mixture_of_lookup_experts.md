---
title: >-
  [论文解读] Mixture of Lookup Experts
description: >-
  [ICML 2025][LLM效率][Mixture-of-Experts] 提出 MoLE（Mixture of Lookup Experts），将 MoE 中的路由专家输入从中间特征改为 embedding token，使专家可在推理前被重参数化为查找表（LUT）并卸载到存储设备，从而在保持 MoE 级别性能的同时实现与 dense 模型相当的推理速度和显存占用。
tags:
  - "ICML 2025"
  - "LLM效率"
  - "Mixture-of-Experts"
  - "查找表"
  - "专家卸载"
  - "推理加速"
  - "模型部署"
---

# Mixture of Lookup Experts

**会议**: ICML 2025  
**arXiv**: [2503.15798](https://arxiv.org/abs/2503.15798)  
**代码**: [有](https://github.com/JieShibo/MoLE)  
**领域**: LLM效率  
**关键词**: Mixture-of-Experts, 查找表, 专家卸载, 推理加速, 模型部署

## 一句话总结

提出 MoLE（Mixture of Lookup Experts），将 MoE 中的路由专家输入从中间特征改为 embedding token，使专家可在推理前被重参数化为查找表（LUT）并卸载到存储设备，从而在保持 MoE 级别性能的同时实现与 dense 模型相当的推理速度和显存占用。

## 研究背景与动机

**领域现状**：MoE 架构通过只激活部分专家来降低推理 FLOPs，已被 Mixtral、DeepSeek-MoE 等大规模语言模型广泛采用。

**核心痛点**：虽然 MoE 降低了计算量，但所有专家参数仍需加载到 VRAM 中。以 Mixtral-8×7B 为例，每个 token 仅激活 13B 参数，但总参数量高达 46B，需要至少 92GB VRAM（FP16）。

**现有方案的问题**：专家卸载（Expert Offloading）方法将专家放到 CPU 内存/磁盘上，按需加载。但存在两大缺陷：

**通信延迟高**：每步解码需加载不同专家，PCIe 4.0 下单步加载延迟可达 0.7s

**批量推理不友好**：不同样本可能选择不同专家，batch size 增大时需加载更多（甚至全部）专家

**核心矛盾**：MoE 的专家需要被加载到 GPU 上计算，而计算正是造成通信开销的根源。如果能让专家"免计算"，就不需要加载庞大的参数。

**本文方案**：MoLE 的核心 idea 是——将专家的输入限制为 embedding token（有限离散集合），训练后将专家预计算为查找表（LUT），推理时只需根据 input id 查表，而非进行矩阵运算。

**切入角度**：利用 embedding 层输出与 input id 一一对应的性质，将 FFN 计算转化为预计算 + 查表操作，从根本上消除专家的通信和计算瓶颈。

## 方法详解

### 整体框架

MoLE 在训练阶段和推理阶段具有不同结构：

- **训练阶段**：类似 MoE，包含 $N$ 个路由专家和一个共享专家，但有两点关键差异：(1) 路由专家的输入为 embedding token 而非中间特征；(2) 激活全部专家而非 top-k。
- **推理阶段**：路由专家被重参数化为查找表（LUT），卸载到存储设备。推理时直接根据 input id 查表获取结果，无需任何专家计算。

### 关键设计

1. **Embedding Token 作为专家输入**：

   传统 MoE 中专家接收中间特征 $\boldsymbol{h}$ 作为输入。MoLE 将路由专家的输入改为 embedding token $\boldsymbol{e} = \text{Embedding}(i)$。由于 embedding 层输出仅由 input id 决定，专家的可能输入被限制为词表大小 $|\mathcal{V}|$ 个离散值。

   训练阶段 MoLE 层的计算为：

    $\boldsymbol{h}' = \sum_{j=1}^{N} \big(g_j \cdot \text{FFN}_j(\boldsymbol{e})\big) + \text{FFN}_{shared}(\boldsymbol{h}) + \boldsymbol{h}$

   其中 $g_j$ 由 router 基于中间特征 $\boldsymbol{h}$ 计算得到。

   **设计动机**：虽然专家不再接收上下文信息，但 router 和共享专家仍使用中间特征，且专家输出会影响后续 attention 层的行为，因此模型仍具备上下文建模能力。

2. **全专家激活策略**：

   传统 MoE 采用 top-k 稀疏激活以降低 FLOPs。MoLE 的路由专家在推理时不需要计算（查表即可），因此可以激活全部 $N$ 个专家：

    $\{g_j\}_{j=1}^{N} = \text{SoftMax}(\{\boldsymbol{h} \cdot \boldsymbol{r}_j\}_{j=1}^{N})$

   **设计动机**：全激活消除了 MoE 的 top-k 选择机制，使模型完全可微分，无需 load balance loss 或 z-loss 等辅助损失来防止路由坍塌。消融实验证明添加辅助损失反而降低性能。

3. **查找表（LUT）重参数化**：

   训练完成后，对每个可能的 input id $i$，预计算所有专家的输出：

    $\boldsymbol{v}_j^i = \text{FFN}_j(\text{Embedding}(i)) \in \mathbb{R}^d$

   构建查找表 $\text{LUT}_l = \{\{\boldsymbol{v}_j^i\}_{j=1}^{N}\}_{i=1}^{|\mathcal{V}|}$，推理时的计算简化为：

    $\boldsymbol{h}' = \sum_{j=1}^{N} (g_j \cdot \boldsymbol{v}_j^i) + \text{FFN}_{shared}(\boldsymbol{h}) + \boldsymbol{h}$

   **设计动机**：LUT 可完全卸载到存储设备。每步推理只需加载 $dN$ 个参数（即当前 token 对应的 $N$ 个专家输出向量），而 MoE 需加载 $2dkD_r$ 个参数。对于 410M 模型，MoLE 每 token 加载量仅为 MoE 的 1/2000。

### 损失函数 / 训练策略

- **训练损失**：仅使用标准语言建模的交叉熵损失，与 dense 模型完全一致
- **无辅助损失**：由于全专家激活，不存在路由坍塌问题，不需要 load balance loss 或 z-loss
- **训练数据**：100B token 的 Pile 数据集子集
- **LUT 量化**：训练后可对 LUT 进行后量化（NF4/NF3），存储量降至原来的 25%，性能几乎无损

## 实验关键数据

### 主实验

| 规模 | 模型 | 卸载参数 | 每token加载量 | ARC-C | ARC-E | BoolQ | HellaSwag | PIQA | AVG |
|------|------|---------|-------------|-------|-------|-------|-----------|------|-----|
| 160M | Dense | 0B | 0M | 20.3 | 45.9 | 57.1 | 29.7 | 64.0 | 38.8 |
| 160M | MoE-10E | 0.3B | 57M | 21.7 | 49.5 | 51.6 | 32.0 | 66.8 | 40.3 |
| 160M | MoLE-4E | 1.8B | 0.037M | 21.9 | 48.5 | 60.7 | 31.2 | 65.1 | 40.8 |
| 160M | MoLE-16E | 7.4B | 0.15M | 22.4 | 48.6 | 60.3 | 32.7 | 68.3 | 41.9 |
| 410M | Dense | 0B | 0M | 21.8 | 50.8 | 56.8 | 33.8 | 66.5 | 41.8 |
| 410M | MoE-34E | 3.4B | 201M | 25.0 | 57.0 | 59.7 | 39.9 | 71.5 | 46.6 |
| 410M | MoLE-16E | 19.7B | 0.39M | 23.6 | 57.0 | 60.9 | 37.6 | 70.8 | 45.7 |
| 1B | Dense | 0B | 0M | 24.1 | 56.9 | 52.8 | 37.6 | 69.5 | 44.3 |
| 1B | MoE-10E | 2.7B | 537M | 25.9 | 57.8 | 53.8 | 40.7 | 72.0 | 46.6 |
| 1B | MoLE-4E | 6.6B | 0.26M | 25.5 | 58.8 | 61.7 | 39.8 | 71.7 | 47.4 |

### 消融实验

| 配置 | AVG | 说明 |
|------|-----|------|
| 仅 LM loss（默认） | 41.9 | MoLE 最优设置 |
| LM loss + load balance loss | 41.7 | 辅助损失导致性能下降 |
| LM loss + load balance + z-loss | 40.6 | 进一步下降，优化目标错位 |
| 专家隐藏维度 $D_r = d$ | 40.8 | 专家容量较小 |
| 专家隐藏维度 $D_r = 4d$ | 41.9 | 最佳性价比 |
| 专家隐藏维度 $D_r = 16d$ | 41.7 | 容量饱和，无额外收益 |
| 专家数 N=2 | 39.7 | 太少，容量不足 |
| 专家数 N=16 | 41.9 | 较好平衡点 |
| 专家数 N=32 | 42.3 | 持续提升，有可扩展性 |
| LUT FP16 | 40.8 | 存储 3.5GB |
| LUT NF4 量化 | 40.9 | 存储 0.9GB，几乎无损 |
| LUT NF3 量化 | 40.5 | 存储 0.7GB，轻微下降 |

### 关键发现

1. **通信量大幅缩减**：MoLE 每 token 加载参数量仅为 MoE 的 1/1500 ~ 1/2000，使低带宽存储（如磁盘、网络存储）的卸载成为可能
2. **推理速度媲美 dense 模型**：在 V100 上测试，MoLE 解码延迟与 dense 模型基本一致，远低于 MoE + offloading
3. **批量推理友好**：MoE 随 batch size 增大延迟急剧上升（更多专家需加载），MoLE 的延迟几乎不随 batch size 变化
4. **全激活优于稀疏激活**：全专家激活的性能提升（+1.5）足以弥补使用 embedding 输入的性能损失（-0.7）
5. **增加专家数量具备可扩展性**，但增加专家隐藏维度在超过一定阈值后饱和——说明 LUT 容量存在上限
6. **LUT 量化潜力大**：NF4 量化后存储减少 75%，性能几乎无损，说明 LUT 存在显著冗余

## 亮点与洞察

1. **核心洞察极为精巧**：利用 embedding 输出与 input id 的固定对应关系，将连续空间的 FFN 计算转化为有限离散集合上的预计算查表，这是一个非常优雅的工程转化
2. **训练-推理结构解耦**：训练时用完整 FFN 保证梯度流动和模型容量，推理时用 LUT 实现零计算，这种重参数化思路值得借鉴
3. **简洁性**：不需要辅助损失、不需要 top-k 选择、不需要复杂的缓存/预取策略，整个方法非常干净
4. **LUT 量化结果惊艳**：NF4 量化几乎不影响性能，暗示专家学到的 token 级表示具有较好的低秩/低精度特性

## 局限与展望

1. **LUT 存储仍然较大**：虽然通信量极低，但 LUT 总存储量可达卸载专家的 2.4-7.4 倍（160M 模型下 7.4GB vs 1.0GB），大词表模型（如 100k+）会更严重
2. **实验规模较小**：最大仅到 1B 激活参数，未在 7B+ 规模上验证，Scaling behavior 存疑
3. **专家不接收上下文信息**：这是实现 LUT 的必要代价，限制了专家的表达能力。论文论证了 router + shared expert 可以补偿，但在更复杂任务上的影响未知
4. **仅使用 MLP 形式的专家**：未探索更多专家架构（如 attention-based expert），也未探索更多样的离散输入空间
5. **缺少与 MoE 压缩方法的对比**：如 Expert Pruning / Expert Merging 等方法也能降低 MoE 的存储和通信开销
6. **预填充阶段分析不足**：论文主要关注解码阶段延迟，预填充阶段的 LUT 批量查询效率未讨论

## 相关工作与启发

- **MoE 系列**：Mixtral、DeepSeek-MoE、OLMoE 是主要对比基线。MoE++ (Jin et al., 2025) 也在探索零计算专家，但思路不同
- **Expert Offloading**：Eliseev & Mazur (2023)、Pre-gated MoE (Hwang et al., 2024) 等通过优化预取和缓存策略加速推理，但受限于通信瓶颈
- **重参数化思想**：类似 RepVGG 的训练-推理结构解耦思路，将复杂训练结构转化为简单推理结构
- **启发**：这种"将计算结果预存为查找表"的方式可以推广到其他具有有限离散输入的模块，例如 positional encoding、特定类型的 adapter 等

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ [核心idea——将MoE专家输入限制为embedding实现LUT重参数化——非常新颖且优雅]
- 实验充分度: ⭐⭐⭐⭐ [多规模、多消融、含效率分析和量化实验，但最大规模仅1B，缺少大模型验证]
- 写作质量: ⭐⭐⭐⭐⭐ [逻辑清晰，从动机到方法到消融环环相扣，图表设计合理]
- 价值: ⭐⭐⭐⭐ [为VRAM受限场景的MoE部署提供了全新方向，但需大规模验证后才能确认实用价值]

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] MoH: Multi-Head Attention as Mixture-of-Head Attention](moh_multi-head_attention_as_mixture-of-head_attention.md)
- [\[ICML 2025\] Autonomy-of-Experts Models (AoE)](autonomy-of-experts_models.md)
- [\[NeurIPS 2025\] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks](../../NeurIPS2025/llm_efficiency/on_the_expressive_power_of_mixture-of-experts_for_structured_complex_tasks.md)
- [\[ICML 2026\] L$^3$: Large Lookup Layers](../../ICML2026/llm_efficiency/l3_large_lookup_layers.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)

</div>

<!-- RELATED:END -->
