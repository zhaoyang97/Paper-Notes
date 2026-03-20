# APB: Accelerating Distributed Long-Context Inference by Passing Compressed Context Blocks across GPUs

**会议**: ACL 2025  
**arXiv**: [2502.12085](https://arxiv.org/abs/2502.12085)  
**代码**: [https://github.com/thunlp/APB](https://github.com/thunlp/APB)  
**领域**: LLM效率  
**关键词**: long-context inference, sequence parallelism, KV cache compression, distributed inference, approximate attention  

## 一句话总结
APB 提出了一种分布式长上下文推理框架，通过在序列并行框架中引入本地 KV cache 压缩和跨 GPU 传递压缩上下文块的机制，在不损失任务性能的前提下实现了相比 FlashAttn/RingAttn/StarAttn 分别高达 9.2x/4.2x/1.6x 的 prefill 加速。

## 研究背景与动机
1. **领域现状**：随着 LLM 支持的上下文长度不断增加（Llama-3.1 128K、Claude-3.5 200K、MiniMax-01 4M），长上下文推理的 prefill 阶段成为关键瓶颈，注意力计算复杂度为 $O(n^2)$
2. **现有痛点**：
   - 序列并行（RingAttn、Ulysses）提升了并行度但计算量不变，仍受限于全注意力计算
   - 近似注意力机制（H2O、SnapKV）减少计算量但依赖全局序列信息进行剪枝，与分布式架构冲突
   - StarAttn 简单结合近似注意力和序列并行，但随 GPU 数增加性能持续下降（每个 host 看到的上下文比例降低）
3. **核心矛盾**：现有 KV cache 压缩方法需要全局注意力分数来判断哪些 KV 重要，但序列并行中每个 host 只持有部分上下文，无法获取全局视图
4. **本文要解决什么？** 设计一种在序列并行框架内同时减少计算量和通信开销的近似注意力机制，且随 GPU 数量增加仍保持稳定性能
5. **切入角度**：使用本地 KV cache 压缩（无需全局视图）+ 压缩后跨 GPU 传递关键 KV 对
6. **核心idea一句话**：每个 GPU 独立压缩本地 KV cache，然后只传递压缩后的关键上下文给其他 GPU，实现计算和通信的双重减少

## 方法详解

### 整体框架
APB 推理分为 4 个阶段：上下文切分（Context Splitting）→ 块压缩（Block Compression）→ 通信（Communication）→ 计算（Computation）。输入序列被分为文档 $d$ 和查询 $q$，文档均匀分配到 $H$ 个 GPU，每个 GPU 上的本地 block 前拼接 anchor block。压缩后的 KV 通过 AllGather 通信，构建 passing block 供后续 GPU 参考。

### 关键设计

1. **Anchor Block（锚块）**:
   - 做什么：为每个 host 提供文档开头信息的全局上下文锚点
   - 核心思路：将 query $q$ 和文档前 $l_a$ 个 token 拼接为 anchor block $\mathbf{A} = \{q_1, \ldots, q_{l_q}, d_1, \ldots, d_{l_a}\}$，前置到每个 host 的本地 block 之前
   - 设计动机：与 StarAttn 不同，APB 使用更小的 anchor（$l_a = \frac{1}{4}l_b$ 或 $\frac{1}{8}l_b$，StarAttn 用 $l_a = l_b$），因为通过 passing block 已经补充了跨 host 信息。将 query 嵌入 anchor 使压缩器能根据查询相关性识别关键 KV

2. **本地 KV Cache 压缩（Block Compression）**:
   - 做什么：在每个 host 上独立压缩本地 KV cache，保留最关键的 $l_p$ 个 KV 对
   - 核心思路：使用 Locret 的 retaining heads $\mathcal{R}$——小型 MLP，输入 $[\mathbf{Q}, \mathbf{K}, \mathbf{V}]$ 输出每个 token 的重要性分数，选 Top-$l_p$ 保留。关键公式：$\mathbf{B}_h^C = \text{Top-}l_p(\mathcal{R}([\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h]))$
   - 设计动机：H2O/SnapKV 需要全局注意力分数，在分布式场景不可用。Locret 基于每个 token 自身的 QKV 判断重要性，可完全本地化执行，解决了 Challenge 1

3. **AllGather 通信 + Passing Block 构建**:
   - 做什么：收集所有 host 的压缩 KV cache，为每个 host 构建包含前序所有压缩上下文的 passing block
   - 核心思路：AllGather 通信后，host $h$ 的 passing block $\mathbf{P}_h = (\mathbf{K}_{[1:h-1]}^C, \mathbf{V}_{[1:h-1]}^C)$，即拼接所有前序 host 的压缩 KV。因为已压缩，通信量大幅减少
   - 设计动机：确保每个 host 能感知全局上下文的关键信息，解决了 Challenge 2——随着 host 数增加，每个 host 仍能通过 passing block 访问所有前序上下文的压缩表示

4. **定制 FlashAttn Kernel + 修改的注意力掩码**:
   - 做什么：使用修改的 attention mask 在 anchor + passing + local 三部分拼接的 KV 上计算注意力
   - 核心思路：$\mathbf{Q} = [\mathbf{Q}_a, \mathbf{Q}_h]$，$\mathbf{K} = [\mathbf{K}_a, \mathbf{K}_p^C, \mathbf{K}_h]$，使用定制 FlashAttn kernel 执行。Passing block 在注意力后丢弃，不参与 FFN 计算
   - 与 StarAttn 的区别：StarAttn 只用 anchor block 提供跨 host 信息，APB 额外通过 passing block 传递压缩上下文，信息量更大但通信量可控

### 训练策略
- APB 框架本身免训练，但 KV cache 压缩器（retaining heads）需在长上下文 SFT 数据上预训练
- 训练成本很低——只是小型 MLP，不需修改原始 LLM 参数

## 实验关键数据

### 主实验

**∞Bench 长上下文任务（128K tokens, Llama-3.1-8B）**:

| 方法 | R.PassKey | R.Number | R.KV | Avg. |
|------|:---------:|:--------:|:----:|:----:|
| FullAttn | 100.00 | 99.49 | 51.00 | 47.45 |
| MInference | 98.47 | 98.81 | 17.40 | 43.61 |
| StarAttn | 100.00 | 98.98 | 40.60 | 46.48 |
| **APB** | **100.00** | **98.81** | **81.80** | **50.91** |

APB 平均分超过 FullAttn 3.46 分，在 R.KV 任务上大幅领先（81.8 vs 51.0）。

**RULER Benchmark（Llama-3.1-8B, 128K）**:

| 方法 | MK2 | MK3 | MV | Avg. |
|------|:---:|:---:|:--:|:----:|
| FullAttn | 87.60 | 67.00 | 94.65 | 82.20 |
| StarAttn | 73.60 | 53.00 | 72.80 | 76.84 |
| **APB** | **91.00** | **89.00** | **95.05** | **81.63** |

**推理速度对比（Llama-3.1-8B, 4 GPUs）**:

| 方法 | 128K tokens/s | 256K tokens/s | 加速比 vs FlashAttn |
|------|:------------:|:------------:|:-------------------:|
| FlashAttn | baseline | baseline | 1.0x |
| RingAttn | ~2.2x | ~2.2x | 2.2x |
| StarAttn | - | - | ~5.8x |
| **APB** | - | - | **~9.2x** |

### 消融实验

| 配置 | ∞Bench Avg. | RULER Avg. | 说明 |
|------|:----------:|:---------:|------|
| Full model (anchor=1/4, pass=1/4) | 50.91 | 81.63 | 完整 APB |
| w/o passing block | ~46.5 | ~77 | 退化为 StarAttn 变种 |
| anchor=1/8, pass=1/4 | ~49 | ~80 | 更小 anchor，轻微下降 |
| Random compressor | 明显下降 | 明显下降 | Locret 压缩器有效 |

### 关键发现
- APB 在多个任务（尤其是 R.KV retrieval）上甚至超过 FullAttn，原因是压缩去除了噪声 KV，反而提升了检索精度
- Passing block 是关键贡献——没有它 APB 退化为类 StarAttn 方案，随 GPU 数增加性能下降
- 压缩比可灵活配置（$l_p/l_b$ 比例），在速度和精度间取得可控权衡
- 支持 TP（张量并行）+ SP（序列并行）联合配置，兼容多种模型架构

## 亮点与洞察
- **压缩+分布的协同设计**：不是简单把 KV cache 压缩叠加到序列并行上，而是设计了完整的 anchor-compress-pass-compute pipeline，确保信息流完整性。这种系统-算法协同设计的思路值得学习
- **本地压缩的巧妙选择**：采用 Locret 这种不依赖全局视图的压缩方法，完美匹配分布式场景约束——问题和解决方案的适配度非常高
- **超越精确注意力**：压缩 KV 在检索任务上反而优于全注意力（去噪效果），这个发现很有启发性——提示精确注意力并非总是最优

## 局限性 / 可改进方向
- **仅优化 prefill**：解码阶段仍使用标准注意力，对长生成场景帮助有限。可考虑将 passing block 机制扩展到解码阶段
- **Locret 需额外训练**：每个新模型需训练对应的 retaining heads MLP，增加部署成本
- **anchor + query 前置假设**：要求输入可以清晰地分为 document 和 query，对于对话等 query 散布在上下文中的场景不太适用
- **GPU 间通信仍是瓶颈**：虽然压缩减少了通信量，但 AllGather 在大规模集群上可能仍是瓶颈

## 相关工作与启发
- **vs RingAttn**: RingAttn 保持精确注意力但无计算量减少，APB 通过近似注意力减少计算；两者可视为精确 vs 近似的 trade-off
- **vs StarAttn**: StarAttn 只靠 anchor block 提供跨 host 信息（全尺寸 anchor），APB 用小 anchor + 压缩 passing block，信息量更大且 overhead 更小
- **vs MInference**: MInference 分析注意力稀疏模式做近似，但在分布式场景效果差；APB 通过 Locret 实现了分布式友好的近似
- 这篇的 compress-then-communicate 范式可以迁移到其他分布式计算场景（如分布式 RAG、多 agent 通信）

## 评分
- 新颖性: ⭐⭐⭐⭐ 本地压缩+分布式传递的组合设计新颖，但各组件（Locret、sequence parallelism）较为成熟
- 实验充分度: ⭐⭐⭐⭐⭐ 三个模型、两个 benchmark、速度和精度双维度评估、消融完整
- 写作质量: ⭐⭐⭐⭐⭐ 问题定义清晰，framework 图示直观，算法伪代码完整
- 价值: ⭐⭐⭐⭐⭐ 长上下文推理加速是核心需求，9.2x 加速且无性能损失有很强实用价值
