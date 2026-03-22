# KVComm: Enabling Efficient LLM Communication through Selective KV Sharing

**会议**: ICLR 2026  
**arXiv**: [2510.03346](https://arxiv.org/abs/2510.03346)  
**代码**: 待确认  
**领域**: Agent / LLM 效率  
**关键词**: LLM communication, KV cache sharing, multi-agent LLM, selective layer, attention importance  

## 一句话总结
提出 KVComm 框架通过选择性共享 KV pairs 实现 LLM 间高效通信，发现 hidden states 存在"信息集中偏差"使其不适合跨模型传递，设计基于注意力重要性 + 高斯先验的层选择策略，仅传输 30% 层即可超越大多数 baseline。

## 研究背景与动机
1. **领域现状**：多 LLM 协作场景需要高效通信机制，现有方法传递 hidden states 或全部 KV cache。
2. **现有痛点**：① Hidden states 的 last token 在深层最关键但传递会覆盖 Receiver 信息；② 全 KV cache 传输量太大。
3. **核心矛盾**：通信效率 vs 信息完整性的平衡。
4. **本文要解决**：找到最适合跨 LLM 传递的表示形式和选择策略。
5. **切入角度**：系统对比 hidden states 和 KV pairs，发现 KV pairs 天然适合——可按层选择传递且不覆盖 Receiver 信息。
6. **核心idea**：KV pairs 是最佳通信介质；选中间层（语义最丰富）+ 高注意力层 → 最优子集。

## 方法详解

### 整体框架
Sender 处理 context → 提取 KV pairs → 层选择策略选子集 → 传输给 Receiver。Receiver 在对应层拼接两方 KV：$\mathbf{k}_r^l \leftarrow [\mathbf{k}_s^{l_i}; \mathbf{k}_r^l]$。

### 关键设计

1. **Hidden States vs KV Pairs 对比**:
   - Hidden states 问题：last token 信息最关键但直接传递替换 Receiver 自身表示
   - KV pairs 优势：可在任意层注入，不覆盖 Receiver 信息，attention 自然选择性使用

2. **层选择策略**:
   - 注意力重要性 $\hat{S}_a^l = \frac{1}{H|Q|}\sum_h\sum_q\sum_c a_{h,q,c}^l$
   - 高斯先验 $P^l = \exp(-\frac{(l-\mu)^2}{2\sigma^2})$（鼓励选中间层）
   - 最终分数 $S^l = \alpha S_a^l + (1-\alpha) P^l$，选 top-M 层
   - 仅需 1 个校准样本即可稳健

3. **两个假设验证**:
   - H1: 中间层 KV 含最可迁移的语义知识
   - H2: 注意力更集中的层 more informative

## 实验关键数据

### 主实验（9 模型对，8 数据集）
| 模型 | 方法 | Countries | HotpotQA | MultiFieldQA |
|------|------|-----------|----------|-------------|
| Llama-3.2-3B | Skyline | 0.57 | 0.73 | 0.47 |
| Llama-3.2-3B | KVComm(0.5) | **0.57** | 0.57 | **0.51** |
| Llama-3.2-3B | NLD | 0.51 | 0.47 | 0.38 |
| Llama-3.2-3B | AC | 0.35 | 0.32 | 0.29 |

### 消融实验
| 传输比例 | 效果 |
|---------|------|
| 30% 层 | 超越 NLD/CIPHER/AC 所有 baseline |
| 50% 层 | 接近 Skyline |
| 70% 层 | 逼近或超越 Skyline |
| 非连续 vs 连续选择 | 非连续显著更优 |

### 关键发现
- 仅 30% 层 KV 即可超越大多数 baseline——选择性 > 全量
- MultiFieldQA 上超越 Skyline（0.51 vs 0.47）——选择性共享有正则化效应
- AC 方法多数数据集接近 no-communication baseline
- 计算量比 NLD 减少 2.5x-6x

## 亮点与洞察
- **Hidden states 信息集中偏差**是重要发现，对所有基于 hidden states 的 LLM 通信方法有警示
- **"少即是多"**——30-50% 层 KV 效果优于全量 hidden states
- **高斯先验选中间层**虽简单但有效
- 1 个校准样本即可确定层选择，部署极其轻量

## 局限性 / 可改进方向
- 仅支持同 base model 间通信，不支持异构模型
- 层索引须一一对应，限制不同规模模型间通信
- 高斯先验的 $\mu$、$\sigma$ 需调参
- 仅验证两个 agent 场景
- 数学推理上提升不明显

## 相关工作与启发
- **vs NLD**: NLD 压缩为自然语言，信息损失大；KVComm 直接传递内部表示
- **vs CIPHER**: CIPHER 传 hidden states，受信息集中偏差影响
- **vs DroidSpeak**: DroidSpeak 连续 chunk 选层不如非连续选择灵活
- 可启发 multi-agent LLM 系统设计：KV cache 共享可能成为标准通信原语

## 补充技术细节

### 为什么 KV Pairs 比 Hidden States 更适合通信？
Hidden states 在每层都是一个完整的表示，直接传递会覆盖 Receiver 的对应层表示。而 KV pairs 是 Attention 机制的输入，拼接到 Receiver 的 KV 后不会破坏原有信息，而是让 Attention 机制自然地决定关注哪些信息。这种"加法而非替换"的特性是 KV 通信的核心优势。

### 中间层为什么最有价值？
研究表明 LLM 的层可以大致分为三个功能区：底层（低级特征、语法）、中层（语义知识、世界知识）、顶层（任务特定表示、下一 token 预测）。中层的语义知识最通用、最可迁移，而底层太低级、顶层太任务特定，都不适合跨模型传递。

### 与 Prompt Compression 的关系
KVComm 可以看作一种“在 KV 空间做 prompt compression”——不是压缩文本，而是压缩内部表示的“层”维度。这比 NLD（将知识压缩为自然语言）保留了更多细粒度信息。未来可以探索在层内进一步压缩（如选择性 token），实现“层 + token”双维度压缩。

### KV 拼接的 Attention 机制
当 Sender 的 KV 拼接到 Receiver 后，Receiver 的 Query 可以自由地 attend 到两方的 Key。由于 Attention 是 softmax 归一化的，无用信息会被自然地低权重化。这比直接替换 hidden states 更「温和」——不会强制覆盖任何信息。

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统对比通信介质，层选择策略合理
- 实验充分度: ⭐⭐⭐⭐ 9 模型对×8 数据集
- 写作质量: ⭐⭐⭐⭐ 假设-验证逻辑清晰
- 价值: ⭐⭐⭐⭐ 对 multi-LLM 协作有实际指导意义
