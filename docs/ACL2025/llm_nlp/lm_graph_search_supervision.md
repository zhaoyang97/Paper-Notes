# Language Models, Graph Searching, and Supervision Adulteration: When More Supervision is Less and How to Make More More

**会议**: ACL 2025  
**arXiv**: [2503.10542](https://arxiv.org/abs/2503.10542)  
**代码**: 无  
**领域**: LLM/NLP  
**关键词**: 图搜索, 监督污染, next-token prediction, 子任务分解, 捷径学习, Clever Hans Cheat, 规划能力  

## 一句话总结

本文证明了 path-star 图搜索任务在 decoder-only LM 上的失败并非 next-token prediction 范式的根本缺陷，而是由"监督污染"（supervision adulteration）导致的——过量的 teacher-forcing 监督信号诱导模型学到 Clever Hans Cheat 捷径，阻碍了子任务分解；通过 token masking、ranking-into-the-future、scratchpad、树形拓扑等六种正交方法均可使任务可学。

## 研究背景与动机

1. **Path-star 任务**：Bachmann & Nagarajan (2024) 提出的极简图搜索任务——星形图有 D 条臂，每条长 M，给定起点 s 和终点 t，模型需生成从 s 到 t 的正确臂。核心难点在于选择正确的 leading node $l_t$。
2. **惊人的失败**：标准 decoder-only LM 通过 teacher-forcing 训练后，在该任务上准确率不超过随机基线 $1/D$，被用来论证 next-token prediction 范式对规划任务存在根本性不足。
3. **引发的后续工作**：此失败促使多个工作提出替代架构（如 Yin et al. 2024 的辅助自编码器、Hu et al. 2025a 的双向编码器），但这些方案改变了模型架构本身。
4. **Clever Hans Cheat (CHC)**：在 teacher-forcing 训练中，模型利用前一个 ground-truth token 进行单边查找（single-edge lookup），而非真正学会从 t 逆向重建臂路径。CHC 吸收了除 $l_t$ 之外的所有序列监督信号，使学习核心规划子任务仅依赖单一 token。
5. **核心洞察——监督污染**：作者提出 supervision adulteration 概念：过量或不当的监督信号之间产生坏的交互，使得目标任务的学习信号被无关的捷径学习所稀释。这不是数据过拟合的问题，而是任务构造方式本身的问题。
6. **反驳动机**：如果标准方法（decoder-only + teacher-forcing + next-token prediction）通过微小修改就能解决该任务，则证明原始声明过于绝对，该失败不构成对范式本身的否定。

## 方法详解

### 整体框架

作者将问题归纳为：path-star 任务的不可学性源于监督污染阻碍了子任务分解（subtask decomposition）。图搜索/臂重建本身是递归定义的，天然包含分解结构，但 teacher-forcing 的过量监督使模型走捷径而非学习分解。因此，只要设计出能诱导子任务分解的训练方式，任务就变得可学。作者提出了六种正交方法验证该理论。

实验设置：decoder-only Transformer，2 头，64 维 embedding，256 维 FFN，8 层，学习率 $5\times10^{-4}$，batch size 1024，在线数据集（每次生成新样本避免过拟合），$|V|=|G|$，训练 100M 样本。

### 模块一：Token Masking

**动机**：打破 teacher-forcing 中前一 ground-truth 与当前预测之间的坏交互，从目标端输入（target-side input）入手。

**方法**：在训练时对目标序列中的 token 进行随机 mask 或替换（scheduled sampling），支持均匀采样和连续 span 采样两种模式。被 mask 的位置无法触发 CHC 的单边查找，迫使模型进行多边查找（multi-edge lookup）。

**分解机制**：当 $l_t$ 被 mask 时，模型被迫学习从 t 逆向推导的子任务（图5c），这恰好是核心规划任务的子集。mask 不仅阻止了 CHC，还直接诱导了子任务分解。

### 模块二：Ranking-into-the-Future (RITF)

**动机**：从损失函数角度重新设计监督信号——让模型在每一步预测未来 token 的分布而非仅下一个 token。

**方法**：在每个时间步 $i$，构造排序目标 $x_i \succ x_{i+1} \succ \dots \succ x_M$，使用 pairwise hinge loss：

$$L_B = \sum_{i=1}^{M}\sum_{j=i}^{M}\sum_{k=j+1}^{M}\max(0, 1-(\sigma_i[j]-\sigma_i[k]))$$

同时加入正确臂内 token 排在其他臂 token 之上的约束。与 BoW（bag-of-words）和 label smoothing 对比，RITF 表现最优。

**分解机制**：多 token 损失天然需要多边查找；每步的未来分布学习构成嵌套子问题（$P_{B,i+1}$ 是 $P_{B,i}$ 的子问题），产生跨序列的密集分解监督。

### 模块三：图拓扑修改（Tree-Star）与通用查询

**Tree-Star**：将训练时的臂从路径改为树结构（split tree），每个分叉创造一个 $D'=2$ 的子 path-star 任务。训练用树、评估用路径——反直觉地，训练与评估不同分布反而有效，因为路径是"过度信息化"的图结构。

**通用查询 (GST)**：在查询中随机采样 $R_t$ 中的单个节点替代固定的 t，使模型在训练时面对不同长度的子路径，直接引入分解。

**通用长度分解**：训练时混合不同 M 值的图，自然提供不同粒度的子任务。

### 训练策略

所有方法均保留标准 teacher-forcing + next-token prediction 范式（RITF 除外，使用了未来分布损失）。使用在线数据集（无固定数据集）避免过拟合。每组实验运行 5 个不同随机种子，报告 SR（>95% 序列准确率的成功率）和 ABB（超过基线 $100/D+10$% 的比例）。

## 实验

### 表1：各方法在不同 (D, M) 下的 Success Rate

| 方法 | D=2,M=5 | D=3,M=5 | D=4,M=5 | D=5,M=5 | D=2,M=7 | D=3,M=7 |
|------|---------|---------|---------|---------|---------|---------|
| Baseline | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Causal-wise shuffle | ✓ | ✓ | 部分 | 部分 | ✓ | 部分 |
| Token masking | ✓ | ✓ | 部分 | ✗ | ✓ | 部分 |
| RITF | ✓ | ✓ | ✓ | 部分 | ✓ | ✓ |
| Split tree | ✓ | ✓ | ✓ | 部分 | ✓ | 部分 |
| GST (通用查询) | ✓ | ✓ | 部分 | — | ✓ | 部分 |

### 表2：Scratchpad 方法对比

| SP 类型 | SP 准确率 | R_t 准确率 | 说明 |
|---------|----------|-----------|------|
| 反向臂序 | ~100% | 高 | trivial，无分析价值 |
| BoW (排序) | 中等 | 低-中 | 排序本身难学，分解效果有限 |
| 图重建 (GR-SP) | 部分学会 | 极低 (4/80) | 能识别节点集但无法完成臂重建 |

### 关键发现

1. **所有六种正交方法均可使任务可学**，充分证明"不可学"结论的脆弱性——微小修改即可打破。
2. **阻止 CHC 并非必要条件**：split tree 和通用长度分解不阻止 CHC 但任务仍可学，关键在于是否诱导了子任务分解。
3. **RITF 优于 BoW 和 label smoothing**：证明指定排序规则比指定具体权重更有效。
4. **GR-SP 的负面结果极具启发性**：模型能正确识别 leading/target 节点集并排序，但无法完成臂重建——说明核心难点是图重建而非规划选择。
5. **因果约束的影响**：causal-wise shuffle 使任务可学，表明 decoder 的因果约束增加了额外难度。
6. **扩展性问题**：所有方法在 D 或 M 增大时性能下降，作者猜测需要更强/一致的分解结构（子任务与主任务同构）才能解决。

## 亮点

- **"监督污染"概念新颖深刻**：将 teacher-forcing 的隐性问题显式化，统一解释了为何更多监督反而有害——不是监督量的问题，而是监督间交互产生的捷径吸收了有用信号。
- **六种正交方法的统一解释**：所有方法从不同角度（输入端/损失端/数据端/拓扑端）诱导子任务分解，强力支撑理论的统一性。
- **负面结果同样有价值**：GR-SP 失败揭示臂重建（而非规划选择）是核心难点，BoW SP 失败揭示反向解法不会被自动发现——打破了"显而易见的解法模型也能找到"的直觉。
- **对先前工作的桥接**：统一解释了 Bachmann & Nagarajan (2024) 和 Saparov et al. (2025) 看似矛盾的结论。

## 局限

1. **扩展性未解决**：所有方法在 D 或 M 增大时均失败，论文只在小规模图上验证了可学性，未给出可扩展方案。
2. **仅使用从头训练的小模型**（2头/64维/8层），未验证在预训练大模型上的适用性，而预训练引入的语义信息可能改变结论。
3. **为何子任务分解是必要的**仍是开放问题——论文展示了经验性必要性但缺乏理论证明。
4. **path-star 任务本身的代表性有限**：作者自己承认该任务不适合作为评估规划能力的基准，图搜索也不能代表一般搜索问题。
5. **RITF 的实用性未验证**：仅在合成任务上测试，未扩展到自然语言任务或更大规模场景。

## 相关工作

- **Path-star 任务起源**：Bachmann & Nagarajan (2024) "The Pitfalls of Next-Token Prediction"，首次提出该任务并声称 decoder-only LM 无法学会。
- **替代架构**：Yin et al. (2024) 的辅助自编码器 planning token、Hu et al. (2025a) 的双向编码器、Ahn et al. (2025) 的最小架构改变。
- **图搜索可学性**：Saparov et al. (2025) 在 encoder-only 和更一般拓扑上取得正面结果但未测试 decoder + path-star，Frydenlund (2024) 证明了表达能力充分但学习仍是开放问题。
- **捷径学习**：Du et al. (2023) 对 LLM 中捷径学习的综述，Bhattamishra et al. (2023) 关于 Transformer 学习稀疏布尔函数的简单性偏差。
- **多 token 预测**：Cai et al. (2024) Medusa 多头推理加速，与本文 RITF 的未来分布学习思想相关但目标不同。

## 评分

- **新颖性**: ⭐⭐⭐⭐ — "监督污染"概念新颖，子任务分解的统一解释框架有理论贡献
- **技术深度**: ⭐⭐⭐⭐ — 六种正交方法设计精巧，实验分析细致（尤其是负面结果的深入讨论）
- **实用性**: ⭐⭐⭐ — 核心发现限于合成任务，向真实 NLP 场景的迁移路径不明
- **表达清晰度**: ⭐⭐⭐⭐⭐ — 逻辑链条严密，图示优秀，概念命名准确直观
