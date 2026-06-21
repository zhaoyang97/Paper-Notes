---
title: >-
  [论文解读] LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error
description: >-
  [ICLR 2026][模型压缩][层级稀疏分配] LSA 用「假设剪掉每层 50% 最不重要权重后的最小线性重建误差」直接刻画各 Transformer 层的冗余度，从而免去 Wanda 式权重打分和人工 reduce 函数，给不同层（乃至 block/projection）分配非均匀稀疏率，在 70% 高稀疏下超越 OWL、DLP 等方法。
tags:
  - "ICLR 2026"
  - "模型压缩"
  - "层级稀疏分配"
  - "线性重建误差"
  - "非均匀稀疏"
  - "无重训剪枝"
  - "大语言模型"
---

# LSA: Layer-wise Sparsity Allocation for Large Language Model Pruning Based on Minimal Linear Reconstruction Error

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=xq3lza5IjN](https://openreview.net/forum?id=xq3lza5IjN)  
**代码**: [https://github.com/BeiYazi0/LSA](https://github.com/BeiYazi0/LSA)  
**领域**: 模型压缩 / LLM 剪枝  
**关键词**: 层级稀疏分配, 线性重建误差, 非均匀稀疏, 无重训剪枝, 大语言模型

## 一句话总结
LSA 用「假设剪掉每层 50% 最不重要权重后的最小线性重建误差」直接刻画各 Transformer 层的冗余度，从而免去 Wanda 式权重打分和人工 reduce 函数，给不同层（乃至 block/projection）分配非均匀稀疏率，在 70% 高稀疏下超越 OWL、DLP 等方法。

## 研究背景与动机
**领域现状**：SparseGPT、Wanda 等一次性（one-shot）剪枝方法无需重训即可压缩 LLM，但它们给所有层施加**统一稀疏率**。近年研究发现各层对模型性能贡献并不均衡，OWL、DLP 等方法因此探索**层级非均匀稀疏分配**。

**现有痛点**：OWL 与 DLP 都依赖 Wanda 风格的权重打分（权重幅度 × 激活范数），再叠加一个人工设计的 reduce 函数（OWL 用均值乘子、DLP 用中位数）来估计层重要性。这带来两个问题：(1) reduce 函数选取靠经验、缺乏理论最优性，OWL 的均值乘子 $m$ 还不能跨模型泛化；(2) 它们只能停留在**层级粒度**，一旦把稀疏分配细化到 block（attention/FFN）或 projection（Q/K/V 等）级别，性能就会大幅崩塌。

**核心矛盾**：要更精细、更鲁棒地刻画「哪层该多剪、哪层该少剪」，却被「必须先逐权重打分 + 选一个好 reduce 函数」这条路径卡住，且无法下探到更细粒度。

**本文目标**：回答两个被忽视的问题——是否需要找更优的 reduce 函数来评估层重要性？能否在更细粒度上施加非均匀稀疏而不损性能？

**核心 idea**：**绕开权重打分**，直接用每层在「剪掉 50% 最不重要权重」假设下的**最小线性重建误差（LRE）**来量化层冗余——误差低的层更易出现关键 outlier、冗余低应少剪，误差高的层误差分布均匀、冗余高可多剪。并把该思路自然延伸到 block/projection 级别。

## 方法详解

### 整体框架
LSA 分三步：先对每个线性层用一个贪心算法算出「剪掉 $p=50\%$ 输入通道时的最小线性重建误差」作为该层冗余度；再把各层误差归一化转成相对重要性分数，并用一个超参 $\beta$ 把重要性映射成落在 $[p_r-\beta, p_r+\beta]$ 区间的稀疏率；最后把同一套逻辑下放到 block / projection 粒度，实现更细的非均匀分配。整个过程只算误差、不真正剪权重，模型保持稠密直到最终统一执行剪枝。

```mermaid
flowchart LR
    A[校准数据 + 各层权重 W] --> B[Algorithm 1<br/>贪心求 p=50% 最小线性重建误差 LRE]
    B --> C[逐层冗余度 E_l]
    C --> D[归一化转重要性<br/>I_l = 1 − E_l/ΣE_i]
    D --> E[β 缩放映射稀疏率<br/>s = p_r + mean(d) − d]
    E --> F{粒度选择}
    F --> G[layer-wise]
    F --> H[block-wise]
    F --> I[projection-wise]
    G & H & I --> J[执行剪枝<br/>误差本身也作为权重重要性指标]
```

### 关键设计

**1. 最小线性重建误差作为冗余度量：把「剪枝后输出差异」化成子矩阵求和。** 对全连接层 $W\in\mathbb{R}^{c_o\times c_i}$ 和输入激活 $X$，剪枝掩码 $M$ 下的线性重建误差为 $E=\lVert WX^T-(M\odot W)X^T\rVert_2^2$。作者推导出：令 $H=X^TX$、$S=H\odot(W^TW)$，被剪输入通道集合为 $P$，则误差可简洁写成 $E=\sum_{i\in P}\sum_{j\in P}S_{i,j}$，即在 $S$ 中选一个 $c_s\times c_s$（$c_s=\lfloor c_i\cdot p\rfloor$）子矩阵使其元素和最小。这把「剪哪些权重」转化成「选哪个子矩阵使和最小」，是后续高效求解的基础。

**2. 贪心 + 分组的高效求解：避免组合爆炸与全矩阵存储。** 穷举所有 $C(c_i,c_s)$ 个子矩阵不可行。LSA 用向量 $\epsilon$ 记录「再剪某通道带来的增量误差」（初值取 $S$ 的对角线），每步贪心选 $\epsilon_j$ 最小的通道、用 $S$ 对应行列更新 $\epsilon$、并把已选通道误差置为 $\infty$。为支持非结构化剪枝，对每个输出通道 $k$ 定义 $S^{(k)}=H\odot(W_{k,:}^TW_{k,:})$，且只在更新时按需计算所需行：$e_{k,:}\leftarrow e_{k,:}+2W_{k,i}(W_{k,:}\odot H_{i,:})$。进一步把所有输出通道**向量化并行**、把 $c_i$ 个输入通道**按大小 $B$ 分组**逐组处理（Algorithm 1），在精度与速度间取得平衡。注意整个过程只产出误差、不真正剪权重。

**3. 误差→重要性→稀疏率的 β 缩放映射：防止过剪崩溃。** 第 $l$ 层重要性定义为 $I_l=1-E_l/\sum_i E_i$，重要性低的层应分更高稀疏。但原始分数尺度需随目标稀疏 $p_r$ 调整，作者引入超参 $\beta$ 把重要性压缩到 $[0,2\beta]$，令 $d=I\times2\beta$，按 $s=p_r+\mathrm{mean}(d)-d$ 分配，使各层稀疏率被约束在 $[p_r-\beta,p_r+\beta]$，从而限制极端剪枝、避免性能塌方。作者还发现计算误差用的 $p$ 不敏感（低于 70% 结果都接近），故全程固定 $p=50\%$。

**4. 细粒度（block / projection）分配：首个不崩塌的更细稀疏方案。** OWL/DLP 只能在层级共享一个重要性分数，下探到 block 或 projection 就会破坏层间信息流而崩溃。LSA 把同一逻辑推广到更细粒度，按 $s=(p_r\times N+(\mathrm{mean}(d)-d)\times\mathrm{mean}(N))/N$（$N$ 为各 projection 权重数）保证整体稀疏率不变。这是首个能在 projection 级做非均匀分配而不出现灾难性退化的方法，block-wise 甚至常优于 layer-wise。

## 实验关键数据

### 主实验：三种粒度对比（WikiText PPL，70% 稀疏）

| 方法 | LLaMA1-7B 层/块/投影 | LLaMA2-7B 层/块/投影 |
|------|---------------------|---------------------|
| Dense | 5.68 | 5.47 |
| SparseGPT+OWL | 18.98 / 25.93 / 29.87 | 20.68 / 27.39 / 29.91 |
| SparseGPT+DLP | 17.78 / 24.78 / 23.26 | 18.68 / 29.64 / 28.05 |
| SparseGPT+**Ours** | **17.57 / 18.25 / 19.46** | **18.63 / 20.40 / 21.15** |
| Wanda+OWL | 24.85 / 57.91 / 85.39 | 30.03 / 52.57 / 80.32 |
| Wanda+DLP | 20.89 / 40.81 / 52.69 | 22.85 / 59.87 / 117.84 |
| Wanda+**Ours** | **20.66 / 21.60 / 24.82** | **22.89 / 25.55 / 34.56** |

关键观察：OWL/DLP 从 layer 下探到 block/projection 时 PPL 暴涨（如 Wanda+DLP 投影级 117.84），而 LSA 在三种粒度间几乎稳定。

### 消融/补充实验

| 设置 | 结论 |
|------|------|
| block-wise on LLaMA3（70%，PPL） | LSA(B) 全面最优：3-8B 32.94 vs DLP 40.12；3.2-1B 87.08 vs DLP 112.29；3.2-3B 46.24 vs DLP 54.86 |
| LOD 与 PPL（LLaMA1/2-13B，70%） | LSA 保留 outlier 最有效，LOD 最高、PPL 最低（如 LLaMA2-13B：LSA 246.05 / 12.56 vs DLP 237.37 / 13.39） |
| $\beta$ 鲁棒区间（block-wise） | OWL/DLP 仅在 $[0,0.07]$ 有效，LSA 在 $[0,0.17]$ 都稳定 |
| 计算误差用的 $p$ | $p<70\%$ 结果接近，对 $p$ 不敏感，固定 50% |

### 关键发现
- LSA 在 70% 高稀疏下于语言建模与七项零样本任务上超越 SOTA。
- 浅层冗余低（少剪）、深层冗余高（可多剪），与「深层往往没想象中重要」的近期结论一致。
- 误差本身比 Wanda 分数更好的权重重要性指标——它考虑了多权重同时剪除时累积的额外误差。

## 亮点与洞察
- **范式转换**：从「逐权重打分 + 人工 reduce 函数」转向「直接用层级重建误差度量冗余」，去掉了一个难调且不可泛化的经验环节。
- **数学化简漂亮**：把剪枝误差化成 $S$ 子矩阵元素和，再用贪心 + 分组 + 向量化把不可行的组合搜索压成可在 LLM 上跑的算法。
- **首探 projection 级非均匀稀疏**：打破了 DLP「细粒度必崩」的论断，且 block-wise 常优于 layer-wise，给后续更细粒度分配开了口。
- **强鲁棒性**：对 $\beta$ 和误差计算用的 $p$ 都不敏感，工程上更易落地。

## 局限与展望
- 仍属非结构化稀疏，实际推理加速依赖专用稀疏算子/硬件支持，未直接给端到端加速数据。
- 需要校准数据计算 $H=X^TX$，对校准集分布有一定依赖。
- $\beta$ 虽鲁棒但仍是需按目标稀疏调的超参，未实现完全自适应。
- 主要在 LLaMA 系与少量 7B 模型验证，超大规模（70B+）与更多架构上的表现待进一步检验。

## 相关工作与启发
- **一次性剪枝**：SparseGPT（幅度 × 逆 Hessian）、Wanda（幅度 × 激活 L2）奠定无重训剪枝基线，但用统一稀疏。
- **层级非均匀稀疏**：OWL（按 outlier 分布 LOD 分配）、DLP（用权重分数中位数刻画冗余）是直接对手，LSA 指出二者共有的「Wanda 打分 + reduce 函数」瓶颈。
- **重建误差剪枝**：ThiNet、Zhuang et al. 在 CNN 上最小化通道重建误差，但需反复前向、复杂度约通道数平方；LSA 通过预计算误差 + 贪心增量更新使其可用于 LLM。
- **启发**：把「重要性度量」从启发式打分换成可推导的误差目标，是值得在量化、KV-cache 压缩等其他 LLM 压缩任务复用的思路。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 用最小线性重建误差替代权重打分度量层冗余，并首次做到 projection 级非均匀稀疏不崩塌，视角新颖。
- **实验充分度**: ⭐⭐⭐⭐ 覆盖 LLaMA1/2/3、Vicuna/Mistral/Qwen 多模型，含三粒度、LOD、$\beta$ 鲁棒性、零样本等多维对比，较为扎实。
- **写作质量**: ⭐⭐⭐⭐ 问题动机清晰，公式推导与算法伪代码完整，图表配合到位。
- **价值**: ⭐⭐⭐⭐ 给无重训 LLM 剪枝提供更优且鲁棒的层级稀疏分配方案，并开启更细粒度方向，实用性强。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Large Language Model Compression with Global Rank and Sparsity Optimization](large_language_model_compression_with_global_rank_and_sparsity_optimization.md)
- [\[ICLR 2026\] RCPU: Rotation-Constrained Error Compensation for Structured Pruning of Large Language Models](rcpu_rotation-constrained_error_compensation_for_structured_pruning_of_large_lan.md)
- [\[ICLR 2026\] MaskPro: Linear-Space Probabilistic Learning for Strict (N:M)-Sparsity on LLMs](maskpro_linear-space_probabilistic_learning_for_strict_nm-sparsity_on_llms.md)
- [\[ACL 2026\] Adaptive Layer Selection for Layer-Wise Token Pruning in LLM Inference](../../ACL2026/model_compression/adaptive_layer_selection_for_layer-wise_token_pruning_in_llm_inference.md)
- [\[ICLR 2026\] SERQ: Saliency-Aware Low-Rank Error Reconstruction for LLM Quantization](serq_saliency-aware_low-rank_error_reconstruction_for_llm_quantization.md)

</div>

<!-- RELATED:END -->
