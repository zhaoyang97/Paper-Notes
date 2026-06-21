---
title: >-
  [论文解读] Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models
description: >-
  [ICLR 2026][LLM效率][投机解码] CAST 在 EAGLE-2/3 的动态草稿树上引入"推理成本"这一被忽视的系统变量（GPU 型号、batch size），把树的宽度、深度和验证 token 数建模成"接受收益 vs 推理代价"的效用最大化问题，从而在单样本场景小幅、在 batch 场景大幅超越现有 SOTA，最高可达 5.2× 加速。
tags:
  - "ICLR 2026"
  - "LLM效率"
  - "投机解码"
  - "动态草稿树"
  - "推理成本感知"
  - "EAGLE"
  - "批处理加速"
---

# Inference-Cost-Aware Dynamic Tree Construction for Efficient Inference in Large Language Models

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iaWyRYthFf](https://openreview.net/forum?id=iaWyRYthFf)  
**代码**: [https://github.com/EAGLE-Research/sglang-eagle4](https://github.com/EAGLE-Research/sglang-eagle4)  
**领域**: LLM 高效推理 / 投机解码 (Speculative Decoding)  
**关键词**: 投机解码, 动态草稿树, 推理成本感知, EAGLE, 批处理加速  

## 一句话总结
CAST 在 EAGLE-2/3 的动态草稿树上引入"推理成本"这一被忽视的系统变量（GPU 型号、batch size），把树的宽度、深度和验证 token 数建模成"接受收益 vs 推理代价"的效用最大化问题，从而在单样本场景小幅、在 batch 场景大幅超越现有 SOTA，最高可达 5.2× 加速。

## 研究背景与动机
**领域现状**：投机解码（speculative decoding）通过"轻量草稿模型提议 + 大目标模型并行验证"在不损失输出分布的前提下加速 LLM 推理。EAGLE 系列把链式草稿升级为树状草稿，EAGLE-2/3 进一步用置信度分数（confidence score）动态构建草稿树，逐层取 top-K、再 rerank 出 top-m 个 token 送验证，是当前 SOTA。

**现有痛点**：EAGLE-2/3 构树规则纯靠启发式，只看"token 接受率"，**完全无视 GPU 硬件与 batch size 这两个关键系统变量**。其隐含假设是"草稿 token 越多越好"，但这只在 batch size=1、GPU 利用率不饱和时成立。

**核心矛盾**：当开启批处理时，GPU 利用率本就被 batch 拉高，再盲目加深加宽草稿树会让 draft/verify 阶段争抢 GPU 资源，**反而拖慢整体速度**——论文实测中 EAGLE-2 在 batch=8 的 CNN/DM 上甚至降到 0.84×（比原始自回归还慢）。即"token 多 ≠ 更快"，存在一个临界点，超过它收益为负。

**本文目标**：把"推理成本"显式纳入草稿树构建，动态决定树深、每层 token 数、以及最终送目标模型验证的 token 数，让加速在不同 GPU、不同 batch 下都稳健。

**核心 idea**：**[成本感知效用最大化]** 借用经济学效用理论——效用函数是凹的、边际效用递减。把"接受 token 的累积置信度"当效用 $u$、把"draft 相对 target 的归一化推理时间"当成本 $c$，只保留**边际效用 $\frac{u_k-u_{k_0}}{c_k-c_{k_0}}$ 超过阈值**的节点。这样 EAGLE-2/3 恰好成为本方法在特定参数下的特例。

## 方法详解

### 整体框架
CAST（Cost-Aware Speculative Tree）不重训任何模型，而是把 EAGLE-2/3 草稿树构建的两个阶段——**扩展（expansion）**与**重排（reranking）**——都改写成"成本感知的效用最大化"。前提是离线预计算一张推理时间查找表 $f(B,c,n)$（batch、上下文长、序列长 → 时间），构建草稿树时用它把"多生成 token 的代价"实时算出来，与"接受收益"做权衡。

```mermaid
flowchart LR
    A[上下文 + 草稿模型] --> B[预计算时间查找表 ST/SD]
    B --> C[动态扩展阶段]
    C --> C1[宽度剪枝: 每层留几个节点]
    C --> C2[深度剪枝: 要不要再长一层]
    C1 --> D[动态重排阶段]
    C2 --> D
    D --> D1[按累积置信度排序+成本约束<br/>定验证 token 数]
    D1 --> E[目标模型并行验证]
```

### 关键设计

**1. 预计算推理成本查找表：把硬件代价变成可查的数**  
方法的根基是把"输入长度 $n$、上下文 $c$、batch $B$ 的一次推理耗时" $f(B,c,n)$ 离线测好存表。为省存储只在 $c=kL$、$n=1{\cdots}N$ 的网格点采样，并配一个量化算子 $\text{select}(c)=(\max(\lfloor c/L\rfloor, M-1)+1)L$ 把任意上下文长映射到最近网格。对每个 batch size 维护目标模型表 $S_T(B)$ 与草稿模型表 $S_D(B)$，构树时 $O(1)$ 查表即可获得"再多算 $k$ 个 token 的真实时间代价"。正是这张表让"GPU 型号 + batch size"从被忽视的隐变量变成构树时显式的优化输入。

**2. 动态扩展之宽度剪枝：用边际效用决定每层留几个节点**  
对第 $i$ 层，把候选节点按置信度降序记为 $v_i^{(j)}(s)$（$j$ 是 batch 内样本），选前 $k$ 个的累积效用按 batch 平均定义为
$$u_k^{(i)}=\frac{1}{B}\sum_{j=1}^{B}\sum_{s=1}^{k}v_i^{(j)}(s).$$
对应的归一化成本用查找表算：$c_k^{(i)}=\frac{S_D(B)[\text{select}(\sum_{j<i}n_j)][k]}{S_T(B)[\text{select}(\sum_{j<i}n_j)][1]}$。因为效用是凹的、边际效用递减，就设阈值 $C_1$，只保留**边际效用 $\frac{u_k-u_{k'}}{c_k-c_{k'}}>C_1$** 的节点数 $n_i$（Algorithm 1 处理离散非严格凹的鲁棒选择）。论文证明（Theorem 4.1）：令 $c_j=\lambda j+\delta$、$C_1=\frac{\sum_j v_i^{(j)}(K)}{B\lambda}$ 时，EAGLE-2/3 的固定 top-K 选择就是本式的特例——这正是"CAST 泛化了 EAGLE-2/3"的数学依据。

**3. 动态扩展之深度剪枝：用预测增益判断要不要再长一层**  
宽度定了每层节点数，还需决定**是否继续生成第 $i{+}1$ 层**。维护一个滑动 buffer $A_i$ 追踪近 $R$ 层的"置信增益比" $\phi_i=\frac{u_{n_{i+1}}^{(i+1)}}{u_{n_i}^{(i)}}$，记 $\alpha_i=\text{Avg}(A_i)$。只有当
$$\alpha_i\cdot\frac{u_{n_i}^{(i)}}{c_{n_i}^{(i)}}\ge C_2$$
即"历史平均增益 × 当前层效用成本比"仍够高时才往下长，否则停止。buffer 用 FIFO 更新、初值置 $\{1\}$ 保证浅层稳定。这让树深随上下文难易和硬件成本自适应，不再是固定 $H$。

**4. 动态重排：在成本约束下定送验证的 token 数**  
扩展后草稿树节点过多需裁剪。论文先实测发现：每个样本的**接受长度（accept length）与全树累积置信度 $v$ 近似线性正相关**（Figure 3），所以"取置信度最高的若干节点"是对的。但取多少？同样复用 Algorithm 1：令 $u_k=\sum_{j=1}^{k}v_{(j)}$、$c_k=\frac{S_T(B)[\text{select}(n_0)][k]}{S_T(B)[\text{select}(n_0)][1]}$、阈值 $C_3$，自动选出送目标模型验证的 token 数。关键洞察是验证 token 数 $m$ 越大接受长度越长，但 speedup 在 $m$ 偏大后**反降**（Figure 4），所以作者刻意不用"平均接受长度"作主指标，直接以实测 speedup 为准。

## 实验关键数据

### 主实验（batch size = 1，speedup ratio，T=0）

| 模型 | 方法 | MT-bench | HumanEval | GSM8K | Alpaca | CNN/DM | NatQ |
|------|------|----------|-----------|-------|--------|--------|------|
| V 13B | EAGLE-3 | 3.70× | 4.73× | 4.00× | 3.86× | 3.68× | 3.31× |
| V 13B | **CAST** | **3.98×** | **5.18×** | 3.98× | 3.80× | **3.76×** | **3.40×** |
| L33 70B | EAGLE-3 | 4.13× | 4.98× | 4.63× | 4.66× | 3.50× | 3.61× |
| L33 70B | **CAST** | **4.23×** | **5.23×** | **4.65×** | **4.83×** | **3.56×** | **3.67×** |
| L31 8B | EAGLE-3 | 3.60× | 4.27× | 3.82× | 4.00× | 3.22× | 3.06× |
| L31 8B | **CAST** | **3.77×** | **4.51×** | **3.95×** | 3.98× | **3.32×** | **3.22×** |

单样本下 CAST 普遍小幅领先 EAGLE-3，且目标模型越大优势越明显，HumanEval 上达 5.23×。

### 批处理实验（batch size = 8，speedup ratio，T=0）

| 模型 | 方法 | MT-Bench | HumanEval | GSM8K | Alpaca | CNN/DM | NatQ |
|------|------|----------|-----------|-------|--------|--------|------|
| Q2 7B | EAGLE-2 | 1.25× | 1.49× | 1.40× | 1.48× | 1.11× | 1.10× |
| Q2 7B | **CAST** | **1.86×** | **2.16×** | **2.19×** | **2.06×** | **1.70×** | **1.72×** |
| L31 8B | EAGLE-3 | 1.72× | 1.97× | 1.92× | 2.16× | 1.34× | 1.72× |
| L31 8B | **CAST** | **2.16×** | **2.62×** | **2.41×** | **2.62×** | **1.76×** | **2.11×** |

批处理才是 CAST 的主战场：EAGLE-2 在 Q2 7B / CNN/DM 上掉到 1.11×、EAGLE 甚至到 0.84×（比原始自回归还慢），而 CAST 稳定在 1.7–2.6×，相对提升常达 40–60%。

### 关键发现
- **"token 多 ≠ 更快"得到实证**：Figure 4 显示验证 token 数 $m$ 增大时接受长度单调上升，但 speedup 过临界点后下降——成本感知正是为找这个临界点。
- **batch 越大、模型越大，成本感知收益越显著**，因为此时 GPU 资源争抢最严重，启发式构树最易"过度投机"。
- 全程 lossless：不微调原模型、不改接受条件，输出分布与目标模型一致。

## 亮点与洞察
- **把"系统侧成本"塞进算法侧构树**：第一次把 GPU 型号、batch size 这类被投机解码文献长期忽视的硬件变量，量化成查找表代价并直接驱动树结构决策。
- **用效用理论统一框架**：凹效用 + 边际收益递减的视角既优雅又给出了 EAGLE-2/3 是特例的严格证明（Theorem 4.1），把一堆启发式收编进一个可调阈值的优化问题。
- **指标选择有反思**：明确弃用"平均接受长度"，因为它在大 batch 下与真实 speedup 脱钩，体现作者对 system-level 评测的清醒。
- **即插即用**：CAST 套在 EAGLE-2 或 EAGLE-3 上都行，工程落地到 SGLang，复现门槛低。

## 局限与展望
- **依赖离线查找表**：$f(B,c,n)$ 需对每种 GPU、每个 batch size 预先标定，换硬件/部署环境要重测，泛化到异构或动态 batch（continuous batching）场景的成本未充分讨论。
- **三个阈值 $C_1,C_2,C_3$ 需调**：虽然有理论支撑，但跨任务/模型的最优阈值如何自动确定仍是开放问题，论文用的是经验设定。
- **接受长度与置信度线性假设**：重排阶段建立在"accept length ∝ 累积置信度"的实测线性关系上，该关系在更长生成、更极端温度下是否仍成立未深究。
- **未覆盖 continuous batching / 变长 batch**：实验用 padding 假设 batch 内等长，真实在线服务的动态拼批场景留待后续。

## 相关工作与启发
- **EAGLE / EAGLE-2 / EAGLE-3**（Li et al. 2024b/2024c/2025）：本文直接的基座，从静态树→置信度动态树，CAST 是其成本感知的泛化。
- **投机解码本源**（Leviathan et al. 2023；Chen et al. 2023）：proposal/verification 分离、residual 分布回退保证 lossless。
- **批处理 × 投机解码**（Su et al. 2023；Qian et al. 2024；Wu et al. 2025）：本文补上"树状草稿 + 批处理"这块此前空白，把链式 batching 分析推广到树。
- **启发**：投机解码的下一步优化重心正从"纯算法接受率"转向"算法 × 硬件协同"，凡是涉及动态计算图 + GPU 资源争抢的场景（MoE 路由、动态 KV、推测式 agent 规划）都可借鉴"把硬件代价建成查找表 → 用边际效用剪枝"的范式。

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 把被忽视的 GPU/batch 成本显式纳入草稿树构建，并用效用理论证明 EAGLE-2/3 是特例，视角清新且有理论支撑。
- **实验充分度**: ⭐⭐⭐⭐ — 6 任务 × 6 模型、单样本/批处理、两种温度、多 GPU（附录），batch 场景对比尤其有说服力；但 continuous batching 与跨硬件迁移成本未覆盖。
- **写作质量**: ⭐⭐⭐⭐ — 动机—矛盾—方法链条清晰，公式与算法完整，弃用接受长度指标的讨论体现严谨。
- **价值**: ⭐⭐⭐⭐ — 即插即用、lossless、已落地 SGLang，对真实批处理推理服务有直接工程价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](dnd_boosting_large_language_models_with_dynamic_nested_depth.md)
- [\[ICLR 2026\] PARD: Accelerating LLM Inference with Low-Cost Parallel Draft Model Adaptation](pard_accelerating_llm_inference_with_lowcost_parallel_draft_model_adaptation.md)
- [\[ICLR 2026\] Neuron-Aware Data Selection in Instruction Tuning for Large Language Models](neuron-aware_data_selection_in_instruction_tuning_for_large_language_models.md)
- [\[ICLR 2026\] Reasoning Language Model Inference Serving Unveiled: An Empirical Study](reasoning_language_model_inference_serving_unveiled_an_empirical_study.md)
- [\[ICLR 2026\] Libra: Effective yet Efficient Load Balancing for Large-scale MoE Inference](libra_effective_yet_efficient_load_balancing_for_large-scale_moe_inference.md)

</div>

<!-- RELATED:END -->
