# Controlling Thinking Speed in Reasoning Models

**会议**: NeurIPS 2025  
**arXiv**: [2507.03704](https://arxiv.org/abs/2507.03704)  
**代码**: 基于 vLLM 实现  
**领域**: LLM推理效率 / 表示工程  
**关键词**: thinking speed, representation engineering, System 1/2, test-time scaling, steering vector  

## 一句话总结
通过表示工程（Representation Engineering）从 LRM 的隐藏空间中提取控制快/慢思考转换的 steering vector，结合基于层间 logit 散度的实时推理难度估计，实现无需训练的自适应推理速度调节，在 4 个 LRM 上平均提升 +1.3% 准确率并减少 -8.6% token 使用。

## 研究背景与动机

1. **领域现状**：大型推理模型（LRM）如 DeepSeek-R1、OpenAI o1 通过长链式思维（Long CoT）实现 System 2 级别的深度推理。但这种模式对所有推理步骤一律使用复杂思考，导致大量冗余计算。

2. **现有痛点**：现有加速方案要么依赖 prompt 控制 token 预算（如 Budget Forcing 的 "Final Answer:" 截断、Thought Extrapolation 的 "Wait" 追加），但 LRM 对 prompt 中的时间/长度约束不敏感；要么需要额外训练（如在快慢推理 trace 上微调），成本高昂。

3. **核心矛盾**：人类推理在一个问题内部也会动态切换快/慢思考——简单步骤快速跳过，关键步骤深入分析。但现有 LRM 缺乏在单次推理中段落级动态调速的能力。

4. **本文要解决什么**：(1) 如何在推理过程中实现快/慢思考的平滑切换，(2) 何时切换才能最优地平衡效率和准确率。

5. **切入角度**：作者发现一个有趣现象——LRM 的短回复和长回复有截然不同的开头词（短回复以 "To"、"First" 开头，长回复以 "Okay"、"Alright" 开头），说明快/慢思考模式在模型表示空间中是可分离的。基于 Representation Engineering（RopE）可以提取控制这种思考模式转换的方向向量。

6. **核心idea一句话**：用 PCA 从 LRM 隐藏层中提取"快→慢"思考的 steering vector，在推理时通过加/减该向量实现 token 级别的思考速度连续控制，并用早晚层 logit 散度作为实时难度信号驱动自适应调速。

## 方法详解

### 整体框架
方法分为两部分：(1) Thinking Speed Control——通过表示工程提取思考速度 steering vector 并在推理时注入；(2) Adaptive Control——基于实时推理难度估计的滑动窗口算法，动态调节 steering intensity。整个方法是纯推理时的 plug-in，无需任何训练。

### 关键设计

1. **快/慢思考 Steering Vector 提取（Representation Reading）**：
   - 做什么：从 LRM 表示空间中找到控制思考速度的方向向量
   - 核心思路：用 MATH 训练集的 7.5k 问题采样快思考（以 "To" 开头）和慢思考（正常开头）的配对响应。截取两种响应的前 2 个推理步骤作为 stimuli，收集最后一个 token 位置的各层隐藏状态 $(h_i^+, h_i^-)$，计算差值向量 $d_i = h_i^+ - h_i^-$（一半正向，一半反向），然后用 PCA 提取第一主成分作为 steering vector $v$
   - 设计动机：RopE 理论认为高层语义概念编码为隐空间中的线性方向。快/慢思考作为高层认知功能，应该也服从此规律。PCA 验证集分类准确率接近 100%，证实了这一假设

2. **推理时表示控制（Representation Controlling）**：
   - 做什么：在每个 token 生成时注入 steering vector 来调节思考速度
   - 核心思路：对目标层 $l \in L$，修改隐状态 $h^l \leftarrow h^l + \alpha \cdot v^l$。$\alpha > 0$ 加速思考（更简洁），$\alpha < 0$ 减速思考（更深入，含反思和回溯）
   - 设计动机：相比 prompt 方法（如追加 "Wait"、截断），表示级操作保留了自然推理流，不会打断模型的推理逻辑。实验显示在相同 token 预算下，表示控制比 Budget Forcing 平均高出 +11.4% Pass@1

3. **实时推理难度估计**：
   - 做什么：在推理过程中逐 token 判断当前推理步骤的难度
   - 核心思路：利用早期层和最终层的 next-token 分布之间的 Jensen-Shannon 散度来度量难度：$d(x_t) = \text{avg}_{l \in L_e} \text{JSD}(p^N(\cdot|x_{<t}) \| p^l(\cdot|x_{<t}))$。高散度意味着需要深层处理，对应反思、计算、逻辑推演等复杂推理行为
   - 设计动机：研究表明 LLM 在处理复杂信息时，早期层和后期层的 logit 差异更大。验证发现 logit 散度最高的 100 个 token 确实对应反思词（Wait, Alternatively）、计算词（equals, multiply）和分析词（analysis, need）

4. **滑动窗口自适应调速算法**：
   - 做什么：根据实时难度信号动态调节 $\alpha$
   - 核心思路：维护最近 $k=8$ 个 token 的难度窗口 $W$，若当前 token 难度超过阈值 $\mu_W + \lambda \cdot \sigma_W$（类似异常检测），则将 $\alpha$ 设为 $\alpha_{\min}$（减速/踩刹车）；否则逐步增大 $\alpha$（加速）直到上限
   - 设计动机：模拟人类推理——快速跳过简单步骤，遇到关键推理点则放慢深入思考

### 损失函数 / 训练策略
无需训练。Steering vector 仅需在少量数据上一次性提取。自适应控制算法作为推理时 plug-in 直接集成到 vLLM 中。

## 实验关键数据

### 主实验（自适应控制）

| 模型 | 方法 | MATH-500 | AIME24 | AIME25 | GPQA Diamond |
|---|---|---|---|---|---|
| DS-R1-Distill-7B | Original | 92.9 / 3404 | 52.5 / 12451 | 40.0 / 13689 | 46.6 / 6189 |
| | 1xWait | 92.7 / 3744 | 52.1 / 12704 | 39.6 / 13867 | 45.9 / 9376 |
| | **Adaptive** | **93.7 / 3123** | **53.8 / 10851** | **42.3 / 12380** | **48.8 / 5422** |
| QwQ-32B | Original | 97.4 / 4305 | 76.7 / 13627 | 65.8 / 15852 | 62.7 / 7969 |
| | **Adaptive** | **97.4 / 4134** | **77.8 / 12365** | **67.4 / 15150** | **64.1 / 7639** |
| Qwen3-8B | Original | 96.8 / 5456 | 75.0 / 14754 | 62.9 / 17797 | 60.1 / 8379 |
| | **Adaptive** | **97.1 / 5171** | **77.5 / 13629** | **65.4 / 17411** | **61.2 / 7894** |

跨 4 个 LRM 和 4 个 benchmark 平均：+1.3% 准确率，-8.6% token，完全无需训练。

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| 随机控制 vs. 难度驱动控制 | 难度驱动在所有指标上优于随机 | 证明 JSD 难度信号的有效性 |
| $\lambda$ 从 1.0→1.5→2.0→2.5 | $\lambda$ 越小，更多 token 被判为困难，准确率↑ token↑ | 直接控制 speed-accuracy trade-off |
| $\alpha_{\max}$ 从 2→4→6→8 | $\alpha_{\max}$ 越大，response 越短，准确率可能降 | 极端值（>16 或 <-8）导致重复生成 |
| 去掉 function-calling 格式 | DA 对简单任务效率至关重要 |  |
| Stimulus 位置选择 | 用初始片段末尾 token 的表示效果最好 | 首 token（开头词嵌入差异）和完整思考链（含 EOS 信号）都不如 |

### 关键发现
- **开头词决定思考模式**：仅在 LRM 的 `<think>` 后追加 "To" 就能实现 5.4× token 压缩（7B），同时保留 60% 准确率。这说明 LRM 内部存在天然的快/慢切换机制
- Thought Extrapolation（"Wait"）效果很差甚至负面：增加反思次数后准确率反而下降，说明 prompt 级控制与模型内部状态脱节
- "Wait" token 与真正的慢思考模式高度不相关：AIME24 上模型平均输出 55.3 个 "Wait"，但仅 1/12.2 对应真正的模式切换
- 与并行搜索（Best-of-N）正交且可组合：在低 token 预算下，加速后的模型在 parallel search 中显著优于 vanilla 和 NoThinking

## 亮点与洞察
- **表示工程 → test-time scaling 的首次实现**：之前表示工程主要用于编辑事实知识或控制情感，本文首次将其用于控制推理速度，实现了 prompt 方法无法做到的 token 级连续调控
- **实时难度估计**的设计非常优雅：利用 Transformer 不同深度层间 logit 散度作为认知复杂度的代理指标，无需额外模型。高散度 token 精确对应反思、计算、推导等行为，是一个非常有启发性的发现
- **完全无训练 + vLLM plug-in**：实用价值极高。可以直接部署到现有 LRM 服务中，无需修改模型权重

## 局限性 / 可改进方向
- 不同 LRM 对 steering intensity 的敏感度不同，目前用统一的 $\alpha$ 范围，未来需要 model-agnostic 的自动调节
- 滑动窗口算法基于启发式规则，理想方案是端到端优化难度信号与 steering intensity 的映射
- 仅在数学推理和编程任务上验证，对开放式生成（如创意写作、对话）的效果未知
- 极端 $\alpha$ 值会导致重复生成，表示空间可能不是完全线性的

## 相关工作与启发
- **vs ARM（2505.20258）**：ARM 通过 RL 训练模型选择推理格式，需要额外训练；本文完全 training-free，通过表示编辑实现更细粒度的 token 级控制。两者互补——ARM 选择宏观格式，本文在微观层面调节推理强度
- **vs Budget Forcing / s1**：这些方法在固定位置截断或延长推理，是 prompt level 的粗粒度控制。本文的表示控制在 token level 平滑调节推理风格，且在相同 token 预算下平均高 11.4%
- **vs Thought Extrapolation**：通过追加 "Wait" 强制模型反思，但实验证实效果不稳定甚至有害。本文揭示原因——"Wait" token 与真正的内部认知转换低度相关

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首次将表示工程用于推理速度控制，发现 LRM 内部快/慢模式的线性可分性和开头词触发机制，均为全新 insight
- 实验充分度: ⭐⭐⭐⭐⭐ 4 个 LRM（DS-R1-7B/32B, QwQ-32B, Qwen3-8B），4+ 个 benchmark，大量消融和案例分析
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰，System 1/2 的类比准确，图表直观
- 价值: ⭐⭐⭐⭐⭐ 无训练 plug-in 设计极具部署价值，JSD 难度信号可迁移到其他推理优化场景
