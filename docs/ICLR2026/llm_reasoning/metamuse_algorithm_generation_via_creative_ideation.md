---
title: >-
  [论文解读] MetaMuse: Algorithm Generation via Creative Ideation
description: >-
  [ICLR 2026][Reasoning][创意构思] 针对 LLM 在生成系统算法时被"可用性偏差"困在 LRU/LFU 等经典启发式上的问题，MetaMuse 提出三条自我反思原则（在性能反馈空间度量多样性、用外部刺激而非内部随机性引导、用路标推理而非自由 CoT 落地），让 LLM 在不连续解空间中做"创意跃迁"，在云厂商真实负载上把缓存缺失最多降 35.76%、装箱用量最多降 30.93%。
tags:
  - "ICLR 2026"
  - "Reasoning"
  - "创意构思"
  - "可用性偏差"
  - "自我反思"
  - "路标推理"
  - "缓存替换"
  - "在线装箱"
---

# MetaMuse: Algorithm Generation via Creative Ideation

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=qJsShqQgWw](https://openreview.net/forum?id=qJsShqQgWw)  
**代码**: 待确认  
**领域**: LLM 推理 / 算法生成 / 系统算法设计  
**关键词**: 创意构思, 可用性偏差, 自我反思, 路标推理, 缓存替换, 在线装箱  

## 一句话总结
针对 LLM 在生成系统算法时被"可用性偏差"困在 LRU/LFU 等经典启发式上的问题，MetaMuse 提出三条自我反思原则（在性能反馈空间度量多样性、用外部刺激而非内部随机性引导、用路标推理而非自由 CoT 落地），让 LLM 在不连续解空间中做"创意跃迁"，在云厂商真实负载上把缓存缺失最多降 35.76%、装箱用量最多降 30.93%。

## 研究背景与动机

**领域现状**：系统算法（缓存替换、作业调度的装箱）直接决定系统性能，但人工设计极其昂贵——作者在某全球云厂商的经验是即便看似简单的算法也要数万工程小时，于是工程师往往退而求其次用 LRU、LFU、First-Fit 这类通用启发式，性能并非最优。一个自然的想法是让 LLM 来自动生成这些算法。

**现有痛点**：系统算法的解空间是**不连续**的——数据结构或控制流上一点小改动就能带来性能的剧烈非线性变化，没有平滑 landscape 可供搜索。作者把 GPT-4o、Llama3.3-70B、DeepSeek-V3 反复采样后发现：LLM 生成的解高度聚集在文献里耳熟能详的启发式（LRU/LFU/FIFO）周围，这是 next-token 预测带来的**可用性偏差**（availability bias）——模型倾向输出训练数据里高频出现的序列。

**核心矛盾**：想用 LLM 在不连续解空间里"跃迁"探索新设计，但 LLM 的生成机制本身就把它往高频经典解上拉。作者证明这个偏差**靠调温度（temperature）治不了**：升温只是对 softmax 概率分布做单调平滑，保留了候选 token 的相对排序，温度推到极端只会让分布趋于均匀、输出语无伦次，并不能真正跳出偏差；让 LLM 基于已有解做变异，它也只会反复微调打分函数，忽视数据结构、层级架构等其他设计维度。

**本文目标**：把算法生成形式化为在解空间上的**采样过程**——每步 LLM 生成一个不同的可执行解，多步累积"既多样又有用"的解后选最优。要实现这个目标需要一个自我反思框架，通过审视"已生成了哪些解"来决定"接下来怎么生成"。

**核心 idea**：**外部刺激驱动的自我反思**——MetaMuse 主动调用 LLM"以为与问题无关"的知识（如随机英文单词），逼它把这些刺激联想映射到问题域，从而做出经典解之外的跃迁，并用性能反馈而非语义嵌入来真正区分解的差异。

## 方法详解

### 整体框架
MetaMuse 是一个迭代式创意构思框架，每次迭代输出一个可执行解。一次迭代依次走三步：先在**性能反馈空间**评估已有解的多样性（§3.1），再据此挑选一组**外部刺激**（关键词）引导构思方向（§3.2），最后用**路标推理**把看似无关的刺激逐步发展成可执行 Python 代码（§3.3）。三步分别对应三条自我反思原则，模型无关。

```mermaid
flowchart LR
    A[已生成解集] --> B[1.评估多样性<br/>性能反馈嵌入]
    B --> C[2.外部刺激引导<br/>RSDict / RSDict-SF]
    C --> D[3.路标推理落地]
    D --> E1[属性提取] --> E2[问题映射] --> E3[方案构造] --> E4[代码生成]
    E4 --> F[可执行解 + 反馈]
    F -.评测hit ratio.-> A
```

### 关键设计

**1. 在性能反馈空间度量多样性：用"结果不同"取代"描述不同"**。MetaMuse 把每个解表示为一个**反馈嵌入** $[p_1, p_2, \dots, p_n]$，其中 $p_i$ 是该解在第 $i$ 条工作负载 trace 上的性能指标（如缓存命中率），$n=30$ 条 trace 由 libCacheSim 从不同 Zipfian/Weibull 分布合成。这么做是因为**语义不同未必是真不同**：GPT-4o 曾从"淘汰访问最少的对象"和"淘汰优先级最低的对象、命中时优先级+1"两段不同描述生成了功能等价的 LFU；用链表还是优先队列实现也可能得到一样的命中率。反馈嵌入有两个好处——每一维都是有量纲的定量指标，两个嵌入的欧氏距离直接反映解的差异（上面两个 LFU 距离为 0，判为等价）；指标天然有界（如命中率 0–100%），使嵌入空间结构化、有界，便于后续计算"引导方向"。

**2. 用外部刺激而非内部随机性引导构思**。MetaMuse 不靠模型内部随机性（温度）探索，而是把**外部刺激**作为构思起点——这些刺激（一组英文关键词）可能与问题域毫不相干，逼 LLM 去联想"概率上看似无关"的知识。实现上去掉停用词得到 2899 个常用词，每个解从 $s=4$ 个关键词构思。挑刺激有两种策略：**RSDict** 是无状态策略，纯随机抽 $s$ 个词，适用于评测不可行或昂贵（如需人工评判）的场景；**RSDict-SF** 在此基础上利用历史解的反馈嵌入算出一个**引导方向**（目标反馈嵌入）——探索时取距已有解集欧氏距离最远的点（求多样），利用时把各维设成高值（如全 trace 100% 命中，求有用）。给定目标嵌入，RSDict-SF 把"哪组刺激能逼近目标"建模为预测问题，用 $n$ 个高斯过程回归（GPR）逐维预测某组刺激将得到的反馈嵌入，无需真生成解。GPR 采用点积核，对刺激顺序不变：

$$K(c_i, c_j) \propto \sum_{p=1}^{s}\sum_{q=1}^{s} \phi(o_{i,p})^\top \phi(o_{j,q})$$

其中 $o_{i,p}$ 是把第 $p$ 个刺激映射到问题域后 LLM 生成的自然语言观察句，$\phi(\cdot)$ 是其 SBERT 语义嵌入。由于 $3000^s$ 种刺激组合无法穷举预测，MetaMuse 用 **power-of-two random choices**：随机抽两组刺激各预测一次反馈嵌入，选更接近目标的那组；RSDict-SF 还需先用 RSDict 生成 $w=100$ 个解 bootstrap 才能起训 GPR。

**3. 用路标推理把无关刺激落地为可执行解**。直接让 LLM 从关键词写代码，它往往只把词塞成变量名敷衍了事。MetaMuse 设置四个**路标**（waypoint）作为中间检查点，强制 LLM 一步步把刺激转化为真正的算法设计：**属性提取**（由"flower"联想出"a ring of petals"等性质）→ **问题映射**（把"一圈花瓣"联想成环形结构，映射到算法里的 circular buffer）→ **方案构造**（综合各观察合成完整解描述）→ **代码生成**（把描述翻成可执行 Python）。相比自由形式 CoT，路标式的结构化分解能防止 LLM 表面化地处理刺激，是多样性提升的关键来源。

## 实验关键数据

两个云厂商真实高价值问题：缓存替换（96 条真实 trace：RetrievalAttention / 腾讯块存储 / 阿里云存储）与在线装箱（288 条 trace：BPPLIB / Weibull / Gaussian）。每个实验各方法生成 350 个可执行解，对比 5 个 LLM 基线 + 16 个人工启发式（共 21 个）。

### 主实验：高性能解

| 任务 | 模型 | 相对 LLM 基线 | 相对人工启发式 |
|------|------|---------------|----------------|
| 缓存替换（90 百分位 trace） | GPT-4o | 缺失率低 5.17%–9.89% | 低 1.75%–13.03% |
| 缓存替换（75 百分位 trace） | GPT-4o | 低 3.62%–6.39% | 低 6.62%–**35.76%** |
| 在线装箱（90 百分位 trace） | GPT-4o | 用量低 9.25%–9.42% | 低 9.25%–20.59% |
| 在线装箱（90 百分位 trace） | DeepSeek-V3 | 低至多 **21.06%** | 低至多 **30.93%** |

MetaMuse 在几乎所有百分位、跨三种 LLM 上都给出更高的性能降幅。

### 多样性与成本

| 指标 | 结果 |
|------|------|
| 缓存替换 distinct 解（vs LLM 基线，跨模型均值） | 1.47×（GPT-4o）/ 1.57×（DeepSeek）/ **1.78×**（Llama3.3-70B） |
| 装箱 distinct 解（vs LLM 基线） | 1.44×（GPT-4o）/ **1.80×**（DeepSeek）/ 1.31×（Llama） |
| LRU 簇密度下降 | 35.28%（6.15→3.98）；ARC 簇下降 77.45%（1.02→0.23） |
| 单解平均成本（GPT-4o） | **2.16 美分**（DeepSeek 2.11 / Llama 2.35），低于 Repeated Sampling 的 3.38 美分 |
| 不安全解占比 | ~2.28%（超时/超内存/非法行为，自动丢弃重做） |

### 消融实验

| 配置 | distinct 解数（GPT-4o，缓存） |
|------|------|
| RSDict | 149 |
| RSDict + 路标推理 | 175 |
| RSDict-SF（无路标 noWR） | 152 |
| RSDict-SF + 路标推理（完整） | **197** |

- **刺激引导有效**：90 百分位 trace 上 67.20% 的 RSDict 解、75.60% 的 RSDict-SF 解优于 Repeated Sampling；RSDict-SF 比 RSDict 多 13.17% distinct 解。
- **路标推理有效**：去掉路标推理后 distinct 解显著下降（如 RSDict-SF 197→152），三种 LLM 上一致。

### 关键发现
- **温度治标不治本**：单调平滑保留 token 排序，无法跳出可用性偏差——这是全文最关键的反直觉论证。
- MetaMuse 产出了工程师"想不到"的设计：MetaMuse-533 用 NSE 计数器追踪对象经历过多少次淘汰事件（反常地偏好久驻对象、防 thrashing），并用饱和计数器累积访问历史识别 bursty 对象；MetaMuse-488 用哈希函数分段缓存做 per-group 替换，而非常见的多级架构。

## 亮点与洞察
- **把"创造力"接地到可度量的反馈空间**：用性能向量而非语义嵌入定义多样性，绕开了"换皮 LFU"这类伪多样性，这是让自动搜索真正有效的前提。
- **外部刺激 + 路标推理的组合很巧**：刺激负责"跳出去"（打破可用性偏差），路标负责"落得下"（防止 LLM 把刺激敷衍成变量名），两者缺一不可，消融数据支撑得很扎实。
- **GPR + power-of-two random choices** 让 RSDict-SF 能在不实际生成解的前提下预筛刺激，把"试错"成本压到单解 2 美分量级，工程可落地性强。
- 结论根植于真实云厂商负载与真实生产问题，而非玩具任务，说服力更高。

## 局限与展望
- 只验证了两类系统算法（缓存替换、在线装箱），是否能推广到更复杂的系统设计（调度、路由、一致性协议）未知。
- 反馈空间多样性依赖能快速仿真评测的环境（libCacheSim 等），对没有廉价仿真器、评测昂贵的问题，只能退回无状态的 RSDict，失去自我反馈引导。
- 刺激来源目前是通用英文词典 + 去停用词，技术术语等"低价值"词仍可能被误用为变量名；刺激选择策略（$s$、词典构造）较启发式，缺少更系统的设计。
- 代码生成统一交给 GPP-4o，构思模型与编码模型解耦，端到端用单一弱模型时表现如何未充分探讨。

## 相关工作与启发
- **自动启发式设计**（FunSearch、ReEvo、MCTS-AHD、OpenEvolve 等）多靠 LLM 对初始种群做变异/交叉演化，受初始种群限制且仍受可用性偏差影响（偏向改打分函数）；MetaMuse 的差异是直接在解空间做跃迁、用外部刺激扩展可达解空间。
- **人机协同产多样输出**需要人参与、引入人类偏差；MetaMuse 主张 LLM 自身就能"think outside the box"，靠三条自我反思原则实现。
- 启发：可用性偏差是 LLM 用于"发现/创造"类任务的根本障碍，而把多样性接地到可度量反馈空间、用结构化路标约束推理，是一套可迁移到其他生成式发现任务（如分子设计、电路设计）的通用配方。

## 评分
- **新颖性**: ⭐⭐⭐⭐ 把算法生成形式化为不连续解空间采样、用外部刺激打破可用性偏差、反馈空间度量多样性，三点组合新颖且论证扎实（尤其温度无效的分析）。
- **实验充分度**: ⭐⭐⭐⭐ 两个真实生产问题、跨三种 LLM、21 个基线、多样性/成本/消融/案例分析齐全；略减分在仅两类系统算法、未含更复杂任务。
- **写作质量**: ⭐⭐⭐⭐ 动机—原则—方法—验证逻辑清晰，三条原则一一对应三步，图 2 pipeline 直观。
- **价值**: ⭐⭐⭐⭐ 真实云厂商场景下显著超越人工启发式（缺失最多降 35.76%）、单解成本仅 2 美分，工程落地价值高，且方法范式可迁移。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] Exposing Weaknesses of Large Reasoning Models through Graph Algorithm Problems](exposing_weaknesses_of_large_reasoning_models_through_graph_algorithm_problems.md)
- [\[ICLR 2026\] Reverse-Engineered Reasoning for Open-Ended Generation](reverse-engineered_reasoning_for_open-ended_generation.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ICLR 2026\] Log-Augmented Generation: Scaling Test-Time Reasoning with Reusable Computation](log-augmented_generation_scaling_test-time_reasoning_with_reusable_computation.md)
- [\[ICLR 2026\] Unleashing Scientific Reasoning for Bio-Experimental Protocol Generation via Structured Component-based Reward Mechanism](unleashing_scientific_reasoning_for_bio-experimental_protocol_generation_via_str.md)

</div>

<!-- RELATED:END -->
