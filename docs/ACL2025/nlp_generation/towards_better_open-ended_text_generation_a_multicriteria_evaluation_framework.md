# Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework

**会议**: ACL 2025 (workshop GEM²)
**arXiv**: [2410.18653](https://arxiv.org/abs/2410.18653)
**代码**: [GitHub](https://github.com/YecanLee/2BeOETG)
**领域**: 文本生成
**关键词**: open-ended text generation, multicriteria evaluation, decoding strategies, Bradley-Terry model, text quality metric

## 一句话总结
针对开放式文本生成中多指标（coherence/diversity/perplexity）之间的权衡问题，提出三种互补的多准则评估方法——Extended Bradley-Terry 模型（序数排名）、Union-Free Generic Depth（允许不可比性的偏序）和 Q*Text（基数评估综合指标），在6个 LLM × 59种解码策略 × 180万+生成文本上验证，发现中等超参配置普遍优于极端配置，小模型+合理解码策略可匹敌大模型。

## 研究背景与动机

1. **领域现状**：LLM 的输出质量不仅取决于模型架构，还取决于推理时的解码策略（beam search、top-k/top-p sampling、contrastive search 等）。评估方法多依赖单一指标或人工评判。
2. **现有痛点**：解码策略天然涉及多指标权衡——优化 coherence 会牺牲 diversity，反之亦然。单一指标评估给出片面结论；现有聚合方法如 Pareto front 对大规模 benchmarking 缺乏信息量，加权求和依赖任意权重选择。
3. **核心矛盾**：如何在多个互相冲突的自动指标间建立有原则的聚合方法，提供可靠的解码策略排名或打分？
4. **本文要解决什么**：(a) 给定多指标，如何建立序数排名（允许不可比性）？(b) 如何设计有统计基础的综合指标进行基数评估？
5. **切入角度**：区分两种实践场景——场景1（只需排名→用偏序理论）和场景2（需量化差距→设计综合指标），分别对应不同方法。
6. **核心 idea**：引入偏序理论中的 depth function 和统计中的 Bradley-Terry 模型到文本生成评估，加上提出 Q*Text 综合指标用高斯惩罚函数平衡极端值。

## 方法详解

### 整体框架
输入：6个 LLM（GPT2-XL~Falcon2-11B）× 5种解码策略 × 59种超参配置 → 180万+生成文本，每条评估 coherence、diversity、generation perplexity 三个指标。输出：解码方法的排名或打分。

### 关键设计

1. **Extended Bradley-Terry Model（场景1：排名）**
   - 做什么：基于成对比较建立解码方法的全序排名
   - 核心思路：对每个 prompt，354个解码方法两两比较（一个方法在**所有三个指标**上不劣于另一个则优胜，否则为 tie）。用 GLM + Poisson 分布估计每个方法的 worth 参数 $\pi_i$，$P(i > j) = \pi_i / (\pi_i + \pi_j + \nu\sqrt{\pi_i\pi_j})$
   - 设计动机：计算效率高 $O(n^2m)$，可扩展到大规模；但**强制全序**可能过度简化

2. **Union-Free Generic (UFG) Depth（场景1：偏序排名）**
   - 做什么：保留不可比性的偏序排名
   - 核心思路：将每个 prompt 产生的成对比较视为一个**偏序观测**，用 depth function 衡量每个偏序的"中心性"——depth 最高的偏序是数据最支持的排名结构
   - 设计动机：不假设比较间独立性，允许方法间不可比；但计算复杂度最坏 $O(2^m)$
   - 关键发现：**四个最佳方法的最高 depth 偏序是"全部不可比"**（depth=0.977）

3. **Q*Text（场景2：基数评估）**
   - 做什么：将 coherence、diversity、perplexity 聚合为单一综合分数
   - 核心思路：$\text{Q*Text} = \frac{\sum_{i=1}^3 w_i M_i P_i(M_i)}{\sum_{i=1}^3 w_i}$，其中 $P_i(x) = \exp(-\alpha_i(x-\mu_i)^2)$ 是高斯惩罚函数——极端偏离最优目标 $\mu_i$ 会被惩罚
   - 参数优化：9个参数通过最大化与人类评分的 Spearman 相关 $\rho_s$ 来优化
   - 设计动机：高斯惩罚避免退化（beam search 的极低 diversity 或乱码生成），自动平衡多指标

## 实验关键数据

### Bradley-Terry 排名 (WikiText-103)

| 排名 | 解码方法 | Worth 参数 |
|------|---------|-----------|
| 1 | Mistral-7B CS (α=0.6, k=15) | 0.0469 |
| 2 | Mistral-7B CS (α=0.4, k=3) | 0.0374 |
| 3 | Mistral-7B CS (α=0.8, k=3) | 0.0346 |
| 最差 | GPT2-XL CS (α=1.0, k=20) | 最低 |

### Q*Text 案例分析

| 解码方法 | Q*Text 分数 | 说明 |
|---------|-----------|------|
| Human 参考文本 | 87.33 | 人类 baseline |
| GPT2-XL CS (0.6, 5) | **86.69** | 小模型+合理参数≈人类 |
| Mistral CS (0.4, 10) | 81.62 | 大模型中等配置 |
| GPT2-XL CS (1.0, 20) | 0.02 | 极端参数→退化乱码 |
| Llama3 beam (3) | 0.02 | beam search→重复退化 |

### 关键发现
- **Contrastive Search 中等参数（α=0.4~0.6, k=5~15）普遍最优**——在 coherence/diversity 间取得最佳平衡
- **Beam Search 几乎总是最差**——diversity 极低导致被 Q*Text 严厉惩罚
- **小模型+好策略 > 大模型+差策略**：GPT2-XL (1.5B) 配合 CS(0.6,5) 的 Q*Text=86.69，接近人类 87.33
- **Top-4 方法实际上不可比**：UFG depth 揭示 Bradley-Terry 的全序排名可能是"强加的"
- **Stochastic 方法偏好高 diversity 配置**：temperature τ>0.7, top-k k>10, nucleus p>0.8

## 亮点与洞察
- **三种方法互补设计**：Bradley-Terry（快速全序）→ UFG depth（保留不可比性）→ Q*Text（基数评估），形成完整的评估工具箱。不同场景选择不同方法——practitioner 用 Bradley-Terry 快速选策略，researcher 用 Q*Text 量化差距
- **Q*Text 的高斯惩罚设计**：用 $\exp(-\alpha(x-\mu)^2)$ 惩罚极端值避免退化，既比简单加权求和有原则，又能自动识别退化生成（重复/乱码得0分）。高斯形状确保中等区间得分最高，是处理多指标权衡的优雅方案
- **"Top 方法实际不可比"的发现**：对 NLP 社区追求"单一最佳"的评测文化是重要提醒——多数情况下方法之间的优劣取决于你更看重哪个指标
- **解码策略比模型大小更重要**：GPT2-XL (1.5B) + CS(0.6,5) 接近人类水平，比 Llama3-8B + beam search 好两个数量级。这对部署有直接指导意义——先优化解码策略再考虑换大模型
- **180万生成文本的实验规模**：覆盖6模型×3数据集×59配置，是目前解码策略评估中规模最大的研究之一

## 局限性 / 可改进方向
- **仅三个自动指标**：排除了 MAUVE（需聚合数据），未考虑事实性/安全性/流畅度等维度
- **UFG depth 计算瓶颈**：最坏 $O(2^m)$ 限制只能用于少量方法子集（本文仅比较4个方法）
- **256 token 最大长度**：未评估长文本生成场景，长文本的 coherence 衡量更复杂
- **模型范围**：最大到 11B，未包含 70B+ 或 GPT-4 级别模型
- **Q*Text 参数泛化性**：高斯惩罚的 $\mu_i, \alpha_i$ 依赖训练数据的人类评分，跨领域使用需重新标注优化
- **未考虑指令遵循场景**：仅evaluates 续写任务，chat 场景的多准则评估可能需要不同指标组合（如 helpfulness、safety）
- **解码策略覆盖不全**：未包含近期流行的 speculative decoding、guided generation 等新策略

## 相关工作与启发
- **vs MAUVE (Pillutla et al., 2021)**：MAUVE 是分布级指标，本文需要 instance-level 指标才能构建偏序；两者互补
- **vs Chatbot Arena (Chiang et al., 2024)**：Arena 也用 Bradley-Terry 排 LLM，本文将其扩展到解码策略+多指标+偏序
- **vs Contrastive Search (Su et al., 2022)**：CS 在本文评估中表现最佳，但最优参数因模型/任务而异
- **启发**：Q*Text 的惩罚函数设计思路可迁移到其他多指标评估场景（如 RAG 评估中平衡 faithfulness/relevance/fluency）

## 评分
- 新颖性: ⭐⭐⭐⭐ 将偏序理论/depth function 引入文本生成评估是新颖的，Q*Text 设计实用
- 实验充分度: ⭐⭐⭐⭐⭐ 6模型×59配置×3数据集×180万生成，规模非常大
- 写作质量: ⭐⭐⭐⭐ 框架清晰（两场景三方法），但数学部分略重
- 价值: ⭐⭐⭐⭐ 为文本生成评估提供了系统化工具箱，对解码策略选择有实用指导
