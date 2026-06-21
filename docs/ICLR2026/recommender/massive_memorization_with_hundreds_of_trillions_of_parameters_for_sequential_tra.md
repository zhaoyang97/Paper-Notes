---
title: >-
  [论文解读] Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders
description: >-
  [ICLR 2026][推荐系统][终身用户序列] VISTA 把候选物品对超长用户历史的 target attention 拆成「先把上百万长度的历史压成几百个摘要 token 并缓存」+「下游只对缓存 token 做轻量注意力」两阶段，让训练/推理成本固定，已在 Meta 服务数十亿用户的推荐平台上线。
tags:
  - "ICLR 2026"
  - "推荐系统"
  - "终身用户序列"
  - "序列摘要"
  - "线性注意力"
  - "Embedding 缓存"
  - "HSTU"
---

# Massive Memorization with Hundreds of Trillions of Parameters for Sequential Transducer Generative Recommenders

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=LSHSaY4gYM](https://openreview.net/forum?id=LSHSaY4gYM)  
**代码**: 待确认  
**领域**: 推荐系统 / 序列建模 / 工业级部署  
**关键词**: 终身用户序列、序列摘要、线性注意力、Embedding 缓存、HSTU  

## 一句话总结
VISTA 把候选物品对超长用户历史的 target attention 拆成「先把上百万长度的历史压成几百个摘要 token 并缓存」+「下游只对缓存 token 做轻量注意力」两阶段，让训练/推理成本固定，已在 Meta 服务数十亿用户的推荐平台上线。

## 研究背景与动机
- **领域现状**：工业推荐高度依赖用户交互历史序列，近年 HSTU、SIM/TWIN 等 transformer 类序列建模把历史拉长到 10k–100k，性能持续提升。
- **现有痛点**：两条主流路线各有死穴。**全序列建模**（HSTU）对 O(100K) 长历史做完整注意力，计算量和延迟在「每天训练 O(10B)–O(100B) 样本、推理有硬延迟上限」的工业环境里难以承受，GPU 卡不够就用不起来；**target-specific 采样**（SIM/TWIN）虽然只取与目标相关的短子序列，但推理成本随候选数量线性增长，且短子序列与全序列之间存在信息鸿沟。
- **核心矛盾**：把历史拉到「终身」（百万级）能带来增益，但每来一个请求都重算 attention 的代价无法接受——**性能要靠长序列，成本却不能随序列/候选数膨胀**。
- **本文目标**：在保持下游训练/推理成本固定的前提下，把用户历史扩展到百万级，且方案要能真正落地工业基础设施。
- **核心 idea**：**【解耦计算】** 把昂贵的「历史摘要」从在线链路中剥离——只在基座模型训练时算一次，把摘要 embedding 量化后写进 KV 缓存；下游训练和推理只做便宜的第二阶段注意力。用「多存储换少算力」，因为 GPU 计算成本比存储高几个数量级。

## 方法详解

### 整体框架
VISTA（VIrtual Sequential Target Attention）把传统「候选物品 → 全部历史」的 target attention 分两阶段：阶段一用自注意力把超长用户交互历史（UIH）摘要成几百个 token embedding；阶段二让候选物品只对这几百个摘要 token 做注意力出预测。关键在于阶段一**只在基座模型训练时运行**，输出量化后导出到 O(100)TB–O(1)PB 级 KV 缓存；推理时直接取缓存、去量化，跳过最贵的摘要计算。

```mermaid
flowchart LR
    UIH[超长用户历史<br/>O(100K)~O(1M) items] --> S1[阶段一: UIH 摘要<br/>虚拟 seed + 准线性注意力]
    S1 --> Q[量化导出]
    Q --> Cache[(KV 缓存<br/>O(100)TB~PB)]
    Cache -.推理时取用.-> S2[阶段二: target-aware 注意力<br/>标准 O(N²) transformer]
    Cand[候选物品] --> S2
    S2 --> Pred[CTR / 多任务预测]
    S1 -. 训练时直连 .-> S2
```

### 关键设计

**1. 虚拟 seed 摘要：把终身历史压成几百个 user embedding。** 阶段一引入一组随机初始化、跨用户共享的「虚拟 seed embedding」作为可学习参数，让它们与 UIH 序列一起过摘要模块的自注意力，输出即可解读为编码个性化偏好的 user embedding（PCA 可视化能看出不同国家用户的分离）。seed 数量构成一条 scaling law——seed 越多性能越好，但存储成本越高，是性能与财务成本的折中。这套摘要被缓存后，下游就不再碰原始百万级历史。

**2. 准线性注意力 QLA：兼顾线性复杂度与表达力。** 软注意力是 $O(N^2)$，超长序列下不可行；原始线性注意力靠矩阵结合律把 $(QK^\top)V$ 改成 $Q(K^\top V)$ 降到 $O(N)$，但被证明表达力不足。VISTA 提出 QLA，用 SiLU 非线性 $\varphi$ 给线性注意力注入非线性复杂度。历史自注意力部分写作 $O[S]=\varphi(Q[S])\,\varphi(\varphi(K[S])^\top V[S])$（去掉了 RowNormalize）。推荐场景有一条铁律——**候选之间不能互相 attend**，否则因为线上候选只是日志候选的子集会造成 label 泄漏；因此对候选 query 部分只保留它对历史的注意力，再单独加一项「候选对自身」的对角项：$O[T]=\varphi(Q[T])\,\varphi(\varphi(K[S])^\top V[S])+\Delta(\varphi(Q[T]),\varphi(K[T]))\,V[T]$，其中 $\Delta$ 把两矩阵的逐行点积放到对角线上。整套用 Triton 手写前反向以榨干 GPU 性能。

**3. 生成式序列重构损失：逼 seed 记住整段历史。** 为强化「记忆」效果，作者加了一个重构损失：用一个因果 transformer 解码器（去掉 softmax 层）吃进 seed embedding $s_1{\dots}s_k$ 和历史 item embedding $u_1{\dots}u_M$，输出 $v_1{\dots}v_M$，再对齐成错一位的均方误差 $L_{\text{reconstruct}}=\sum_{i=1}^{M-1}\lVert v_i-u_{i+1}\rVert_2^2$。因果掩码保证 $v_i$ 只依赖 $u_1{\dots}u_i$，无未来信息泄漏，从而强迫个性化 seed 尽量保留整段历史的信息。思想上源自 VAE，但据作者所知是首次显式用于推荐。

**4. Embedding 交付系统：让方案真能上线。** VISTA 不止是模型，更是模型-系统协同设计。端到端三段：源模型在线训练 → 摘要 embedding 经 Kafka 实时消息队列 + Hive 持久化双路交付 → 地理多副本内存 KV store 服务。摘要 embedding 以 2 小时节奏更新，A/B 显示与实时直算摘要模块性能相当；并刻意把历史压到 O(100)TB 量级，使其可部署进既有系统。

## 实验关键数据

### 主实验表格
公开数据集（Amazon-Electronics、KuaiRand-1K）+ Minimal Production，统一 FuxiCTR 框架做 CTR 预测，只替换注意力层：

| 模型 | Amazon AUC↑ | Amazon NE↓ | KuaiRand AUC↑ | Minimal Prod AUC↑ | Minimal Prod NE↓ |
|------|-------------|------------|---------------|-------------------|------------------|
| DIN | 0.873 | 0.656 | 0.744 | 0.632 | 1.048 |
| HSTU | 0.884 | 0.628 | 0.743 | 0.668 | 1.099 |
| VISTA-w/o-QLA | **0.886** | **0.621** | **0.744** | 0.627 | **1.038** |
| VISTA-w/-QLA | 0.884 | 0.623 | 0.743 | 0.632 | 1.062 |

工业级离线（相对 HSTU 基线的 NE 降幅，越负越好）：VISTA 在 C/E1/E2/E3 四任务上 Eval NE 分别 -0.40% / -1.19% / -2.98% / -2.23%。

### 消融实验表格

| 配置 | C-Task Eval NE↓ | E2-Task Eval NE↓ | 说明 |
|------|-----------------|------------------|------|
| VISTA（128 seed, 256D） | -0.40% | -2.98% | 最优配置 |
| VISTA-128D | -0.29% | -2.51% | embedding 维度减半 → 退化 |
| VISTA-64Seed | -0.37% | -3.01% | seed 减半 |
| VISTA-w/o-Recon | -0.29% | -3.00% | 去重构损失 → C-Task 明显退化 |

QLA vs 标准自注意力：QLA 把序列从 6,000 撑到 16,000、层数从 3 到 5，QPS 反而 +5%，NE 还略降。

### 关键发现
- **线上 A/B**（5% 流量、15 天，视频推荐）：C-Task / O1 / O2 三项分别显著 +0.5% / +0.2% / +0.04%（O2 的 0.01% 即视为大改进）。
- **推理 GPU 降 94%**：靠缓存+服务 embedding 而非每请求重算，序列越长优势越大。
- seed 数量呈 scaling law；QLA 在长序列上大幅缩短单 epoch 训练/评估时间且 AUC/NE 几乎不变。

## 亮点与洞察
- **用存储换算力的工业洞察**很硬核：把「该不该重算历史」的问题转化为「算一次缓存起来反复取」，直接命中工业推荐 GPU 紧张的真实约束。
- **解耦基座训练与下游训练/推理**是真正可复用的范式——摘要 token 像一份「冻结的 user 表征」，可被任意下游模型当 feature 用，2 小时刷新即可。
- QLA 里「候选不能互相 attend」的对角项处理，把推荐特有的 label 泄漏约束精确地写进了线性注意力公式，细节考究。

## 局限与展望
- **离线实验受限于 GPU**：线上用 12,000 长度，离线只跑 2,000，公开数据集序列更短（Amazon 均长仅 8.9），所谓「百万级终身历史」的增益主要靠工业数据和系统论证，公开可复现证据偏弱。
- **存储成本与刷新滞后**：PB 级缓存与 2 小时更新节奏带来存储开销和轻微 de-sync，对中小团队不友好；论文也承认这是性能-成本折中。
- **强绑定自家基础设施**（Kafka/Hive/多区 KV store），方法-系统耦合度高，迁移到其他平台需要重做交付链路。
- 未来方向：进一步优化压缩技术、探索跨域泛化。

## 相关工作与启发
- **HSTU**（Zhai et al. 2024）：把推荐重构成序列转导、可扩到万亿参数，是 VISTA 的主要基线和共训对象。
- **SIM/TWIN/TWIN-V2**：target-specific 采样路线，VISTA 正是为解决其「候选数线性成本 + 短/全序列鸿沟」而生。
- **线性注意力谱系**：Katharopoulos 线性注意力、Lightning Attention(GLA/SGLU)、Mamba/Hydra(SSM)——QLA 借了 SGLU 门控并加非线性补表达力。
- 对从业者的启发：当「更长上下文」与「在线成本」冲突时，**把重计算前移到训练期 + 缓存中间表征**是一条通用解法，可迁移到长上下文检索、用户画像等场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ 两阶段解耦+缓存摘要的范式在工业推荐里确属首创，QLA 与重构损失虽借自既有思想但组合落地扎实。
- 实验充分度: ⭐⭐⭐⭐ 公开+工业离线+线上 A/B 三层覆盖，消融完整；扣分在公开数据集序列太短，终身历史增益难独立复现。
- 写作质量: ⭐⭐⭐⭐ 动机-方法-系统-实验链路清晰，公式与系统图齐全，工程细节交代到位。
- 价值: ⭐⭐⭐⭐⭐ 已服务数十亿用户、推理 GPU 降 94%，对工业推荐有直接、可观的落地价值。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] On the Mechanisms of Collaborative Learning in VAE Recommenders](on_the_mechanisms_of_collaborative_learning_in_vae_recommenders.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](../../ACL2026/recommender/what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[ICLR 2026\] Continual Low-Rank Adapters for LLM-based Generative Recommender Systems](continual_low-rank_adapters_for_llm-based_generative_recommender_systems.md)
- [\[AAAI 2026\] Inductive Generative Recommendation via Retrieval-based Speculation](../../AAAI2026/recommender/inductive_generative_recommendation_via_retrieval-based_speculation.md)

</div>

<!-- RELATED:END -->
