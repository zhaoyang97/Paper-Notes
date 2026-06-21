---
title: >-
  [论文解读] MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG
description: >-
  [ICLR 2026][信息检索/RAG][Parametric RAG] MergePRAG 用超网络把每一跳检索到的段落翻译成"段落专家"参数，并通过基于 Gram–Schmidt 正交化的持续合并机制把它们逐跳叠加进 LLM 的关键层，从而第一次把参数化 RAG（PRAG）从单跳扩展到多跳推理场景。
tags:
  - "ICLR 2026"
  - "信息检索/RAG"
  - "Parametric RAG"
  - "Multi-hop QA"
  - "超网络"
  - "正交合并"
  - "Gram–Schmidt"
  - "知识注入"
---

# MergePRAG: Orthogonal Merging of Passage-experts for Multi-hop Parametric RAG

**会议**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=FSL1J2gmJV](https://openreview.net/forum?id=FSL1J2gmJV)  
**代码**: [https://github.com/Liu-Xuebing/MhQA_hypernetwork](https://github.com/Liu-Xuebing/MhQA_hypernetwork)  
**领域**: 信息检索 / 检索增强生成（参数化 RAG）  
**关键词**: Parametric RAG, Multi-hop QA, 超网络, 正交合并, Gram–Schmidt, 知识注入  

## 一句话总结
MergePRAG 用超网络把每一跳检索到的段落翻译成"段落专家"参数，并通过基于 Gram–Schmidt 正交化的持续合并机制把它们逐跳叠加进 LLM 的关键层，从而第一次把参数化 RAG（PRAG）从单跳扩展到多跳推理场景。

## 研究背景与动机
**领域现状**：给 LLM 注入外部知识有两条主线——RAG 把检索到的段落塞进上下文，参数化适配（PKA）直接改模型权重。近期兴起的参数化 RAG（PRAG）走中间路线：用一个"超网络"把检索段落翻译成 LoRA 参数增量注入模型，让知识真正"内化"，从而绕开 RAG 的长上下文低效和噪声敏感问题，并被证明能稳定超过标准 RAG。

**现有痛点**：PRAG 至今只在**单跳**设定下被验证——一次检索、一次回答就结束。但真实的复杂问答是**多跳**的：一个复杂问题要拆成若干子问题，每个子问题独立检索、生成子答案，段落是**逐跳增量到来**的。单跳 PRAG 的超网络没有处理"知识跨跳持续累积"的能力。

**核心矛盾**：如何让 PRAG 在多跳场景下把每一跳新到的段落知识**持续注入**模型，又**不需要重训或重建**一个专门为多跳设计的超网络？如果每跳都简单地把新段落参数做算术平均叠加，不同段落专家之间会产生**知识冲突**（冗余、互相覆盖），破坏已积累的知识。

**本文目标**：把单跳 PRAG 推广到多跳 RAG，复用同一个段落级超网络，让段落专家可以一跳跳无冲突地累积，成为通往"推理增强 RAG"（IRCoT、Self-RAG、DeepRAG 等）的桥梁。

**核心 idea**：**正交持续合并** —— 把每个段落看作一个"专家"参数，新专家只以**与已有专家子空间正交的分量**被加入，从而消除冗余、保留互补信息；同时只更新一个预选的**关键层**来高效编码段落、稳定推理。

## 方法详解

### 整体框架
MergePRAG 把多跳 QA 组织成"子问题生成 → 检索 → 参数化 → 持续合并 → 生成子答案"的循环。每一跳 $t$，子问题生成器从已有推理链生成子问题 $sq_t$，检索器返回 top 段落 $SP_t$；每个段落经超网络 $H_\phi$ 翻译成键值记忆，先通过 **inner-merging** 合成本跳的段落专家 $H_\phi(SP_t)$，再通过**正交持续合并** $\text{Merge}_{seq}$ 把它叠进累积参数 $F(SP_{1:t-1})$ 得到 $F(SP_{1:t})$，注入基座 LLM 的关键层 $l^*$ 生成子答案。循环直到不再产生子问题，输出最终答案。

```mermaid
flowchart LR
    A[复杂问题 q] --> B[子问题生成 Msq<br/>sq_t = Msq(C_t-1)]
    B --> C[检索器 R<br/>返回段落 SP_t]
    C --> D[超网络 Hϕ<br/>段落→键值记忆 Kp,Vp]
    D --> E[Inner-merging<br/>合成本跳专家 Hϕ(SP_t)]
    E --> F[正交持续合并 Merge_seq<br/>F(SP_1:t-1) ⊕ Hϕ(SP_t)]
    F --> G[注入关键层 l*<br/>生成子答案 sa_t]
    G --> H{还有子问题?}
    H -->|是| B
    H -->|否| I[生成最终答案 a]
```

### 关键设计

**1. 持续合并机制（Continual Merging）：复用单跳超网络拼出多跳能力。** 多跳的核心难题是要把"段落序列 → 参数"的映射 $F$ 表达出来，但直接对不同长度的段落序列训练 $F$ 既昂贵又不通用。MergePRAG 的巧思是**不训练 $F$，而是用一个递归的合并算子从单跳超网络 $H_\phi$ 推导出它**：每来一跳，就把前面累积的参数 $F(SP_{1:t-1})$ 和本跳新段落参数 $H_\phi(SP_t)$ 合并，$F(SP_{1:t}) = \text{Merge}_{seq}\big(F(SP_{1:t-1}), H_\phi(SP_t)\big)$。生成子答案时直接用合并后的参数 $sa_t = M_{\theta_0 \oplus F(SP_{1:t})}(sq_t)$，无需把段落再放进上下文。这样一个段落级超网络就能服务任意跳数，避免为多跳重新设计/重训超网络。变体 MergePRAG+ 进一步像 PRAG-Combine 那样把参数注入和上下文段落互补使用，兼得两者之长。

**2. 正交持续合并（Gram–Schmidt Orthogonal Merging）：让段落专家互不覆盖。** 这是全文的灵魂。若直接把新专家加到累积专家上，方向重叠会导致冲突和冗余。MergePRAG 借鉴 Gram–Schmidt 正交化：设累积参数（键或值矩阵）为 $W_F^{t-1}$，先算它张成子空间上的投影矩阵 $P_{t-1} = W_F^{t-1}\big((W_F^{t-1})^\top W_F^{t-1}\big)^{-1}(W_F^{t-1})^\top$，再**只把新参数 $W_t$ 与该子空间正交的分量加进去**：$W_F^{t} = W_F^{t-1} + (I - P_{t-1})W_t$。这样新段落只贡献"已有知识里没有的部分"，既消冗余又促互补，从而在跨跳累积时保住已学知识不被覆盖。该正交合并同时用于 inner-merging 和 sequence-merging。

**3. 关键层键值记忆参数化（Critical-layer KV-Memory）：在最该改的那层高效注入。** MergePRAG 不在所有层注入，而是仿照 ROME/PMET 等"定位再编辑"思路，只更新**单个关键层 $l^*$**。超网络为每个段落生成 $k$ 对键值向量 $\{K_p, V_p\}$ 作为"压缩的段落记忆"，通过一个**记忆注意力**插进关键层的 FFN：以基座 FFN 输出 $\text{MLP}_{\theta_0}(x)$ 为 query，对 $(K_p,V_p)$ 做标准注意力得到段落专家输出 $E_{H_\phi(p)}(x)=\text{Attention}(\text{MLP}_{\theta_0}(x), K_p, V_p)$，再以残差形式叠加 $\text{MLP}_{\theta_0 \oplus H_\phi(p)}(x) = \text{MLP}_{\theta_0}(x) + E_{H_\phi(p)}(x)$。关键层 $l^*$ 通过逐层扫描——注入段落向量后测困惑度变化——来定位，实验发现早中层贡献最大；只动一层既省算力又稳定推理。

**4. 序列到记忆的超网络架构与训练。** 超网络 $H_\phi$ 把段落 token 序列变成键值记忆：先用注意力池化得到段落嵌入 $\text{Emd}(p)$，过两层 MLP（含 LayerNorm + ReLU）得隐表示 $h_b$，再用两组线性投影分别映射成键矩阵 $K_p = W_K h_b + b_K$ 和值矩阵 $V_p = W_V h_b + b_V$。训练用交叉熵 $L_{CE}(\phi) = -\sum_{(q,p,a)} \log P_{M_{\theta_0 \oplus H_\phi(p)}}(a \mid q)$ 让注入段落参数后的模型能正确答题；子问题生成器 $M_{sq}$ 则用冷启动方式在 $[sq_1, sa_1, \dots, \langle\text{EOS}\rangle]$ 序列上自回归训练。

## 实验关键数据

### 主实验表格
LLaMA3.1-8B / Qwen2.5-7B 在三个多跳 QA 数据集上对比 SOTA（EM/F1，节选 |SP|=4）：

| 模型 | 方法 | HotpotQA EM/F1 | 2WikiMhQA EM/F1 | MuSiQue EM/F1 |
|------|------|----------------|-----------------|---------------|
| LLaMA3.1-8B | RAG-CoT (E5) | 43.7 / 50.4 | 36.2 / 40.0 | 5.9 / 12.5 |
| LLaMA3.1-8B | R3-RAG† (E5) | 45.6 / 58.8 | 52.9 / 60.9 | **21.2** / **32.8** |
| LLaMA3.1-8B | **MergePRAG+** (E5) | **52.4** / **60.7** | **73.2** / **79.3** | 16.7 / 27.7 |
| Qwen2.5-7B | R3-RAG† (E5) | 46.4 / 59.7 | 54.2 / 62.7 | **21.4** / **34.0** |
| Qwen2.5-7B | **MergePRAG+** (E5) | **50.8** / **58.4** | **77.4** / **81.5** | 12.3 / 21.6 |

MergePRAG+ 在 HotpotQA 和 2WikiMhQA 上几乎全面领先（2WikiMhQA 涨幅最大，EM 比 R3-RAG 高 20+ 点），仅在多跳更难、推理链更长的 MuSiQue 上不及强 RL 基线 R3-RAG。

### 消融实验表格
关键消融（LLaMA3.1-8B, HotpotQA / 2WikiMhQA, EM/F1）：

| 设置 | HotpotQA | 2WikiMhQA |
|------|----------|-----------|
| MultihopRAG（无超网络，不微调） | 37.8 / 47.6 | 23.3 / 35.6 |
| MultihopRAG 微调 [sq→sa] | 43.7 / 50.2 | 58.1 / 62.6 |
| MultihopRAG 微调 [(P,sq)→sa] | 40.1 / 46.8 | 60.3 / 62.0 |
| MergePRAG (|SP|=0) | 28.4 / 35.5 | 45.6 / 50.1 |
| **MergePRAG+ (|SP|=1)** | **47.4 / 55.3** | **65.6 / 70.5** |

合并方法对比（序列合并，|SP|=1）：正交合并比 TIES-merging 高约 **2.4%**，比算术平均也稳定高约 **1% EM**。

### 关键发现
- **超网络注入显著优于直接微调**：直接用段落微调 $[(P_{gold},sq)\to sa]$ 反而比无段落微调 $[sq\to sa]$ 更差，说明朴素微调会损害泛化；而 MergePRAG 在注入参数化知识的同时保住了 RAG 能力。
- **正交合并最稳**：在多种 |SP| 和合并位置下都优于算术/TIES 合并，作者预期在知识冲突更严重时优势更大。
- **段落数 |SP| 越多越好**：从 |SP|=2 到 12 单调提升、不退化；KV 向量数 $k$ 增大也持续提升（更大记忆容量）。

## 亮点与洞察
- **"复用单跳超网络拼多跳"是优雅的工程哲学**：不去训一个段落序列到参数的大映射，而用递归合并算子从单跳能力推导多跳，省训练且天然支持任意跳数。
- **把模型合并（model merging）里的正交化思想迁到 RAG 知识注入**，给"段落专家冲突"这个具体问题找到了干净的数学工具（Gram–Schmidt 投影）。
- **参数化 + 上下文互补（MergePRAG+）**比纯参数化（|SP|=0）大幅更好，说明二者捕获的知识并不冗余，而是叠加增益。

## 局限与展望
- **MuSiQue 上不敌 R3-RAG**：在跳数最多、推理最难的数据集上，基于 RL 训练完整推理轨迹的方法仍更强，说明 MergePRAG 的子问题生成/推理控制还偏弱。
- **正交合并的开销**：每跳要算投影矩阵（含矩阵求逆），跳数和 KV 维度大时可能成为瓶颈；论文把效率分析放在附录，主文未充分展开。
- **关键层只取单层**：靠逐层扫困惑度选 $l^*$，是否对所有任务/模型都最优、多层注入会不会更好，留有空间。
- **正交带来的"非负面"假设**：作者预期知识冲突越严重正交优势越大，但当前数据集冲突程度有限，强冲突场景的验证仍待补。

## 相关工作与启发
- **参数化知识增强**：从全量微调 → LoRA 等 PEFT → ROME/MEMIT/PMET 定位编辑 → MEND/MALMEN 超网络注入 → T-Patcher/MEMoE 外部记忆，MergePRAG 站在"超网络注入 + 关键层 + 外部专家"的交汇点。
- **检索增强生成**：从经典 RAG 到 PRAG/DyPRAG（把检索映射成参数），再到强调推理的 FLARE/IRCoT/DeepRAG/R3-RAG；MergePRAG 的定位是"PRAG 通往推理增强 RAG 的踏脚石"。
- **启发**：模型合并领域的去冲突技术（TIES、正交化）可以反哺 RAG 的知识注入；"逐跳累积参数"这一范式或可推广到工具调用、长程 agent 记忆等需要增量内化知识的场景。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次把 PRAG 扩展到多跳，且用 Gram–Schmidt 正交合并解决段落专家冲突，组合新颖。
- 实验充分度: ⭐⭐⭐⭐ 两个基座、三个 QA 数据集 + 知识编辑、丰富消融（合并方法/段落数/KV数/层位），覆盖全面；MuSiQue 略弱。
- 写作质量: ⭐⭐⭐⭐ 公式与流程清晰，Figure 1 概览到位；部分关键细节（效率、超网络架构）下放附录。
- 价值: ⭐⭐⭐⭐ 为参数化 RAG 打开多跳大门，并提供可复用的去冲突合并工具，对 RAG + 推理融合方向有实际推动力。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] FrugalRAG: Less is More in RL Finetuning for Multi-hop Question Answering](frugalrag_less_is_more_in_rl_finetuning_for_multi-hop_question_answering.md)
- [\[ICLR 2026\] Demystifying Deep Search: A Holistic Evaluation with Hint-free Multi-Hop Questions and Factorised Metrics](demystifying_deep_search_a_holistic_evaluation_with_hint-free_multi-hop_question.md)
- [\[AAAI 2026\] REAP: Enhancing RAG with Recursive Evaluation and Adaptive Planning for Multi-Hop Question Answering](../../AAAI2026/information_retrieval/reap_enhancing_rag_with_recursive_evaluation_and_adaptive_planning_for_multi-hop.md)
- [\[ACL 2025\] Mitigating Lost-in-Retrieval Problems in RAG Multi-Hop QA](../../ACL2025/information_retrieval/mitigating_lost-in-retrieval_problems_in_retrieval_augmented_multi-hop_question_.md)
- [\[NeurIPS 2025\] Think Straight, Stop Smart: Structured Reasoning for Efficient Multi-Hop RAG](../../NeurIPS2025/information_retrieval/think_straight_stop_smart_structured_reasoning_for_efficient_multi-hop_rag.md)

</div>

<!-- RELATED:END -->
