---
title: >-
  [论文解读] MemEIC: A Step Toward Continual and Compositional Knowledge Editing
description: >-
  [NEURIPS2025][知识编辑][LVLM] 提出 MemEIC 框架，通过外部双模态检索记忆 + 内部模态分离 LoRA 适配器 + 仿脑 Knowledge Connector 三层架构，实现大视觉语言模型的持续、组合式知识编辑，在新提出的 CCKEB 基准上大幅超越现有方法。 大视觉语言模型 (LVLM) 编码的…
tags:
  - "NEURIPS2025"
  - "知识编辑"
  - "LVLM"
  - "持续学习"
  - "组合推理"
  - "LoRA"
  - "external memory"
---

# MemEIC: A Step Toward Continual and Compositional Knowledge Editing

**会议**: NEURIPS2025  
**arXiv**: [2510.25798](https://arxiv.org/abs/2510.25798)  
**代码**: [MemEIC/MemEIC](https://github.com/MemEIC/MemEIC)  
**领域**: 知识编辑  
**关键词**: 知识编辑, LVLM, 持续学习, 组合推理, LoRA, external memory

## 一句话总结

提出 MemEIC 框架，通过外部双模态检索记忆 + 内部模态分离 LoRA 适配器 + 仿脑 Knowledge Connector 三层架构，实现大视觉语言模型的持续、组合式知识编辑，在新提出的 CCKEB 基准上大幅超越现有方法。

## 背景与动机

大视觉语言模型 (LVLM) 编码的事实知识会过时或出错，需要高效的知识编辑机制进行更新。现有的知识编辑工作存在三个关键局限：

1. **单模态孤立编辑**：大多数方法只关注视觉或文本单一模态的编辑，忽视了 LVLM 天然的多模态特性。例如只能更正"图中人物身份"（视觉编辑），或只能更新"某人的职位"（文本编辑），却无法同时处理两者
2. **缺乏持续编辑评估**：现有基准不评估模型在多次顺序编辑后的知识保留能力，忽视了灾难性遗忘问题
3. **缺乏组合推理评估**：现实场景中的查询往往需要整合视觉和文本两种编辑结果，例如"照片中的人最近担任了什么职位？"——需要先识别图中人物（视觉编辑），再查找其新职位（文本编辑）

现有方法可分为两类，各有本质缺陷：

- **外部记忆方法**（如 SERAC, IKE）：不修改模型参数，适合长期保留，但直接沿用文本 LLM 的检索策略，忽略视觉线索，导致视觉编辑检索失败；且模型过度依赖内部陈旧知识，当外部检索信息与内部知识冲突时表现差
- **内部记忆方法**（如 LoRA, MEND）：通过微调将编辑嵌入模型参数，能有效内化新知识，但视觉和文本编辑共享同一参数空间，导致跨模态干扰和表示坍塌；顺序编辑还会引发灾难性遗忘

## 核心问题

如何让 LVLM 在持续接收交替的视觉和文本知识编辑时，既能稳定保留历史编辑、避免遗忘，又能进行跨模态的组合推理？

## 方法详解

### 整体架构

MemEIC 采用三层协同架构：查询分解 → 外部记忆检索 (Mem-E) + 内部模态分离适配器 (Mem-I) → Knowledge Connector 融合。

### 1. 查询分解

使用 GPT-4o 将输入查询 $Q$ 自动分解为视觉子查询 $Q_v$ 和文本子查询 $Q_t$，实现各模态独立处理。例如"照片中的人最近担任了什么职位？"被分解为视觉部分"照片中的人是谁？"和文本部分"[某人]最近担任了什么职位？"。

### 2. 模态感知外部记忆 (Mem-E)

维护两个独立的外部记忆存储：

- **文本记忆** $M_t = \{(q_i, a_i)\}$：存储文本 QA 编辑对
- **视觉记忆** $M_v = \{(I_j, q_j, a_j)\}$：存储包含图像的视觉编辑三元组

检索时的关键创新是**多模态融合检索**：视觉查询不仅使用文本相似度，还结合 CLIP 图像编码器计算图像相似度，加权系数 $\alpha = 0.5$。文本检索使用 DistilBERT 的 [CLS] 表示。

对于组合查询，先通过视觉检索获取目标实体，再将实体名替换到文本子查询的占位符中，实现级联检索。

### 3. 内部分离式知识集成 (Mem-I)

受**人脑偏侧化**启发（左脑处理语言、右脑处理视觉），设计双 LoRA 适配器：

- **视觉适配器** $\theta_v$（对应右脑）：仅处理视觉知识更新
- **文本适配器** $\theta_t$（对应左脑）：仅处理文本知识更新
- 原始预训练 FFN 权重 $\theta$ 保持冻结，保留编辑前知识

根据查询类型选择性激活对应适配器，编辑时也只更新对应模态的适配器，从根本上防止跨模态干扰和表示坍塌。

### 4. Knowledge Connector（仿胼胝体机制）

灵感来自脑科学中连接左右脑半球的**胼胝体 (corpus callosum)**。当组合查询同时激活双适配器时，Knowledge Connector 通过在自注意力模块的 $Q$, $K$ 投影上添加 LoRA 适配，促进不同模态 token 表示之间的信息交换：

$$Q^\ell = h^\ell(W_Q + \mathbb{I}_{v,t} \cdot \Delta W_Q^L)$$

其中 $\mathbb{I}_{v,t}$ 是指示函数，仅在两个适配器同时激活时为 1。对单模态查询，Connector 退化为恒等操作，不影响独立模态表示。

### 训练流程

- **阶段一**：冻结 LVLM，训练外部记忆模块的检索和对齐能力
- **阶段二**：激活双适配器，使用**对抗性检索器**（混合正确和错误证据）训练 Knowledge Connector，鼓励模型学会选择性整合内外部证据，避免过度依赖外部记忆

## 实验关键数据

### 基准与设置

- 提出 CCKEB 基准，扩展 VLKEB，每个样本配对视觉编辑 + 文本编辑
- 骨干模型：LLaVA-1.5 (7B)、MiniGPT-4
- 顺序执行 500 次编辑，在 gap = 0, 10, 20, 50, 100 处评估

### 主实验结果（LLaVA-1.5，gap 平均）

| 方法 | Visual Rel | Text Rel | Comp Rel |
|------|-----------|----------|----------|
| SERAC | 较低（文本检索不足） | 稳定 | 低（无法融合） |
| LoRA | 0-gap 满分，gap↑急剧下降 ~30pt | 同左 | 62.05 |
| WISE | 较稳定但视觉编辑差 | 较稳定 | 较低 |
| **MemEIC** | **98.93** | **92.48** | **80.56** |

MemEIC 相比 WISE 在视觉可靠性上 +16.94，组合可靠性上 +32.35；相比最佳基线 LoRA 在 Comp Rel 上 +18.51。

### 消融实验关键发现

1. **外部记忆融合视觉线索至关重要**：Mem-E (tex+vis) vs Mem-E (tex)，Reliability 从 48.02 → 96.51，Image Locality 从 4.02 → 57.10
2. **双 LoRA 优于单 LoRA**：在保持总参数量相同 (r=8×2 vs r=16) 的条件下，Dual-LoRA 在 T-Loc 上 +17.77%，I-Loc 上 +2.86%（p<0.05 显著）
3. **Knowledge Connector 是组合推理的关键**：
    - Base+RAG（完美检索）：Comp Rel 仅 64.93%——说明完美检索不够
    - Dual-LoRA+RAG：gap=0 时 78.16%，gap=100 时降至 63.39%——双适配器缺乏交互
    - **Dual-LoRA+RAG+Connector**：gap=0 时 **99.21%**，gap=100 时仍 **97.01%**——接近 oracle 水平

## 亮点

1. **问题定义新颖**：首次形式化持续组合知识编辑 (CCKE) 问题，提出 CompRel 指标和 CCKEB 基准，填补了多模态知识编辑评估的空白
2. **仿脑设计有说服力**：偏侧化双 LoRA 对应左右脑、Knowledge Connector 对应胼胝体，生物学类比自然且有效
3. **三层架构各司其职**：外部记忆负责精准检索、内部适配器负责模态隔离编辑、Connector 负责按需融合，层次清晰
4. **对抗训练策略聪明**：阶段二用混合正确/错误证据训练 Connector，有效缓解模型对外部记忆的过度依赖
5. **消融实验充分**：逐步叠加组件，清晰展示每个模块的贡献

## 局限与展望

1. **查询分解依赖 GPT-4o**：增加推理成本和延迟，实际部署中可能成为瓶颈，可探索轻量级分解器
2. **实验仅限 paired 设置**：视觉编辑后紧跟对应文本编辑，真实场景中编辑顺序可能更随机和复杂
3. **CCKEB 规模有限**：基于 VLKEB 扩展，实体类型和关系种类可能不够多样
4. **仅验证两个骨干模型**：LLaVA-1.5 和 MiniGPT-4 均相对较老，在更新更强的 LVLM 上效果待验证
5. **Knowledge Connector 冻结后的适应性**：部署阶段 Connector 冻结，若编辑分布与训练差异大，融合效果可能下降
6. **外部记忆检索的扩展性**：随着编辑量增长，余弦相似度检索的效率和准确性可能下降

## 与相关工作的对比

| 维度 | SERAC | WISE | LoRA/FT | MemEIC |
|------|-------|------|---------|--------|
| 编辑方式 | 外部记忆 | 内部 (side-FFN) | 内部 (微调) | 外部+内部混合 |
| 多模态检索 | 仅文本 | N/A | N/A | 图文融合 |
| 遗忘抵抗 | 强（不改参数） | 中（路由机制） | 弱（共享空间） | 强（模态分离） |
| 跨模态干扰 | 无 | 有 | 严重 | 无（双 LoRA 隔离） |
| 组合推理 | 差 | 差 | 中 | 强（Knowledge Connector） |
| 持续编辑稳定性 | 稳定 | 较稳定 | 急剧退化 | 稳定 |

## 启发与关联

1. **模态分离 + 按需融合**范式有普适性：不仅适用于知识编辑，对多模态持续学习、多任务学习中的跨模态干扰问题都有参考价值
2. **脑科学启发的网络设计**：偏侧化和胼胝体的类比为多模态架构设计提供了新视角，可推广到多模态 adapter 的通用设计中
3. **对抗训练缓解检索依赖**：用噪声检索结果训练模型，提升对检索错误的鲁棒性，这一策略可用于任何 RAG 系统
4. 与 continual learning 领域高度相关：双 LoRA 隔离编辑本质上是任务/模态特定参数的扩展策略，与 progressive networks 思路一脉相承

## 评分

- 新颖性: ⭐⭐⭐⭐ — CCKE 问题定义和仿脑三层架构均有明确创新
- 实验充分度: ⭐⭐⭐⭐ — 消融细致，但骨干模型偏旧、基准规模有限
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，脑科学类比增强可读性
- 价值: ⭐⭐⭐⭐ — 为多模态知识编辑建立了新基准和强 baseline

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](../../ACL2025/knowledge_editing/sake_steering_activations_for_knowledge_editing.md)
- [\[ACL 2025\] ScEdit: Script-based Assessment of Knowledge Editing](../../ACL2025/knowledge_editing/scedit_script-based_assessment_of_knowledge_editing.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](../../ACL2025/knowledge_editing/efficient_knowledge_editing.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[NeurIPS 2025\] Edit Less, Achieve More: Dynamic Sparse Neuron Masking for Lifelong Knowledge Editing in LLMs](edit_less_achieve_more_dynamic_sparse_neuron_masking_for_lifelong_knowledge_edit.md)

</div>

<!-- RELATED:END -->
