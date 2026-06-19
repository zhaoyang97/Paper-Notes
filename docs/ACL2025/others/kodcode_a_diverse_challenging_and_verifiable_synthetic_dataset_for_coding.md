---
title: >-
  [论文解读] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding
description: >-
  [ACL 2025][合成数据集] KodCode 提出一套三阶段合成数据管线（编程题目合成→解决方案+单元测试自验证→后训练数据合成），构建了 447K 经过验证的编程 question-solution-test 三元组，微调后的模型在 HumanEval、MBPP、BigCodeBench、LiveCodeBench 等基准上超越 Qwen2.5-Coder-32B-Instruct 和 DeepSeek-R1-Distill-Llama-70B。
tags:
  - "ACL 2025"
  - "合成数据集"
  - "代码生成"
  - "自验证"
  - "强化学习"
  - "推理模型"
---

# KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding

**会议**: ACL 2025  
**arXiv**: [2503.02951](https://arxiv.org/abs/2503.02951)  
**代码**: [有](https://kodcode-ai.github.io) / [模型](https://huggingface.co/KodCode)  
**领域**: 其他  
**关键词**: 合成数据集, 代码生成, 自验证, 强化学习, 推理模型

## 一句话总结

KodCode 提出一套三阶段合成数据管线（编程题目合成→解决方案+单元测试自验证→后训练数据合成），构建了 447K 经过验证的编程 question-solution-test 三元组，微调后的模型在 HumanEval、MBPP、BigCodeBench、LiveCodeBench 等基准上超越 Qwen2.5-Coder-32B-Instruct 和 DeepSeek-R1-Distill-Llama-70B。

## 研究背景与动机

训练高性能代码 LLM 需要高质量、可验证的训练数据，但现有资源存在三大缺陷：

**人工数据集规模受限**：TACO (26K)、APPS (10K)、CodeContests (13K) 虽然质量高但规模小

**合成数据集质量不足**：
   - Code Alpaca (20K)：多样性低、难度低、无单元测试
   - Evol Instruct (111K)：多样性低、无测试验证
   - OSS Instruct (75K)：中等多样性、无测试

**缺乏统一的大规模、多难度、可验证数据集**：没有同时满足高多样性、混合难度和经单元测试验证的数据集

KodCode 的核心目标：构建一个**大规模**（447K）、**多样**（12个子集）、**有挑战**（简单到竞赛级）、**可验证**（自带单元测试）的合成编码数据集。

## 方法详解

### 整体框架

三阶段管线：Step 1 合成多样化编程题目 → Step 2 生成解决方案和单元测试并自验证 → Step 3 后训练数据合成（格式转换 + 推理模型生成 CoT 回答）。

### 关键设计

1. **Step 1: 编程题目合成（12个子集，5种方法）**：

    - **Magpie-Prefill**: 用预填充后缀（"Write a Python function that"）+ Qwen2.5-Coder-7B 补全，高效生成简单题目
    - **评估题目扩展**: 用 GPT-4o 作为教师 LLM，分析种子题目结构后生成新题（种子来自 LeetCode、Codeforces、APPS、TACO、Code Contests）
    - **DSA 知识转题目**: 从 Python DSA 代码片段提取，生成数据结构和算法题目
    - **技术文档转题目**: 将 flask、pandas、pytorch 等库的文档转化为编程题，内置质量控制
    - **更多题目**: 用 Magpie + 7个开源 LLM 生成，LLM 分类器过滤保留高质量题目
    - **去重**: 使用 all-mpnet-base-v2 嵌入 + FAISS 最近邻距离过滤
    - 设计动机：多来源多方法确保题目的多样性和难度覆盖

2. **Step 2: 解决方案与测试生成（自验证机制）**：

    - 用 GPT-4o 同时生成 solution 和 unit test
    - 执行单元测试验证 solution 正确性
    - 使用 pytest-cov 进行分支覆盖分析，仅保留 100% 分支覆盖的三元组
    - **关键创新 — 对困难题目分配额外尝试次数**：
        - 每个题目最多 n=10 次尝试
        - 每次从头重新生成 solution 和 test（避免错误测试导致后续全部失败）
        - 仅保留分支覆盖不低于之前尝试的新版本
        - 经过 n 次仍失败的题目被丢弃（可能本身有缺陷）
        - 自然产生难度标签：基于通过率分为 easy/medium/hard
    - 最终获得 279K 验证三元组
    - 设计动机：避免简单丢弃难题引入易题偏差

3. **Step 3: 后训练数据合成**：

    - **格式转换器（Style Converter）**: 将自然语言题目重写为 Python 函数签名格式，配合 solution 和 test 输入
    - 产生额外 168K 三元组（总计 447K），直接可用于 RL 训练
    - **SFT 数据生成**: 用 DeepSeek R1 生成 CoT 回答，每题 3 次生成 + test-based reject sampling
    - 设计动机：弥合编程题目格式与训练数据格式的差距

### 训练策略

- **SFT**: Qwen2.5-Coder-32B-Instruct，学习率 1e-5，最大序列长度 16384
- **RL (GRPO)**: Qwen2.5-7B-Instruct-1M / Qwen2.5-Coder-7B-Instruct，256 步，16 rollouts/题，二元奖励（通过全部测试=1，否则=0）

## 实验关键数据

### 主实验（SFT 模型 vs 基线）

| 模型 | HumanEval | MBPP | BCB-C Full | BCB-C Hard | BCB-I Full | BCB-I Hard | 平均 |
|------|-----------|------|-----------|-----------|-----------|-----------|------|
| Qwen2.5-Coder-32B-Inst | 90.9 | 90.2 | 57.6 | 31.1 | 49.4 | 25.7 | 59.25 |
| DeepSeek-R1-Distill-70B | 89.0 | 81.7 | 53.5 | 25.7 | 43.9 | 25.7 | 57.79 |
| Bespoke-Stratos-32B | 88.4 | 88.1 | 56.2 | 33.1 | 47.3 | 27.0 | 59.64 |
| **KodCode-32B-SFT-50K** | **92.7** | 89.9 | **59.8** | **37.8** | **51.1** | **32.4** | 61.22 |
| **KodCode-32B-SFT-Hard-18K** | 90.9 | 89.2 | 59.7 | 37.2 | 50.5 | 31.1 | **61.26** |

### 消融实验

| 数据选择 | BCB-C Hard | BCB-I Hard | LCB Hard |
|---------|-----------|-----------|---------|
| KodCode-SFT-Hard-10K | **39.9** | **31.8** | **6.3** |
| KodCode-SFT-10K (随机) | 38.5 | 27.7 | 4.8 |
| KodCode-SFT-NoConvert-10K | 35.1 | 28.4 | 5.6 |

### RL 实验

| 模型 | 步数 | BCB-C Full | BCB-I Full | HumanEval | 平均 |
|------|------|-----------|-----------|-----------|------|
| Qwen2.5-Coder-7B-Inst (基线) | - | 52.0 | 41.8 | 91.5 | 52.32 |
| + GRPO KodCode | 128 | 52.5 | 42.2 | 90.9 | 53.56 |
| + GRPO KodCode | 256 | **53.7** | **42.9** | 90.2 | **53.99** |

### 管线分析关键数据

| 验证指标 | MBPP | LiveCodeBench-V5 |
|---------|------|-----------------|
| 自验证通过率 | 88.9% (80/90) | 49.9% (190/381) |
| 通过人工测试率 | **97.5%** (78/80) | **99.47%** (189/190) |
| Pass@1 → Pass@5 | 平均提升 20%+ | - |
| 潜在污染数 | 94/447K (0.02%) | - |

### 关键发现

1. **KodCode SFT 模型全面 SOTA**：在 BigCodeBench Hard 上超越最强基线 4.7%（Complete）和 5.4%（Instruct）
2. **困难题目的关键价值**：Hard-10K 在 BCB-I Hard 上比随机 10K 高 4.1%（31.8 vs 27.7）
3. **格式转换器有效**：移除格式转换后 BCB-C Hard 从 38.5 降到 35.1（-3.4%）
4. **额外尝试次数的价值**：Pass@1→Pass@5 提升 20%+，Pass@5→Pass@10 再提升 4%
5. **自验证可靠**：保留的解决方案在人工测试上通过率 97.5-99.5%
6. **RL 训练有效**：GRPO 持续提升性能，更多步数进一步改善

## 亮点与洞察

- **核心创新在管线设计而非模型架构**：三阶段管线的每一步都有清晰的设计考量
- **对困难题目的处理策略**巧妙：额外尝试而非丢弃，既保留了难题又自然产生难度标签
- **自验证 + reject sampling** 实现了低成本高可靠的数据质量保证
- **数据集对 SFT 和 RL 的双重适用性**：solution-test 对天然适合 RL 的奖励信号设计
- t-SNE 可视化清楚展示了 KodCode 的多样性优势：覆盖整个空间而非聚集在一角

## 局限与展望

1. **LiveCodeBench-Hard 上表现有限**：竞赛级编程题仍是短板，需要更多高难度题目
2. 数据合成依赖 GPT-4o 和 DeepSeek R1，成本较高
3. 仅限 Python 语言，缺乏多语言支持
4. 未探索最优的后训练数据选择策略
5. 缺乏仓库级别（repository-level）代码的合成数据

## 相关工作与启发

- 与 OpenCoder (Huang et al., 2024) 的教师模型测试生成思路类似，但 KodCode 增加了难题保留和格式转换
- EvalPlus (Liu et al., 2024) 的 mutation-based 测试扩展可与 KodCode 的自验证互补
- 管线设计可推广到数学推理等其他需要可验证数据的领域
- 证明了合成数据可以让小模型超越大模型的观点

## 评分

- **新颖性**: ⭐⭐⭐⭐ — 管线设计新颖（尤其是难题保留策略），但核心组件均基于已有技术
- **实验充分度**: ⭐⭐⭐⭐⭐ — 管线分析（自验证、Pass@k、污染、多样性）+ 性能评估（SFT/RL/消融）非常全面
- **写作质量**: ⭐⭐⭐⭐⭐ — 逻辑清晰，图表丰富（t-SNE、Sankey、难度分布），管线描述详尽
- **价值**: ⭐⭐⭐⭐⭐ — 数据集开源、模型开源、方法可复现，对代码 LLM 社区有重要的基础设施价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] SynDaCaTE: A Synthetic Dataset for Evaluating Part-Whole Hierarchical Inference](../../ICML2025/others/syndacate_a_synthetic_dataset_for_evaluating_part-whole_hierarchical_inference.md)
- [\[CVPR 2026\] What Is Wrong with Synthetic Data for Scene Text Recognition? A Strong Synthetic Engine with Diverse Simulations and Self-Evolution](../../CVPR2026/others/what_is_wrong_with_synthetic_data_for_scene_text_recognition_a_strong_synthetic_.md)
- [\[CVPR 2025\] BenDFM: A taxonomy and synthetic CAD dataset for manufacturability assessment in sheet metal bending](../../CVPR2025/others/bendfm_a_taxonomy_and_synthetic_cad_dataset_for_manufacturability_assessment_in_.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] TARGA: Targeted Synthetic Data Generation for Practical Reasoning over Structured Data](targa_targeted_synthetic_data_generation_for_practical_reasoning_over_structured.md)

</div>

<!-- RELATED:END -->
