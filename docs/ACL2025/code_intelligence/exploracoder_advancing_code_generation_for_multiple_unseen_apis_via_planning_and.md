---
title: >-
  [论文解读] ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration
description: >-
  [ACL 2025][代码智能][未知API代码生成] 提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。
tags:
  - "ACL 2025"
  - "代码智能"
  - "未知API代码生成"
  - "探索式编程"
  - "链式API探索"
  - "任务规划"
  - "RAG增强代码生成"
  - "Torchdata"
---

# ExploraCoder: Advancing Code Generation for Multiple Unseen APIs via Planning and Chained Exploration

**会议**: ACL 2025  
**arXiv**: [2412.05366](https://arxiv.org/abs/2412.05366)  
**代码**: [https://github.com/greenlight2000/ExploraCoder](https://github.com/greenlight2000/ExploraCoder)  
**作者**: Yunkun Wang, Yue Zhang, Zhen Qin, Chen Zhi, Binhua Li, Fei Huang, Yongbin Li, Shuiguang Deng
**机构**: Zhejiang University, Alibaba Group
**领域**: 代码生成 / 模型压缩（API调用）  
**关键词**: 未知API代码生成, 探索式编程, 链式API探索, 任务规划, RAG增强代码生成, Torchdata

## 一句话总结

提出无需额外训练的 ExploraCoder 框架，通过任务规划将复杂多 API 编程问题分解为子任务，再通过链式 API 探索（CoAE）逐步实验并积累正确的 API 用法经验，在多 API 不可见库基准上 pass@10 绝对提升最高 17.28%。

## 研究背景与动机

**领域现状**：LLM 在代码生成上展现出强大能力，但面对训练数据中未见过的库 API 时表现急剧下降。库持续更新、私有库的存在使得穷尽式重训练不现实。

**现有方法的不足**：
   - **持续预训练**：新库训练数据稀缺，重训成本高
   - **标准 RAG**（如 DocPrompting）：检索 API 文档后一次性生成代码，在涉及多个 API 交互的复杂场景下效果差——检索器难以为复合需求找到所有相关 API，且 LLM 一次性协调多个不熟悉 API 的交互容易产生幻觉
   - **改进的检索方法**（CAPIR、EpiGen）：改善了 API 文档检索和预规划，但忽略了 LLM 在多 API 交互推理上的局限性和 API 文档的歧义性
   - **反应式 Agent（ReAct）**：端到端代码构建仍暴露 LLM 在多 API 协调上的短板

**核心动机**：模拟**人类探索式编程范式**——开发者面对不熟悉的库时，先阅读文档理解库的整体能力，再通过逐步实验单个 API 调用积累实践经验，最终组合出完整解决方案。

## 方法详解

### 整体框架（Figure 1）

ExploraCoder 由三个模块组成：

1. **任务规划（Task Planning）**：将复杂编程问题分解为多个 API 调用子任务
2. **API 推荐（API Recommendation）**：为每个子任务检索并排序相关 API 文档
3. **链式 API 探索（CoAE）**：逐步实验每个子任务的 API 调用，积累经验用于最终代码生成

### 1. 任务规划

利用 LLM 的 ICL 能力进行规划：
- 提供库的高层概述文本 s（由 GPT-3.5 自动摘要自库文档）
- 从库代码示例中自动提取 few-shot 规划示例 D
- LLM 基于问题 ψ 和上下文（D, s）生成 n 个 API 相关子任务

**设计要点**：规划粒度瞄准每个子任务仅需 1-2 个未见 API 调用的级别。

### 2. API 推荐

两级检索流程：
1. **密集检索**：将 API 文档处理为表格形式（路径、签名、描述），使用密集检索器按语义相似度为每个子任务检索 top-k 个 API
2. **LLM 重排序**：提示 LLM 对检索到的 API 进行子任务级重排序和去噪 → 得到子任务级 API 集合 $\tilde{\mathcal{A}}_i$
3. **全局重排序**：同时进行跨任务全局 API 推荐 → 得到全局 API 集合 $\tilde{\mathcal{A}}_G$
4. 最终提供给生成器的 API 集合为两者并集

### 3. 链式 API 探索（CoAE）——核心创新

按子任务序列逐步探索（灰色块示意）：

**Step 1: 实验代码生成**
- 对每个子任务 $t_i$，LLM 基于子任务描述、库概述、推荐 API 文档和**前序子任务的探索经验** $\mathcal{E}_{1:i-1}$ 生成 m=5 个多样化实验代码片段

**Step 2: 代码执行与观察**
- 在沙箱环境中直接执行每个实验代码
- 收集可执行性 δ、错误信息 ε、程序输出 γ
- 鼓励 LLM 在实验代码中 print 关键信息（如 API 返回对象的格式）以扩展 API 使用知识

**Step 3: 经验选择策略**
- 优先随机选择成功执行且有有效输出的候选
- 若全部失败则随机选择一个失败的候选
- 选中的经验传递给下一个子任务并累积

**Step 4（可选）: 中间自调试**
- 当某子任务的所有候选代码都执行失败时，提示 LLM 进行调试
- 修复常见的简单错误（如缺少 import）和复杂的跨子任务 API 交互问题

最终获得完整的 API 探索轨迹 $\hat{\mathcal{E}} = \{\hat{\mathcal{E}}_i\}_{i=1}^n$，连同推荐 API 文档一起提供给最终代码生成器。

### 基准构建

**Torchdata-Manual**（新建）：
- 100 个手工设计的编程问题，每个涉及 **8-14 个** Torchdata API
- 随机组合 API 后由专业程序员筛选合理组合并设计问题
- 目前公开的执行式库代码基准中 API 调用序列最长

**Torchdata-Github**（已有）：
- 50 个来自 GitHub 客户端代码的问题，涉及 3-8 个 API 调用
- 论文补充了原始工作缺失的外部资源

## 实验关键数据

### 主实验：GPT-3.5（API 未训练模型）在 Torchdata-Manual 上

| 方法 | pass@1 | pass@5 | pass@10 | pass@20 |
|------|--------|--------|---------|---------|
| Direct Generation | 0% | 0% | 0% | 0% |
| DocPrompting (naive RAG) | 0.19% | 0.89% | 1.66% | 2.81% |
| CAPIR | 3.01% | 6.75% | 8.21% | 9.66% |
| EpiGen | 2.16% | 4.40% | 5.23% | 5.86% |
| **ExploraCoder** | **7.00%** | **11.54%** | **13.84%** | **15.67%** |
| ExploraCoder + Self-Debug | **11.5%** | **18.32%** | **20.87%** | **23.51%** |

- ExploraCoder 在 pass@10 上绝对超越 naive RAG **12.18%**，超越 CAPIR **5.63%**
- 加入自调试后进一步提升至 20.87%（pass@10）

### API 未训练 vs 预训练模型对比

在 Torchdata-Github 上：
- GPT-3.5 + ExploraCoder (pass@10 = 21.67%) 超越 GPT-4-1106-preview 直接生成 (21.34%)
- GPT-4-0613 + ExploraCoder 达到 28.11%（pass@10），比 naive RAG 高 4.04%

### API 预训练模型也受益

GPT-4-1106-preview 在 Torchdata-Manual 上：
- 直接生成：pass@1 仅 0.16%
- + naive RAG：pass@1 = 3.19%
- + ExploraCoder：pass@1 = **14.62%**（绝对提升 11.43%）

### 与 Agent 方法对比

| 方法 | pass@10 | success@10 |
|------|---------|-----------|
| ReAct | 2.95% | 12.45% |
| KnowAgent | 11.01% | 23.29% |
| ExploraCoder | 13.84% | 25.40% |
| ExploraCoder* (+ debug) | **20.87%** | **36.81%** |

ExploraCoder 的链式探索比端到端的 Agent 方法更有效。

## 亮点与洞察

1. **探索式编程范式的优雅模拟**：将人类程序员面对不熟悉库时的"读文档→试代码→积累经验→组合方案"工作流形式化为可自动执行的 CoAE 链，设计自然且有效
2. **分治策略解决检索瓶颈**：将复合需求分解为简单子任务后检索 API，自然避免了复杂查询的检索退化问题
3. **经验传递机制**：每个子任务的执行结果（包括输出格式信息）传递给后续子任务，形成递进式知识积累
4. **Torchdata-Manual 基准**：每任务 8-14 个 API 的复杂度远超现有基准（3-8 个），更贴近真实世界编程需求
5. **无需训练**：整个框架基于 prompt 和代码执行，不需要额外训练或微调

## 局限性

1. **基于单一库评估**：仅在 Torchdata 库上验证，泛化到其他编程语言/库的效果需进一步确认
2. **规划质量依赖 LLM**：任务规划的粒度和准确性高度依赖 LLM 能力，对小模型可能不适用
3. **计算开销**：CoAE 需要对每个子任务生成多个实验代码并执行，API 调用成本和时间开销较高
4. **经验选择策略简单**：随机选择成功候选可能不是最优策略，更智能的选择（如基于代码覆盖率）可能进一步提升
5. **标题中的 "Model Compression" 领域标签可能不太准确**：更像是代码生成/工具使用方向

## 相关工作

- **库导向代码生成**：Zan et al. (2022, 2023, 2024) 系统研究了使用外部库 API 的代码生成任务
- **RAG 代码生成**：DocPrompting (Zhou et al. 2023)、CAPIR (Ma et al. 2024) 改进 API 检索
- **CoT 代码生成**：EpiGen (Li et al. 2024) 用自然语言预规划辅助代码生成
- **Agent 方法**：ReAct (Yao et al. 2022)、KnowAgent 采用反应式规划和调试
- **探索式编程**：Sheil (1986)、Beth Kery & Myers (2017) 研究了人类程序员的探索式编程行为

## 评分

⭐⭐⭐⭐ (4/5)

- **创新性** ⭐⭐⭐⭐：CoAE 的设计模拟人类探索式编程范式，巧妙地将 API 使用经验在子任务间传递
- **实验充分性** ⭐⭐⭐⭐：两个基准 × 多个模型 × 与多种 baseline 对比 + 消融实验
- **实用价值** ⭐⭐⭐⭐⭐：对 LLM 使用未知 API 库编程的实际场景有直接帮助，框架易于集成
- **基准贡献** ⭐⭐⭐⭐：Torchdata-Manual 填补了复杂多 API 基准的空白

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] GeoTikzBridge: Advancing Multimodal Code Generation for Geometric Perception and Reasoning](../../CVPR2026/code_intelligence/geotikzbridge_advancing_multimodal_code_generation_for_geometric_perception_and_.md)
- [\[ACL 2026\] OmniDiagram: Advancing Unified Diagram Code Generation via Visual Interrogation Reward](../../ACL2026/code_intelligence/omnidiagram_advancing_unified_diagram_code_generation_via_visual_interrogation_r.md)
- [\[ACL 2026\] PaT: Planning-after-Trial for Efficient Test-Time Code Generation](../../ACL2026/code_intelligence/pat_planning-after-trial_for_efficient_test-time_code_generation.md)
- [\[ACL 2026\] Bootstrapping Code Translation with Weighted Multilanguage Exploration](../../ACL2026/code_intelligence/bootstrapping_code_translation_with_weighted_multilanguage_exploration.md)
- [\[ACL 2026\] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation](../../ACL2026/code_intelligence/ro-slm_onboard_small_language_models_for_robot_task_planning_and_operation_code_.md)

</div>

<!-- RELATED:END -->
