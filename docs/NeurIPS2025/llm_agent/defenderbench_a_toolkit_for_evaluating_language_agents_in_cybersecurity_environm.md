---
title: >-
  [论文解读] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments
description: >-
  [NeurIPS 2025][LLM Agent][网络安全] 提出 DefenderBench，一个开源模块化工具包，用于在攻防和知识理解三类网络安全任务上系统评估 LLM Agent 的能力，覆盖网络入侵模拟、恶意内容检测、代码漏洞检测/修复、CTI 知识问答五大场景，基准测试显示 Claude-3.7-sonnet 综合最强（81.65 分）。
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "网络安全"
  - "评测基准"
  - "漏洞检测"
  - "网络入侵模拟"
---

# DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments

**会议**: NeurIPS 2025  
**arXiv**: [2506.00739](https://arxiv.org/abs/2506.00739)  
**代码**: [https://github.com/microsoft/DefenderBench](https://github.com/microsoft/DefenderBench)  
**领域**: LLM Agent  
**关键词**: 网络安全, LLM Agent, 评测基准, 漏洞检测, 网络入侵模拟

## 一句话总结
提出 DefenderBench，一个开源模块化工具包，用于在攻防和知识理解三类网络安全任务上系统评估 LLM Agent 的能力，覆盖网络入侵模拟、恶意内容检测、代码漏洞检测/修复、CTI 知识问答五大场景，基准测试显示 Claude-3.7-sonnet 综合最强（81.65 分）。

## 研究背景与动机

**领域现状**：LLM Agent 在软件开发、文档翻译、事实核查等领域已展现强大能力，但在网络安全领域的评估仍然不足。现有安全评测（Cybench 做 CTF、CyberMetric 做知识问答、CyberSecEval 做代码漏洞）各自只关注单一任务。

**核心痛点**：
   - 缺乏**统一的**涵盖攻防和知识理解的综合评测平台
   - 不同工作使用不同评测框架，难以公平比较不同 LLM 的安全能力
   - 现有基准大多成本高、不易复现

**本文切入点**：构建一个**实用、开源、模块化**的一站式评测工具包，让研究者能以低成本公平评估 LLM Agent 在网络安全任务上的表现。

## 方法详解

### 整体框架

DefenderBench 由三大模块组成：
1. **数据预处理模块**：自动下载、清洗、划分数据集，缓存到本地
2. **任务环境模块**：为每个任务构建交互式环境（提供 instruction、定义 action space、管理对话历史）
3. **Agent 接口模块**：统一的 LLM Agent 接口，支持开源和闭源模型的即插即用

### 关键设计

**五类网络安全任务**：

1. **网络入侵模拟 (CyberBattleSim)**

    - 基于 CyberBattleSim 仿真工具，转化为文本交互游戏
    - Agent 可执行三种操作：`local_vulnerability`（本地漏洞利用）、`remote_vulnerability`（远程攻击）、`connect`（凭证连接）
    - 两种网络拓扑：Chain（链式，较简单）和 CTF（捕获旗帜，更复杂）
    - 指标：节点接管率（winning rate）

2. **恶意内容检测**

    - Malicious-Text：钓鱼邮件/短信检测（20,137 样本，500 测试）
    - Malicious-Web：钓鱼网页检测（15,612 样本，500 测试）
    - 指标：Macro-F1

3. **CTI 知识问答 (MCQA)**

    - 基于 CTI-MCQA 数据集，2,338 个网络威胁情报相关四选一问题
    - 500 测试样本 + 20 few-shot 样本池
    - 指标：Macro-F1

4. **代码漏洞检测**

    - Vulnerable-CG：基于 CodeXGLUE 的 C 语言函数漏洞检测
    - Vulnerable-DV：基于 Devign（FFmpeg + Qemu）的漏洞检测
    - 指标：Macro-F1

5. **代码漏洞修复 (CVEFix)**

    - 240 个单方法漏洞修复样本，覆盖 C/C++/Go/Java/JS/PHP/Python/Rust
    - 给定漏洞代码，要求 Agent 生成修复后的代码
    - 指标：CodeBLEU

**全局指标**：DefenderBench Score = 所有任务指标的无权重平均值

### Agent 基线设计

采用**最小化脚手架**的 baseline agent：
- 提供任务说明（instruction）+ 响应格式要求
- 每步提供完整历史轨迹（prior actions + observations）
- Agent 生成一个 action → 发送到环境 → 获取 observation → 判断是否终止
- 检测/QA 任务最多 5 步，网络入侵最多 100 步

### 损失函数 / 训练策略
本文是评测基准而非训练方法，不涉及损失函数设计。所有 LLM 以零微调的方式直接评测。

## 实验关键数据

### 主实验

| 模型 | CBS-Chain | CBS-CTF | Mal.Text | Mal.Web | MCQA | Vuln-CG | Vuln-DV | CVEfix | DefB |
|------|-----------|---------|----------|---------|------|---------|---------|--------|------|
| Naive Baseline | 19.4 | 22.2 | 52.4 | 50.4 | 25.0 | 50.0 | 47.8 | 83.2 | 43.8 |
| Llama 3.3 70B | **100.0** | 33.3 | 96.0 | 82.8 | 69.6 | 58.0 | 57.4 | 77.3 | 71.8 |
| GPT-4-turbo | 90.0 | 46.7 | 93.4 | 83.2 | 73.8 | **58.2** | 57.6 | 73.7 | 72.1 |
| Claude-3.5-sonnet | 100.0 | 56.7 | 93.8 | 88.2 | 72.4 | 56.4 | 56.8 | 75.7 | 75.0 |
| Claude-3.7-sonnet | **100.0** | **100.0** | 96.2 | 90.0 | 74.2 | 56.6 | 56.0 | 80.2 | **81.7** |
| Claude-3.7-sonnet-think | 100.0 | 76.7 | 94.4 | **91.0** | **78.2** | 54.6 | 52.8 | 79.5 | 78.4 |
| o3 | 83.3 | 20.0 | 92.4 | 88.0 | 76.4 | 30.8 | **59.6** | 55.6 | 63.9 |

### 消融实验

**模型规模效应**：
- Llama 3.1 8B → 70B：DefB 54.7 → 68.7（+14.0）
- Llama 3.2 1B → 3B：DefB 38.3 → 50.2（+11.8）
- GPT-4.1 → 4.1-mini → 4.1-nano：63.9 → 58.9 → 47.5（规模越大越好）

**Few-shot 增强**：
- 大部分大模型从 few-shot ICL 中受益显著
- 小模型（Llama 3.2 1B/3B, Phi-3.5-mini）反而因长输入而性能下降

**CoT 效果**：
- 交互式任务（网络入侵）中 CoT 最有效：GPT-4o 提升 17.0 分
- 静态任务中 CoT 效果有限，部分模型甚至略降

### 关键发现
- **Claude-3.7-sonnet 是综合最强模型**（81.65），尤其在两个网络入侵环境均达 100% 胜率
- **推理增强模型（o1/o3/o4-mini）并未超越基础模型**——在安全任务上推理能力不是唯一关键
- **漏洞检测仍是最难的任务**——大多数模型仅略强于随机基线，说明 LLM 在精细程序理解上的局限
- **小模型在长输入场景（如 HTML 网页检测）表现极差**——Llama 3.2 1B 甚至低于随机基线
- CodeBLEU 作为漏洞修复评测指标可能不够理想——copy-paste baseline 反而得分最高

## 亮点与洞察
- **全面性**：是目前最完整的 LLM 网络安全评测工具包，覆盖攻/防/知识三个维度、五类任务
- **模块化设计**：用户可轻松接入自己的 LLM、Agent、新任务，支持 Weights & Biases 可视化
- **公平对比**：统一 Agent 框架 + 标准化数据处理，消除了不同工作间的评测偏差
- **实用洞察**：揭示了推理模型在安全任务上的意外弱点，以及模型规模对安全能力的关键影响
- **成本友好**：有意控制测试集规模（500 样本），使中小团队也能负担得起评测

## 局限与展望
- **Agent 设计过于简单**：仅使用最小化脚手架的 baseline agent，未探索更复杂的 tool-augmented agent（如集成静态分析工具）
- **CVEFix 评测指标不理想**：CodeBLEU 不能准确反映小范围代码修改的质量，需要更好的评测指标
- **任务覆盖可扩展**：未包含社会工程、取证分析、日志分析等重要安全场景
- **网络入侵环境较受限**：CyberBattleSim 的拓扑结构较简化，与真实网络环境差距较大
- **未考虑 Agent 的安全风险**：作为双用技术，未深入讨论 LLM Agent 被滥用的风险对策

## 相关工作与启发
- **vs AgentBench/SWE-bench**：这些通用 Agent 基准不覆盖安全领域，DefenderBench 填补空白
- **vs Cybench**：Cybench 只关注 CTF，DefenderBench 覆盖更广（攻防+知识）
- **vs CyberSecEval**：CyberSecEval 聚焦代码安全，DefenderBench 增加了网络入侵和恶意内容检测
- **启发**：未来可以将 DefenderBench 与 red-teaming 框架结合，评估 LLM Agent 在对抗场景下的鲁棒性

## 评分
- 新颖性: ⭐⭐⭐ 工程贡献大于方法创新，但填补了重要评测空白
- 实验充分度: ⭐⭐⭐⭐ 覆盖 17+ 个模型、5 类任务、多种增强策略，对比充分
- 写作质量: ⭐⭐⭐⭐ 结构清晰，任务描述详细
- 价值: ⭐⭐⭐⭐ 对 LLM 安全能力评估有重要参考价值，开源工具包实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](../../ACL2026/llm_agent/feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[NeurIPS 2025\] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement](the_lighthouse_of_language_enhancing_llm_agents_via_critique-guided_improvement.md)
- [\[ACL 2025\] LegalAgentBench: Evaluating LLM Agents in Legal Domain](../../ACL2025/llm_agent/legalagentbench_evaluating_llm_agents_in_legal_domain.md)
- [\[ICLR 2026\] Gaia2: Benchmarking LLM Agents on Dynamic and Asynchronous Environments](../../ICLR2026/llm_agent/gaia2_benchmarking_llm_agents_on_dynamic_and_asynchronous_environments.md)
- [\[ACL 2025\] ToolHop: A Query-Driven Benchmark for Evaluating Large Language Models in Multi-Hop Tool Use](../../ACL2025/llm_agent/toolhop_multi_hop_tool_use.md)

</div>

<!-- RELATED:END -->
