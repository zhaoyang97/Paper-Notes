---
title: >-
  [论文解读] PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer
description: >-
  [NEURIPS2025][视频理解][video anomaly detection] 提出 PANDA，一个基于 MLLM 的 Agentic AI 工程师框架，通过自适应场景感知策略规划、目标驱动启发式推理、工具增强自反思和链式记忆四大能力，实现无需训练和人工干预的通用视频异常检测。 现有问题： 视频异常检测（VAD）…
tags:
  - "NEURIPS2025"
  - "视频理解"
  - "video anomaly detection"
  - "agentic AI"
  - "RAG"
  - "self-reflection"
  - "chain-of-memory"
---

# PANDA: Towards Generalist Video Anomaly Detection via Agentic AI Engineer

**会议**: NEURIPS2025  
**arXiv**: [2509.26386](https://arxiv.org/abs/2509.26386)  
**代码**: [GitHub](https://github.com/showlab/PANDA)  
**领域**: LLM Agent  
**关键词**: video anomaly detection, agentic AI, RAG, self-reflection, chain-of-memory

## 一句话总结

提出 PANDA，一个基于 MLLM 的 Agentic AI 工程师框架，通过自适应场景感知策略规划、目标驱动启发式推理、工具增强自反思和链式记忆四大能力，实现无需训练和人工干预的通用视频异常检测。

## 研究背景与动机

**现有问题**: 视频异常检测（VAD）方法按范式分为训练依赖型和无训练型，两者都需要针对新场景的人工参与（标注数据/手工提示模板/后处理规则），无法实现真正的通用化

**训练依赖方法局限**: 半监督/弱监督/指令微调方法在分布外环境和新异常类型上性能急剧下降，且标注和训练成本高

**无训练方法局限**: 虽然利用预训练 LLM/VLM 免去训练，但仍依赖手工设计的预处理步骤、提示模板、规则管理和后处理，静态管线缺乏自适应能力

**复杂场景的挑战**: 低分辨率、弱光照、高噪声、长时间跨度异常等复杂条件下，传统方法普遍失效

**Agent 范式的机遇**: 类比人类工程师系统性分析问题、适应复杂环境、通过工具使用和经验迭代改进的能力，Agent 框架可实现自主感知、策略制定、推理、工具调用和自我改进

**核心目标**: 实现 Generalist VAD——自动处理任意场景、任意异常类型，零训练数据、零人工干预

## 方法详解

### 整体框架

PANDA 基于 LangGraph 构建，以 VLM（Qwen2.5VL-7B）做感知和推理、MLLM（Gemini 2.0 Flash）做规划和反思。工作流程：(1) 环境感知 + RAG 策略规划 → (2) 目标驱动启发式推理 → (3) 遇到不确定时触发工具增强自反思 → (4) 链式记忆持续积累经验。支持离线和在线两种推理模式。

### 关键设计 1: 自适应场景感知策略规划

**环境感知**：均匀采样 M 个关键帧，结合用户查询构造感知提示送入 VLM，获取结构化环境信息：
- Scene Overview（场景类型和活动）
- Potential Anomalies（场景可能的异常类型）
- Weather Condition（时间/天气）
- Video Quality（分辨率/清晰度）

**RAG 策略规划**：
1. 根据用户查询，由 MLLM 生成结构化异常知识库 $\kappa_a$（事件类型 + 异常规则 + 适用场景），每类异常预定义 H=20 条规则-场景对
2. 用环境信息作为 query，通过 FAISS 检索 top-k=5 条最相关的异常规则
3. 整合用户查询、环境信息和检索规则，由 MLLM 生成检测策略计划，包含：预处理步骤、潜在异常列表、启发式推理提示

### 关键设计 2: 目标驱动启发式推理

在策略计划引导下，VLM 对每个视频片段（每片段 s=5 帧）进行推理：

- **输入**：增强后的视频片段 + 视觉记忆 + 推理提示（含潜在异常、规则、启发式提示、反思信息）
- **短期记忆**：维护最近 l=5 步的文本推理记录和对应视觉帧
- **输出**：三元组 {Status, Score, Reason}
    - Status: Normal（正常）/ Abnormal（异常）/ Insufficient（信息不足）
    - Score: [0,1] 异常概率
    - Reason: VLM 给出的判断理由
- **关键机制**：当 Status 为 Insufficient 时，触发反思模块获取更多信息后重新推理

### 关键设计 3: 工具增强自反思

当推理结果为 Insufficient 时激活，配备工具集 $\tau$（图像去模糊、去噪、亮度增强、图像检索、目标检测、网络搜索等）。

**经验驱动反思**：
1. 查询长期链式记忆（Long CoM），检索最相似的历史反思案例
2. 构建反思提示（融合用户查询、环境信息、策略、规则、短期记忆、不确定原因、历史经验）
3. MLLM 分析不确定原因并生成反思计划：需调用的工具及参数 + 新异常规则 + 新启发式提示

**工具调用与精炼推理**：
- 执行反思计划中的工具，获取文本增强信息（检测结果、搜索结果）和视觉增强信息（处理后的帧 + 检索的历史关键帧）
- 用增强信息更新推理提示，VLM 重新推理
- 最多反思 r=3 轮，若仍 Insufficient 则赋默认分数跳过

### 关键设计 4: 自改进链式记忆（CoM）

- **短期 CoM**：推理阶段记录最近 l 步文本+视觉记忆；反思阶段记录历史反思输出
- **长期 CoM**：维护跨时间步的完整决策轨迹 $\text{Long-CoM} = \{M_1, M_2, \ldots, M_T\}$，每个 $M_t = \{\text{Result}_\text{reasoning}, \text{Result}_\text{reflection}, \text{Result}_\text{reasoning}^\text{refined}\}$
- **作用**：初始时 Long CoM 为空依赖短期记忆；随着处理更多片段，长期记忆渐进积累，支持记忆一致的推理和反思规划

### 损失函数

PANDA 是无训练框架，不涉及可学习损失函数。核心是 RAG 检索（all-MiniLM-L6-v2 编码 + FAISS 索引）和 VLM/MLLM 的 prompt engineering。

## 实验关键数据

### 主实验：多场景/开放集/复杂场景性能对比

| 方法 | 监督方式 | 可解释 | 免人工 | UCF-Crime AUC% | XD-Violence AP% | UBnormal AUC% | CSAD AUC% |
|------|---------|--------|--------|---------------|-----------------|---------------|-----------|
| HL-Net | 弱监督 | ✗ | ✗ | 82.44 | 73.67 | - | - |
| RTFM | 弱监督 | ✗ | ✗ | 84.30 | 77.81 | 64.94 | - |
| VadCLIP | 弱监督 | ✗ | ✗ | 88.02 | 84.51 | - | - |
| Holmes-VAU | 指令微调 | ✓ | ✗ | 88.96 | 87.68 | 56.77 | 72.47 |
| LAVAD | 无训练 | ✓ | ✗ | 80.28 | 62.01 | 64.23 | 57.26 |
| AnomalyRuler | 无训练 | ✓ | ✗ | - | - | 71.90 | - |
| **PANDA (离线)** | **无训练** | **✓** | **✓** | **84.89** | **70.16** | **75.78** | **73.12** |
| PANDA (在线) | 无训练 | ✓ | ✓ | 82.57 | 63.57 | 72.41 | 71.25 |

PANDA 在所有无训练方法中全面最优；在 UBnormal 开放集上超过所有训练方法；在复杂场景 CSAD 上超过 Holmes-VAU（指令微调方法）。

### 消融实验：PANDA 核心能力贡献

| Planning | Reflection | Memory | UCF-Crime AUC% |
|----------|-----------|--------|---------------|
| ✗ | ✗ | ✗ | 75.25 |
| ✓ | ✗ | ✗ | 80.37 (+5.12%) |
| ✓ | ✓ | ✗ | 82.63 (+2.26%) |
| ✓ | ✓ | ✓ | **84.89 (+2.26%)** |

### 超参数分析（UCF-Crime）

| 反思轮数 r | AUC% | 规则数 k | AUC% | 短期记忆长度 l | AUC% |
|-----------|------|---------|------|--------------|------|
| 1 | 83.83 | 1 | 82.79 | 1 | 82.92 |
| **3** | **84.89** | **5** | **84.89** | **5** | **84.89** |
| 5 | 84.91 | 9 | 83.92 | 9 | 84.03 |

### 关键发现

1. **策略规划是最大增益点**：+5.12% AUC，RAG 检索异常规则 + 场景感知策略规划对推理精度提升最显著
2. **反思和记忆贡献相当**：各 +2.26%，工具增强反思解决模糊案例，链式记忆提供跨时间一致性
3. **超参数甜蜜点明确**：r=3, k=5, l=5 为最优平衡；过多过少都有害（信息不足 vs 噪声干扰）
4. **CSAD 复杂场景优势**：73.12% vs Holmes-VAU 的 72.47%，说明 PANDA 在低质量视频和长时间异常上更鲁棒
5. **开放集泛化最强**：UBnormal 上 75.78% 超过所有方法，12 类未见异常类型的检测能力验证了 Generalist 定位

## 亮点与洞察

1. **Generalist VAD 新范式**：首次实现零训练、零人工的通用视频异常检测，开创了 Agent 范式在 VAD 领域的先河
2. **RAG + 场景感知的巧妙结合**：不是简单的 RAG，而是先感知环境再以环境信息作为 query 检索相关规则，确保策略与场景匹配
3. **三层推理架构（推理-反思-精炼）**：Normal/Abnormal/Insufficient 三状态设计优雅地处理了置信度不足的情况，避免强行判断
4. **短期+长期双记忆机制**：短期记忆提供局部时序上下文，长期记忆积累全局经验，实现 "越用越聪明"
5. **CSAD 基准贡献**：构建了专门的复杂场景异常检测基准（低分辨率、弱光、高噪声、长时间异常），填补了评测空白

## 局限性

1. **推理成本高**：每帧需调用 VLM 推理，反思时还需 MLLM + 工具调用，实时性和 API 成本是实际部署的瓶颈
2. **在 UCF-Crime 和 XD-Violence 上不及训练方法**：AUC 84.89% vs VadCLIP 88.02%、AP 70.16% vs Holmes-VAU 87.68%，有训练数据时仍有差距
3. **工具集固定**：当前工具（去模糊、检索、检测等）预定义，面对全新场景可能缺少所需工具
4. **依赖特定模型组合**：Qwen2.5VL-7B + Gemini 2.0 Flash，不同模型的效果和替代性未验证

## 相关工作与启发

- **vs LAVAD / AnomalyRuler**：PANDA 从静态 prompt 升级为动态 Agent，自适应感知+反思+记忆使其在所有场景全面超越无训练方法
- **vs Holmes-VAU**：Holmes-VAU 需要指令微调数据，PANDA 完全免训练但在复杂场景上性能相当甚至更好
- **vs 通用 Agent（AutoGPT/HuggingGPT）**：PANDA 为 VAD 量身设计了 RAG 知识库、启发式推理和 CoM 记忆机制，不是简单套用通用 Agent 框架
- **启发**：Agent 范式从 "工具调用" 升级到 "策略规划 + 推理 + 反思 + 记忆" 的完整闭环，可推广到其他视频理解任务（如行为识别、事件预测）

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (首个 Generalist VAD Agent，四大能力设计系统性强)
- 实验充分度: ⭐⭐⭐⭐ (4 个数据集、3 种设置、详细消融和超参分析，但缺少效率分析)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，公式详尽，图示直观)
- 价值: ⭐⭐⭐⭐⭐ (开创 VAD Agent 新方向，CSAD 基准有社区价值，代码开源)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] VADTree: Explainable Training-Free Video Anomaly Detection via Hierarchical Granularity](vadtree_explainable_training-free_video_anomaly_detection_via_hierarchical_granu.md)
- [\[NeurIPS 2025\] MoniTor: Exploiting Large Language Models with Instruction for Online Video Anomaly Detection](monitor_exploiting_large_language_models_with_instruction_for_online_video_anoma.md)
- [\[CVPR 2026\] Your One-Stop Solution for AI-Generated Video Detection](../../CVPR2026/video_understanding/your_one-stop_solution_for_ai-generated_video_detection.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/video_understanding/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[CVPR 2025\] Anomize: Better Open Vocabulary Video Anomaly Detection](../../CVPR2025/video_understanding/anomize_better_open_vocabulary_video_anomaly_detection.md)

</div>

<!-- RELATED:END -->
