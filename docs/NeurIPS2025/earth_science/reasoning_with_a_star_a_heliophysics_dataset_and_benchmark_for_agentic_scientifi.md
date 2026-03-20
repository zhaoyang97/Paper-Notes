# Reasoning With a Star: A Heliophysics Dataset and Benchmark for Agentic Scientific Reasoning

**会议**: NeurIPS 2025  
**arXiv**: [2511.20694](https://arxiv.org/abs/2511.20694)  
**代码**: HuggingFace - SpaceML/ReasoningWithAStar  
**领域**: Agent  
**关键词**: scientific reasoning, multi-agent, heliophysics, systems engineering, benchmark

## 一句话总结
提出 Reasoning With a Star (RWS)，一个源自 NASA 太阳物理暑期学校问题集的 158 道科学推理 benchmark（含数值/符号/文本三类答案），配合 unit-aware 评分器，比较了四种多 agent 协调模式（HMAW/PACE/PHASE/SCHEMA），发现没有单一模式在所有任务上占优——系统工程启发的 SCHEMA 在需要严格约束验证的任务上最强。

## 研究背景与动机

1. **领域现状**：LLM 在推理 benchmark（GSM8K、MATH、GPQA）上表现出色，但在需要物理假设、单位一致性和科学格式的领域推理上仍有不足。太阳物理作为跨学科领域涉及气候、通信、空间安全，但在 LLM 推理 benchmark 中严重缺失。
2. **现有痛点**：(1) 现有 benchmark 不测试科学推理——不要求声明假设、保持单位一致、提供正确格式；(2) 多 agent 系统缺乏在真实科学推理任务上的系统性比较；(3) 单次推理（single-shot）的 LLM 存在"推理幻觉"和代数错误。
3. **核心矛盾**：科学推理不是单一逻辑跳跃——需要领域专业知识、迭代精炼和假设验证。但多 agent 系统的设计没有清晰的指导原则。
4. **本文要解决什么？** (1) 构建一个面向真实科学推理的 benchmark；(2) 系统比较多 agent 模式的适用场景。
5. **切入角度**：将系统工程原则（复杂性必须被赚取，不能被假设）应用于 agent 设计，在不使用 RAG 的前提下评估纯科学推理能力。
6. **核心 idea 一句话**：不同 agent 协调模式适合不同类型的推理任务——算术推理适合轻量自我批判，科学推理需要系统工程式的约束跟踪和验证。

## 方法详解

### 整体框架
RWS benchmark（158 道太阳物理题，数值/符号/文本三类） -> 程序化评分器（unit-aware 数值容差、CAS 符号等价、schema 验证） -> 评估 single-shot + 4 种多 agent 模式 -> 跨 benchmark 比较（+ GSM8K/MATH/GPQA/HumanEval/SWE-bench）。

### 关键设计

1. **RWS 数据集构建**:
   - 做什么：从 NASA/UCAR LWS 暑期学校问题集构建 benchmark。
   - 核心思路：OCR -> 手动清洗 -> JSONL 格式。每道题包含问题描述、中间推理步骤、最终答案、答案类型（numeric 38题/symbolic 52题/textual 68题）、格式提示和元数据。物理假设保留在问题和推理步骤文本中。
   - 设计动机：太阳物理要求声明假设（如绝热膨胀、忽略某些损耗项）、保持单位一致性、提供正确科学格式——这些是现有 benchmark 不测试的。

2. **程序化评分器**:
   - 做什么：自动检查科学答案的正确性。
   - 核心思路：数值答案——5% 容差 + 单位检查；符号答案——用 SymPy 检查代数等价性；文本答案——语义等价判断。自动评分失败时，用两个 LLM agent（Parser + Judge）做二次验证。
   - 设计动机：科学答案的正确性不能用简单字符串匹配——代数等价的表达式可能形式不同，单位必须正确。

3. **四种多 Agent 模式**:
   - **HMAW**（Hierarchical Multi-Agent Workflow）：CEO/manager/worker 层级交接。
   - **PACE**（Plan-Answer-Critique-Enclose）：生成答案后自我批判循环。
   - **PHASE**（Plan-Hypothesize-Analyze-Solve-Evaluate）：先提假设再求解验证。
   - **SCHEMA**：系统工程启发，包括需求跟踪、假设管理、接口检查。
   - 设计动机：覆盖从简单层级到复杂工程流程的不同协调策略，系统比较其适用场景。

### 损失函数 / 训练策略
无训练。所有模式使用 Gemini 2.5 Pro 作为 base model。使用 SWE-agent 适配器处理代码 benchmark。

## 实验关键数据

### 主实验（Single-shot 基线）

| 模型 | RWS 准确率 |
|------|-----------|
| Gemini 2.5 Pro | **35.44%** |
| OpenAI OSS 120B | 32.91% |
| Meta Llama 3.3 | 31.01% |
| Mistral 24.11 | 27.22% |

### 多 Agent 模式对比

| 数据集 | HMAW | PACE | PHASE | SCHEMA |
|--------|------|------|-------|--------|
| GSM8K | 91.1 | **93.4** | 92.4 | 86.4 |
| MATH | 78.3 | **81.5** | 77.8 | 71.4 |
| GPQA | **79.0** | 77.1 | 77.2 | 73.4 |
| RWS | 39.5 | 41.9 | 42.5 | **44.3** |
| HumanEval | 30.5 | 37.8 | 36.0 | **43.3** |
| SWE-bench Verified | 53.8 | 55.7 | 60.5 | **63.2** |

### 关键发现
- **没有单一模式在所有任务上最优**——这是最重要的结论。
- **PACE 在数学推理上最强**（GSM8K 93.4%, MATH 81.5%）：轻量自我批判循环足以纠正计算错误。
- **HMAW 在 GPQA 上最强**（79.0%）：简单层级交接足以处理分类式科学 QA。
- **SCHEMA 在 RWS/HumanEval/SWE-bench 上最强**（44.3%/43.3%/63.2%）：需要格式约束、假设跟踪和需求验证的任务受益于系统工程方法。
- **所有多 agent 模式在 RWS 上都超过 single-shot**（39-44% vs 35%）——即使简单的协调也能增强科学推理。
- **RWS 上的绝对准确率仍然很低**（<45%）——科学推理对当前 LLM 仍然很难。

## 亮点与洞察
- **"复杂性必须被赚取，不能被假设"的系统工程原则**应用于 agent 设计非常有智慧：添加更多 agent 不一定更好，关键是匹配任务需求。
- **RWS 的评分器设计**很有价值：unit-aware 数值容差 + CAS 符号等价 + 语义判断检验 + LLM 二次验证。可以迁移到其他科学 benchmark。
- **跨 benchmark 比较**的方法论值得学习：在统一条件下比较不同模式在不同类型任务上的表现，提供了有解释力的分类指导。

## 局限性 / 可改进方向
- **数据集规模小**（158 道题）：可能不足以得出可靠的统计结论。
- **仅太阳物理领域**：泛化到其他科学领域需验证。
- **no RAG 设置限制了实用性**：真实科学推理需要访问参考资料和公式表。
- **建议方向**：扩展 RWS 到其他空间科学领域，添加 RAG 支持。

## 相关工作与启发
- **vs GSM8K/MATH**: 这些测试数学推理，RWS 测试科学推理（需要假设声明、单位一致）——不同维度。
- **vs GPQA**: GPQA 是科学 QA 但主要是选择题，RWS 要求生成具体数值/公式/解释。
- **vs SWE-bench**: SWE-bench 测试代码能力，RWS 测试科学推理——但 SCHEMA 在两者上都最强，说明约束验证能力是通用技能。

## 评分
- 新颖性: ⭐⭐⭐⭐ 太阳物理推理 benchmark + 多 agent 模式系统比较
- 实验充分度: ⭐⭐⭐⭐ 跨 6 个 benchmark 比较，但 RWS 本身规模小
- 写作质量: ⭐⭐⭐⭐ 结构清晰，系统工程原则贯穿全文
- 价值: ⭐⭐⭐⭐ 提供了有解释力的 agent 模式选择指导
