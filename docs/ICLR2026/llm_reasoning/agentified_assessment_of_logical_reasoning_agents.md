# Agentified Assessment of Logical Reasoning Agents

**会议**: ICLR2026  
**arXiv**: [2603.02788](https://arxiv.org/abs/2603.02788)  
**代码**: [HuggingFace数据集](https://huggingface.co/datasets/yfxiao/folio-refined)  
**领域**: llm_reasoning  
**关键词**: 逻辑推理评测, Agent-to-Agent评估, 一阶逻辑, 自动形式化, SMT求解  

## 一句话总结
提出基于Agent的评测框架(AAA)，用assessor agent标准化地评估逻辑推理agent，并以自动形式化agent（NL→Z3Py+SMT求解）在清洗后的FOLIO上达到86.70%准确率，大幅超过CoT基线73.89%。

## 背景与动机
1. 评估推理agent时，运行失败(超时/解析错误)与推理错误常被混淆在单一准确率数字中
2. 传统评测harness将benchmark逻辑与agent实现紧耦合，集成成本O(n)随benchmark数线性增长
3. FOLIO数据集存在潜在标签错误和NL-FOL翻译质量问题，需要系统化清洗
4. 需要一个可复现、可审计、对执行失败鲁棒的评测框架
5. Agent评测缺乏标准化接口，不同agent难以即插即用地参与多个benchmark
6. 一阶逻辑推理是LLM的重要能力，但现有评测方法不够可靠

## 方法详解
**Agentified Agent Assessment (AAA) 框架**：将评测逻辑封装为assessor agent，与被测agent通过标准A2A接口通信。Assessor负责下发任务、执行预算控制(超时/重试)、输出解析、结构化失败类型记录(Timeout/RuntimeError/ParseError)，并生成机器可读的评测报告。集成成本从O(n)降到O(1)。

**FOLIO数据清洗流水线**：(1) 用Vampire定理证明器对FOL表示做形式化验证，通过可满足性检查判断True/False/Uncertain标签；(2) 当验证结果与标签冲突时，用critique agent诊断翻译错误，refiner agent执行修复，迭代直到标签一致；(3) 超过阈值仍未解决则标记人工审查。训练集标签错误率3.8%，验证集1.5%。

**自动形式化Agent**：两阶段流水线——Stage 1: LLM将自然语言前提/结论生成Z3Py代码；Stage 2: 沙箱执行(60s超时)，通过可满足性检查判断逻辑蕴含关系。包含最多3次自修复循环，遇语法错误时提取错误信息做定向修复。

## 实验关键数据
| 方法 | True准确率 | False准确率 | Uncertain准确率 | 总体准确率 |
|------|-----------|------------|----------------|-----------|
| Chain-of-Thought | 89.04% | 44.26% | 84.06% | 73.89% |
| Auto-formalization | 90.41% | **77.05%** | **91.30%** | **86.70%** |

- 清洗后FOLIO验证集203例；backbone: Gemini 2.5 Flash, T=0.0
- False类别提升最大(+32.79%)，说明形式化验证在矛盾检测上优势显著
- Uncertain类别也有明显提升(+7.24%)，solver擅长处理逻辑不确定性

## 亮点
- AAA框架将评测本身agent化，实现即插即用评估，集成成本O(1)
- 数据清洗流水线系统化修复FOLIO标签错误，提高benchmark可靠性
- 结构化失败类型记录使评测结果可审计、可追溯
- 自动形式化+SMT求解在False/Uncertain类别上优势显著

## 局限性 / 可改进方向
- 仅在单一数据集(FOLIO 203例验证集)上验证，规模较小
- 仅比较了CoT和Auto-formalization两种agent，缺乏更多方法对比
- 数据清洗流水线仍有30.4%训练样本标记为"problematic"未解决
- 未探索更丰富的tool-using agent场景
- A2A接口的实际互操作性和开销未详细分析

## 与相关工作的对比
- 相比传统静态评测harness，AAA解耦评估逻辑与agent实现
- 基于AgentBeats框架和A2A协议，统一agent间通信标准
- FOLIO清洗工作使用Vampire定理证明器做形式化验证，比人工标注更可靠
- 自动形式化思路与NL→逻辑程序的经典方法一脉相承，但结合了LLM生成和自修复

## 评分
- 新颖性: ⭐⭐⭐ (AAA框架概念新颖，但实际技术较简单)
- 实验充分度: ⭐⭐ (单数据集、少对比方法)
- 写作质量: ⭐⭐⭐⭐ (清晰规范)
- 价值: ⭐⭐⭐ (评测范式有启发性)
