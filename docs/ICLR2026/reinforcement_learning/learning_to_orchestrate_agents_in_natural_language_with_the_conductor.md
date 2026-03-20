# Learning to Orchestrate Agents in Natural Language with the Conductor

**会议**: ICLR2026  
**arXiv**: [2512.04388](https://arxiv.org/abs/2512.04388)  
**代码**: 随论文提交  
**领域**: llm_reasoning  
**关键词**: 多Agent协调, RL训练, 工作流编排, 推理时扩展, 递归拓扑  

## 一句话总结
用RL训练7B的Conductor模型，通过自然语言输出Agent工作流(子任务分配+通信拓扑)来协调GPT-5/Claude/Gemini等大模型，在LiveCodeBench和GPQA等benchmark上超越所有单模型和多Agent基线，达到SOTA(平均77.27 vs GPT-5的74.78)。

## 背景与动机
1. 不同LLM在不同领域有专长，没有单一模型在所有任务上最优
2. 手动设计的Agent工作流是商业AI产品关键组件，但设计成本高且缺乏适应性
3. 有效的prompting和自修正策略仍是核心研究方向
4. 现有多Agent方法(MoA/RouterDC等)仅做模型路由或使用固定拓扑，表达力有限
5. RL已被证明能在LLM中激发推理能力，但尚未用于学习Agent间协调策略
6. 推理时扩展(test-time scaling)需要新的计算轴——递归拓扑提供了一种可能

## 方法详解
**Conductor框架**：7B LLM通过GRPO训练，输入问题，输出完整的Agent工作流，包含：
- **子任务列表**：自然语言描述的聚焦指令
- **Agent分配**：每个子任务分配给哪个worker LLM
- **访问列表**：每个Agent可以看到哪些先前步骤的输出(定义通信拓扑)

**训练**：960个问题(MATH/MMLU/RLPR/LiveCodeBench)，仅200次GRPO迭代。奖励：格式错误=0，答案正确=1，答案错误=0.5。Worker池: Gemini 2.5 Pro, Claude Sonnet 4, GPT-5, DeepSeek-R1-Distill-Qwen-32B, Gemma3-27B, Qwen3-32B。

**扩展**：
- **自适应Worker池**：训练时随机采样k个模型子集，推理时适配任意开源/闭源组合
- **递归拓扑**：Conductor可指定自己为worker，产生递归调用，实现动态推理时扩展

## 实验关键数据
| 模型 | MATH500 | LiveCodeBench | AIME25 | GPQA-D | 平均 |
|------|---------|--------------|--------|--------|------|
| GPT-5 | 99.0 | 82.90 | 90.8 | 82.3 | 74.78 |
| Gemini 2.5 Pro | 96.0 | 67.24 | 78.3 | 84.8 | 70.97 |
| **Conductor** | **99.4** | **83.93** | **93.3** | **87.5** | **77.27** |
| Conductor-Recursive | - | - | 66.67* | 82.32* | 63.00* |

- 7B Conductor超越所有单模型(包括GPT-5)和多Agent基线(MoA/MASRouter等)
- 仅用开源模型时，仍超Claude Sonnet 4约10%
- 递归拓扑在BigCodeBench上额外提升2.2%
- 涌现行为：任务自适应(MMLU用2步，LiveCodeBench用3-4步)

## 亮点
- 用RL端到端学习Agent协调策略，无需人工设计拓扑——prompt工程/验证/辩论自然涌现
- 7B小模型协调远大于自身的frontier模型，达到前所未有的集体智能
- 自然语言作为工作流规范的统一接口，表达力远超路由分类器
- 递归拓扑开创推理时扩展的新计算轴
- 仅960题×200迭代即收敛，训练效率极高

## 局限性 / 可改进方向
- 依赖昂贵的闭源API(GPT-5/Claude/Gemini)，每次评估成本高
- 训练数据仅960题，可能限制对更多样任务的泛化
- 递归深度人工设限，未探索最优递归策略
- 未分析Conductor在何种情况下失败(错误分配/糟糕prompt)
- Worker池固定于7个模型，扩展到更大池的效率未知

## 与相关工作的对比
- vs MoA/MASRouter/RouterDC: 这些方法用路由分类器选择固定拓扑，Conductor用自然语言自由构建
- vs 自修正(self-reflection): Conductor通过跨模型协调显著超越5轮自修正
- vs RL tool-use: 扩展了RL+工具范式，将"工具"泛化为LLM API调用
- vs 人工设计Agent框架: 端到端RL替代手工scaffold，更灵活且效果更好

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (RL学习Agent协调的范式创新，递归拓扑新颖)
- 实验充分度: ⭐⭐⭐⭐⭐ (7个benchmark+多基线+消融+scale分析)
- 写作质量: ⭐⭐⭐⭐⭐ (清晰流畅，涌现行为分析引人入胜)
- 价值: ⭐⭐⭐⭐⭐ (开创性工作，定义了Agent协调的新范式)
