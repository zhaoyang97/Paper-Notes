# Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning

**会议**: ACL 2025  
**arXiv**: [2506.03939](https://arxiv.org/abs/2506.03939)  
**代码**: [https://github.com/Graph-Counselor](https://github.com/Graph-Counselor)  
**领域**: LLM Agent  
**关键词**: 图检索增强生成, 多智能体协作, 知识图谱推理, 自反思, 图结构推理  

## 一句话总结
Graph Counselor 提出了一个多智能体协作的 GraphRAG 推理框架，通过 Planning/Thought/Execution 三个 Agent 自适应提取图结构信息，并引入多视角自反思机制纠正推理偏差，在多个图推理任务上超越现有方法。

## 研究背景与动机
1. **领域现状**：GraphRAG 通过建模知识关系提升 LLM 的事实准确性，但现有方法存在两大缺陷
2. **现有痛点**：
   - **信息聚合低效**：依赖单 Agent + 固定迭代模式，无法自适应捕获图数据中的多层次信息（文本/结构/度数）
   - **推理机制僵化**：预设推理方案无法根据任务复杂度动态调整推理深度，也缺乏语义纠正能力
3. **核心矛盾**：图结构的非线性本质与 LLM 线性文本理解之间存在天然鸿沟，导致语义理解偏差
4. **本文要解决什么**：设计灵活的图信息提取和自纠正推理框架
5. **切入角度**：将图推理分解为规划→思考→执行三步，让不同 Agent 各司其职
6. **核心idea一句话**：三个专用 Agent 自适应提取图信息 + 多视角自反思纠正推理偏差

## 方法详解

### 整体框架
两层架构：内层是 AGIEM（三Agent迭代推理），外层是 SR（自反思纠错）。输入问题 → Planning Agent 分析推理路径 → Thought Agent 确定需要什么图信息 → Execution Agent 调用图操作提取信息 → 迭代直到推理完成 → SR 验证并反思 → 若有错误则回传修正上下文重新推理。

### 关键设计
1. **Adaptive Graph Information Extraction Module (AGIEM)**:
   - **Planning Agent**：分析问题语义，制定后续推理路径或判断已有信息是否足够推断答案
   - **Thought Agent**：基于规划结果确定当前推理步骤需要什么具体图信息
   - **Execution Agent**：调用四种图操作组件（Retrieve/Feature/Neighbor/Degree），支持串行组合和并行执行。如 $\text{Retrieve}(t) \circ \text{Feature}(I_v, \mathcal{T}_v)$ 先检索再取特征
   - 设计动机：三 Agent 各司其职，避免单 Agent 在规划/信息需求判断/实际操作之间角色混淆

2. **Self-Reflection with Multiple Perspectives (SR)**:
   - 做什么：检查推理结果的逻辑一致性，纠正语义和图结构之间的对齐偏差
   - 核心思路：三阶段反思——(1) Recap & Understanding 回顾推理目标 (2) Analysis & Adjustment 发现遗漏/冗余/不一致 (3) Refinement & Update 优化推理策略。结合反向推理和多视角分析
   - 设计动机：对抗 LLM 在图推理中的语义漂移问题，通过发散思维从多角度纠错

3. **LLM 状态转换机制**:
   - 单个 LLM 通过上下文切换在三种 Agent 角色间转换，实现多Agent效果
   - 外层 judgment module 判断推理正确性，不正确则触发 SR 反思循环

### 损失函数 / 训练策略
无需训练/微调。纯基于提示的方法，在推理时使用。

## 实验关键数据

### 主实验（GRBENCH 数据集，QwenScore）
| 模型 | Base | TextRAG | GraphRAG-1hop | Graph-CoT | Graph Counselor |
|------|------|---------|---------------|-----------|----------------|
| Llama-3.1-70B (Academic) | 10.82 | 14.00 | 34.94 | ~30 | **38+** |
| Llama-3.1-70B (Legal) | 16.11 | 28.89 | 37.22 | ~35 | **42+** |

### 消融实验
| 配置 | 说明 |
|------|------|
| w/o AGIEM (单Agent) | 性能显著下降，验证多Agent协作的必要性 |
| w/o SR (无反思) | 推理准确率下降，特别是复杂多跳问题 |
| w/o Degree组件 | 度数相关查询失败 |

### 关键发现
- 在多跳推理（需要跨越多个图关系）的问题上优势最大
- SR 反思机制对复杂问题贡献显著，对简单问题影响较小
- Execution Agent 的组件组合机制是灵活性的关键
- 不同领域（学术/电商/医疗/法律）都有提升，泛化性好

## 亮点与洞察
- 三Agent分工的设计比单Agent做所有事情更清晰有效——规划和执行分离是Agent设计的最佳实践
- 图操作组件的串并行组合提供了类似编程的灵活性，比固定跳数的 GraphRAG 更适应复杂查询
- SR 的多视角反思比简单的自检更有效，因为它引入了发散思维避免重复犯错
- 用单个 LLM 角色切换实现多Agent效果，降低了部署复杂度

## 局限性 / 可改进方向
- 多轮迭代+反思的计算开销较大，推理延迟高
- 仅在 KGQA 类任务上验证，未探索其他图任务（如链接预测、图分类）
- 图操作组件是手工定义的，缺乏自动发现新操作的能力
- SR 的判断模块（correctness flag）可能不可靠，影响反思触发的准确性

## 相关工作与启发
- **vs Graph-CoT**: Graph-CoT 用固定迭代模式，Graph Counselor 三Agent自适应调整推理深度
- **vs KG-Agent**: 都是图上的Agent推理，Graph Counselor 加入了更完整的自反思机制
- **vs ToG (Think-on-Graph)**: ToG 依赖单一Agent遍历图，Graph Counselor 的多Agent分工更高效

## 评分
- 新颖性: ⭐⭐⭐ 多Agent分工+图操作组合+自反思的组合有新意，但各个组件独立看并不全新
- 实验充分度: ⭐⭐⭐⭐ 多个数据集+多个基线模型+消融+不同领域
- 写作质量: ⭐⭐⭐⭐ 框架图清晰，Agent角色描述明确
- 价值: ⭐⭐⭐⭐ 为GraphRAG提供了实用的多Agent推理范式
