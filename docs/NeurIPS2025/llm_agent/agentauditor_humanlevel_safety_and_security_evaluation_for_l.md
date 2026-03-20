# AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents

**会议**: NeurIPS 2025  
**arXiv**: [2506.00641](https://arxiv.org/abs/2506.00641)  
**代码**: [https://github.com/Astarojth/AgentAuditor](https://github.com/Astarojth/AgentAuditor)  
**领域**: LLM Agent / AI 安全  
**关键词**: agent safety, security evaluation, LLM-as-judge, memory-augmented reasoning, RAG, benchmark, risk assessment  

## 一句话总结
提出 AgentAuditor，一个通用的无训练记忆增强推理框架，使 LLM 评估者能模拟人类专家评估 agent 的安全与安全性——通过自适应提取结构化语义特征并生成CoT推理轨迹构建经验记忆，多阶段上下文感知 RAG 检索相关经验指导新案例评估，在自建的 ASSEBench（2293条记录×15类風险×29场景）上达到人类水平准确率。

## 背景与动机
LLM agent 安全评估面临四大挑战：
1. **逐步动作的危险遗漏**：规则或LLM评估者常忽略agent分步执行中的隐蔽风险
2. **微妙语义理解不足**：无法捕捉含糊或隐含的不安全行为
3. **复合效应忽视**：小问题逐步累积形成严重风险但现有方法无法识别
4. **模糊的安全/安全规则**：现实中安全边界不清晰——同一行为在严格标准下不安全但在宽松标准下可接受

## 核心问题
如何构建一个**无需训练、通用的**LLM agent安全评估框架，使其评估准确率达到人类专家水平？

## 方法详解

### 关键设计
1. **经验记忆构建**: 让 LLM 对历史交互自适应提取结构化语义特征（场景、风险类型、行为模式），并生成关联的 CoT 推理轨迹。这些组成"经验记忆"——类似人类专家积累的评估经验。

2. **多阶段上下文感知 RAG**: 评估新案例时，动态检索最相关的历史推理经验。多阶段设计：先粗检索（场景匹配）→ 再精检索（风险模式匹配）→ 最终用检索到的经验指导评估推理。

3. **ASSEBench 基准**: 
   - 2293 条精细标注的交互记录
   - 15 种风险类型（包括安全和安全两个维度）
   - 29 个应用场景
   - **关键创新**：引入"严格"和"宽松"双标准判断——同一模糊案例提供两种评价，更贴近现实的灰色地带

### 训练策略
完全无训练。通过 prompt engineering 和 RAG 实现，适用于任何 LLM base。

## 实验关键数据
AgentAuditor 在所有基准上一致改善 LLM 的评估性能，在 ASSEBench 上达到人类水平准确率。

### 消融实验要点
- 经验记忆的作用：移除后性能显著下降
- 多阶段RAG vs 单阶段：多阶段检索更精准
- 不同LLM base的效果：在多种LLM上一致有效
- Strict vs Lenient标准的影响

## 亮点
- **首个同时覆盖安全（safety）和安全（security）的agent评估基准**
- 经验记忆+RAG的框架无需训练即可达到人类水平——实用性极强
- Strict/Lenient双标准设计反映了现实中安全评估的灰色地带
- 2293条高质量标注 × 15类风险 × 29场景 = 大规模全面的评估覆盖
- 与 AgentMisalignment（同系列笔记）互补——后者检测不对齐倾向，前者评估安全违规

## 局限性 / 可改进方向
- 依赖初始经验记忆的质量——冷启动时可能不够全面
- RAG检索的计算开销随经验库增长
- 安全标准的定义可能因文化和法律环境而异
- 仅评估agent的文本交互，未涉及多模态agent

## 与相关工作的对比
- **vs SafetyBench（LLM安全基准）**: SafetyBench评估LLM本身；AgentAuditor评估agent的交互安全——更贴近部署场景
- **vs AgentMisalignment（同系列笔记）**: AgentMisalignment测量不对齐倾向（检测）；AgentAuditor做逐案安全评估（审计）

## 启发与关联
- 经验记忆+RAG的框架可用于任何"LLM-as-Judge"场景——如代码安全审计、内容审核
- Strict/Lenient双标准的设计思路可迁移到VLM的幻觉"严重程度"评估
- ASSEBench可作为agent对齐训练的标准化数据源

## 评分
- 新颖性: ⭐⭐⭐⭐ 记忆增强LLM评估框架+首个安全安全双维基准
- 实验充分度: ⭐⭐⭐⭐⭐ 2293条标注、15类风险、29场景、多LLM验证
- 写作质量: ⭐⭐⭐⭐ 问题定义清晰、基准设计周全
- 价值: ⭐⭐⭐⭐⭐ Agent安全评估是当前最关键的AI安全方向
