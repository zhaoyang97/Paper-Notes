# ARise: Towards Knowledge-Augmented Reasoning via Risk-Adaptive Search

**会议**: ACL 2025  
**arXiv**: [2504.10893](https://arxiv.org/abs/2504.10893)  
**代码**: [https://opencausalab.github.io/ARise](https://opencausalab.github.io/ARise)  
**领域**: LLM 推理  
**关键词**: 知识增强推理, MCTS, 风险评估, RAG, 多跳问答  

## 一句话总结

提出 ARise 框架，将贝叶斯风险评估与动态 RAG 集成到蒙特卡洛树搜索中，解决知识增强推理中的错误传播和验证瓶颈问题，在多跳QA任务上平均准确率超 SOTA KAR 方法 23.10%，超 RAG-equipped 推理模型（DeepSeek-R1）25.37%。

## 研究背景与动机

1. **领域现状**:
   - LLM 推理能力提升主要依赖 test-time compute scaling（如 System 2 慢思考）
   - 大型推理模型（LRM）如 DeepSeek-R1 在数学和代码上达到专家水平
   - RAG 是获取外部知识的有效方式，CoT prompting 可以将检索嵌入推理步骤

2. **现有痛点**:
   - **推理方法的局限**: LRM 隐式假设模型已拥有所有推理所需知识，在开放领域（法律、医学）场景下失效
   - **错误传播（Error Propagation）**: 基于 CoT 的知识增强推理中，早期步骤的错误会在链式推理中级联放大
   - **验证瓶颈（Verification Bottleneck）**: 多分支决策过程中的 explore-exploit 权衡难以有效解决；现有验证方案要么依赖不可靠的自验证，要么需要特定的验证器训练

3. **核心矛盾**:
   - 知识增强（RAG）和推理增强（search/reasoning）需要协同工作，但现有方法未能有效结合
   - 在多分支搜索中如何评估中间推理状态的质量？自验证不可靠，外部验证器成本高

4. **本文要解决什么？**
   - 如何在开放领域、知识密集的复杂推理场景中有效结合知识检索与推理搜索
   - 如何在树搜索中动态评估推理路径的风险，平衡探索与利用

5. **切入角度**:
   - 将贝叶斯风险最小化引入 MCTS 的节点评估，用 "问题生成似然" 作为中间状态质量的代理指标
   - 每一步包含分解+检索推理两个动作，精细化推理粒度

6. **核心idea一句话**:
   - 用贝叶斯风险评估指导 MCTS 中知识增强推理的 explore-exploit 权衡

## 方法详解

### 整体框架

ARise 由三个核心组件组成：
1. **Reasoning State Generation（推理状态生成）**: 每步包含问题分解 + 检索后推理
2. **Monte Carlo Tree Search（蒙特卡洛树搜索）**: 将线性推理扩展为树结构
3. **Risk Assessment（风险评估）**: 贝叶斯风险最小化评估中间推理状态

### 关键设计

1. **推理状态生成（Reasoning State Generation）**:
   - 做什么: 每步 LLM 进行问题分解和基于检索文档的推理，中间结果依次追加到推理状态中
   - 核心思路: 第 $i$ 步输入为原始问题 $\mathbf{q}$ + 前序结果 $\mathbf{s_{i-1}}$，先生成子问题 $\mathbf{d_i}$，再结合检索文档得到推理结果 $\mathbf{r_i}$
   - 设计动机: 分解+检索推理的交替进行提供更细粒度的知识获取，每步都有明确定义的（状态, 动作）对

2. **蒙特卡洛树搜索（MCTS）**:
   - 做什么: 包含 Selection（UCT）、Expansion（多角度分解）、Simulation（想象性 rollout）、Backpropagation（自底向上更新）四阶段
   - 核心思路: 
     - **Selection**: $\text{UCT}(\mathbf{s}, \mathbf{a}) = Q(\mathbf{s}, \mathbf{a}) + w\sqrt{\frac{\ln N(Pa(\mathbf{s}))}{N(\mathbf{s}, \mathbf{a})}}$
     - **Backpropagation**: $Q(\mathbf{s}, \mathbf{a}) = \frac{\sum_{\mathbf{c}} Q(\mathbf{c}) \cdot N(\mathbf{c})}{\sum_{\mathbf{c}} N(\mathbf{c})}$
   - 设计动机: 将线性 CoT 推理扩展为树结构，允许回溯和多路径探索，缓解错误传播

3. **风险评估（Risk Assessment）**:
   - 做什么: 用贝叶斯公式将节点的中间结果质量转化为可计算的"问题生成似然"
   - 核心思路: 
     - 相关性: $\log p(\mathbf{r}|\mathbf{q}) \propto \log p(\mathbf{q}|\mathbf{r})$
     - 风险: $\text{Risk}((\mathbf{s}, \mathbf{a}) \to \mathbf{r}|\mathbf{q}) = -\frac{1}{|\mathbf{q}|}\sum_t \log p(q_t | \mathbf{q}_{<t}, \mathbf{r}; \Theta)$
     - 价值: $Q(\mathbf{s}, \mathbf{a}) = 1 - \frac{1}{1+e^{\alpha(\text{Risk} - \beta)}}$
   - 设计动机: 利用策略模型本身计算风险，无需额外训练验证器；风险低说明中间结果与原始问题高度相关

### 损失函数 / 训练策略

- **无需训练**: ARise 是一个纯推理时框架，不需要微调模型
- **关键超参数**: UCT 中的探索权重 $w$，sigmoid 中的平移/缩放因子 $\alpha, \beta$
- **策略模型**: 使用 Qwen2.5-7B/14B-Instruct 和 Llama3.1-8B-Instruct
- **检索器**: 使用标准检索系统动态获取相关文档

## 实验关键数据

### 主实验

在三个多跳QA基准上测试（Qwen2.5-14B-Instruct）：

| 方法 | HotpotQA (EM/F1) | 2Wiki (EM/F1) | MusiQue (EM/F1) | 平均 (EM/F1) |
|------|-----------------|--------------|----------------|-------------|
| Vanilla | 59.50/63.63 | 37.00/50.33 | 14.50/47.07 | 37.00/53.68 |
| Self-Ask | 58.50/64.74 | 38.50/53.45 | 25.00/58.59 | 40.67/58.93 |
| Auto-RAG | 68.00/66.64 | 53.00/55.13 | 35.50/59.05 | 52.17/60.27 |
| RATT | 64.50/73.91 | 43.00/57.48 | 24.00/63.76 | 43.83/65.05 |
| **ARise** | **73.50/75.39** | **56.50/62.61** | **40.50/65.87** | **56.83/67.96** |

- ARise 平均 EM 比最佳基线 Auto-RAG 高 4.66%，F1 比 RATT 高 2.91%
- 在最难的 MusiQue 上 EM 40.50 vs Auto-RAG 35.50（+5.0%）

Qwen2.5-7B-Instruct 结果：
- ARise 平均 EM 47.67，F1 65.83，显著优于所有基线

Llama3.1-8B-Instruct 结果：
- ARise 平均 F1 68.12，即使在小模型上也保持优势

### 关键发现

1. **ARise 显著优于 SOTA KAR 方法**: 平均准确率提升 23.10%，F1 提升 15.52%
2. **超越 RAG-equipped LRM**: 比装备 RAG 的 DeepSeek-R1 平均准确率和 F1 分别提升 4.04% 和 25.37%
3. **搜索型宽推理优于学习型深推理**: 实验证明在开放领域中，基于搜索的多路径探索比深度推理模型的单路径深思更有效
4. **模型规模 scaling**: ARise 的性能随模型规模增大逐步趋近最优，显示良好的扩展性
5. **风险评估有效**: 消融实验验证了贝叶斯风险评估在引导搜索方向上的关键作用

## 亮点与洞察

- **知识获取是推理的必要条件**: 深刻指出 LRM 的 "隐式完备知识假设" 在开放领域不成立
- **风险评估的数学优美**: 用贝叶斯公式将验证问题转化为可计算的策略模型自评估，避免外部验证器
- **错误传播 vs 验证瓶颈的深刻分析**: 清晰地刻画了 KAR 的两大核心挑战
- **MCTS + RAG 的自然结合**: 树搜索的每个节点都包含检索操作，搜索和知识获取无缝融合
- **无需训练、即插即用**: 推理时框架，适用于任意 LLM

## 局限性 / 可改进方向

1. MCTS 的推理成本较高：多次展开、模拟和回溯显著增加 LLM 调用次数
2. 风险评估依赖策略模型自身的条件似然，小模型可能估计不准确
3. 仅在多跳 QA 上评估，其他知识密集推理任务（如科学推理、法律推理）未验证
4. 检索质量的上限限制了推理质量，底层检索器的性能未深入讨论
5. UCT 和风险函数的超参数（$w, \alpha, \beta$）需要调优

## 相关工作与启发

- **RATT (Zhang et al., 2024)**: 基于树结构的 RAG，但验证机制简单
- **Auto-RAG (Yu et al., 2024)**: 自动化检索增强，prompt-based
- **DeepSeek-R1 (2025)**: 最新推理模型，但在知识密集场景下不如 RAG-equipped 方法
- 启发: test-time compute scaling 不应局限于深度思考，宽度搜索（多路径探索）+ 知识检索在开放领域可能更有效

## 评分

| 维度 | 分数 (1-10) |
|------|-----------|
| 创新性 | 8 |
| 技术深度 | 8 |
| 实验充分性 | 8 |
| 写作质量 | 8 |
| 实用价值 | 8 |
| **总分** | **8.0** |
