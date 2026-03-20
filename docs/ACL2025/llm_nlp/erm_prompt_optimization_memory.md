# Efficient and Accurate Prompt Optimization: the Benefit of Memory in Exemplar-Guided Reflection

**会议**: ACL 2025  
**arXiv**: [2411.07446](https://arxiv.org/abs/2411.07446)  
**代码**: 无  
**领域**: llm_nlp  
**关键词**: prompt optimization, feedback memory, exemplar retrieval, automatic prompt engineering, LLM  

## 一句话总结
提出 ERM 方法，通过指导性元提示生成带详细解题过程的 exemplar 来增强 feedback 质量，并引入 Feedback Memory 和 Exemplar Factory 两种长期记忆机制来高效存储和复用历史反馈与示例，在多个任务上以约一半的优化步数超越了 SOTA prompt 优化方法。

## 研究背景与动机
1. **领域现状**：自动 prompt 优化旨在无需人工干预地找到最优 prompt，主流方法包括进化式（EvoPrompt）、轨迹式（OPRO、GPO）和反馈式（ProTeGi）三类。
2. **现有痛点**：反馈式方法存在两个核心问题——（a）只使用当前步骤的 feedback，历史 feedback 和未选中的 feedback 被直接丢弃，导致需要更多优化步数才能收敛；（b）推理时检索 exemplar 仅基于语义相似度，未评估其对实际任务性能的影响。
3. **核心矛盾**：有价值的 feedback 信息被浪费，exemplar 的选择与任务性能脱节。
4. **本文要解决什么？** 如何高效利用所有历史 feedback？如何选择真正有助于任务性能的 exemplar？
5. **切入角度**：借鉴人类记忆机制（Ebbinghaus 遗忘曲线），对 feedback 和 exemplar 建立带优先级分数的长期记忆存储，通过评估效果动态调整优先级并选择性遗忘。
6. **核心 idea 一句话**：用记忆机制管理 feedback 和 exemplar，让有价值的信息持续被利用而无价值的被遗忘。

## 方法详解

### 整体框架
ERM（Exemplar-Guided Reflection with Memory）包含三个核心组件：输入是错误样本集合 $\mathcal{B}$ 和当前 prompt $p^t$，经过 Exemplar-Guided Reflection 生成 exemplar 和 feedback，存入 Feedback Memory 和 Exemplar Factory 两种记忆存储，最终输出优化后的 prompt $p^{t+1}$。推理时从 Exemplar Factory 检索 exemplar 拼接到 prompt 中提升预测准确率。

### 关键设计
1. **Exemplar-Guided Reflection（指导性反思）**:
   - 做什么：设计指导性元提示，引导 prompt optimizer 从错误样本中选出典型样本并提供详细解题过程（CoT 风格），再基于这些 exemplar 生成更有信息量的 feedback。
   - 核心思路：$\mathcal{E}^t = M_e(p^t, \mathcal{B}; p^{meta}_{ref*})$，先生成 exemplar 集合（含 question、answer、CoT），再基于 exemplar 生成 feedback $\mathcal{F}^t = M_e(p^t, \mathcal{B}, \mathcal{E}^t; p^{meta}_{ref*})$。
   - 设计动机：传统方法直接在错误样本上生成 feedback 信息量有限，加入详细解题过程使 feedback 更具针对性，为后续 prompt 优化提供更精准的改进方向。

2. **Feedback Memory（反馈记忆）**:
   - 做什么：存储历史 feedback 并为每条分配优先级分数，周期性检索高优先级 feedback 指导 prompt 优化。
   - 核心思路：存储时过滤（仅存储能带来性能提升的 feedback + 去重）；检索时按优先级概率采样 $P_f = \text{softmax}(\{e^{s_p(f_i)/\tau_f}\})$；使用后更新优先级 $s_p^t(f) = (1-\beta)s_p(f)^{t-1} + \beta \mathbb{I}(f)$，低于阈值 $\theta$ 的 feedback 被遗忘。
   - 设计动机：避免有价值的历史 feedback 被丢弃，同时通过选择性遗忘机制确保记忆中只保留有效信息，加速优化收敛。

3. **Exemplar Factory（示例工厂）**:
   - 做什么：存储、评估和检索 exemplar，推理时选择最优 exemplar 拼接到 prompt 中增强预测。
   - 核心思路：同样使用优先级分数管理，检索时综合考虑优先级和与当前问题的语义相似度 $P_e^r = \text{softmax}(\{e^{s_p(e_i) \cdot s_s^j(e_i)/\tau_e}\})$。存储时验证解题过程正确性并去重，使用后根据对预测的帮助程度更新优先级。
   - 设计动机：纯语义相似度检索不一定选出对任务最有帮助的 exemplar，需要通过实际效果反馈来优化检索策略。

### 损失函数 / 训练策略
- 使用 beam search 在候选 prompt 中选择验证集上表现最好的 $k$ 个进入下一步优化
- 相似度计算使用 BGE-M3 模型
- Task model 使用 Doubao-Pro，Prompt optimizer 使用 GPT-4o

## 实验关键数据

### 主实验
| 数据集 | 指标 | ERM | 之前SOTA | 提升 |
|--------|------|-----|----------|------|
| LIAR | F1 | 68.6 | 58.5 (ProTeGi) | +10.1 |
| BBH | F1 | 86.1 | 81.9 (CoT) | +4.2 |
| ETHOS | F1 | 98.0 | 96.5 (ProTeGi) | +1.5 |
| WebNLG | Rouge-L | 59.6 | 55.7 (ProTeGi) | +3.9 |
| GSM8K | Acc. | 93.3 | 91.7 (Promptbreeder) | +1.6 |
| WSC | Acc. | 86.0 | 84.0 (GPO) | +2.0 |

### 消融实验
| 配置 | LIAR F1 | BBH F1 | 说明 |
|------|---------|--------|------|
| Baseline (ProTeGi) | 58.5 | 73.6 | 无任何组件 |
| +Exemplar-Guided Reflection | 62.9 | 75.7 | +4.4 |
| +Reflection +Feedback Memory | 67.2 | 84.7 | +8.7 |
| +Reflection +Exemplar Factory | 66.6 | 82.6 | +8.1 |
| Full ERM | 68.6 | 86.1 | +10.1 |

### 关键发现
- Feedback Memory 贡献最大（LIAR 上 +5.7），说明历史 feedback 的复用非常重要
- Exemplar Factory 中过滤和选择性遗忘缺一不可，仅检索不过滤反而无效
- ERM 达到最佳性能所需优化步数约为 ProTeGi 的一半（LIAR：7步 vs 13步）
- Few-shot 设置下依然显著优于其他方法

## 亮点与洞察
- 将人类记忆的"遗忘曲线"概念引入 prompt 优化，用优先级分数动态管理 feedback/exemplar 的生命周期，思路可迁移到其他需要长期知识管理的场景
- Exemplar-Guided Reflection 的设计很巧妙：先生成 CoT 风格的示例解答，再用这些解答辅助 feedback 生成，形成"教学相长"的闭环

## 局限性 / 可改进方向
- 未探索人类介入优化过程的场景（当模型持续犯错时，人工提供解答可能更高效）
- 实验受限于预算，任务类型有限
- Feedback Memory 和 Exemplar Factory 的超参数（$\beta$, $\theta$, $\tau$）需要调优

## 相关工作与启发
- **vs ProTeGi**: ProTeGi 只用当前步 feedback，ERM 通过记忆机制复用历史 feedback，LIAR 上 +10.1
- **vs OPRO/GPO**: 轨迹式方法基于历史 prompt+分数优化，但无法从错误中学习具体改进方向
- **vs EvoPrompt**: 进化式方法随机变异 prompt，缺乏针对性的反馈指导

## 评分
- 新颖性: ⭐⭐⭐⭐ 记忆机制应用于 prompt 优化是有新意的组合，但各组件单独看并不算全新
- 实验充分度: ⭐⭐⭐⭐ 7个数据集、充分的消融实验和效率分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图示说明到位
- 价值: ⭐⭐⭐⭐ 实用性强，prompt 优化效率提升明显
