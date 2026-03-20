# Agent-SAMA: State-Aware Mobile Assistant

**会议**: AAAI 2026  
**arXiv**: [2505.23596v3](https://arxiv.org/abs/2505.23596v3)  
**代码**: [Zenodo](https://doi.org/10.5281/zenodo.15430187)  
**领域**: LLM Agent / 移动端GUI自动化  
**关键词**: GUI Agent, 有限状态机, 多Agent协作, 错误恢复, 移动端任务自动化  

## 一句话总结
提出Agent-SAMA，首次将有限状态机（FSM）引入移动端GUI Agent，将UI屏幕建模为状态、用户操作建模为转移，通过四个专门化Agent协作实现状态感知的任务规划、执行验证和错误恢复，在跨App基准上成功率提升最高12%、恢复率提升13.8%。

## 背景与动机
移动端GUI Agent利用MLLM理解UI截图并执行点击/滑动等操作，已有AppAgent、Mobile-Agent系列等工作。但现有Agent本质上是**反应式（reactive）**的——仅根据当前屏幕决策下一步操作，缺乏对App导航流程的结构化表示。这就像游客逐条街道走，知道去过哪里但对整个路线没有全局理解。这导致三个关键缺陷：1）无法理解执行上下文（当前处于任务的哪个阶段）；2）无法检测操作结果是否符合预期；3）错误恢复无结构化支撑，容易陷入重复失败循环。

## 核心问题
如何为GUI Agent提供结构化的App导航表示，使其能够追踪执行进度、预判操作结果、在出错时精准回退到稳定状态？这在跨App长步骤任务中尤为重要，因为操作链条长、出错概率高，反应式Agent难以胜任。

## 方法详解
### 整体框架
Agent-SAMA是一个多Agent框架，包含四个阶段：规划（Planning）→ 执行（Execution）→ 验证与恢复（Verification & Recovery）→ 知识保留（Knowledge Retention）。核心创新在于用FSM $\mathcal{M} = (S, A, T, s_0, G)$ 建模App交互：UI屏幕为状态$S$，用户操作为动作$A$，屏幕间的转换为转移函数$T$。FSM在执行过程中实时增量构建。

### 关键设计
1. **Planner Agent + LLM-as-Judge 规划**：将高层任务分解为子任务序列$\pi = [(g_1, r_1), ..., (g_k, r_k)]$，每个子任务附带理由说明。关键创新是生成5个候选计划，然后用LLM-as-judges根据目标相关性、执行效率、鲁棒性等维度评估选出最佳方案，避免单路径规划的次优问题。
2. **State Agent + FSM 实时构建**：执行阶段的核心模块。Screen Parser提取UI元素坐标和描述，State Agent将每个屏幕映射为FSM节点，包含三个要素：当前状态描述$d_i$、下一状态预测$d_{i+1}$、前置/后置条件(pre/post-condition)。为避免状态爆炸，引入**State Beacon**机制——为每个状态生成简洁语义标签（如"Homepage of Walmart"），新状态先匹配已有beacon，匹配则复用节点。跨App任务中每个App维护独立FSM。
3. **Reflection Agent 错误恢复**：利用FSM进行结构化验证与恢复。比较FSM预测的转移（含后置条件）与实际屏幕，输出三种判定：Success/NoChange/Fail。失败时利用FSM找到之前验证过的稳定状态$s_j$，生成恢复计划回退重试。若连续恢复失败（$n=2$次），升级到Planner重新规划，防止陷入恢复死循环。
4. **Mentor Agent 知识保留**：任务结束后提取可复用知识$K$（动作序列、指导线索、构建的FSM），存入长期记忆。新任务开始时检索相关知识作为上下文（如Walmart的购物FSM可迁移给Amazon），提升规划效率和鲁棒性。

### 损失函数 / 训练策略
无需训练，全部通过prompt engineering驱动MLLM（GPT-4o），温度设为0以减少变异。Screen Parser使用DBNet做OCR、GroundingDINO做图标定位、Qwen-VL-Plus生成图标描述。

## 实验关键数据
| 数据集 | 指标 | Agent-SAMA | Mobile-Agent-E+Evo | 提升 |
|--------|------|-----------|-------------|------|
| Mobile-Eval-E | Success Rate | 84.0% | 72.0% | +12.0% |
| Mobile-Eval-E | Recovery Success | 71.88% | 67.34% | +4.53% |
| Mobile-Eval-E | Action Accuracy | 83.24% | 76.65% | +6.59% |
| Mobile-Eval-E | Satisfaction Score | 86.15% | 78.97% | +7.18% |
| SPA-Bench | Success Rate | 80.0% | 75.0% | +5.0% |
| SPA-Bench | Recovery Success | 66.67% | 52.86% | +13.81% |
| AndroidWorld | Success Rate | 63.7% | 53.4% | +10.3% |

### 消融实验要点
- **规划模块影响最大**：移除Planner后，Mobile-Eval-E的SR从84%骤降到52%，SPA-Bench从80%降到45%
- **多计划选择有效**：单路径规划 vs 5候选+Judge选择，SR差异约12%
- **Pre/Post条件**：移除后SR从84%降到72%（M-Eval-E），对验证和恢复质量影响明显
- **Mentor知识保留**：移除后SR从84%降到68%（M-Eval-E），说明跨任务知识迁移很重要
- 四个组件互补增强，缺一不可

## 亮点
- **首次将FSM引入GUI Agent**：将移动App交互建模为状态机是自然且有效的抽象，为agent提供了结构化记忆和推理支撑
- **State Beacon去重机制**：简洁实用，有效缓解状态爆炸问题
- **错误恢复流程设计精巧**：FSM提供稳定锚点、分级恢复策略（先局部回退，再全局重规划），实际减少了执行错误次数（32 vs 49 on M-Eval-E）
- **Agent-SAMA在弱MLLM下也有竞争力**：使用Claude 3.5时仍优于GPT-4o版本的baseline，说明框架本身的增益独立于底座模型
- **方法是模型无关的**：FSM层可作为轻量级记忆层插入任何现有GUI Agent

## 局限性 / 可改进方向
- **State Beacon依赖LLM文本匹配**：可能存在误匹配或漏匹配，可考虑替换为视觉-语义嵌入做高效匹配
- **扁平FSM在超长链任务中可能状态爆炸**：跨5+个App、20+步操作时缺乏层次化抽象
- **仅评估了三个基准**：缺少对动态内容（广告弹窗等）、外部中断等场景的评估
- **不同运行结果有波动**：与原论文报告的baseline结果有差异，虽做了5次重复取平均但仍有不确定性
- **知识保留中的FSM跨App迁移**的有效性未单独量化

## 与相关工作的对比
- **vs Mobile-Agent-E+Evo**：同样的多Agent架构和长期记忆，但Agent-SAMA增加了FSM结构化表示，使得恢复更精准（恢复率高4.5%~13.8%）、执行错误更少
- **vs GUI-Xplore**：两者都用图建模App导航，但GUI-Xplore从离线视频构建静态图用于推理评估；Agent-SAMA在线实时构建FSM并用于执行决策和恢复，是"在线+实用"的版本
- **vs AgentS2/V-Droid**：这些是AndroidWorld上的强baseline（54.3%/59.5%），Agent-SAMA以63.7%超越，且Agent-SAMA更侧重跨App长链任务

## 启发与关联
- **与已有idea的直接关联**：ideas/llm_nlp/20260317_hierarchical_fsm_gui_agent.md 提出了层次化FSM + 嵌入式状态匹配的思路，正是针对本文的两个核心瓶颈（扁平FSM的状态爆炸 + LLM文本匹配的开销）。本文的实验结果（state beacon的有效性、跨App独立FSM设计）进一步验证了层次化改进的可行性
- **FSM作为通用Agent记忆层**：本文论证了FSM可以作为"轻量级、模型无关的记忆层"——这个思路可以推广到Web Agent、桌面端Agent甚至具身Agent中
- **LLM-as-Judge用于计划选择**：生成多个候选计划再用Judge打分选择，比单次生成更鲁棒，这个设计模式在其他Agent场景（如代码生成、科研自动化）中也值得借鉴
- **Pre/Post Condition的形式化思路**：来自传统软件工程的契约式设计（Design by Contract），在LLM Agent中重新焕发价值

## 评分
- 新颖性: ⭐⭐⭐⭐ （FSM建模App交互的思路很自然，真正创新在于工程化落地和多Agent协作设计）
- 实验充分度: ⭐⭐⭐⭐ （三个基准、消融实验、多MLLM对比，但跨App知识迁移效果缺少定量分析）
- 写作质量: ⭐⭐⭐⭐ （结构清晰、图示直观，附录含完整prompt和案例，但部分表格排版略乱）
- 价值: ⭐⭐⭐⭐ （FSM作为Agent记忆层的思路具有通用性，代码已开源，可作为后续研究的起点）
