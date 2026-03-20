# Co-EPG: A Framework for Co-Evolution of Planning and Grounding in Autonomous GUI Agents

**会议**: AAAI 2026  
**arXiv**: [2511.10705](https://arxiv.org/abs/2511.10705)  
**代码**: 未开源  
**领域**: Agent / LLM / GUI自动化  
**关键词**: GUI Agent, Planning-Grounding协同进化, GRPO, 自迭代训练, 奖励集成  

## 一句话总结
提出Co-EPG框架，将GUI Agent解耦为Planning和Grounding两个模型，通过GRPO协同训练和基于置信度的动态奖励集成机制（C-DREM）建立正反馈循环，使两个模型自迭代协同进化，仅用基准数据集（无需外部数据）即在Multimodal-Mind2Web（58.4%）和AndroidControl（83.1%）上达到SOTA。

## 背景与动机
GUI任务自动化是AI的重要前沿方向。当前GUI Agent需要两个核心能力：**规划（Planning）**——根据屏幕状态决定做什么操作；**定位（Grounding）**——在界面上找到对应元素的精确位置。

现有方法存在两大范式：
1. **端到端单模型**：直接训练一个VLM同时做规划和定位。但在多样化GUI环境中泛化能力差，感知和交互都有局限。
2. **模块化多模型**：将规划和定位解耦为独立模型或采用多Agent协作。但现有的协作架构存在两个根本问题：
   - **协同不足**：各模型独立优化，没有利用规划与定位之间的互相促进关系
   - **数据低效**：过度依赖大规模合成数据生成，对已有数据的利用不充分

因此，需要一种能让Planning和Grounding模型互相促进、持续提升的新范式。

## 核心问题
如何在解耦的Planning-Grounding架构中实现两个模型的**协同进化**？具体来说：
1. 规划模型生成的plan质量如何用定位模型来评估和引导？
2. 规划模型的改进如何反过来产出更高质量的训练数据增强定位模型？
3. 如何在不依赖外部数据的前提下，通过自迭代实现持续提升？

这个问题的关键难点在于：plan的好坏无法直接评估（不像坐标可以直接和gt比），需要通过定位模型的执行结果来间接衡量。

## 方法详解

### 整体框架
Co-EPG是一个自迭代训练框架，核心是建立Planning和Grounding之间的**正反馈循环**：

**输入**：GUI截图 + 任务描述 + 历史操作  
**输出**：完整的操作动作（坐标 + 动作类型 + 动作值）

整体流程分为两个交替阶段：
1. **迭代训练（Iterative Training）**：先用SFT初始化两个模型，然后用C-DREM驱动的GRPO协同训练优化规划模型，再用优化后的规划模型生成的高质量数据进一步SFT定位模型
2. **数据增强（Data Enhancement）**：利用更新后的模型扩充和提升训练数据的质量与多样性

通过多轮迭代（论文做了3轮），两个模型不断螺旋上升。

### 关键设计

1. **P-G双模型架构**：将决策解耦为两步：
   - 规划模型 $\pi$：输入当前观察 $o_t$、任务描述 $Q$、历史 $h_t$，输出文本plan $p_t$、动作类型 $a_t^{type}$、动作值 $a_t^{value}$
   - 定位模型 $\phi$：输入截图 $o_t^{vision}$ 和plan $p_t$，输出目标元素坐标 $a_t^{coor}$
   
   Plan作为两个模型之间的**交互媒介**——它既是规划模型的输出，又是定位模型的输入。这是协同进化的关键桥梁。

2. **C-DREM（基于置信度的动态奖励集成机制）**：解决GRPO训练中"单一奖励模型有偏差"的问题。核心思路：
   - 使用多个定位模型（包含两个大型开源VLM：Qwen2.5-VL-72B和32B，以及迭代训练的定位模型 $\phi_k$）共同评估plan质量
   - 每个模型的权重由两部分决定：**静态先验** $\sigma_j$（自训练模型设更高权重）和**动态置信度** $c_j$（通过预测坐标token的对数似然均值计算）
   - 权重通过softmax归一化：$w_j = \frac{\exp(\sigma_j \cdot c_j)}{\sum_n \exp(\sigma_n \cdot c_n)}$
   - Plan奖励 $r_{plan}$：若定位模型预测坐标落在目标bbox内则为1，否则为0；最终加权求和

3. **自增强数据演化**：
   - 初始化（k=0）：用开源VLM池（Planner Π和Verifier Φ）生成plan并验证，保留成功的构成 $D_0$
   - 后续迭代（k≥1）：将新训练的 $\pi_k'$ 加入Planner池增加多样性，$\phi_k$ 加入Verifier池提升验证可靠性
   - 只保留最新两代模型以平衡效率

### 损失函数 / 训练策略

**奖励设计**（三部分）：
- Plan奖励 $r_{plan}$：定位模型预测坐标是否在目标bbox内，通过C-DREM加权集成
- 动作类型奖励 $r_{type}$：精确匹配gt，正确为1错误为0
- 动作值奖励 $r_{value}$：预测值与gt的F1分数 > 0.5则为1，否则为0

**最终奖励**：若 $r_{type}=0$ 或 $r_{value}=0$，则 $r_i=0$；否则 $r_i = r_{plan}$。这意味着动作类型和值是"门控"条件，plan奖励才是核心优化信号。

**GRPO训练**：每组生成 $G=7$ 个rollout，通过组内归一化计算advantage：$A_i = \frac{r_i - \text{mean}(\{r_1,...,r_G\})}{\text{std}(\{r_1,...,r_G\})}$

**训练细节**：
- 骨干模型：Qwen2.5-VL-3B/7B-Instruct
- SFT：batch size 96, lr $1 \times 10^{-6}$, 3 epochs, 8 GPUs
- GRPO：batch size 294, lr $5 \times 10^{-7}$, temperature 0.9, KL系数 $\beta=0.01$, clip $\epsilon=0.2$, 7 GPUs, 约48小时
- C-DREM静态先验权重比：Qwen2.5-VL-72B : 32B : $\phi_k$ = 1:1:2

## 实验关键数据

### Multimodal-Mind2Web（Web任务）

| 方法 | 模型规模 | Cross-Task Step SR | Cross-Website Step SR | Cross-Domain Step SR | Avg Step SR |
|------|----------|------|------|------|------|
| GPT-4 + Choice | - | 40.2 | 32.4 | 36.8 | 36.5 |
| GPT-4V + OmniParser | - | 39.4 | 36.5 | 42.0 | 39.3 |
| Explorer-7B | 7B | 53.2 | 56.7 | 53.0 | 54.3 |
| AgentTrek-7B | 7B | 55.7 | 51.4 | 52.6 | 53.2 |
| AGUVIS-7B | 7B | 60.4 | 54.6 | 56.6 | 57.2 |
| **Co-EPG-Web-3B** | 3B | 53.1 | 51.1 | 50.0 | **51.4** |
| **Co-EPG-Web-7B** | 7B | 61.9 | 58.1 | 55.3 | **58.4** |

### AndroidControl（移动任务）

| 方法 | High Step Acc | Low Step Acc | Avg Step Acc |
|------|------|------|------|
| UI-TARS-2B | 68.9 | 89.3 | 79.1 |
| UI-TARS-7B | 72.5 | 90.8 | 81.7 |
| InfiGUI-R1-3B | 73.2 | 90.0 | 81.6 |
| **Co-EPG-Mob-3B** | 73.4 | 90.2 | **81.8** |
| **Co-EPG-Mob-7B** | 74.2 | 92.0 | **83.1** |

### OmniACT（桌面+Web跨平台）

| 方法 | Avg Acc |
|------|------|
| GPT-4o + UGround-V1-7B | 34.0 |
| **Co-EPG-Des-7B-M2** | **53.2** (+19.2%) |

### 消融实验要点

**P-G解耦 vs 端到端**：
| 架构 | Avg Step SR |
|------|------|
| End-to-End | 50.1 |
| P-G Dual-Model | 53.5 (+3.4%) |

**C-DREM消融**：
| 变体 | Avg Step SR |
|------|------|
| w/o C-DREM（单一定位模型做奖励） | 56.50 |
| w/o Confidence & Prior Weights（均匀加权） | 57.01 (+0.51) |
| w/o Confidence Weights（仅静态先验） | 57.67 (+1.17) |
| 完整C-DREM | **58.41** (+1.91) |

**迭代效果**（Co-EPG-Web-7B Avg Step SR）：
| 阶段 | w/o GRPO | w/ GRPO |
|------|------|------|
| Iteration 1 | 52.6 | 53.5 |
| Iteration 2 | 54.5 | 55.0 |
| Iteration 3 | 56.5 | **58.4** |

每轮迭代都有稳定提升，GRPO始终优于纯SFT，说明数据迭代和GRPO协同训练是双驱动力。

**数据效率**：Co-EPG-Web-7B仅使用**6862个标注step**（AGUVIS的2.42%即283500），即超越AGUVIS-7B的性能。

## 亮点
- **正反馈循环设计非常优雅**：Plan作为Planning和Grounding的桥梁，定位模型的执行结果自然地成为规划模型的奖励信号，不需要人工设计复杂的reward函数
- **C-DREM解决了RL训练中奖励模型单点失效的问题**：用多个定位模型集成 + 置信度加权，比单一奖励模型更稳定，收敛更快。置信度用token对数似然计算，计算开销很小
- **数据效率极高**：仅用基准数据集的2.42%数据就超越了依赖大规模合成数据的AGUVIS，验证了"深度挖掘数据价值"比"粗暴堆数据"更有效
- **跨平台泛化**：在Web（Mind2Web）、Mobile（AndroidControl）、Desktop（OmniACT）三类环境都有效

## 局限性 / 可改进方向
- **仅做了step-level评估**：论文只评估单步动作的准确率，没有做完整任务的端到端成功率评估（trajectory-level），而实际应用中需要连续多步都正确
- **定位模型仍需大模型做奖励**：C-DREM依赖Qwen2.5-VL-72B和32B作为辅助奖励源，GRPO训练的计算开销很大（7 GPUs × 48小时）
- **迭代收益递减**：从3轮迭代的数据看，提升幅度在逐渐减小（Iter1→2 vs Iter2→3），论文也没有探索更多轮次是否会饱和
- **Plan的语义质量没有直接度量**：plan的好坏完全通过定位结果间接评估，但一个好的plan被差的定位模型执行失败也会得到负奖励，可能引入噪声
- **仅在离线数据上验证**：没有在线环境交互的实验，不知道在真实GUI环境中连续执行的效果
- **对Qwen2.5-VL的依赖较强**：骨干和辅助奖励模型都基于Qwen系列，在其他VLM上的效果未知

## 与相关工作的对比
- **vs AGUVIS**（ICLR 2025）：AGUVIS采用两阶段（先grounding预训练再planning微调）的端到端单模型方案，需要大量合成数据（283K steps）。Co-EPG用解耦+协同进化的方式，数据量仅为其2.42%就超越它。本质区别是Co-EPG让两个模型互相提升，而非单向的先后训练。
- **vs WebRL / WebEvolver**：这些方法也用RL做self-evolution，但都是端到端单模型设计，没有利用Planning-Grounding的解耦特性。Co-EPG的贡献在于将解耦架构与自进化训练结合。
- **vs Agent-SAMA**（AAAI 2026）：Agent-SAMA用状态感知的FSM做GUI导航决策但不涉及训练优化，更偏工程设计。Co-EPG聚焦训练范式创新。

## 启发与关联
- 与ideas中的 [Hierarchical FSM GUI Agent](../../../ideas/llm_nlp/20260317_hierarchical_fsm_gui_agent.md) 想法相关：Co-EPG的P-G解耦思路可以和层次化FSM结合——上层FSM做任务级规划，下层用Co-EPG的方式训练每个App的Planning-Grounding模块
- **协同进化的思路可迁移**：不仅限于GUI Agent，任何可以分解为"策略+执行"的任务（如机器人操作中的高层规划+低层控制）都可以借鉴这种正反馈循环
- **C-DREM的思路有通用价值**：用多个模型的置信度加权做奖励集成，可以推广到其他RL-from-AI-feedback的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ 将Planning-Grounding解耦与GRPO协同训练结合是新的，C-DREM的置信度加权集成也有巧思，但每个单独组件不算全新
- 实验充分度: ⭐⭐⭐⭐ 在Web/Mobile/Desktop三个benchmark上验证，消融实验覆盖了主要组件，但缺少trajectory-level评估和在线交互实验
- 写作质量: ⭐⭐⭐⭐ 整体逻辑清晰，框架图易懂，公式推导完整。但Section 4与Table 1的排版位置有些混乱
- 实用价值: ⭐⭐⭐⭐ 数据效率高是很大的优点，但GRPO训练需要多个大模型做奖励评估，部署门槛较高
