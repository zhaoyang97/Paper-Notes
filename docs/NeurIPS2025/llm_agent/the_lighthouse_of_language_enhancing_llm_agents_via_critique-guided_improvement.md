# The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement

**会议**: NeurIPS 2025  
**arXiv**: [2503.16024](https://arxiv.org/abs/2503.16024)  
**代码**: https://github.com/rhyang2021/CGI  
**领域**: Agent  
**关键词**: 自然语言反馈, Actor-Critic, 迭代精炼, 交互式环境, Agent训练

## 一句话总结
提出 CGI（Critique-Guided Improvement）双角色框架，训练专门的 Critic 模型为 Actor Agent 提供结构化自然语言反馈（判别+修正建议），并通过迭代动作精炼让 Actor 学会利用这些反馈，在 WebShop/ScienceWorld/TextCraft 三个环境中平均得分 74.20%，超越 GPT-4o（45.46%）和 Iterative SFT（58.21%）。

## 研究背景与动机

1. **领域现状**：LLM Agent 在交互式环境中需要迭代探索和改进。目前的反馈机制分两类：数值反馈（奖励模型/验证器打分）和自然语言反馈（自我纠正/LLM-as-judge）。
2. **现有痛点**：
   - **数值反馈信息量有限**：Best-of-N 等方法只能从候选动作中筛选最优，无法提供"为什么错"和"怎么改"的上下文信息
   - **自我反馈质量差**：Self-refinement 严重依赖模型自身能力，容易产生幻觉和低质量反馈，在复杂任务中甚至导致性能下降
   - **语言反馈难以利用**：即使有高质量的自然语言反馈，Agent 也常常无法正确解读和落实建议——经过 SFT 微调后的模型这个问题更为突出
3. **核心矛盾**：自然语言反馈比数值信号信息更丰富，但"生成高质量反馈"和"有效利用反馈"这两个问题同时存在且相互耦合
4. **本文要解决什么**：① 如何训练一个专门的 Critic 生成高质量的结构化语言反馈？② 如何让 Actor 在迭代过程中真正学会利用这些反馈来改进行为？
5. **切入角度**：将问题拆分为 Actor-Critic 双角色，分别训练——Critic 从 GPT-4 蒸馏高质量反馈，Actor 通过迭代 SFT 学习在反馈指导下精炼动作
6. **核心 idea 一句话**：训练专门的 Critic 生成"判别+修正"结构化反馈，再通过迭代精炼让 Actor 学会将语言反馈转化为更好的动作。

## 方法详解

### 整体框架
CGI 是一个两阶段双角色框架：**Critique Generation**（训练 Critic）+ **Action Refinement**（迭代训练 Actor）。在推理时，每一步 Actor 先生成 $M$ 个候选动作，Critic 对每个候选给出结构化评价和修改建议，Actor 根据这些反馈生成最终的精炼动作并执行。整个过程在 POMDP 框架下形式化。

### 关键设计

1. **结构化批评生成（Structured Critique）**:
   - 做什么：Critic 对每个候选动作生成包含"判别"和"修正"两部分的结构化反馈
   - 核心思路：判别部分从三个维度评估——**贡献度**（动作是否推进任务）、**可行性**（动作是否合法）、**效率**（动作是否最优路径）。修正部分给出整体评级（Excellent/Good/Neutral/Poor/Very Poor）和具体的改进建议
   - 设计动机：非结构化的自由反馈太模糊，agent 难以执行；三维度评估覆盖了"做什么"、"能不能做"、"值不值得做"三个关键问题，确保反馈的全面性和可操作性

2. **Critic 模型训练**:
   - 做什么：用 GPT-4o 作为专家 Critic 生成高质量反馈数据，然后蒸馏到小模型
   - 核心思路：给定专家轨迹 $\tau^{exp}$ 作为参考，GPT-4o 评估候选动作与最优动作的对齐程度，生成结构化 critique。只收集成功轨迹（$\mathcal{R}(\tau')=1$）的反馈，用标准语言建模损失 $\mathcal{L}_{critic}(\phi) = \mathbb{E}[\log \pi_\phi(c_t | \tau'_t, a_t, e)]$ 训练 Critic 模型
   - 设计动机：仅 8B 的 Critic 模型就能超越 GPT-4o 作为 critic 的表现（平均 61.44% vs 32.28%），说明专门化训练远优于通用 LLM

3. **迭代动作精炼（Iterative Action Refinement）**:
   - 做什么：通过多轮 exploration + learning 让 Actor 学会利用 Critique
   - 核心思路：每轮迭代中，Actor 在 Critic 指导下与环境交互，收集两类数据——$\mathcal{D}_{correct}$（正确轨迹，增强推理能力）和 $\mathcal{D}_{refine}$（critique-action 对，增强反馈利用能力）。混合 $\mathcal{D}_{general}$（ShareGPT 通用数据）防止过拟合。每轮从原始基模型开始训练而非上一轮模型
   - 设计动机：直接 SFT 后的模型反而不善于利用外部反馈（"policy misalignment"问题），迭代精炼让 Actor 的策略分布与 Critic 反馈保持对齐

### 损失函数 / 训练策略
Actor 损失函数结合三类数据：$\mathcal{L}_{actor}(\theta) = \beta[\mathbb{E}_{\mathcal{D}_{train}}[\log \pi_\theta(\tau|x,e)] + \mathbb{E}_{\mathcal{D}_{refine}}[\log \pi_\theta(a'_t|\tau'_t,c_t,e)]] + (1-\beta)\mathbb{E}_{\mathcal{D}_{general}}[\log \pi_\theta(y|x)]$。Backbone 为 Llama-3-8B-Instruct，每轮从基模型重新训练以避免过拟合。推理时默认 $M=5$ 个候选动作。

## 实验关键数据

### 主实验

| 方法 | WebShop | ScienceWorld | TextCraft | 平均 |
|------|---------|-------------|-----------|------|
| GPT-4o | 25.48 | 46.91 | 64.00 | 45.46 |
| Llama-3-70B-Instruct | 8.35 | 49.20 | 2.00 | 19.85 |
| AgentLM-70B | 49.50 | 10.68 | 4.00 | 21.39 |
| Iterative SFT (8B) | 78.21 | 41.42 | 55.00 | 58.21 |
| **CGI (8B, Ours)** | **76.17** | **78.43** | **68.00** | **74.20** |

CGI 比 Iterative SFT 高 +15.99%，比 GPT-4o 高 +28.74%。

### 消融实验

| 配置 | WebShop | ScienceWorld | TextCraft | 平均 |
|------|---------|-------------|-----------|------|
| CGI #Iter1 (Full) | 73.22 | 66.27 | 66.00 | 68.50 |
| w/o $\mathcal{D}_{refine}$ | 74.33 | 39.33 | 37.00 | 50.22 |
| w/o $\mathcal{D}_{correct}$ | 66.25 | 60.93 | 52.00 | 59.72 |
| w/o $\mathcal{D}_{general}$ | 67.88 | 67.23 | 62.00 | 65.70 |

### 关键发现
- **$\mathcal{D}_{refine}$ 最关键**：去掉 critique-action 对后平均下降 18.28%，在长轨迹任务上（ScienceWorld -26.94%）影响最大
- **语言反馈远优于数值信号**：仅 8B Critic 模型指导 8B Actor 达到 61.44%，而 DGAP 数值判别器只有 23.64%
- **8B Critic 超越 GPT-4**：在三个环境中平均 61.44% vs GPT-4o 的 32.28%，证明专门化训练的小模型可以远超通用大模型
- **SFT 后的模型反而不善于利用反馈**：SFT 后的 Llama-3-8B 在 Critic 指导下只有 55.94%（ScienceWorld），而原版在 Critic 指导下达 68.51%
- **CGI 在长轨迹任务上优势最大**：难任务在 3 轮迭代后提升 +28.75%，而简单任务首轮就能适应

## 亮点与洞察
- **"反馈利用"比"反馈质量"更重要**：这是一个反直觉的发现——即使反馈质量很高，如果 Agent 不会用，效果依然有限。CGI 通过迭代精炼解决这一根本问题
- **每轮从基模型重训**：避免了迭代 SFT 中常见的分布漂移和过拟合问题，这个 trick 简单但有效
- **Critique 主要在轨迹早期起作用**：Revision Ratio 在 stage 1 最高，说明好的反馈帮助 Agent 在早期就进入正确的探索方向，避免无效搜索

## 局限性 / 可改进方向
- **依赖 GPT-4 生成 Critic 训练数据**：数据蒸馏成本较高，且受限于 GPT-4 的 critique 质量天花板
- **推理开销大**：每步生成 $M=5$ 个候选动作 + Critic 评估，推理成本是正常的 $\sim$6倍
- **仅在模拟环境验证**：WebShop/ScienceWorld/TextCraft 都是相对简单的文本交互环境，未在真实软件工程、网页操作等场景验证
- **Critic 和 Actor 共享相同 backbone**：未探索不同大小 Critic 和 Actor 的最优组合
- **通用数据混合比例 $\beta$ 未详细分析**

## 相关工作与启发
- **vs Reflexion**: Reflexion 用自我总结作为下轮反馈，但容易陷入局部最优，在三个环境中几乎没有提升。CGI 用外部 Critic 提供更客观的反馈
- **vs Best-of-N / DGAP**: 数值信号只能做筛选（"哪个好"），不能做修正（"怎么改好"），CGI 的语言反馈信息密度高得多
- **vs Self-Critique**: 自我批评在三个环境中平均只有 10.19%（8B模型），甚至比 no-critique 还差（12.65%），证明小模型自我反馈不可靠

## 评分
- 新颖性: ⭐⭐⭐⭐ 双角色框架和迭代精炼的结合新颖，"SFT后模型不会用反馈"这一观察有价值
- 实验充分度: ⭐⭐⭐⭐⭐ 三个环境、多种baseline、详细消融、轨迹分析、候选数影响等分析全面
- 写作质量: ⭐⭐⭐⭐ 结构清晰，三个 Finding 的组织方式很好，但部分符号定义分散
- 价值: ⭐⭐⭐⭐ Actor-Critic 语言反馈范式对 Agent 训练有实用价值，8B超越GPT-4的结论很实用

## 实验关键数据
