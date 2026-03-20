# Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve

**会议**: NeurIPS 2025  
**arXiv**: [2505.23946](https://arxiv.org/abs/2505.23946)  
**代码**: https://github.com/MITIBM-FastCoder/LessonL  
**领域**: LLM Agent  
**关键词**: 多智能体协作, 代码优化, Lesson机制, 互相学习, 性能优化

## 一句话总结
提出 LessonL 框架，使多个小 LLM 智能体通过相互学习的"课程"(lesson)对成功和失败案例进行反思，协同优化代码性能，3 个 7B-14B 模型组合达到 GPT-4o 甚至接近 o3 的代码优化效果。

## 研究背景与动机

1. **领域现状**：代码优化是软件开发关键环节，但 AI 研究中被严重忽视。现有工作要么专注代码生成，要么依赖专门的 HPC 模型。

2. **现有痛点**：
   - LLM 在细粒度任务上有互补优势（如 Qwen7B 在 geometry 任务上超过 GPT-4o 2.5 倍），但这些优势未被利用
   - 多智能体协作要么采用角色分工（planner/coder/debugger），要么独立提议聚合
   - 缺乏可解释的知识共享机制

3. **核心矛盾**：如何利用多个小 LLM 的互补优势进行代码优化，同时保持可解释和成本高效。

4. **本文要解决什么？** 设计一个多智能体学习框架，允许 agents 从彼此的成功和失败中学习。

5. **切入角度**：学生互助学习的比喻——从教科书学习 + 从同学学习。

6. **核心idea一句话**：三阶段迭代——lesson 生成（solicitation）、lesson 存储（banking）、lesson 选择（selection），让 agent 间共享可解释的优化经验。

## 方法详解

### 整体框架
LessonL 框架的核心循环：初始解生成 → Lesson 提取 → Lesson 存储 → Lesson 选择 → 使用 selected lesson 迭代改进 → 循环。T 轮迭代，每轮 n 个智能体各生成一个 lesson，选择 k 个 lesson 用于下一轮。

### 关键设计

1. **Lesson 提取（Solicitation）**:
   - 做什么：从代码修改结果中提取自然语言形式的优化经验
   - 核心思路：基于四种代码修改场景提取不同类型 lesson：
     - (a) 加速：代码更快且正确 → 正向 lesson（如"重新排序循环改进缓存局部性"）
     - (b) 减速：代码正确但更慢 → 警告 lesson
     - (c) 功能错误：不通过测试 → 错误 lesson
     - (d) 语法错误：编译失败 → 语法 lesson
   - 设计动机：不仅从成功中学习，也从失败中学习，防止重复犯错

2. **Lesson 存储与选择（Banking & Selection）**:
   - 做什么：管理 lesson 池，为下一轮选择最有用的 lesson
   - 核心思路：混合选择策略——取 top ⌈k/2⌉ 高 speedup 的 lesson（优选）+ top ⌊k/2⌋ 高相关性（CodeBERT 余弦相似度）的 lesson
   - 设计动机：纯高 speedup 选择可能导致轨迹固化，混合策略引入多样性

3. **有效性动态调整**:
   - 做什么：随时间调整 lesson 的权重
   - 核心思路：定义调整因子 $f$：当 lesson 在后续轮次应用时，实际 speedup 与原始 speedup 比较，累积更新 $f = c/n$
   - 设计动机：lesson 创建时有效不代表一直有效，需要自适应降级在实际中表现不佳的 lesson

## 实验关键数据

### 主实验

| 方法 | ParEval Serial 正确率 | ParEval >2x 加速比例 | ParEval 串行加速 | ParEval 并行加速 |
|------|---------------------|---------------------|-----------------|-----------------|
| Qwen14B 基线 | 0.67 | 0.14 | 1.60x | 2.28x |
| GPT-4o mini | 0.77 | 0.14 | 1.57x | 2.72x |
| GPT-4o | 0.80 | 0.16 | 1.72x | 2.93x |
| OpenAI o3 | 0.77 | **0.23** | **2.21x** | **3.55x** |
| MapCoder (Qwen14B×3) | 0.88 | 0.15 | 1.85x | 3.43x |
| **LessonL (Qwen×3)** | **0.91** | **0.21** | **2.16x** | **3.46x** |

### 消融实验

| 方案 | ParEval S 正确率 | >2x 比例 | 串行加速 | 并行加速 |
|------|-----------------|---------|---------|---------|
| 完整 LessonL | 0.91 | 0.21 | **2.16x** | **3.46x** |
| 仅高 speedup lesson | 0.92 | 0.21 | 2.08x | 3.20x |
| 仅高相关性 lesson | 0.91 | 0.18 | 1.96x | 3.40x |
| 无 speedup 调整 | 0.92 | 0.21 | 2.05x | 3.28x |
| 随机选择 lesson | 0.91 | 0.19 | 2.03x | 3.47x |
| 无 lesson | 0.89 | 0.20 | 1.95x | 3.01x |

### 关键发现
- **小模型的超能力**：3 个 7B-14B 模型的 LessonL 达到 GPT-4o 甚至接近 o3 级别，并行加速 3.46x vs o3 的 3.55x
- 混合选择策略最优——纯高 speedup 或纯高相关性都不如混合
- 无 lesson 基线性能显著下降（并行 3.01x vs 3.46x），验证了 lesson 机制的必要性
- 3 个 agents 最优，6 个时改进饱和甚至略差
- 成本优势明显：相比 MapCoder 和 MoA，LessonL 在 Pareto 最优边界

## 亮点与洞察
- **可解释的知识共享**：lesson 是自然语言可理解的优化策略（如"使用 OpenMP 并行化"），比黑箱 embedding 融合更有教育价值
- **自适应 lesson 权重**：通过 f 因子动态调整，避免了固定权重导致的盲目学习
- **从失败中学习**：负 lesson（错误和减速案例）防止了重复错误，这在 MapCoder/MoA 中无法实现
- **细粒度互补性发现**：框架自动发现各模型的强项，无需预知

## 局限性 / 可改进方向
- Lesson 提取和迭代增加延迟，对实时应用不利
- 仅在函数级代码上验证，难以扩展到 repo 级 SWE 任务
- 多轮后 lesson 边际收益显著下降
- 自动提取的 lesson 质量参差不齐，可能包含 LLM 幻觉

## 相关工作与启发
- **vs MapCoder/MoA**：MapCoder 采用独立提议聚合，MoA 采用层级聚合，LessonL 引入显式的 lesson 机制填补知识共享空白
- **vs Self-Refine/Reflexion**：单 agent 自我反思，LessonL 推广到多 agent 间的相互学习
- **vs PIE/HPC-Coder**：专门微调或检索方法，LessonL 是轻量级的即时知识注入

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ lesson 概念新颖，设计优雅简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 多 benchmark、6 大消融项、成本分析、案例研究
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑递进清晰，案例生动
- 价值: ⭐⭐⭐⭐⭐ 对多智能体学习的新视角，成本效益优势明显
