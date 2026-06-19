---
title: >-
  [论文解读] R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization
description: >-
  [NeurIPS 2025][多智能体][多智能体框架] 提出 R&D-Agent(Q)，一个数据驱动的多智能体框架，通过五个协作模块（Specification、Synthesis、Implementation、Validation、Analysis）自动化量化策略的因子挖掘与模型创新联合优化，在真实股票市场上以不到 $10 的成本实现约 2× 于传统因子库的年化收益。
tags:
  - "NeurIPS 2025"
  - "多智能体"
  - "多智能体框架"
  - "量化因子挖掘"
  - "模型优化"
  - "数据驱动"
  - "自动化研发"
---

# R&D-Agent-Quant: A Multi-Agent Framework for Data-Centric Factors and Model Joint Optimization

**会议**: NeurIPS 2025  
**arXiv**: [2505.15155](https://arxiv.org/abs/2505.15155)  
**代码**: [microsoft/RD-Agent](https://github.com/microsoft/RD-Agent)  
**领域**: LLM Agent / 量化金融  
**关键词**: 多智能体框架, 量化因子挖掘, 模型优化, 数据驱动, 自动化研发

## 一句话总结

提出 R&D-Agent(Q)，一个数据驱动的多智能体框架，通过五个协作模块（Specification、Synthesis、Implementation、Validation、Analysis）自动化量化策略的因子挖掘与模型创新联合优化，在真实股票市场上以不到 $10 的成本实现约 2× 于传统因子库的年化收益。

## 研究背景与动机

### 1. 领域现状

金融市场是一个高维、非线性、非平稳的动态系统，资产收益呈现厚尾分布、时变波动率和复杂的截面依赖关系。量化投资正从经验驱动向数据驱动范式转型，核心流程包括：数据处理 → 因子挖掘 → 模型训练 → 回测评估。微软开源项目 Qlib 已简化了数据处理和回测环节，使研究重心转向因子挖掘和模型创新这两大核心。

### 2. 现有痛点

- **自动化程度低**：当前工作流依赖大量人工干预（假设生成、编码调参），迭代缓慢，半自动系统无法满足快速市场的响应需求
- **可解释性差**：现有 LLM 代理直接从语言交互生成交易信号，缺乏显式的因子构造和透明模型逻辑，容易产生幻觉，难以在实盘交易中落地
- **优化割裂**：数据处理、因子挖掘、模型训练、评估各环节缺乏系统性任务分解和代理级协调，跨阶段反馈受限

### 3. 核心矛盾

因子挖掘和模型创新是量化研究的两大关键，但它们互相依赖——好的因子需要好的模型来验证，好的模型也需要好的因子作为输入。现有方法要么只优化因子，要么只优化模型，缺乏联合优化的自动化框架。

### 4. 本文目标

设计一个端到端自动化的多智能体框架，能够自主地进行因子挖掘与模型创新的联合迭代优化，形成闭环的"假设→实现→验证→反馈"循环。

### 5. 切入角度

将量化研发流程分解为 5 个 LLM 驱动的功能单元，模拟人类量化研究员的试错过程，每个单元有明确的输入输出接口，通过持续存储实现知识积累。

### 6. 核心 idea

用五个协同的 LLM Agent 模块形成闭环 R&D 循环，通过知识森林驱动假设生成、Co-STEER 代码生成代理实现编码、多臂老虎机调度器自适应选择优化方向，完成因子-模型联合优化。

## 方法详解

### 整体框架

R&D-Agent(Q) 将量化研发分解为五个功能模块，分属 Research 和 Development 两个阶段：

1. **Research 阶段**：Specification Unit（任务定义）→ Synthesis Unit（假设生成）
2. **Development 阶段**：Implementation Unit（代码实现）→ Validation Unit（回测验证）→ Analysis Unit（结果分析与调度）

五个模块形成持续迭代的闭环，支持因子和模型的动态联合优化。

### 关键设计

#### 1. Specification Unit（规范单元）

- **功能**：动态配置任务上下文和约束条件，确保后续模块一致性
- **核心思路**：形式化为元组 $\mathcal{S}=(\mathcal{B}, \mathcal{D}, \mathcal{F}, \mathcal{M})$，分别编码先验知识、数据接口、输出格式、执行环境（Qlib 回测平台）
- **设计动机**：通过统一的输入输出规范减少歧义，让 Agent 不需要关注底层预处理和基础设施

#### 2. Synthesis Unit（合成单元）

- **功能**：基于历史实验自动生成新假设
- **核心思路**：维护实验轨迹 $e^t = \{h^t, f^t\}$（假设+反馈），维护 SOTA 集合，通过条件子集提取构建知识森林，用生成式随机映射 $G(\mathcal{H}_t^{(a)}, \mathcal{F}_t^{(a)})$ 产生新假设
- **自适应机制**：成功时增加复杂度/范围，失败时调整结构或引入新变量——形成"idea 森林"
- **设计动机**：模拟人类分析师在理论先验与实证反馈之间的综合推理过程

#### 3. Implementation Unit — Co-STEER（实现单元）

- **功能**：将假设翻译为可执行代码
- **核心思路**：构建 DAG 表示任务依赖关系，拓扑排序确定执行顺序；引导式 Chain-of-Thought 推理提高代码生成质量
- **知识库机制**：记录 (任务, 代码, 反馈) 三元组 $\mathcal{K}^{(t+1)} = \mathcal{K}^{(t)} \cup \{(t_j, c_j, f_j)\}$，通过相似度检索实现跨任务知识迁移
- **设计动机**：量化编码任务有结构依赖，需要系统化的调度和反馈驱动的代码优化

#### 4. Validation Unit（验证单元）

- **功能**：评估因子/模型的实际效果
- **因子去重**：计算新因子与 SOTA 因子库的 IC 相关性，IC_max ≥ 0.99 的判定为冗余并剔除
- **回测评估**：在 Qlib 平台上进行真实市场回测

#### 5. Analysis Unit — 多臂老虎机调度

- **功能**：多维评估实验结果，决定下一轮优先优化因子还是模型
- **核心思路**：将方向选择建模为上下文两臂老虎机问题，观测 8 维性能状态向量，采用线性 Thompson Sampling 自适应平衡探索与利用
- **设计动机**：因子优化和模型优化的边际收益随迭代变化，需要动态选择最优方向

### 损失函数 / 训练策略

本框架不涉及传统意义上的端到端训练。核心优化目标是最大化累积实现质量 $\pi_I = \arg\max_\pi \mathbb{E}[\sum_{j=1}^n R_I(c_j)]$，其中 $R_I(c_j)$ 评估代码正确性和性能。方向调度通过贝叶斯线性回归维护后验，线性奖励函数 $r = \mathbf{w}^\top \mathbf{x}_t$ 指导方向选择。

## 实验关键数据

### 主实验

数据集：CSI 300（沪深 300），训练 2008-2014，验证 2015-2016，测试 2017-2020.08。

| 方法类型 | 模型 | IC | ICIR | ARR | IR | MDD | CR |
|---------|------|-----|------|-----|-----|------|-----|
| ML | LightGBM | 0.0277 | 0.2211 | 3.97% | 0.57 | -8.55% | 0.46 |
| DL | TRA | 0.0404 | 0.3197 | 6.49% | 1.01 | -8.60% | 0.75 |
| DL | MASTER | 0.0215 | 0.1925 | 8.96% | 1.34 | -8.51% | 1.05 |
| 因子库 | Alpha 158 | 0.0341 | 0.2952 | 5.70% | 0.85 | -7.71% | 0.74 |
| 因子库 | Alpha 360 | 0.0420 | 0.3290 | 4.38% | 0.67 | -7.21% | 0.61 |
| **R&D** | **R&D-Factor(GPT-4o)** | **0.0489** | **0.4050** | **14.61%** | **1.68** | -7.50% | **1.95** |
| **R&D** | **R&D-Model(o3-mini)** | **0.0469** | 0.3688 | 10.09% | **1.70** | **-6.94%** | 1.45 |
| **R&D** | **R&D-Agent(Q)(o3-mini)** | **0.0532** | **0.4278** | **14.21%** | **1.74** | -7.42% | **1.92** |

**关键发现**：R&D-Agent(Q) 联合优化达到最高 IC=0.0532, ARR=14.21%, IR=1.74，全面超越所有基线。

### 消融实验

| 配置 | 核心观察 |
|-----|---------|
| R&D-Factor only | 动态因子优化以 ~22% 因子数量达到 Alpha 158 同等 IC |
| R&D-Model only | 自适应模型配置在风险控制（MDD）上优于所有手工 DL 架构 |
| R&D-Agent(Q) joint | 联合优化释放互补增益，IC + 策略表现全面最优 |
| GPT-4o vs o3-mini | o3-mini 在结构化编码任务上 pass@k 收敛更快（CoT 推理更强）|
| 因子假设聚类分析 | 36 轮中 8 个入选 SOTA，跨越 5/6 个聚类 → 多样探索产生互补信号 |

### 泛化性验证

在 CSI 500 和 NASDAQ 100 上（测试期 2024-2025）验证，R&D-Agent(Q) 在中美两国市场均保持顶级表现，IC/ICIR/IR/MDD 指标均处于最优或次优，确认框架跨市场泛化能力和对知识截止问题的鲁棒性。

### 关键发现

1. **因子效率**：R&D-Factor 用不到 Alpha 158 的 22% 因子数量即达到同等 IC，且在 2019-2020 基线退化时保持稳定
2. **模型效率**：R&D-Model 变体在收益-回撤空间中显著优于 baseline DL 模型，达到收益-风险比最优前沿
3. **联合优化增益**：R&D-Agent(Q) 的联合因子-模型优化释放互补性能提升，最终 ARR 约为经典因子库的 2 倍
4. **成本**：整个自动化 R&D 过程的 API 调用成本不到 $10
5. **Co-STEER 自修复**：代码生成 pass@k 在几轮迭代内快速收敛，o3-mini 展现更强的链式推理优势

## 亮点与洞察

1. **首个金融量化全栈自动化多智能体框架**：将因子挖掘和模型创新纳入统一的闭环 R&D 流程，这是"AI for Quant R&D"的重要范式探索
2. **数据驱动而非直接交易**：LLM 不直接接触原始市场数据或交易，而是在 schema 层面生成假设和代码，规避幻觉和数据泄露风险
3. **知识森林 + 多臂老虎机**：Research 阶段的 idea 森林支持"精炼-转向-复用"模式，Analysis 阶段的 Thompson Sampling 自适应调度因子/模型优化方向
4. **Co-STEER 代码代理**：专为结构化数据任务设计的代码生成代理，DAG 调度 + 知识迁移显著提升因子/模型编码效率
5. **极低成本高回报**：$10 以下的 API 成本实现 2× ARR 提升，展示了 LLM Agent 在量化研发中的高性价比

## 局限与展望

1. **市场覆盖**：主实验集中在中国 A 股市场，虽然补充了美股验证，但尚未涵盖更多新兴市场和不同资产类别（期货、期权、加密货币等）
2. **实盘验证缺失**：所有实验基于 Qlib 回测平台，未在实际交易环境中验证（滑点、流动性、市场冲击等现实因素）
3. **LLM 依赖**：框架性能受底层 LLM 能力限制，对 LLM 升级/降级的敏感性值得更深入研究
4. **因子可解释性**：虽然框架生成有假设描述的因子，但自动生成因子的经济学含义是否真正有意义尚需验证
5. **调度策略**：两臂老虎机仅在因子/模型间切换，若扩展到更多优化维度（如风控、执行策略），调度复杂度将显著增加

## 相关工作与启发

- **Qlib (Microsoft)**：提供标准化回测和数据处理基础设施，R&D-Agent(Q) 在此基础上自动化了核心研究环节
- **AlphaFactor 系列**（Alpha 101/158/360）：静态因子库，本文证明自动化动态因子生成可用更少因子达到更好效果
- **LLM for Finance**（FinGPT, BloombergGPT 等）：主要做信号提取或直接生成交易决策，缺乏透明的因子构造和模型验证流程
- **Multi-Agent 模拟**（模拟对冲基金、金融专家协作）：侧重模拟而非自动化研发迭代
- **启发**：这种"Research-Development 闭环"范式有望推广到其他数据驱动的研发场景（药物研发、材料科学等）

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首个量化金融全栈自动化多智能体框架，因子-模型联合优化 + 多臂老虎机调度的设计独特
- 实验充分度: ⭐⭐⭐⭐ — CSI 300 主实验全面，补充了 CSI 500 和 NASDAQ 100 泛化验证，消融和分析详尽，但缺少实盘验证
- 写作质量: ⭐⭐⭐⭐ — 框架描述清晰，模块化设计易于理解，形式化适度，图表质量高
- 价值: ⭐⭐⭐⭐⭐ — 对量化金融自动化研发具有重要范式意义，代码开源（微软 RD-Agent），实用性强

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2026\] MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems](../../ICML2026/multi_agent/maspo_joint_prompt_optimization_for_llm-based_multi-agent_systems.md)
- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](../../ACL2026/multi_agent/masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[NeurIPS 2025\] Lessons Learned: A Multi-Agent Framework for Code LLMs to Learn and Improve](lessons_learned_a_multi-agent_framework_for_code_llms_to_learn_and_improve.md)
- [\[ICML 2026\] OMAC: A Holistic Optimization Framework for LLM-Based Multi-Agent Collaboration](../../ICML2026/multi_agent/omac_a_holistic_optimization_framework_for_llm-based_multi-agent_collaboration.md)
- [\[ICML 2025\] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML](../../ICML2025/multi_agent/automl-agent_a_multi-agent_llm_framework_for_full-pipeline_automl.md)

</div>

<!-- RELATED:END -->
