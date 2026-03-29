# A Survey of Automatic Prompt Optimization with Instruction-focused Heuristic-based Search Algorithm

**会议**: ACL 2025
**arXiv**: [2502.18746](https://arxiv.org/abs/2502.18746)
**代码**: [GitHub](https://github.com/jxzhangjhu/Awesome-LLM-Prompt-Optimization)
**领域**: LLM / Prompt 工程
**关键词**: 自动提示优化, 启发式搜索, 指令优化, Prompt工程, 分类体系

## 一句话总结
系统综述基于启发式搜索算法的自动 Prompt 优化方法——提出五维分类体系（Where/What/What criteria/Which operators/Which algorithms），覆盖优化空间、目标、评价标准、候选生成算子和搜索算法，并综述支撑数据集和工具框架。

## 研究背景与动机
1. **领域现状**：手动 Prompt 工程（如 CoT、"step by step"）依赖人类直觉和反复试错，无法系统性地发现最优 prompt。自动 Prompt 优化将 prompt 设计视为搜索问题，通过算法迭代精炼。
2. **现有痛点**：自动 Prompt 优化领域快速发展但研究碎片化——有的用进化算法、有的用贝叶斯优化、有的用 LLM 自身作为优化器——缺乏统一分类框架。
3. **核心矛盾**：如何将方法论差异巨大的各种自动 prompt 优化方法纳入一个连贯的分类体系？
4. **本文要解决什么？** 提供一个全面而有组织的综述，用五维分类将碎片化研究统一起来。
5. **切入角度**：聚焦两个范围限制——(a) 指令式（instruction-focused）而非示例选择；(b) 启发式搜索而非 RL 或集成方法，避免泛泛而谈。
6. **核心idea一句话**：五维分类体系（Where×What×What criteria×Which operators×Which algorithms）是理解自动 Prompt 优化的完整坐标系。

## 核心内容

### 五维分类体系

1. **Where（优化空间）**：
   - **软提示空间**：连续嵌入空间，可用梯度优化（Prefix Tuning、P-Tuning）。子类：梯度→嵌入、梯度→目标 token（如 GCG 用梯度选择替换位置）、梯度→词表、零阶优化（ZOPO 用 NTK 近似的高斯过程估计梯度）
   - **离散提示空间**：直接优化文本字符串，不可微分。如 ProTeGi 用 LLM 反馈生成"伪梯度"，EvoPrompt 用遗传算法

2. **What（优化目标）**：
   - 仅指令优化（最常见）：直接精炼指令文本
   - 指令+示例联合优化：三种范式——示例→指令（MoP：先聚类示例再生成专门指令）、指令→示例（MIPRO：用指令生成匹配的示例对）、并行优化（EASE：用 bandit 算法同时搜索最佳指令-示例组合）
   - 指令+可选示例（PhaseEvo：根据任务动态决定是否需要示例）

3. **What criteria（优化标准）**：不仅是任务精度——还包括鲁棒性、效率、可解释性、安全性。多目标优化越来越受关注

4. **Which operators（候选生成算子）**：
   - 零父本：从零生成（Lamarckian、模型驱动）
   - 单父本：语义改写（部分/全局）、LLM/人类/梯度反馈、增删替换
   - 多父本：交叉（组合多 prompt）、差异、EDA（估计分布采样）

5. **Which algorithms（搜索算法）**：Bandit 算法、束搜索、启发式采样、蒙特卡洛树搜索、元启发式算法（进化算法/GA/差分进化/模拟退火）、迭代精炼

### 代表方法回顾

| 方法 | 优化空间 | 核心思路 | 搜索算法 |
|------|---------|---------|---------|
| APE (Zhou et al.) | 离散 | LLM 生成+筛选指令 | 迭代精炼 |
| OPRO (Yang et al.) | 离散 | LLM 自身作为优化器 | 迭代精炼 |
| ProTeGi (Pryzant et al.) | 离散 | LLM 反馈生成伪梯度 | 束搜索 |
| EvoPrompt (Fernando et al.) | 离散 | 遗传算法变异+交叉 | 进化算法 |
| PromptBreeder | 离散 | 自进化 prompt + 策略也进化 | 进化算法 |
| InstructZero (Chen et al.) | 软 | 贝叶斯优化调整软 prompt | 贝叶斯优化 |
| GCG (Zou et al.) | 软→离散 | 梯度定位最优替换 token | 贪心坐标梯度 |

### 数据集与工具
- 数据集覆盖推理（BBH）、常识（CSQA）、数学（GSM8K）、代码、对话等多领域基准
- 工具框架：DSPy（声明式 prompt 编程）、PromptFoo（测试框架）、TextGrad（文本梯度）、PromptBench（鲁棒性测试）

## 亮点与洞察
- **五维分类体系清晰全面**：将碎片化的研究统一到一个框架下，每篇论文都可以在五个维度上定位
- **LLM 自身作为优化器（OPRO 范式）**打破了传统优化-评估分离的思路——这可能是最有前景的方向
- **优化标准应超越精度**：鲁棒性和安全性同等重要——GCG 原本用于对抗攻击但其技术可反向用于防御

## 局限性 / 可改进方向
- **对多模态 prompt 优化覆盖不足**：主要关注文本 prompt，视觉/音频 prompt 优化未涉及
- **缺少方法间的定量对比**：分类但未提供统一实验对比——各方法在相同基准上的性能对比仍缺失
- **快速发展的领域**：某些 2024 年底的最新方法可能未覆盖

## 相关工作与启发
- **vs 其他 Prompt 综述**：本文范围更精确（启发式搜索 + 指令），其他综述覆盖更广但深度不足
- Prompt 优化本质是搜索问题——搜索算法的选择至关重要

## 评分
- 新颖性: ⭐⭐⭐ 综述贡献在分类体系，方法为回顾性
- 实验充分度: ⭐⭐ 无实验，纯综述
- 写作质量: ⭐⭐⭐⭐ 分类清晰，覆盖全面，Figure 1 的树状分类图直观
- 价值: ⭐⭐⭐⭐ 对 Prompt 工程从业者有实用参考价值，五维分类是有用工具
