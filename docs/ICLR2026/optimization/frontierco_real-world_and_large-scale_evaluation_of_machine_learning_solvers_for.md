# FrontierCO: Real-World and Large-Scale Evaluation of Machine Learning Solvers for Combinatorial Optimization

**会议**: ICLR 2026  
**arXiv**: [2505.16952](https://arxiv.org/abs/2505.16952)  
**代码**: [HuggingFace](https://huggingface.co/datasets/CO-Bench/FrontierCO)  
**领域**: Agent / 组合优化  
**关键词**: combinatorial optimization, ML solver, benchmark, real-world instances, LLM agent  

## 一句话总结
FrontierCO 是一个涵盖 8 类组合优化问题（TSP、MIS、CVRP 等）的大规模真实世界基准测试，评估了 16 个 ML 求解器（神经网络方法 + LLM Agent）与 SOTA 传统求解器的差距，发现 ML 方法在结构复杂和极大规模实例上仍显著落后于传统方法，但在部分场景有超越潜力。

## 研究背景与动机
1. **领域现状**：ML 用于组合优化（CO）近年发展迅速，包括端到端神经求解器（GNN、RL、扩散模型）和 LLM Agent 方法（FunSearch、ReEvo 等），在小规模合成基准上展示了promising结果。
2. **现有痛点**：三大局限——① **规模**：大多数评估在 toy 级别实例上进行（如 TSP ≤ 10K 节点），而实际应用需要处理百万级节点；② **真实性**：合成数据集无法捕捉真实世界的结构多样性（如非欧几里得图、竞赛级不规则实例）；③ **覆盖度**：缺乏跨问题类型的统一评估协议。
3. **核心矛盾**：ML 方法在合成基准上的"进展"可能是因为问题太简单/太规则，而非方法真正有效。需要在真实结构和极端规模下检验。
4. **本文要解决**：提供一个严格的、基于真实世界的 CO 基准测试套件，统一评估 ML 求解器与传统 SOTA 求解器的差距。
5. **切入角度**：从竞赛库（DIMACS、TSPLib、PACE Challenge）收集真实实例，分为 easy（已可解决）和 hard（开放问题）两个测试集，将规模推到 TSP 1000万节点、MIS 800万节点。
6. **核心idea**：ML for CO 的进展需要在真实结构和极大规模下检验，而非仅在合成数据上刷分。

## 方法详解

### 整体框架
FrontierCO 涵盖 8 类 CO 问题（路由: TSP/CVRP; 图: MIS/MDS; 设施选址: CFLP/CPMP; 调度: FJSP; 树: STP），统一使用 primal gap 作为评价指标：$\text{pg}(x;s) = |cost(x;s) - c^*| / \max(|cost(x;s)|, |c^*|)$，范围 [0,1]，0 为最优。每个问题提供：easy set（历史上困难但现在可解）+ hard set（开放/计算密集型实例）+ 标准化训练/验证集。

### 关键设计

1. **实例规模与数据来源**:
   - TSP: 最大 1000万节点（此前 ML 评估最大 1万）
   - MIS: 最大 800万节点（此前最大 1.1万）
   - 数据源: TSPLib, DIMACS Challenge, PACE Challenge 2025, BHOSLib 等
   - 设计动机: 真实世界的 CO 实例具有不规则结构，合成均匀分布无法代表

2. **Hard Set 的构建逻辑**:
   - 不仅仅是"更大"——强调结构复杂性
   - 如 STP 的 PUC 超立方体图、MIS 的 SAT-induced 实例
   - 很多实例没有已知最优解，防止"启发式黑客"（heuristic hacking）和记忆化
   - 设计动机: 迫使 ML 方法展示真正的泛化能力

3. **评估体系**:
   - 16 个 ML 求解器: 含神经求解器（DiffUCO, SDDS, LEHD, DIFUSCO, SIL, DeepACO 等）和 LLM Agent（FunSearch, Self-Refine, ReEvo）
   - 传统 SOTA: KaMIS(MIS), LKH-3(TSP), HGS(CVRP), Gurobi(MDS/CFLP), CPLEX(FJSP) 等
   - 统一时间限制: 每实例 1 小时
   - 统一硬件: 单 CPU 核 + 单 GPU

4. **标准化训练/验证数据**:
   - 解决跨论文比较中合成数据不一致的问题
   - 包含数据加载器、评估函数和 LLM Agent 模板
   - 对 LLM 隐藏评估函数以防数据泄露

## 实验关键数据

### 主实验——Easy Set 上的 Primal Gap (%)
| 领域 | SOTA 传统 | 最佳 Neural | 最佳 LLM |
|------|----------|------------|---------|
| TSP | **0.00** (LKH-3) | 0.16 (LEHD) | 3.82 (ReEvo) |
| MIS | **0.00** (KaMIS) | 0.37 (SDDS) | 7.21 |
| CVRP | **0.14** (HGS) | 1.73 (SIL) | 12.5 |
| CFLP | **0.00** (Gurobi) | 0.91 (SORREL) | 5.4 |
| FJSP | **0.00** (CPLEX) | 8.2 (MPGN) | 15.3 |

### Hard Set 上的 Gap 分析
- 在 Hard Set 上差距**急剧扩大**
- TSP 10M 节点: 传统方法 gap ~1%, 最佳 Neural gap ~15%
- MIS 结构化实例: Neural 方法在 SAT-induced 图上几乎完全失败
- LLM Agent 方法普遍高方差，有时能偶然超越传统方法，但不稳定

### 消融/分析
| 维度 | 发现 |
|------|------|
| 规模扩展 | Neural 性能随规模指数退化，传统方法线性退化 |
| 结构复杂度 | 非欧/不规则结构对 Neural 方法打击最大 |
| 泛化 | 合成→真实的分布迁移导致显著性能下降 |
| LLM 稳定性 | ReEvo 等方法方差极大，同问题不同 run 差异巨大 |

### 关键发现
- **ML 方法在 easy 实例上有竞争力但在 hard 实例上全面落后**，差距在结构复杂和大规模实例上加剧
- **Neural 方法能增强简单启发式**但无法替代精细工程的专用求解器
- **LLM Agent 偶尔能超越 SOTA 传统方法**但高方差，因为它们无法理解自己生成的算法中哪些真正有效
- 合成→真实的**分布迁移**是 Neural 方法的核心瓶颈
- **调度和设施选址**等约束复杂问题对 ML 方法尤其困难

## 亮点与洞察
- **规模对比震撼**: TSP 从 10K 推到 10M，MIS 从 11K 推到 8M——暴露了 ML 方法在规模上的根本缺陷
- **"Hard ≠ Large"的设计哲学**: Hard set 强调结构复杂性而非仅仅规模，如 PUC 超立方体和 SAT-induced 图——这比单纯增大规模更有意义
- **LLM Agent 的双刃剑特性**: 高方差但偶尔超越 SOTA——说明 LLM 的代码生成能力有创造性但缺乏稳定性和深层理解
- **标准化的价值**: 提供统一的训练数据/评估协议，终结了跨论文比较中"苹果比橘子"的混乱

## 局限性 / 可改进方向
- 每实例 1 小时的时间限制可能不足以让某些方法收敛
- 仅评估单 GPU 设置，未考虑分布式并行
- 某些问题（如 STP）的 Neural baseline 较弱，可能低估了 ML 的潜力
- 未包含一些新兴方法（如 Neural 引导的 Branch-and-Bound 最新进展）
- Hard set 的 BKS 可能不是真正最优解，primal gap 受参考解质量影响

## 相关工作与启发
- **vs CO-Bench**: FrontierCO 聚焦真实世界实例和极端规模，CO-Bench 更偏向 LLM Agent 评估
- **vs 现有 Neural CO 评估**: 此前评估都在合成数据和小规模上，FrontierCO 暴露了这种评估方式的不可靠性
- **vs DIMACS Competition**: 借鉴竞赛评估思路但面向 ML 社区，降低了参与门槛
- 对 ML for CO 研究方向有重要指引：应更关注泛化能力和结构适应性，而非在合成基准上刷分

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个真正大规模真实世界的 ML for CO 基准
- 实验充分度: ⭐⭐⭐⭐⭐ 8 类问题 × 16 个 ML 求解器 + SOTA 传统方法的全面评测
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据呈现直观
- 价值: ⭐⭐⭐⭐⭐ 对 ML for CO 社区有"照妖镜"般的警示价值
