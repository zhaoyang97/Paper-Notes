# VReST: Enhancing Reasoning in Large Vision-Language Models through Tree Search and Self-Reward Mechanism

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2506.08691](https://arxiv.org/abs/2506.08691)  
**代码**: [https://github.com/GaryJiajia/VReST](https://github.com/GaryJiajia/VReST)  
**领域**: 多模态VLM / LLM推理  
**关键词**: MCTS, 视觉推理, 自奖励, 测试时缩放, 多模态CoT  

## 一句话总结
提出VReST，首次将蒙特卡洛树搜索（MCTS）应用于多模态CoT推理：每个节点是一个推理步骤，通过多模态自奖励机制（sub-question有用性+答案正确性+视觉-语言线索相关性）评估推理质量，无需训练即在MathVista上达到64.50%（超越CoT的54.60%和ToT的60.20%），并展示出多模态测试时缩放定律。

## 背景与动机
LVLM的CoT推理在复杂视觉数学任务上效果有限——在MathVista等基准上，CoT甚至不如直接回答（Direct QA: 55.70% vs CoT: 54.60%）。原因：(1) 推理步骤有限，无法充分探索解空间；(2) 缺乏对推理链质量的评估和修正机制。已有的Tree-of-Thoughts等方法使用启发式选择，容易陷入局部最优。

## 核心问题
如何在测试时无需训练地充分探索LVLM的推理空间，并可靠地评估和选择最优推理链？

## 方法详解

### 整体框架
构建推理搜索树，每个节点是一个推理步骤（sub-question + sub-answer），通过MCTS的Selection→Expansion→Rewarding→Backpropagation四步迭代K次，最终选择累积奖励最高的推理路径。

### 关键设计

1. **MCTS推理搜索**: 
   - **Selection**: 使用UCT（Upper Confidence Bound for Trees）平衡探索和利用：UCT(v) = R(v) + c·√(ln N(parent)/N(v))
   - **Expansion**: 对选中的叶节点，用LVLM在较高temperature下生成w个候选推理步骤（宽度w的树扩展）
   - **Rewarding**: 用自奖励机制评估每个新推理步骤
   - **Backpropagation**: 将奖励值沿路径回传更新祖先节点

2. **多模态自奖励机制（Self-Reward）**: 不引入额外模型，用LVLM自身评估推理质量，融合两个维度：
   - **R₁**: sub-question的有用性——"这些子问题对解决原始问题有用吗？"（Yes/No概率）
   - **R₂**: 最后一步答案的正确性——"这个答案正确吗？"（Yes/No概率）
   - 最终奖励：R = √(R₁ × R₂)，用几何平均确保两者都要高

3. **推理链选择**: 搜索完成后，有两种策略：
   - **VReST**: 选择累积奖励最高的推理链
   - **VReST-Vote**: 将top-k推理链的最终答案做投票，多数决

### 损失函数 / 训练策略
- 完全无需训练（training-free），纯推理时方法
- 默认K=8次MCTS迭代，w=3宽度，D_max=10最大深度
- 基于InternVL2-8B和Qwen2-VL-7B进行实验

## 实验关键数据

**MathVista (testmini)**:

| 方法 | ALL | MWP | VQA | SCI | STA |
|------|-----|-----|-----|-----|-----|
| Direct QA | 55.70 | 60.75 | 50.28 | 59.84 | 67.44 |
| CoT | 54.60 | 56.99 | 48.04 | 59.02 | 70.43 |
| CoT-Vote | 62.30 | 69.89 | 56.98 | 60.66 | 79.07 |
| ToT | 60.20 | 63.44 | 54.19 | 57.38 | 74.09 |
| **VReST** | **64.50** | **72.04** | 58.10 | **67.21** | 75.75 |
| **VReST-Vote** | **65.40** | **75.81** | **64.25** | **68.03** | 77.74 |

**MATH-Vision (testmini)**: VReST: 26.64% vs ToT: 20.39% vs CoT: 14.47%

**测试时缩放**: 随着MCTS迭代次数从1增加到16，性能持续提升且不饱和——展示了多模态测试时缩放定律。

### 消融实验要点
- **奖励函数**: R₁和R₂都不可或缺，几何平均优于算术平均
- **MCTS K次迭代**: K=1时已略优于基线，K=8时显著领先
- **树宽度w**: w=3是效率和效果的最佳平衡
- **VReST vs VReST-Vote**: Vote在答案多样性高的任务更有效
- **CoT反而不如Direct QA**: 验证了复杂推理中简单CoT的局限性

## 亮点
- **首次MCTS用于多模态推理**: 填补了MCTS在视觉推理中的空白
- **优雅的自奖励设计**: 不需要额外reward model，LVLM自评即可——用Sub-Q有用性和答案正确性两个维度
- **测试时缩放定律**: 在多模态任务中首次展示：增加推理时计算=持续提升性能
- **CoT < Direct QA的揭示**: 证明简单CoT在复杂视觉推理中确实不够用

## 局限性 / 可改进方向
- 计算成本高：K=8的MCTS需要数十次LVLM推理，延迟增加10x+
- 自奖励可能不够可靠——LVLM评估自己生成内容时可能有偏差
- 仅在数学/视觉推理任务验证，通用VQA或开放式生成可能需要不同的奖励设计
- 树搜索假设推理可以分解为离散步骤，连续推理场景不适用
- 未与OpenAI o1/o3等内置推理模型对比

## 与相关工作的对比
- **vs CoT/CoT-SC**: VReST是系统性搜索 vs CoT的线性/采样策略
- **vs ToT**: ToT用启发式评估导致局部最优，VReST用MCTS+UCT做全局探索
- **vs Cantor**: Cantor用多角色分步推理但无搜索机制
- **vs Improve VLM CoT Reasoning (本批次)**: 那篇用SFT+DPO改进CoT质量，VReST用测试时搜索——两者正交互补

## 启发与关联
- MCTS的推理时缩放定律可以指导VLM部署策略——简单问题用少迭代、难问题用多迭代（自适应MCTS）
- 自奖励机制可以与VHR（视觉注意力头增强）结合——增强视觉感知后再做树搜索
- VReST的思路可以推广到Agent任务——GUI操作也可以用MCTS搜索最优动作序列

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次在多模态推理中应用MCTS，自奖励机制设计合理
- 实验充分度: ⭐⭐⭐⭐ 3个benchmark、详细消融、缩放分析
- 写作质量: ⭐⭐⭐⭐⭐ 图2的框架图极其清晰，方法描述精确
- 价值: ⭐⭐⭐⭐ 展示了测试时缩放在多模态中的潜力，启发后续研究
