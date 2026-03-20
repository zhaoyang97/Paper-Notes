# Contrastive Representations for Temporal Reasoning

## 基本信息
- **arXiv**: 2508.13113
- **会议**: NeurIPS 2025
- **作者**: Alicja Ziarko, Michal Bortkiewicz, Michal Zawalski, Benjamin Eysenbach, Piotr Milos
- **领域**: Temporal Reasoning / Representation Learning / Planning

## 一句话总结
论文研究能否用纯表示学习替代显式搜索来承担部分时序推理，指出标准 temporal contrastive learning 容易抓住伪特征而失去时序结构，进一步提出 CRTR（Combinatorial Representations for Temporal Reasoning），通过特制负采样从理论上去除伪特征，学到同时编码感知与时序结构的表示，在 Sokoban 和 Rubik's Cube 上取得强结果，甚至可在不依赖外部搜索算法的情况下求解任意初始魔方状态。

## 背景与动机
经典 AI 常把问题拆为：
- perception 学状态表示；
- planning 依赖搜索。

但一个根本问题是：**如果表示本身已经编码足够强的时序结构，是否可以减少甚至摆脱显式搜索？** 现有 temporal contrastive learning 虽流行，但常靠数据中的静态捷径或伪特征完成任务，并未真正学会 temporal reasoning。

## 核心问题
如何设计一种表示学习方法，使学到的 representation 真正对动作序列、状态转移和时序组合结构敏感，从而支持复杂推理和求解？

## 方法详解

### 1. 标准 temporal contrastive learning 的失败模式
作者指出传统方法的问题在于：
- negative sampling 太弱或太随意；
- 模型可以利用与时序无关的 spurious features 区分样本；
- 最终得到的表示不具备真正的规划能力。

### 2. CRTR：Combinatorial Representations for Temporal Reasoning
CRTR 的关键在于特制负采样机制：
- 从组合结构层面构造更难、更“反捷径”的 negative samples；
- 从理论上去除能被伪特征利用的判别路径；
- 迫使模型关注时序依赖与状态转移规律。

### 3. 学到可用于求解的表示
作者展示这种表示不仅可用于分类或检索，还可支持复杂任务求解：
- 在 Sokoban 等具有长时依赖的任务上表现强；
- 在 Rubik's Cube 上可泛化到任意初始状态；
- 所需 search steps 少于 BestFS，但解路径更长。

更强的一点是：这是首个仅依赖学得表示、无需外部搜索算法就能高效求解任意魔方状态的方法之一。

## 实验结论
- 标准 temporal contrastive learning 不能可靠捕获 temporal structure；
- CRTR 在 Sokoban、Rubik's Cube 等复杂时序任务上显著更强；
- 学得的表示具备跨初始状态泛化能力。

## 亮点
1. **问题切得很准**：直接挑战“contrastive 表示真的学到时序了吗”这个核心问题。
2. **理论与实验结合**：不仅提出负采样方案，还给出理论去伪特征分析。
3. **结果很硬**：Rubik's Cube 泛化求解是强信号。
4. **对 agent 很有启发**：说明 representation 本身可以承担部分 planning 责任。

## 局限性
1. 任务仍以结构化环境为主，向开放世界视觉-语言 agent 迁移还需探索。
2. 虽然搜索步数更少，但解更长，说明策略最优性仍有限。
3. 负采样设计在更复杂数据域上的构造成本可能上升。

## 与相关工作的对比
- 相比普通 temporal contrastive learning：CRTR 显式针对 spurious features 设计。
- 相比经典搜索规划：CRTR 希望把部分规划能力“编进表示”。
- 相比 world model 路线：CRTR 更强调可解耦的时序表示而非完整环境生成。

## 启发
- 可以探索把 CRTR 思路迁移到视频理解或视觉规划中的 latent state learning。
- 对 VLA/robotics 场景，若能学出时序结构表示，可能显著减轻 search/planning 负担。
- 与 RICL 这类 temporal credit assignment 方法有潜在互补性：一个提升表示，一个提升监督信号。

## 评分
- 新颖性：★★★★★
- 技术深度：★★★★☆
- 实验说服力：★★★★★
- 研究启发性：★★★★★