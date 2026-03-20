# Cheems: A Practical Guidance for Building and Evaluating Chinese Reward Models from Scratch

**会议**: ACL 2025  
**arXiv**: [2502.17173](https://arxiv.org/abs/2502.17173)  
**代码**: [https://github.com/AlignRM/CheemsRM](https://github.com/AlignRM/CheemsRM)  
**领域**: 对齐RLHF  
**关键词**: reward model, Chinese preference, benchmark, distant supervision, RLHF  

## 一句话总结
为弥补中文 Reward Model 资源的空白，本文构建了 CheemsBench（首个大规模中文 RM 评测基准）和 CheemsPreference（首个大规模中文偏好数据集），通过人机协作标注 + 远程监督过滤策略训练的 CheemsRM 在中文场景显著超越现有所有开源 RM。

## 研究背景与动机
1. **领域现状**：Reward Model 是 RLHF 的核心组件，但当前 RM 研究高度集中于英文场景（如 RewardBench、UltraRM、Skywork-Reward），中文 RM 发展严重滞后
2. **现有痛点**：
   - 现有中文偏好数据集规模小（Huozi 仅几千条）、领域受限（知乎问答等特定场景）
   - 现有 RM benchmark 均为英文（RewardBench），无法评估 RM 在中文场景的表现
   - 大量依赖 GPT 合成标注数据，难以准确反映中文用户的真实偏好
3. **核心矛盾**：缺乏高质量的中文偏好数据和评测基准，导致中文 RM 无法有效学习并捕获中文用户的偏好
4. **本文要解决什么？** 从零构建中文 RM 资源体系——评测基准 + 偏好数据集 + 训练方法
5. **切入角度**：以人类标注为核心，辅以远程监督策略扩大规模
6. **核心idea一句话**：先用全人工标注构建高质量小数据集和 benchmark，再用其训练的 RM 过滤 GPT 标注的大规模数据，实现质量与规模的平衡

## 方法详解

### 整体框架
分为三个部分：(1) CheemsBench 评测基准：2492 条 prompt × 5 responses，全人工五轮三选比较 + 冲突消解算法生成可靠偏序排名；(2) CheemsPreference 偏好数据集：27K 人类指令 + 多模型采样 + 人机协作标注（人工小集 + GPT 大集 + RM 过滤）；(3) CheemsRM：基于 Qwen2.5-72B-Instruct 在 CheemsPreference 上训练。

### 关键设计

1. **CheemsBench 构建 — 多回复三选比较 + 冲突消解**:
   - 做什么：为每个 prompt 采样 5 个回复，进行 5 轮三选比较（triple-wise comparison），生成可靠的偏序排名
   - 核心思路：将标注结果转为有向偏好图，用 DFS 检测环（冲突）→ 将环中节点合并为大节点 → 重复直到无环 → 拓扑排序得到偏序。使用 Accuracy 和 Exact Match 两种评测指标
   - 设计动机：传统 pairwise comparison 在反映下游任务性能方面存在局限（Wen et al., 2024）。多回复评估更贴合实际使用场景（best-of-N sampling 等）。三选比较比两两比较的信息密度更高，同时避免了全排序的标注成本
   - 数据来源：1146 条开源 prompt + 1346 条真实人类指令，覆盖推理、理解、创作、复杂指令等类别

2. **CheemsPreference 构建 — 远程监督策略**:
   - 做什么：用人工标注的小数据集训练的 RM 来过滤 GPT 标注的大数据集
   - 核心思路：(a) 人工标注 3260 条 prompt（37K comparisons），(b) GPT-4o 标注 27861 条 prompt（332K comparisons），对 $C_N^2$ 对做 pairwise 比较。(c) 用人工数据训练的 RM 过滤 GPT 标注中的冲突和错误，保留一致的偏好链
   - 设计动机：纯人工标注成本过高（3K 已是极限），纯 GPT 标注质量不可靠（存在位置偏差、不一致性）。远程监督在两者间取得平衡
   - 长度去偏：按 chosen 比 rejected 长/短分两组，下采样较大组以平衡长度偏差

3. **CheemsRM 训练 — 多回复 Bradley-Terry 损失**:
   - 做什么：在多回复偏序数据上训练判别式 RM
   - 核心思路：Bradley-Terry 损失 $\mathcal{L}' = -\mathbb{E}[\log\sigma(r(x, y_w) - r(x, y_l))]$，加高斯正则化 $\mathcal{L} = \mathcal{L}' + \mathbb{E}[r^2(x, y)]$ 稳定训练。使用贪心 sample-based batch 策略，尽量将同一 prompt 的所有回复放入一个 batch
   - 设计动机：相比标准 pairwise 训练，多回复提供更丰富的比较信号；高斯正则防止 reward score 爆炸

## 实验关键数据

### 主实验

**CheemsBench 上各 RM 表现**:

| 模型 | RewardBench | Open Prompt Acc. | Human Instr. Acc. | Overall |
|------|:-----------:|:----------------:|:-----------------:|:-------:|
| Skywork-Reward-Gemma-27B | **0.938** | 0.754 | 0.748 | 0.535 |
| Nemotron-70B-Reward | 0.941 | 0.750 | 0.722 | 0.515 |
| Skywork-Critic-70B (gen) | 0.933 | 0.755 | 0.731 | 0.516 |
| GPT-4o (gen) | 0.846 | 0.640 | 0.727 | 0.457 |
| **CheemsRM (Ours)** | 0.919 | **0.857** | **0.832** | **0.657** |

CheemsRM 在 Overall 上以 **0.657** 大幅领先第二名 0.535（+12.2%），Exact Match 分别达到 0.508 和 0.431，远超其他模型（最好<0.33）。

### 消融实验

**偏好数据来源消融**:

| 数据来源 | Open Acc. | Human Acc. | Overall |
|---------|:---------:|:----------:|:-------:|
| 仅 GPT 标注 | 0.815 | 0.789 | 0.590 |
| 仅 Human 标注 | 0.829 | 0.811 | 0.614 |
| GPT + Human | 0.839 | 0.820 | 0.633 |
| GPT + Human + 远程监督过滤 | **0.857** | **0.832** | **0.657** |

**偏好数据集对比（训练基座: Qwen2.5-72B）**:

| 数据集 | Open Acc. | Human Acc. |
|-------|:---------:|:----------:|
| Huozi (中文最佳现有) | 0.728 | 0.682 |
| HH-RLHF (英文) | 0.753 | 0.740 |
| Ultrafeedback (英文最佳) | 0.769 | 0.749 |
| **CheemsPreference** | **0.857** | **0.832** |

### 关键发现
- 现有最强英文 RM（Skywork-Reward-Gemma-27B，RewardBench 0.938）在中文场景大幅降级——Overall 仅 0.535
- 人类标注数据虽然只有 3K，但训练效果优于 28K 的 GPT 标注数据，说明数据质量远比规模重要
- 远程监督过滤在 GPT + Human 基础上再提升 2.4% Overall，验证了过滤策略的有效性
- RM 在"理解"类任务上表现最差，在"推理"类任务上表现最好——暗示当前 RM 更擅长判断客观正确性，而非主观质量

## 亮点与洞察
- **远程监督策略的精妙设计**：用小量人工标注数据训练的 RM 来过滤大量 GPT 标注数据——这个"以小博大"的策略非常实用，可推广到其他语言和领域的偏好数据构建
- **冲突消解算法**：将标注分歧形式化为图中的环路问题，用 DFS + 节点合并 + 拓扑排序解决，优雅且可扩展。这个算法可以复用到任何多标注者场景
- **首次系统性揭示中英文 RM 差距**：即使是 RewardBench 顶级模型在中文场景也表现不佳，意义重大

## 局限性 / 可改进方向
- **基座模型依赖 Qwen2.5-72B**：CheemsRM 计算成本高，可探索在更小模型（7B-14B）上的效果
- **偏好分类体系依赖人工设计**：8 大类数十小类的分类可能遗漏某些中文特有场景（如古文理解、方言处理）
- **仅评估判别式用法**：未验证 CheemsPreference 用于 DPO/PPO 训练的下游效果
- **GPT-4o 标注存在成本**：28K prompt × $C_5^2$ 对的 GPT-4o 调用成本不低，可探索更便宜的替代

## 相关工作与启发
- **vs RewardBench**: RewardBench 是英文 RM 标准 benchmark，CheemsBench 填补了中文空白；但 CheemsBench 额外引入了多回复评估和 Exact Match 指标
- **vs Skywork-Reward**: Skywork-Reward 在 RewardBench 上 SOTA（0.938），但在 CheemsBench 上仅 0.535——说明英文 RM 能力不能直接迁移到中文
- **vs UltraFeedback**: UltraFeedback 是英文最强通用偏好数据集之一，CheemsPreference 在中文场景远超其训练效果
- 远程监督 + 冲突消解策略可直接用于构建日文、韩文等其他语言的 RM 资源

## 评分
- 新颖性: ⭐⭐⭐ 资源贡献为主，技术新颖性中等（远程监督策略有新意）
- 实验充分度: ⭐⭐⭐⭐⭐ 评测全面（16+ RM 对比、多数据集消融、下游任务相关性分析）
- 写作质量: ⭐⭐⭐⭐ 结构清晰，数据详实，图表丰富
- 价值: ⭐⭐⭐⭐ 填补中文 RM 资源空白，对中文 LLM 对齐社区有重要参考价值
