# Checklists Are Better Than Reward Models For Aligning Language Models

**会议**: NeurIPS 2025  
**arXiv**: [2507.18624](https://arxiv.org/abs/2507.18624)  
**代码**: 已开源（WildChecklists 数据集 + 模型 + 代码）  
**领域**: LLM 对齐 / 指令跟随  
**关键词**: RLCF, checklist, reward model, DPO, instruction following, alignment  

## 一句话总结
提出 Reinforcement Learning from Checklist Feedback (RLCF)，将指令分解为动态生成的 yes/no checklist，结合 AI judge 和代码验证器逐项评分后做 DPO 训练，在 5 个 benchmark 上一致性提升 Qwen2.5-7B-Instruct，是唯一在所有 benchmark 上都有正收益的方法（FollowBench +4pt, InFoBench +6pt, Arena-Hard +3pt）。

## 研究背景与动机

1. **领域现状**：指令跟随是 LLM 实用化的核心要求。当前后训练 pipeline 普遍采用 SFT + RLHF（用 reward model 打分做偏好优化），但 reward model 的 reward 信号是单一标量，对复杂多需求指令的评估过于粗糙。

2. **现有痛点**：三种主流自动反馈方式都有明显缺陷——(1) 可验证指令只覆盖格式约束，忽略风格/内容等主观维度；(2) Reward model 灵活但任意性强，容易被 reward hacking，且在 RewardBench 上排名高不代表 RLHF 效果好；(3) 用大模型当 judge 推理，但 judge 需要自己推断评估标准，模糊了 generator-verifier gap。

3. **核心矛盾**：RM 和 AI judge 都用一个单一标量/固定维度评估所有指令的所有方面，但用户指令的需求维度是高度动态和多样的。例如 "翻译成西班牙语" 和 "写一个包含3个逗号的论点" 需要完全不同的评估标准。

4. **本文要解决什么**：设计一种自动、灵活、直观且通用的回复评分方式，为每条指令生成专属的多维度评估标准（checklist），替代 reward model 做 RL 训练。

5. **切入角度**：将复杂的回复质量评估分解为一系列简单的 yes/no 问题（checklist），每个问题要么由 AI judge 回答，要么由生成的验证代码自动检查。这利用了"多个简单判断的组合优于一个复杂判断"的原则。

6. **核心idea一句话**：从指令动态生成 instruction-specific 的 checklist（通过 candidate-based 方法），逐项用 AI judge + 代码验证器评分，加权汇总后用 DPO 做偏好训练。

## 方法详解

### 整体框架
RLCF Pipeline: (1) 从 130K WildChat 指令生成 candidate-based checklists → (2) 从学生模型采样回复对 → (3) 用 checklist 逐项评分（AI judge 25次采样 + 代码验证器） → (4) 加权汇总各项得分为总分 → (5) 筛选差异最大的 40% 回复对做 DPO。

### 关键设计

1. **Candidate-Based Checklist 生成**：
   - 做什么：为每条指令生成高质量、细粒度的评估标准
   - 核心思路：两阶段——先用不同大小的模型（0.5B~7B）生成多个质量各异的候选回复，再让 72B 模型分析这些回复的所有可能失败模式，总结为 checklist。每项附带 0-100 的重要性权重
   - 设计动机：直接从指令提取 checklist 容易产生"复述指令"的问题，覆盖面不够。通过观察不同质量的回复，模型能发现更细微的质量差异（如格式、事实准确性、语气等）。实验证实 candidate-based 方法在客观性（+0.8%）、原子性（+22%）上显著优于直接方法

2. **混合评分：AI Judge + 代码验证器**：
   - 做什么：可靠地评估每个 checklist 项
   - 核心思路：对每个 checklist 项，(a) 让 72B judge 模型采样 25 个 0-100 分数取平均，(b) 同时让模型判断该项是否能被代码精确验证（如"是否包含3个逗号"），如果能则生成 Python 验证函数，代码评分与 judge 评分取平均
   - 设计动机：LLM 在判断硬约束（如字母出现次数）时不可靠，但代码验证器对此完美。同时代码无法评估软约束（如"是否连贯"），所以需要 AI judge。Table 8 的案例分析清楚展示了两者互补

3. **Universal Requirements（防 Reward Hacking）**：
   - 做什么：防止模型通过堆砌冗余内容来满足 checklist
   - 核心思路：对所有 checklist 附加两条通用要求：(1) 回复是否直接回应请求而无过度/离题信息，(2) 回复是否匹配指令所需的语气/风格
   - 设计动机：初始实验发现模型学会在回复开头生成冗长概述来提高 checklist 得分，类似 reward hacking

### 损失函数 / 训练策略
- DPO 训练，batch size 1024，max seq len 2048，cosine lr schedule（max 3e-6，min 2e-6），2 epochs
- 130K instructions（WildChat），筛选 40% 差异最大的回复对
- 在 8×H100 上训练约 3 小时；评分约 4 天（25 samples/item）

## 实验关键数据

### 主实验（Qwen2.5-7B-Instruct）

| 方法 | IFEval (Avg) | InFoBench | FollowBench HSR | Arena-Hard | AlpacaEval LC |
|---|---|---|---|---|---|
| Baseline | 77.3 | 78.1 | 71.4 | 42.8 | 36.2 |
| + DPO (Skywork RM) | 76.0 | 82.0 | 69.5 | 50.3 | 41.5 |
| + DPO (ArmoRM) | 76.0 | 83.5 | 70.4 | 46.4 | 38.1 |
| + DPO (Ultrafeedback) | 74.6 | 80.0 | 72.6 | 47.9 | 38.7 |
| + DPO (AI Judge) | 75.2 | 76.1 | 70.3 | 44.4 | 33.4 |
| **+ DPO (RLCF)** | **78.6** | **84.1** | **75.3** | **48.4** | **37.1** |

RLCF 是唯一在所有 5 个 benchmark 上都有正向提升的方法。RM-based 方法在某些 benchmark 上有提升但在其他 benchmark 上退化（如 Skywork 在 IFEval 和 FollowBench 上退化）。

### 消融实验

| 配置 | 效果 | 说明 |
|---|---|---|
| Direct vs Candidate-based checklist | Candidate-based 在 IFEval/FollowBench 上更好 2-3% | 更客观更原子化的 checklist 产出更好训练信号 |
| 去掉代码验证器 | FollowBench 略降但 Arena-Hard 反升 | 代码验证器对硬约束有帮助但不是决定性因素 |
| 去掉 prompt-based scoring | Format 类约束表现下降 | AI judge 和代码验证器互补 |
| 减少采样次数（25→5） | IFEval/InFoBench 基本不变，FollowBench content 类下降 | 5 次采样可节省 55% 时间，是实用的折中 |
| Off-policy（Llama/OLMo） | 一致提升，无退化 | Checklist 捕获通用标准，不绑定特定模型族 |

### 关键发现
- **RewardBench 准确率 ≠ RLHF 效果**：Skywork-27B 在 RewardBench 上远超 checklist-based reward（96.1 vs 90.0 Chat），但在实际 RLHF 训练中 checklist 效果更好。这一发现与 Malik et al. 2025 和 Razin et al. 2025 的结论一致
- **RLCF 最帮助 "content" 类约束**：FollowBench 分类分析显示，RLCF 在限定答案空间的内容约束上提升最大（+6.4%），而非格式约束。这说明 checklist 让模型学会关注指令的完整含义
- **泛化性强但有领域偏差**：WildChat 主要是日常咨询类指令，导致在数学（GSM8K -1%）和事实性（TruthfulQA -1.5%）上略有退化。但扩展 prompt 分布比重训练 RM 容易得多

## 亮点与洞察
- **"极端混合评估器"视角**：RLCF 可以被理解为一个无限大的 mixture-of-evaluators，其中每条指令动态选择一组专属评估器。这比固定 4 个维度（UltraFeedback）或单一标量（RM）提供了指数级更丰富的反馈信号
- **Reward Model 的悖论**：在 RewardBench 上评估更好的 RM 在实际 RLHF 中效果更差，因为 RM 学到的"好回复"模式可能不匹配特定指令的需求。Checklist 绕过了这个问题——它不学习什么是"好"，而是直接检查具体条件
- **代码验证器的选择性使用**：不是对所有 checklist 项都生成代码，而是只在模型有 100% 信心能精确检查时才生成，其余交给 AI judge。这种自适应组合是一个实用且优雅的设计

## 局限性 / 可改进方向
- 评分成本高：130K 指令的评分需要 4 天在 8×H100 上（72B 模型 25 次采样/项），但可通过减少到 5 次采样节省 55% 时间
- 仅探索了 DPO（off-policy），未尝试 GRPO 等 on-policy 方法，后者可能利用 checklist 反馈效果更好
- 依赖 72B 教师模型做 judge，是 strong-to-weak generalization
- 训练数据 domain 偏向日常咨询，safety 和数学推理等领域覆盖不足

## 相关工作与启发
- **vs UltraFeedback**：UltraFeedback 用 4 个固定维度（instruction following, helpfulness, truthfulness, honesty）评估所有回复。RLCF 用 instruction-specific 的动态维度，实验证明显著更有效
- **vs AutoIF / IFBench**：这些工作合成带有可验证约束的指令来训练模型。RLCF 不创造新指令，而是从已有自然指令中提取评估维度，泛化性更强
- **vs SPARKLE (2506.04723)**：SPARKLE 发现 RL 增强知识整合能力。RLCF 则揭示了 RL 反馈信号本身的质量问题——好的反馈不一定来自更好的 RM，而是来自更结构化的评估方式

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 将 reward 信号从单一标量重新定义为 instruction-specific 的 checklist 是重要范式转变；candidate-based 生成方法巧妙简洁
- 实验充分度: ⭐⭐⭐⭐⭐ 5 个 benchmark，6 个 baseline，跨模型族验证，RewardBench 对比，多个消融实验
- 写作质量: ⭐⭐⭐⭐⭐ Figure 1 和 Table 2-4 清晰展示只有 RLCF 一致性正收益；Table 8 的案例分析非常说服力
- 价值: ⭐⭐⭐⭐⭐ 揭示了 RM 在 RLHF 中的根本局限性，提供了可行的替代方案，WildChecklists 数据集有直接复用价值
