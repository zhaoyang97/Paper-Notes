# Align to Structure: Aligning Large Language Models with Structural Information

**会议**: AAAI 2026  
**arXiv**: [2504.03622v2](https://arxiv.org/abs/2504.03622v2)  
**代码**: [https://github.com/minnesotanlp/struct_align](https://github.com/minnesotanlp/struct_align) (有)  
**领域**: llm_nlp  
**关键词**: 结构对齐, 篇章结构, PPO强化学习, 长文本生成, 密集奖励  

## 一句话总结
提出 Structural Alignment 方法，通过将语言学篇章结构框架（表层文本结构评分 + 基于RST的篇章motif分类器）融入PPO强化学习训练，并设计基于篇章motif的密集奖励机制，使LLM生成更连贯、更具人类写作风格的长文本，在论文写作和长文档摘要任务上均优于标准RLHF模型。

## 背景与动机
- 现有LLM对齐方法（RLHF/RLAIF）主要关注"有用性"和"无害性"，即人类*偏好什么*，但忽略了人类*如何组织*连贯的篇章结构
- 人类写作者天然使用"问题-解决"、"因果"、"描述"等文本结构来保持局部连贯和逻辑流，而LLM缺乏这种层次化规划能力
- 长文本生成（>1000 token）面临主题漂移、浅层衔接、缺乏修辞一致性等问题，标准RLHF训练中稀疏的episodic奖励导致信用分配困难、训练不稳定
- 现有长文本评估困难，缺大规模人工标注数据集，且主观性强

## 核心问题
如何让LLM不仅在内容层面对齐人类偏好，还能在**篇章结构层面**对齐人类写作惯例，从而生成逻辑连贯、组织良好的长文本？具体挑战包括：(1) 如何量化篇章结构质量并转化为可训练的奖励信号；(2) 如何在长序列PPO训练中解决稀疏奖励带来的不稳定性。

## 方法详解
### 整体框架
在RLAIF+PPO框架下，以Qwen2-1.5B-Instruct作为策略模型，探索两条互补的结构对齐路径：
1. **Surface-Level Text Structure Scoring (SA_S)**：利用外部大模型（Qwen2-72B-Instruct-AWQ）作为评估器，对生成文本进行表层结构打分
2. **Graph-Level Discourse Structure Scoring (SA_G)**：基于RST篇章分析提取层次化篇章motif，训练作者身份分类器作为奖励模型

此外还引入密集奖励机制和长度惩罚归一化，增强长文本PPO训练稳定性。

### 关键设计
1. **Surface-Level Text Structure Evaluator（表层结构评估器）**：使用Qwen2-72B作为RLAIF评判器，从三个维度给生成文本打0-5分：(a) Logical Flow and Structure（逻辑流与结构）——评估思路是否逻辑推进、整体组织是否连贯；(b) Hierarchical Organization（层次组织）——评估内容是否从一般到具体有层次；(c) Balance and Emphasis（平衡与重点）——评估关键观点是否突出、不同论点是否均衡覆盖。最终奖励为三项评分的均值。

2. **Graph-Level Discourse Structure Scorer（图级篇章结构评分器）**：基于Kim et al. (2024)的方法，使用DMRST解析器将文本解析为RST篇章树，转化为递归超图并提取discourse motif（篇章motif）——即区分人类写作和机器生成文本的结构特征模式。将motif分布与Longformer的[CLS]嵌入拼接，输入二分类器判断文本是否"像人类写的"，分类器输出作为奖励信号。由于RST解析器限制约512 token输入，对长文本按段落分段（每段400-512 token），对非重叠段分别解析后聚合motif分布。

3. **Dense Reward Shaping（密集奖励塑形）**：标准PPO仅在序列末尾给一个episodic奖励，长序列中信用分配困难。本文在episodic奖励之外，额外为属于"人类特征篇章motif"的token分配 $\frac{1}{2 \cdot \text{num\_tokens}}$ 的奖励。特征motif通过MF-IDF（Motif Frequency-Inverse Document Frequency）在语料库级别识别，选取超过一个标准差阈值的motif。以EDU（基本篇章单元，平均约20个subword token）为粒度计算。

4. **Length-Penalty Normalization（长度惩罚归一化）**：针对LLM常常不遵循目标长度的问题，当生成长度不足时按比例惩罚：$S_n = S_o \times [1 - \alpha \times \max(0, \frac{L_d - L_r}{L_d})]$，训练过程中逐步增加目标生成长度。

### 损失函数 / 训练策略
- 使用标准PPO目标函数（clipped surrogate objective），KL系数0.03
- 训练配置：per-device batch size 2，gradient accumulation 4，local rollout forward batch size 12
- 生成序列长度400-2K token，训练中逐epoch递增目标长度
- SA_S使用8个A100 GPU运行SGLang服务Qwen2-72B-AWQ评估器实例，另外8个A100做PPO训练
- 还尝试了两阶段训练：先SA_S再SA_G（SA_S→G）或反向（SA_G→S），利用两种对齐的互补性

## 实验关键数据
| 任务/数据集 | 指标 | SA_G (图级) | SA_S (表层) | Base (Qwen2-1.5B) | RLHF_OA | 说明 |
|------------|------|-----------|-----------|-------------------|---------|------|
| GovReport摘要 | ROUGE-1 | **55.86** | 55.45 | 53.21 | 53.25 | 图级对齐最优 |
| GovReport摘要 | ROUGE-2 | **21.72** | 21.43 | 20.13 | 20.25 | +1.59 vs Base |
| GovReport摘要 | ROUGE-L | **52.81** | 52.30 | 50.39 | 50.47 | +2.42 vs Base |
| Essay生成 | GPT-4o Pairwise | SA_G > SA_S > RLHF > Base | - | - | - | 图级对齐在pairwise评估中持续胜出 |

- Text Structure Evaluator验证：与人类评分的Pearson相关性——coherence 0.47, organization 0.44, balance 0.39, purpose 0.37, variability 0.39；MSE 0.79（0-4范围）

### 消融实验要点
- **两阶段训练**：SA_S→G和SA_G→S略优于单独SA_S或SA_G，但提升幅度相对额外计算成本较小
- **表层 vs 图级互补性**：两种评分之间相关性很低（散点图近乎无相关），说明它们捕获不同层面的结构信息，互补性强
- **标准RLHF的不足**：使用OpenAssistant奖励模型的RLHF训练不仅未提升篇章结构质量，生成文本的人类特征motif比例甚至呈下降趋势
- **密集奖励效果**：图级对齐训练中，人类特征motif比例在约50个batch step后达到稳定的高水平；而标准PPO中motif比例持续下降
- **表层对齐效果**：生成文本中出现更多篇章连接词（therefore, however, moreover等）、更清晰的段落标题（Introduction, Claim 1, Conclusion等）、Joint和Hyperedge等层次关系motif显著增加

## 亮点
- **新颖的对齐目标**：不对齐人类"偏好"而是对齐人类"写作结构"，将篇章语言学理论引入LLM对齐，视角新颖
- **密集奖励设计巧妙**：基于篇章motif的token级奖励塑形，利用语言学结构的天然粒度解决长序列PPO信用分配问题，简单有效
- **两种互补评分方法**：表层评分提升可读性和显式结构化，图级评分强化深层连贯性和修辞复杂度，两者低相关且互补
- **长度惩罚归一化**：简洁地解决了LLM不遵循目标长度的问题
- **RST分段扩展**：通过段落级分段和motif聚合，将512 token限制的RST解析器扩展到处理长文本

## 局限性 / 可改进方向
- 仅在正式essay写作和摘要任务上评估，未涉及创意写作、对话等非正式场景（这些场景的篇章结构更松散且难以形式化）
- 策略模型仅用1.5B参数的Qwen2，未验证对更大模型的效果（受计算资源限制）
- RST解析器限制单次输入约512 token，需要分段处理，可能割裂全局连贯模式
- 仅使用RST一种篇章理论，SDRT等其他形式体系可能捕获更细微的关系模式
- Essay质量评估依赖GPT-4o作为judge，缺乏大规模人工评估
- 多奖励对齐策略仅尝试了简单的两阶段串联，更先进的多奖励方法（如Pareto优化）可能进一步提升

## 与相关工作的对比
- **vs 标准RLHF（OpenAssistant RM）**：标准RLHF仅优化通用人类偏好，不强化篇章组织；实验中RLHF模型在essay生成和摘要任务上与Base差异甚微，且human-distinctive motif占比甚至下降
- **vs Kim et al. (2024) Discourse Motifs**：该工作发现篇章motif可区分人类与AI生成文本（ACL 2024），本文创新性地将其从检测工具转化为RL训练的奖励信号和密集奖励源
- **vs Dense Reward方法（Chan et al., 2024; Cao et al., 2024）**：这些工作提出了一般性的密集奖励方法，本文的贡献在于利用语言学结构（篇章motif + EDU粒度）提供领域感知的密集奖励，更适合长文本生成场景

## 启发与关联
- 篇章结构对齐的思路可迁移到其他需要长文本输出的场景，如学术论文写作、报告生成、代码文档生成等
- 密集奖励设计利用了任务本身的结构特征（篇章树），这种"领域感知的奖励塑形"思路具有通用性——可以借鉴到其他具有内在结构信息的任务中
- 两种互补评分方法的低相关性发现暗示：在多目标对齐中，选择正交维度的奖励信号可能比同质化的多奖励更有价值
- 将原本用于"检测AI文本"的分类器反向用于"训练AI生成更像人类"，这种逆向应用思路值得关注

## 评分
- 新颖性: ⭐⭐⭐⭐ 将篇章语言学理论引入LLM对齐是新颖视角，但基础框架（PPO+RLAIF）较为标准
- 实验充分度: ⭐⭐⭐ 仅用1.5B模型，仅评估essay和摘要两个任务，缺少大规模人工评估，下游任务多样性不足
- 写作质量: ⭐⭐⭐⭐ 论文结构清晰，方法描述详细，语言学动机阐述充分
- 价值: ⭐⭐⭐⭐ 提出了从"对齐偏好"到"对齐结构"的新范式，密集奖励设计可复用，但实际应用影响受限于模型规模和场景覆盖

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 对我的价值: ⭐⭐⭐
