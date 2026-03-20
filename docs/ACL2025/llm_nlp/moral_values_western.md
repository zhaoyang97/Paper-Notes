# Comparing Moral Values in Western English-speaking Societies and LLMs with Word Associations

**会议**: ACL 2025  
**arXiv**: [2505.19674](https://arxiv.org/abs/2505.19674)  
**代码**: [https://github.com/ChunhuaLiu596/Word_Association_Generation](https://github.com/ChunhuaLiu596/Word_Association_Generation)  
**领域**: AI Safety / Moral Alignment  
**关键词**: moral values, word association, Moral Foundation Theory, mental lexicon, LLM alignment  

## 一句话总结

提出通过词语联想（word association）而非直接提问来比较 LLM 与西方英语社会的道德价值观，发现 LLM 在正面道德维度上与人类更一致，但在情感多样性和具体性上存在系统性差异。

## 研究背景与动机

随着 LLM 日益广泛部署，理解其反映的道德价值观变得越来越重要。但直接用道德问卷提示 LLM 面临多重问题：（1）道德问卷可能泄露到 LLM 训练数据中（Abdulhai et al., 2023）；（2）LLM 对提示措辞敏感，回答不稳定（Almeida et al., 2024）；（3）二元回答（同意/不同意）无法捕捉细微的道德推理；（4）LLM 的 next-token prediction 本质使得直接评估道德判断本身就存在概念困难。作者提出利用词语联想测试——一种在心理学中已被证明能反映道德推理的低层次认知范式——来更鲁棒地探测 LLM 的道德概念组织方式。词语联想的优势在于它不直接询问道德立场，而是通过概念关联间接推断。

## 方法详解

### 整体框架

两阶段框架：

1. **词语联想图构建**：从人类和 LLM 分别收集 12K cue 词的联想响应，构建两个独立的联想图。
   - 人类联想图（wa-h）：来自 Small World of Words 数据集
   - LLM 联想图（wa-l）：通过提示 Llama 生成
2. **道德信息传播**：基于 Moral Foundation Theory (MFT) 的 626 个种子词，通过随机游走在联想图中传播 5 维道德值，得到两个 Global Moral Network（gmn-h 和 gmn-l），然后进行系统性对比分析。

### 关键设计

- **词语联想采集**：使用 Llama-3.1-8B-Instruct 模型，对 12K cue 每个提示 100 次（Monte-Carlo 近似），每次生成最多 3 个联想词。采用与人类实验 (Small World of Words, ~90K 参与者) 完全相同的指令，确保实验的可比性。
- **温度调优**：平衡多样性（response types 种类数）和可靠性（split-half reliability），最终选定 temperature=2.1 使 wa-l 的多样性和可靠性均趋近 wa-h。
- **Global Moral Network**：用带归一化的随机游走传播道德信息：F_{t+1} = αSF_t + (1-α)F_0，其中 S 是对称归一化邻接矩阵，F_0 包含 MFD 种子词的 5 维道德值（Care, Fairness, Loyalty, Authority, Sanctity）。
- **Alpha 优化**：gmn-h 最佳 α=0.75，gmn-l 最佳 α=0.9。人类图需要更小的 α（信息传播更容易），因为人类图直径更小（3 vs 4）、密度更高（0.013 vs 0.007）、连通性更强（114 vs 77）。这意味着 LLM 的概念网络更稀疏，需要更强的传播力才能将道德信息传递到远处节点。
- **评估方式**：使用 eMFD（众包扩展版道德词典，包含 2186 个评估词）的 Spearman 相关系数评估，对比 MAG baseline。为避免数据泄露，从评估集中划出 277 个词用于调参。

### 损失函数 / 训练策略

本文不涉及模型训练。使用已有的 Llama-3.1-8B-Instruct（15T token 预训练 + RLHF），关键超参数：
- 联想生成温度 T=2.1
- 传播参数 α（gmn-h: 0.75, gmn-l: 0.9）

## 实验关键数据

### 主实验

**道德值预测（Spearman 相关）与 eMFD 对比：**

| 道德维度 | MAG (baseline) | gmn-h (人类图) | gmn-l (LLM图) |
|---------|----------------|---------------|---------------|
| Care (n=1895) | 0.29 | **0.47** | 0.46 |
| Sanctity (n=1893) | 0.25 | 0.39 | **0.44** |
| Fairness (n=1514) | 0.23 | 0.29 | **0.32** |
| Authority (n=1737) | 0.21 | 0.19 | **0.25** |
| Loyalty (n=1714) | 0.30 | 0.26 | **0.30** |
| 总体 (n=8753) | 0.20 | 0.28 | **0.29** |

gmn-l 在 4/5 维度上优于 gmn-h，gmn-h 仅在 Care 上略优。两者均大幅超越 MAG baseline。

**联想重叠度**：wa-l 在 top-1 联想上与 wa-h 几乎完全一致（Precision@1 ≈ 100%），top-10 保持约 80% 的精度，优于 Word2Vec baseline。Word2Vec 在小 k 时精度尤其差，说明联想任务不等同于简单的语义相似度。平均关联强度相关性也由 wa-l 显著优于 Word2Vec。

### 关键发现

1. **正面道德一致性高于负面**：LLM 与人类在正面道德概念（如 church, religion, God 等）上高度一致，但在负面道德概念上出现分歧（如人类产生 "betrayal, cheating" 等更情感化的响应，LLM 产生 "prejudice, discrimination" 等更抽象的响应）。
2. **人类联想更具体、更情感化**：例如 "prejudice" 人类联想 "pride, black, race, racist"（具体种族关联），LLM 联想 "stereotypes, biases, stereotyping, bigoted"（抽象概念关联）。
3. **系统性差异概念揭示价值取向**：LLM 将 "abortion, immigrant, politician" 等概念评为更正面，而人类将 "jail, air, plastic" 等评为更正面，反映出 LLM 可能在训练中吸收了特定的社会价值倾向。
4. **全局传播优于局部方法**：基于全局图传播的 gmn 相比 MAG 的局部子图方法，更好地捕获了远距离的道德关联，相关性提升 0.08-0.18。
5. **LLM 联想更抽象**：人类用具体实例联想道德概念（如 "church" → "catholic, synagogue, stone"），LLM 倾向产生范畴/功能性联想（如 "church" → "altar, minister, baptism, service"），反映不同的概念组织方式。

## 亮点与洞察

- **方法论创新**：将词语联想这一心理学经典范式应用于 LLM 道德评估，巧妙回避了直接提问的各种偏差问题（训练数据泄露、提示敏感性、社会期望偏差）。
- **全局道德网络**是对 MAG 局部方法的重要改进，能够捕获更复杂的远距离道德推理，相关性平均提升 0.08-0.09。
- **温度调优方法**：同时优化多样性和可靠性两个指标，确保 LLM 联想数据与人类数据具有可比性，这一方法论贡献可推广到其他需要 LLM 模拟人类行为的研究。
- **图结构差异有解释力**：人类图更密集→传播更容易→需要更小的 α，这一发现揭示了人类和 LLM 概念组织方式的结构性差异。
- **定性分析揭示深层差异**：如 LLM 将 "abortion" 和 "immigrant" 评为更正面，而人类图中这些概念的道德判断更中性或负面，暗示 RLHF 训练可能引入了特定的价值倾向。
- **间接探测比直接提问更可靠**：避免了"社会期望效应"（LLM 倾向于给出社会可接受的答案），为 AI 对齐研究提供了更真实的道德画像。

## 局限性

- 仅使用 Llama-3.1-8B-Instruct 一个模型，未验证不同 LLM（GPT、Claude 等）的差异。
- 聚焦于西方英语文化，不能推广到其他文化和语言。
- MFT 本身作为道德框架存在争议（如 fairness 维度的拆分问题），且种子词数量有限（626 个）。
- 联想实验的 Monte-Carlo 近似（100 次重复）是否充分未严格验证。
- 随机游走传播可能受图中 hub 节点（高度连接的通用词）影响，这些词可能稀释道德信号。
- Precision@k 在 k>10 后下降明显，说明 LLM 的联想长尾分布与人类仍有差距。

## 相关工作

- **Moral Foundation Theory**：Graham et al. (2013) 五维道德框架；Frimer et al. (2017) MFD 字典；Hopp et al. (2021) eMFD 众包扩展。
- **词语联想研究**：Small World of Words (De Deyne et al., 2019)；Ramezani & Xu (2024) 的 MAG 用联想推断道德值。
- **LLM 道德评估**：Ji et al. (2024) 发现 LLM 对道德的理解是表面的；Scherrer et al. (2023) 道德问卷评估；Abdulhai et al. (2023) 发现问卷泄露问题。
- **LLM 词语联想**：Abramski et al. (2024) 发现 LLM 联想多样性显著低于人类。

## 评分

- **新颖性**: 5/5 — 词语联想 + 全局道德网络的方法论组合非常新颖
- **技术深度**: 4/5 — 随机游走传播、温度调优、图结构分析等技术扎实
- **实验充分性**: 3/5 — 仅一个 LLM，跨模型泛化性未知
- **实用价值**: 4/5 — 为 LLM 道德对齐提供了新的评估范式
- **推荐指数**: ⭐⭐⭐⭐
