# Unveiling the Key Factors for Distilling Chain-of-Thought Reasoning

**会议**: ACL 2025
**arXiv**: [2502.18001](https://arxiv.org/abs/2502.18001)
**代码**: https://github.com/EIT-NLP/Distilling-CoT-Reasoning
**领域**: LLM Reasoning / CoT 蒸馏
**关键词**: Chain-of-Thought, Knowledge Distillation, Reasoning Granularity, Small Language Models, Teacher-Student

## 一句话总结
系统研究影响 CoT 蒸馏的三大因素（粒度、格式、教师模型），发现 SLM 与粒度呈非单调关系、格式影响较小、强教师不总是更好。

## 研究背景与动机
1. **领域现状**：CoT prompting 大幅提升 LLM 推理能力，但计算开销大，需将 CoT 能力蒸馏到小模型。
2. **现有痛点**：CoT 蒸馏中教师和生成方法的选择往往是随意的，缺乏系统性指导。
3. **核心矛盾**：对 LLM 有效的 CoT 策略（更细粒度、特定格式）是否同样适用于 SLM？
4. **本文要解决什么？** 什么是训练学生模型获得推理能力的最有效 CoT 监督？
5. **切入角度**：借鉴人类教学类比——教师选择、教学粒度、教学格式三个维度的系统实验。
6. **核心idea一句话**：CoT 蒸馏需要"因材施教"——针对学生模型能力定制粒度和教师选择。

## 方法详解

### 整体框架
4 个教师模型 × 7 个学生模型 × 6 个粒度等级 × 多种格式 × 7 个数据集，全面交叉实验。

### 关键设计
1. **粒度实验 (Granularity)**:
   - 做什么：设计 6 个粒度等级，从极简到极细的推理步骤
   - 核心思路：用 1-shot prompting 控制 GPT-4o 生成不同粒度的 CoT 标注
   - 设计动机：验证 SLM 是否像 LLM 一样从更细粒度中获益

2. **格式实验 (Format)**:
   - 做什么：测试自然语言、Least-to-Most、RaR 等不同推理格式
   - 核心思路：保持内容不变，仅改变推理链的呈现结构
   - 设计动机：探究 SLM 对推理格式的敏感度

3. **教师选择实验 (Teacher)**:
   - 做什么：对比 GPT-4o、Gemini-1.5-Flash、LLaMA-3-70B 和人类标注
   - 核心思路：固定粒度和格式，只变教师来源
   - 设计动机：验证"最强教师=最好学生"假设是否成立

### 损失函数 / 训练策略
标准 SFT：$\mathcal{L}_{distill} = \sum_i \mathcal{L}(S(x_i), \mathcal{C}_{T,g,f}(x_i) \oplus y_i)$

## 实验关键数据

### 主实验（粒度影响，GPT-4o 作教师）

| 模型 | GSM8K L1 | GSM8K L3 | GSM8K L5 | GSM8K 最佳 |
|------|---------|---------|---------|-----------|
| Gemma 2B | 49.66 | 53.37 | 53.42 | L5 (53.45) |
| LLaMA 3.2 3B | 59.59 | 62.57 | 62.29 | L4 (63.48) |
| BLOOM 3B | 18.20 | 23.81 | 22.47 | L3 (23.81) |

### 消融实验（粒度 vs 长度）

| 设置 | GSM8K Acc | 序列长度 |
|------|----------|---------|
| Level 1 | 47.61 | 100.93 |
| Level 1 + Padding | 46.62 | 143.43 |
| Level 5 | 52.92 | 138.16 |

### 关键发现
- **发现1**：SLM 与粒度呈非单调关系——强学生受益于细粒度，弱学生偏好中等粒度
- **发现2**：CoT 格式对 LLM 影响大，但对 SLM 影响小（因 SFT 适应能力）
- **发现3**：强教师不总能产出更好学生——人类标注准确率近完美但常逊于 LLM 生成的 CoT（多样性和复杂性更重要）

## 亮点与洞察
- 教学类比非常自然，三个维度（谁教、教什么、怎么教）清晰易懂
- 长度 vs 粒度的去耦实验设计巧妙
- "人类标注不一定最好"的发现反直觉但合理——LLM 标注的多样性补偿了准确率

## 局限性 / 可改进方向
- 仅用 1-shot prompting 生成标注，其他生成策略可能改变结论
- 未探讨混合粒度（对简单题低粒度、难题高粒度）的策略
- 学生模型最大只到 3B，7B+ 模型的行为可能不同

## 相关工作与启发
- **vs Magister et al. (2023)**: 早期 CoT 蒸馏工作未研究粒度因素
- **vs Zong et al. (2023)**: 他们认为强教师总是更好，本文证伪


## 补充细节
- 教师模型：GPT-4o、Gemini-1.5-Flash、LLaMA-3-70B、人类标注
- 学生模型：BLOOM (560M/1.1B/1.7B/3B)、Gemma 2B、LLaMA 3.2 (1B/3B)
- 数学数据集：SVAMP、GSM8K、AQuA-RAT、MATH
- 常识推理数据集：CommonsenseQA、OpenBookQA、StrategyQA
- 粒度从 Level 1 到 Level 6，通过 prompt 控制
- CoT 格式包括：原始 CoT、Least-to-Most、RaR
- Padding 实验证明纯长度增加无用
- 核心结论：SLM 需要因材施教的 CoT 蒸馏策略
- 人类标注虽然准确率高但多样性不如 LLM 生成
- Only Answer baseline 可揭示学生模型的预训练隐式知识
- 较弱学生在复杂数据集上表现接近随机猜测
- 用 1-shot prompting 确保粒度控制的一致性

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个系统研究 CoT 蒸馏三大因素的工作
- 实验充分度: ⭐⭐⭐⭐⭐ 4 教师 × 7 学生 × 7 数据集 × 6 粒度，实验量巨大
- 写作质量: ⭐⭐⭐⭐ 教学类比框架清晰，结论可操作
- 价值: ⭐⭐⭐⭐ 为 CoT 蒸馏提供了实用的配置指南
