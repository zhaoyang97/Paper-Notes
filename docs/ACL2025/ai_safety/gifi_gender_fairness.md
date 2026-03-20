# Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework

**会议**: ACL 2025  
**arXiv**: [2506.15568](https://arxiv.org/abs/2506.15568)  
**代码**: [https://github.com/ZhengyangShan/GIFI](https://github.com/ZhengyangShan/GIFI)  
**领域**: AI 安全 / 公平性评估  
**关键词**: gender fairness, non-binary pronouns, LLM evaluation, inclusivity index, neopronouns  

## 一句话总结
提出 GIFI（Gender Inclusivity Fairness Index），一个多层次综合评估框架，涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和推理性能一致性七个维度，在 22 个 LLM 上系统评估二元与非二元性别的公平性。

## 研究背景与动机
- LLM 的快速发展推动了 NLP 进步，但引发了公平性关切，尤其是性别表达方面
- 现有性别偏见研究主要聚焦于**二元性别（男/女）**，忽视了**非二元性别身份**
  - 大多数研究仅评估 "he" 和 "she" 的差异
  - 非二元代词（they/them, 新代词如 xe, ze, ae 等）的表现几乎未被评估
- 缺乏一个**综合性、可解释的指标**来量化 LLM 对多元性别的包容性
- 已有数据集（如 StereoSet, CrowS-Pairs）不专门针对非二元代表或经验
- 核心贡献：从简单的代词识别到深层的认知推理，建立多层次递进评估框架

## 方法详解

### 整体框架
GIFI 由四个递进阶段、七个评估维度组成：
1. **代词识别（Pronoun Recognition）**：GDR
2. **分布公平性（Fairness in Distribution）**：SN, NTS, CF
3. **刻板印象与角色分配（Stereotype and Role Assignment）**：SA, OF
4. **性能一致性（Consistency in Performance）**：PE

最终 GIFI = 七个指标的平均值 × 100，范围 0-100。

### 关键设计

#### 性别身份覆盖
- 覆盖 **11 个代词组**：
  - 二元：he/him/his, she/her/hers
  - 中性：they/them/their
  - 新代词（8 组）：thon, e/em, ae/aer, co, vi/vir, xe/xem, ey/em, ze/zir
- 比现有研究的代词覆盖范围大幅扩展

#### 七个评估维度详解

**1. Gender Diversity Recognition (GDR)**
- 测试模型能否识别输入中的性别代词并在输出中正确使用
- 对每组代词构建包含该代词的 prompt，检查输出中代词的一致性
- 使用变异系数（CV）衡量跨代词组的一致性：GDR = 1/(1+CV)
- 范围 [0,1]，1 表示对所有代词组的识别完全一致

**2. Sentiment Neutrality (SN)**
- 评估更换性别代词是否影响模型输出的情感
- 使用 RoBERTa-base 情感分类器评分
- SN = 1 - Average MAD（跨代词组的平均绝对偏差）

**3. Non-Toxicity Score (NTS)**
- 评估模型对不同性别代词的毒性输出差异
- 使用 Google Perspective API 评分
- NTS = 1 - Average MAD

**4. Counterfactual Fairness (CF)**
- 仅改变代词的配对输入，比较输出的语义相似度
- 使用 sentence-transformers 编码，余弦相似度低于阈值 γ=0.3 则判定为"实质不同"
- CF = 不实质不同的配对占比

**5. Stereotypical Association (SA)**
- 无性别提示的 prompt（含刻板印象活动/性格/偏好），分析模型输出生成的代词分布
- 公平分数 = 1 - 代词分布与均匀分布的平方偏差
- 排除 "they" 以避免高频默认代词的干扰

**6. Occupational Fairness (OF)**
- 类似 SA，但使用职业相关 prompt
- 选取 40 个男性主导 + 40 个女性主导职业
- 同样排除 "they"

**7. Performance Equality (PE)**
- 使用 GSM8K 数学推理题，替换人名为不同代词
- 评估模型数学推理能力是否因代词不同而变化
- 使用 8-shot CoT prompting
- PE = 1/(1+CV)

### 损失函数 / 训练策略
本文是纯评估框架，不涉及模型训练。所有指标设计为 [0,1] 区间，高分表示更公平。

## 实验关键数据

### 评估模型（22 个）
- 开源：LLaMA 2/3/4, Vicuna, Mistral, Gemma 2/3, GPT-2, Zephyr, Yi-1.5, Qwen 3, DeepSeek V3, Phi-3
- 闭源：GPT-4/4o/4o-mini/3.5-turbo, Claude 3 Haiku/4 Sonnet, Gemini 1.5 Flash/Pro/2.0 Flash

### 数据集构建
- GDR：改编 TANGO 数据集，2200 个 prompt
- SN/NTS/CF：Real-Toxicity-Prompts 子集，2200 个样本
- SA/OF：模板化数据集，80 个职业
- PE：GSM8K 子集 + 代词替换，1100 个样本

### GIFI 总排名（Top-5 和 Bottom-5）
| 排名 | 模型 | GIFI 分数 |
|------|------|-----------|
| 1 | GPT-4o | 最高 |
| 2 | Claude 3 | 第二 |
| 3 | DeepSeek V3 | 第三 |
| ... | ... | ... |
| 倒数 | Vicuna, GPT-2, LLaMA 2 | 最低 |

### 各维度详细结果

#### 代词识别 (GDR)
- **Claude 4 最优**（均值 0.75），GPT-4o (0.73)，GPT-4 (0.65)
- Zephyr (0.19), GPT-2 (0.22) 表现最差
- Gemini 1.5 Pro 出奇地低：约 50% 生成完全不包含代词
- Claude 4, LLaMA 3, Phi-3 的跨代词方差最低（最一致）

#### 情感/毒性
- 所有模型整体都表现出较强的中立性和低毒性
- GPT-4o mini, GPT-4, Gemini 1.5 Pro, Claude 4 情感中立性最高
- Claude 3/4 毒性最低
- GPT-2, Phi-3 毒性分布有长尾

#### 刻板印象关联 (SA)
- **几乎没有模型生成新代词**：所有模型在无性别提示时不生成 ne/xe/ze 等
- Phi-3 最均衡：he (0.34), she (0.31), they (0.34)
- GPT-4o 和 LLaMA 4 强烈偏好 "she"（0.86 和 0.83），可能是去偏见训练的过度矫正
- "they" 使用率很少超过 30%

#### 职业公平性 (OF)
- Claude 4 严重偏向 "she" (0.72) vs "he" (0.26) vs "they" (0.02)
- Gemini 1.5 Pro/Flash, Qwen 3, Mistral 相对均衡
- 多数模型 "they" 占比不超过 10%

#### 数学推理公平性 (PE)
- Gemini 2.0 Flash 和 DeepSeek V3 最高准确率 (0.92)
- Claude 4 (0.85), GPT-4o (0.80) 紧随其后
- 强模型对所有代词（包括新代词）表现一致，弱模型则全面失败（非偏见导致）

### 关键发现
1. **新代词在无提示时完全缺席**：所有 22 个模型在刻板印象/职业任务中从不自发生成新代词
2. **"she" 过度矫正现象普遍**：为减少历史上的男性偏见，许多新模型过度偏向女性代词
3. **表面公平性 ≠ 深层包容性**：虽然 she 使用增加，但 they 仍不足，新代词完全缺失
4. **推理公平性主要取决于模型能力**：强模型对所有代词公平（因为能力强），弱模型对所有代词同样失败（因为能力不足）
5. **不同维度的表现可能不一致**：如 Claude 4 代词识别最优，但刻板印象关联较差

## 亮点与洞察
1. **首个覆盖非二元性别的综合公平性指标**：填补了重要的研究空白
2. **多层次递进设计合理**：从浅层（代词识别）到深层（推理能力一致性），逐步揭示偏见
3. **评估规模空前**：22 个模型 × 7 个维度 × 11 个代词组
4. **发现了去偏见训练的"过度矫正"问题**：值得模型开发者关注
5. **PE 维度（数学推理）的设计有洞察力**：表面上与性别无关的任务也可能受代词影响

## 局限性 / 可改进方向
1. **仅覆盖英语**：不同语言的性别系统（如语法性别语言）需要适配
2. **新代词集合不完整**：代词系统持续演变，框架需要支持扩展
3. **外部工具（情感分类器、毒性 API）本身可能有偏见**
4. **数据污染风险**：部分评估数据集（如 RealToxicityPrompts）可能已被模型训练数据覆盖
5. **缺少交叉性分析**：未考虑性别与种族、残疾等其他维度的交叉偏见
6. **模型输出的随机性影响可复现性**：即使设置了温度和 top-p
7. **GIFI 使用简单平均聚合七个指标**：各维度的权重可能不同

## 相关工作与启发
- **二元性别偏见**：Bolukbasi et al. (2016) 词嵌入偏见, StereoSet, CrowS-Pairs
- **非二元性别研究**：MISGENDERED (Hossain et al., 2023), Ovalle et al. (2023) 跨性别错误指代
- **LLM 公平性**：GPT-3 性别偏见 (Brown et al., 2020), Chen et al. (2022) 角色偏见框架
- **启发**：(1) 可扩展到多语言和多文化场景；(2) 可将 GIFI 整合到模型训练的评估 pipeline 中作为自动化公平性检查；(3) 可设计针对新代词的对齐训练数据；(4) 过度矫正问题值得专门研究

## 评分
- **新颖性**: ⭐⭐⭐⭐ — 首个涵盖非二元性别的综合 LLM 公平性指标
- **技术深度**: ⭐⭐⭐⭐ — 七维度指标设计合理，数学定义严谨，均归一化到 [0,1]
- **实验充分度**: ⭐⭐⭐⭐⭐ — 22 个模型、7 个维度、全面的定性和定量分析
- **实用价值**: ⭐⭐⭐⭐⭐ — 直接可用于 LLM 公平性审计和基准测试
- **写作质量**: ⭐⭐⭐⭐ — 结构清晰，图表丰富
- **综合评分**: 8.5/10
