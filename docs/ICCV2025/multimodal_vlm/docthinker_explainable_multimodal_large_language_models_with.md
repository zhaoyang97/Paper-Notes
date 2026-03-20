# DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding

**会议**: ICCV 2025  
**arXiv**: [2508.08589](https://arxiv.org/abs/2508.08589)  
**代码**: [https://github.com/wenwenyu/DocThinker](https://github.com/wenwenyu/DocThinker)  
**领域**: 多模态VLM / 文档理解 / 强化学习推理  
**关键词**: rule-based RL, GRPO, document understanding, explainability, chain-of-thought  

## 一句话总结
提出DocThinker，首个将GRPO（Group Relative Policy Optimization）强化学习应用于文档理解的框架，通过四目标规则奖励（格式、答案准确度、RoI IoU、问题改写质量）训练MLLM自主生成可解释的推理过程，仅用4K训练数据在DocVQA上将Qwen2.5-VL-7B从0.355提升到0.579（RL vs SFT: 0.579 vs 0.355），并在视觉定位任务上达到82.4%精度。

## 研究背景与动机
1. **领域现状**：MLLM在文档理解上表现出色，但推理过程是黑盒的，在法律/金融/医疗等高风险领域缺乏可信度。现有方法主要用固定的CoT模板进行推理，如ReFocus用外部工具编辑图像、Visual CoT做多轮处理、MVoT生成交错的视觉-文本推理链。
2. **现有痛点**：(1) 固定CoT模板不够灵活，跨任务泛化差；(2) SFT训练容易遗忘（catastrophic forgetting），在新文档类型上表现下降；(3) 现有方法只输出最终答案，缺少中间推理过程的可解释性。
3. **核心矛盾**：SFT让模型记住训练数据中的推理模式，但无法自主探索更好的推理策略。DeepSeek-R1等工作已证明纯RL可以激发模型的涌现推理能力，但RL在文档理解领域几乎未被探索。
4. **本文要解决什么**：用RL替代SFT来训练文档理解MLLM，让模型自主学习灵活的推理策略，同时生成可解释的中间步骤（推理过程、改写问题、感兴趣区域、最终答案）。
5. **切入角度**：受DeepSeek-R1和MedVLM-R1启发，将GRPO应用于多模态文档理解，设计四个可验证的规则奖励函数来引导模型学习。
6. **核心idea一句话**：用GRPO强化学习+四目标规则奖励替代SFT来训练文档理解MLLM，实现自适应推理和可解释输出。

## 方法详解

### 整体框架
基于Qwen2.5-VL（3B/7B），输入文档图像+问题，模型生成结构化输出：`<think>推理过程</think><answer>{"rephrase_question": ..., "bbox_2d": ..., "final_answer": ...}</answer>`。用GRPO算法采样G=6个候选输出，四个规则奖励评估后计算组内相对优势，通过KL约束优化策略。

### 关键设计

1. **四目标规则奖励函数**:
   - **格式奖励 $R_{\text{format}}$**：检查输出是否遵循XML-style schema（`<think>...</think>`和`<answer>...</answer>`），JSON是否有效且包含所需键值对。二值奖励。
   - **准确度奖励 $R_{\text{accuracy}}$**：最终答案是否与ground truth匹配。二值奖励。
   - **RoI IoU奖励 $R_{\text{RoI}}$**：预测的bounding box与ground truth的IoU是否≥0.5。鼓励模型精确定位文档中与答案相关的区域，增强视觉可解释性。
   - **问题改写奖励 $R_{\text{rephrase}}$**：评估改写后问题的语义相似度和新词多样性。只有答案正确时才计算（避免奖励错误答案的好改写）。
   - 总奖励：$R_{\text{total}} = \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{accuracy}} + \lambda_3 R_{\text{RoI}} + \lambda_4 R_{\text{rephrase}}$，所有$\lambda_i=1$。
   - 设计动机：每个奖励信号都可自动验证（不需要人工标注偏好），且分别优化不同方面（格式规范、答案准确、视觉定位、问题理解）。

2. **GRPO训练策略（替代PPO）**:
   - 做什么：对每个问题采样G=6个候选输出，用规则奖励评估后归一化为组内相对优势$A_i = (r_i - \text{mean}) / \text{std}$，优化策略使高优势回复的概率增大。
   - 核心思路：不需要额外的critic网络（PPO需要），直接用组内相对比较来估计优势。KL散度正则化$\beta=0.04$防止策略偏离参考模型太远，缓解灾难性遗忘。
   - 设计动机：GRPO计算效率高，不需要训练value network，且已在DeepSeek-R1中被验证能激发涌现推理能力。

3. **结构化可解释输出**:
   - 做什么：模型输出包含四个部分——`<think>`中的自由推理过程 + `rephrase_question`（改写后的清晰问题）+ `bbox_2d`（支持答案的文档区域坐标）+ `final_answer`。
   - 核心思路：推理过程让人了解模型"怎么想"，改写问题让人了解模型"怎么理解问题"，bbox让人看到模型"看的是哪里"——三个维度的可解释性。
   - 设计动机：比单纯输出答案有更丰富的可解释信息，且RL训练中推理过程会不断优化（模型自我反思和修正），这是VoT等方法无法做到的。

### 训练配置
- 基于Qwen2.5-VL 3B/7B，8×A100 80G
- 训练数据：Visual CoT数据集中4K或8K样本（远少于VisCoT的438K）
- 训练2个epoch，lr=1e-6，AdamW
- 输入分辨率：336²或1536²

## 实验关键数据

### 主实验（Visual CoT Benchmark）

| 模型 | 策略 | 数据量 | DocVQA | TextVQA | InfoQA | GQA | VSR |
|------|------|--------|--------|---------|--------|-----|-----|
| Qwen2.5VL-7B (原始) | - | - | 0.350 | 0.735 | 0.325 | 0.455 | 0.616 |
| Qwen2.5VL-7B* | SFT | 4K | 0.355 | 0.740 | 0.334 | 0.467 | 0.619 |
| **DocThinker-7B** | **RL** | **4K** | **0.579** | **0.802** | **0.347** | **0.546** | **0.656** |
| VisCoT-7B | SFT | 438K | 0.476 | 0.775 | 0.324 | 0.631 | 0.614 |
| DocThinker-7B (高分辨率) | RL | 4K | 0.795 | 0.827 | 0.689 | 0.694 | 0.721 |

视觉定位（TextREC）:

| 模型 | Precision@1 |
|------|------------|
| TAMN | 80.8% |
| MDETR | 63.3% |
| **DocThinker-7B** | **82.4%** |

### 消融实验

| 配置 | DocVQA | TextVQA | InfoQA | 说明 |
|------|--------|---------|--------|------|
| DocThinker-7B (Full) | 0.795 | 0.827 | 0.689 | 完整模型 |
| w/o RoI IoU | 0.775 | 0.803 | 0.637 | 去掉视觉定位奖励，InfoQA下降明显 |
| w/o Rephrase | 0.763 | 0.772 | 0.658 | 去掉问题改写奖励，TextVQA下降明显 |
| w/o both | 0.741 | 0.758 | 0.602 | 两个都去掉，下降最大 |
| w/o KL ($\beta=0$) | 0.780 | 0.803 | 0.676 | 去掉KL约束，性能下降 |

### 关键发现
- **RL显著优于SFT**：同样4K数据，RL (DocThinker) 在DocVQA上0.579 vs SFT 0.355，提升63%。说明RL能让模型学到更灵活的推理策略而非记忆训练样本。
- **数据效率极高**：仅4K数据RL训练的DocThinker-7B在多个指标上超过用438K数据SFT训练的VisCoT-7B（如TextVQA 0.802 vs 0.775）。
- **RoI IoU和Rephrase Question奖励互补**：RoI IoU更影响需要精确视觉定位的任务（InfoQA），Rephrase更影响需要理解问题的任务（TextVQA/DocVQA）。
- **KL约束对稳定训练至关重要**：$\beta=0$时性能下降，$\beta=0.04$最优。
- 零样本泛化能力强：在未见数据（DUDE 0.568, SROIE 0.814）上也优于基线。

## 亮点与洞察
- **首次将GRPO-RL应用于文档理解**：证明了RL在多模态文档理解中的有效性，且数据效率远超SFT（4K vs 438K）。DeepSeek-R1的成功可以迁移到垂直领域。
- **四维可解释性**：推理过程+改写问题+视觉区域+最终答案，比以往任何方法都提供了更全面的可解释信息。尤其是bbox输出让用户能直观看到模型"看"了文档的哪里。
- **规则奖励的设计哲学**：不用人工偏好标注，而是设计可自动验证的规则奖励。这大大降低了RL训练的成本，也避免了偏好标注的主观性。

## 局限性 / 可改进方向
- 仅在Qwen2.5-VL上验证，未在其他MLLM上测试泛化性。
- 四个奖励的权重都设为1，更精细的权重调整可能带来进一步提升。
- 训练数据来自Visual CoT（有bbox标注），如果没有bbox标注数据，RoI IoU奖励无法使用。
- 推理效率：`<think>`过程增加了输出长度，推理时间增加。
- 仅在文档理解任务验证，是否可以推广到更一般的MLLM推理场景值得探索。

## 相关工作与启发
- **vs DeepSeek-R1**: 将DeepSeek-R1的GRPO框架成功迁移到多模态文档理解领域，证明了RL推理增强的跨领域可行性。
- **vs Visual CoT**: Visual CoT用SFT训练438K数据，DocThinker用RL训练4K数据就超过了它，说明RL的数据效率远超SFT。
- **vs MedVLM-R1**: MedVLM-R1将RL应用于医学图像理解，DocThinker将其应用于文档理解，两者都证明了RL在垂直MLLM领域的有效性。

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将GRPO应用于文档理解，四目标规则奖励设计有创意
- 实验充分度: ⭐⭐⭐⭐ Visual CoT benchmark + TextREC + 消融全面，RL vs SFT对比清晰
- 写作质量: ⭐⭐⭐⭐ 结构清晰，方法描述详细
- 价值: ⭐⭐⭐⭐ 证明了RL在文档理解中的潜力，4K数据超过438K SFT的结果令人印象深刻
