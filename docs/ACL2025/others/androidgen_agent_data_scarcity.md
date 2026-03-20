# AndroidGen: Building an Android Language Agent under Data Scarcity

**会议**: ACL 2025 (Long Paper)  
**arXiv**: 见ACL Anthology  
**代码**: [https://github.com/THUDM/AndroidGen](https://github.com/THUDM/AndroidGen)  
**领域**: Agent / 多模态VLM  
**关键词**: Android Agent, 数据稀缺, 轨迹生成, 自动检查, 开源Agent  

## 一句话总结
提出AndroidGen——一个在数据稀缺条件下增强Android Agent能力的框架（ExpSearch+ReflectPlan+AutoCheck+StepCritic四模块），在AndroidWorld上以GPT-4o达到46.8%成功率（vs M3A的27.7%），并能自动生成高质量轨迹数据训练开源模型达到竞争力水平。

## 背景与动机
手机Agent面临数据稀缺困境：(1) 人工标注轨迹既耗时又昂贵；(2) 场景高度多样化，泛化难；(3) 复杂任务需要多步操作，成功率低导致数据收集效率低；(4) 自动生成的轨迹质量参差不齐，缺乏有效过滤机制。即使GPT-4o自主操作手机成功率也不高。

## 核心问题
如何在缺乏人工标注轨迹数据的情况下，构建强大的Android Agent？

## 方法详解

### 整体框架
四模块协作框架：ExpSearch（经验检索）→ ReflectPlan（反思规划）→ AutoCheck（自动检查）→ StepCritic（步级评估）。可用于增强推理时性能，也可生成训练数据。

### 关键设计

1. **ExpSearch（经验搜索）**: 不同于传统固定数据集的ICL，ExpSearch从已完成的轨迹库中检索与当前任务最相似的成功案例作为few-shot示例。可以从简单任务泛化到复杂任务——先积累简单任务的经验，再用这些经验指导复杂任务。

2. **ReflectPlan（反思规划）**: 在每步操作前，Agent对当前环境进行自我反思，更新计划状态（哪些子目标已完成、哪些待完成）。增强长程推理能力，避免重复操作或遗漏步骤。

3. **AutoCheck（自动检查）**: 在每步操作执行后进行规则化验证：点击的element ID是否存在于屏幕上、输入的文本是否正确填入、滑动方向是否合法等。比让LLM自检更可靠（自检有假阳性问题），简单规则检查反而更有效。

4. **StepCritic（步级评价）**: 基于GPT-4o的细粒度轨迹评估器。将任务分解为子目标，逐步评估轨迹中每一步是否有助于完成对应子目标。为训练数据过滤和augmentation提供细粒度标签。

### 数据生成Pipeline
AndroidGen生成轨迹 → StepCritic评估 → 过滤高质量轨迹 → 训练开源LLM → 得到无需人工标注的开源Android Agent。

## 实验关键数据

**AndroidWorld（18个App，成功率）**:

| Agent | Base Model | Success Rate |
|-------|-----------|-------------|
| SeeAct | GPT-4o | 15.9% |
| M3A | Gemini-1.5-Pro | 19.8% |
| M3A | GPT-4o | 27.7% |
| **AndroidGen** | GLM-4-9B* | **29.2%** |
| **AndroidGen** | Llama-3-70B* | **35.3%** |
| **AndroidGen** | GPT-4o | **46.8%** |

- AndroidGen + GPT-4o: 46.8%, 比M3A+GPT-4o高69%的相对提升
- 开源GLM-4-9B微调后(29.2%)甚至超越GPT-4o的M3A(27.7%)

**AitW (General/Web Shopping)**:

| 方法 | General | Web Shopping |
|------|---------|-------------|
| DigiRL (RL) | 71.9% | 67.2% |
| **AndroidGen** Llama-3-70B* | 74.0% | 79.2% |
| **AndroidGen** GPT-4o | **85.4%** | **81.3%** |

### 消融实验要点
- **ExpSearch**: 最关键模块，去掉后性能下降最大（有经验检索 vs 无经验）
- **ReflectPlan**: 对长步骤任务贡献显著
- **AutoCheck**: 规则检查 > LLM自检（避免假阳性）
- **StepCritic数据过滤**: 过滤后训练数据质量显著提升开源模型性能

## 亮点
- **全链路设计**: 从推理增强到数据生成到开源模型训练，闭环解决数据稀缺问题
- **开源模型平GPT-4o闭源**: 微调后的GLM-4-9B超越M3A+GPT-4o
- **AutoCheck的务实设计**: 放弃"让LLM检查自己"的思路，用简单规则反而更可靠
- **StepCritic细粒度评估**: 不是只看最终成功/失败，而是逐步评估每个操作

## 局限性 / 可改进方向
- StepCritic依赖GPT-4o，成本不可忽视
- 某些App（Chrome, VLC）成功率仍为0%——涉及复杂交互的场景尚未解决
- 文本模式（a11y tree）为主，未充分利用截图视觉信息
- ExpSearch的轨迹库需要逐步积累，冷启动可能有困难
- 人类成功率80%仍有差距——Agent能力上限受LLM基础能力限制

## 与相关工作的对比
- **vs AndroidLab**: AndroidLab提供评估框架+少量标注数据，AndroidGen提供自动数据生成框架——互补
- **vs DigiRL**: DigiRL用RL从在线交互中学习（71.9%），AndroidGen用SFT+经验检索（74.0%微调版）
- **vs AppAgent**: AppAgent是单步推理框架，AndroidGen加入了经验检索和反思规划

## 启发与关联
- ExpSearch的"从简单任务泛化到复杂任务"策略可以推广到任何Agent学习场景
- AutoCheck的"规则检查优于自检"发现对Agent安全性设计有重要启示
- 与AndroidLab结合：用AndroidLab的评估框架 + AndroidGen的数据生成框架 = 完整的Agent训练评估闭环

## 评分
- 新颖性: ⭐⭐⭐⭐ 四模块协作设计和数据生成pipeline有价值
- 实验充分度: ⭐⭐⭐⭐⭐ AndroidWorld+AitW+自建benchmark，多个模型
- 写作质量: ⭐⭐⭐⭐ 图2的实际操作示例和Table 2的per-app分析清晰
- 价值: ⭐⭐⭐⭐⭐ 解决了Agent训练数据稀缺的核心痛点，全套开源
