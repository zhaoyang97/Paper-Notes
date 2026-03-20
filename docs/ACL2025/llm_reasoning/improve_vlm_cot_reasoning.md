# Improve Vision Language Model Chain-of-thought Reasoning

**会议**: ACL 2025 (Long Paper)  
**arXiv**: [2410.16198](https://arxiv.org/abs/2410.16198)  
**代码**: [https://github.com/RifleZhang/LLaVA-Reasoner-DPO](https://github.com/RifleZhang/LLaVA-Reasoner-DPO)  
**领域**: 多模态VLM / LLM推理  
**关键词**: VLM, Chain-of-thought, DPO, 推理增强, 数据蒸馏  

## 一句话总结
通过GPT-4o蒸馏193k CoT数据做SFT + 基于答案正确性构建偏好对做DPO，显著提升VLM的CoT推理能力（LLaVA-Reasoner在8个benchmark上CoT平均提升12.6%），且CoT训练还能反哺直接预测性能。

## 背景与动机
当前VLM训练数据以短答案为主（如"14"），缺乏详细的推理过程。作者发现一个关键现象：在ChartQA上用26k直接预测数据训练后，直接预测准确率提升2.9点（70.2→73.1），但CoT准确率仅提升0.6点（71.2→71.8）。这说明模型无法从短答案训练中隐式学会推理链——需要显式的CoT训练数据。

## 核心问题
1. 仅用短答案训练VLM，能否隐式学会CoT推理？（答：不能，差距显著）
2. 如何在缺乏高质量CoT标注的情况下生成推理数据？
3. 如何进一步校准模型生成的推理链质量？

## 方法详解

### 整体框架
三阶段pipeline：(A) 从GPT-4o蒸馏CoT数据 → (B) SFT训练VLM → (C) DPO强化学习校准推理质量。基于LLaMA3-LLaVA-NeXT-8B架构。

### 关键设计

1. **ShareGPT-4o-Reasoning数据集（193k）**: 利用9个VQA数据集的短答案标注，让GPT-4o生成对应的推理过程。覆盖常识推理（A-OKVQA）、图表理解（ChartQA）、文档理解（DocVQA/InfoVQA/TextVQA）、数学科学推理（MathVision/G-LLaVA/SQA/AI2D）。蒸馏后过滤掉GPT-4o预测与GT不一致的样本。CoT答案峰值约100 tokens，远长于直接答案的5 tokens以下。

2. **双模板SFT策略**: 训练时同时使用直接预测和CoT预测两种prompt模板。直接预测用"Answer with a short answer"，CoT用"Generate a reason first and then output a short answer"。答案以"### Answer:"格式标记，便于评测时提取。关键发现：同时用CoT+Direct数据（方案④）效果最好，CoT平均74.4 vs 只用CoT的73.2。

3. **基于答案正确性的DPO偏好学习**: 用SFT模型对每个问题生成32个候选推理链（temperature=1.0/1.2），将预测与GT比对得到正确/错误标签。选取准确率在0.25-0.85之间的样本（确保有正有负），随机配对构建最多3对偏好数据。共生成64.8k偏好对（ChartQA 24.5k + A-OKVQA 18.3k + Math 22.0k）。一个有用的trick：将响应截断到90 tokens做DPO训练。

### 损失函数 / 训练策略
- SFT：1 epoch, lr=5e-6, batch=32, 8×H100
- DPO：标准DPO目标，β=0.1, lr=5e-7, batch=32, 1 epoch
- DPO模型还可作为verifier，通过 log(πdpo/πsft) 计算reward score做Best-of-N或Weighted Voting重排

## 实验关键数据

| 数据集 | 指标 | LLaVA-Reasoner-DPO (CoT) | LLaVA-Next基线 (CoT) | 提升 |
|--------|------|--------------------------|---------------------|------|
| A-OKVQA | Acc | 87.0 | 84.3 | +2.7 |
| ChartQA | Acc | 84.2 | 71.2 | +13.0 |
| DocVQA | Acc | 82.7 | 67.0 | +15.7 |
| InfoVQA | Acc | 52.7 | 34.9 | +17.8 |
| TextVQA | Acc | 71.5 | 62.2 | +9.3 |
| AI2D | Acc | 79.5 | 67.4 | +12.1 |
| SQA | Acc | 92.6 | 74.4 | +18.2 |
| MathVista | Acc | 52.1 | 40.3 | +11.8 |
| **平均** | | **75.3** | **62.7** | **+12.6** |

DPO相比SFT的增量（CoT）：平均74.4→75.3（+0.9），其中ChartQA +1.2, InfoVQA +1.1, MathVista +1.5。

### 消融实验要点
- **CoT数据 vs 直接数据**: 只用CoT训练（③）CoT性能73.2，只用Direct训练（②）CoT性能65.6，差距巨大
- **CoT训练反哺直接预测**: 只用CoT数据训练的模型（③），在DocVQA/InfoVQA/TextVQA的直接预测上反而超过只用Direct数据的模型（②）
- **数学数据组合**: ChartQA对MathVista贡献最大（+5.5），纯文本数学数据（MathPlus/MathInstruct）效果甚微
- **DPO数据**: 自构建偏好数据（⑥）优于RLAIF-V（⑤），CoT平均75.3 vs 74.6
- **DPO作为Verifier**: Weighted Voting + DPO verifier在64候选时，ChartQA达到85.4（vs单次84.2），MathVista达到53.3（vs单次52.1）

## 亮点
- **关键发现驱动**: 先实验验证"短答案不能隐式教CoT"，再针对性设计解决方案，逻辑清晰
- **答案正确性作为弱监督**: 巧妙利用短答案标注作为推理链的"outcome reward"，不需要人工标注推理过程
- **DPO模型双重身份**: 训练好的DPO模型既是generator也是verifier，可进一步通过重排提升性能
- **数据组合消融**: 详细分析了不同任务数据对不同能力的交叉影响（如ChartQA对MathVista有大贡献）

## 局限性 / 可改进方向
- 仅在8B模型上验证，更大模型（如70B/Mixture-of-Experts）结论是否一致？
- DPO只用了3个领域的偏好数据，未完全扩展到所有9个领域
- 推理链截断到90 tokens是一个magic number，缺乏分析为何这个值最优
- 未与同期STaR/ReST等自我改进方法做对比
- GPT-4o蒸馏有成本和许可问题

## 与相关工作的对比
- **vs LLaVA-CoT (Xu et al.)**: LLaVA-CoT也做VLM推理增强，但本文更聚焦于SFT+DPO两阶段pipeline且有详细消融
- **vs RLAIF-V (Yu et al.)**: RLAIF-V的偏好数据面向减少幻觉，本文的偏好数据面向推理正确性，在推理任务上本文更优
- **vs Cambrian-7B**: 在相同8B规模下，LLaVA-Reasoner-SFT在CoT任务上全面优于Cambrian

## 启发与关联
- 与 [ideas/model_compression/20260316_efficient_surgical_reasoning.md](../../../ideas/model_compression/20260316_efficient_surgical_reasoning.md) 关联：如果能在更小的模型上实现类似的CoT增强，结合模型压缩将很有价值
- "答案正确性→推理链质量"的弱监督范式可以迁移到其他需要过程监督但缺乏标注的场景
- DPO作为verifier的思路可以和Best-of-N、MCTS等推理时计算方法结合

## 评分
- 新颖性: ⭐⭐⭐ 方法各组件（蒸馏、SFT、DPO）都是已有技术，创新主要在组合和实验发现
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其详细，数据组合分析、verifier分析等非常扎实
- 写作质量: ⭐⭐⭐⭐ 结构清晰，实验设计逻辑性强
- 价值: ⭐⭐⭐⭐ 193k CoT数据集和实验结论对社区有实际参考价值
