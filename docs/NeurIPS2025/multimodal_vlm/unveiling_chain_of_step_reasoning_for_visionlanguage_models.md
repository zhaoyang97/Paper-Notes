# Unveiling Chain of Step Reasoning for Vision-Language Models with Fine-grained Rewards

**会议**: NeurIPS 2025  
**arXiv**: [2509.19003](https://arxiv.org/abs/2509.19003)  
**代码**: [https://github.com/baaivision/CoS](https://github.com/baaivision/CoS)  
**领域**: 多模态VLM / 视觉推理 / 过程奖励模型  
**关键词**: chain-of-step, process reward model, step-level reasoning, iterative DPO, inference-time scaling  

## 一句话总结
提出Chain-of-Step (CoS)推理框架：将VLM的推理链分解为结构化步骤（Name+Thought+Reflection），训练Process Reward Model (PRM)提供步骤级精细奖励，通过迭代DPO和step-level beam search显著提升VLM推理能力——在InternVL-2.5-MPO-8B上平均提升4.0%达到73.4%，并揭示"对VLM而言推理质量比长度更重要"。

## 背景与动机
现有VLM的CoT推理是粗粒度的——输出一长段thought without结构化步骤划分，导致推理容易变得冗长混乱，更关键的是无法评估中间推理步骤的质量。这使得RL训练和inference-time scaling都缺乏有效的reward信号。LLM领域的PRM（如Math-Shepherd、Let's Verify Step by Step）已证明step-level reward的价值，但在VLM领域尚未被充分探索。

## 核心问题
如何将VLM的推理链分解为结构化步骤？如何提供精细的步骤级reward信号？step-level reward能否比outcome-level reward更有效地指导RL训练和inference-time scaling？

## 方法详解

### 整体框架
三阶段pipeline（Figure 1）：
1. **SFT on ShareGPT-Step-300K**：教模型输出结构化步骤推理
2. **训练PRM**：用Monte Carlo估计+GPT-4o标注训练step-level reward model
3. **Iterative DPO with PRM**：用PRM选择正负样本对做3轮迭代DPO

### 关键设计

1. **结构化推理步骤设计**：每个推理步包含三个组件：
   - **Name**：步骤概要（如"识别几何形状"）
   - **Thought**：详细推理内容
   - **Reflection**：与视觉内容和前序步骤的关联，缓解幻觉
   
   用11个特殊token标记步骤边界，确保输出格式稳定可解析。步骤数量和长度由模型自主决定。

2. **ShareGPT-Step-300K数据集**：用GPT-4o从17个数据集的QA pairs生成结构化推理链。覆盖数学、科学、图表、文档、知识问答等多样任务。"从结果推理"——给GPT-4o答案参考可大幅降低生成难度、提高质量。

3. **Process Reward Model (PRM)**：
   - 训练数据：用Math-Shepherd (MC估计, N=16)和GPT-4o-as-Judge两种方法各标注100K步骤级数据
   - 标注粒度：每步标注Good/Neutral/Bad
   - 基座：InternVL-2.5-MPO-38B，BCE loss训练2 epochs
   - Step accuracy 87.3% on unseen data

4. **Iterative DPO**：每轮对每个问题生成16条推理路径，用PRM评估（step score 20% + answer score 80%加权），选择分差超过阈值t的正负对做DPO。3轮迭代，每轮20K preference pairs。

5. **Step-level Beam Search**（推理时）：
   - 对每一步采样N个候选
   - 用PRM打分选最佳步骤
   - 基于最佳步骤继续采样下一步
   - 与Best-of-N sampling成本相同但效果更好

## 实验关键数据

| 方法 | MathVista | MMStar | MMMU | M3CoT | AI2D | ChartQA | Avg |
|------|-----------|--------|------|-------|------|---------|-----|
| InternVL2.5-MPO-8B | 65.0 | 60.7 | 53.8 | 67.5 | 84.2 | 85.0 | 69.4 |
| + SFT (CoS) | 65.9 | 61.0 | 53.7 | 75.7 | 81.6 | 88.3 | 71.0 |
| + Iterative DPO (CoS) | **67.8** | **63.5** | **55.5** | **81.0** | **84.9** | **87.4** | **73.4** |
| LLaVA-NeXT-8B | 45.9 | 43.1 | 36.9 | 45.6 | 71.5 | 69.4 | 52.1 |
| + CoS (SFT+DPO) | **54.7** | **58.9** | **41.8** | **71.7** | **79.2** | **79.1** | **64.2** |

GRPO验证：CoS GRPO (PRM reward) > Outcome GRPO，平均63.0 vs 61.2。

### 消融实验要点
- **Step weight最优20%**：纯step score或纯answer score都不是最优，20% step + 80% answer效果最好
- **PRM > Self-Consistency > Outcome**：step-level PRM选择的Best-of-N显著优于Self-Consistency投票
- **Step-level beam search > Best-of-N**：同等计算量下beam search一致性更好
- **PRM DPO > Outcome DPO**：step & answer综合reward的DPO比仅看最终答案正确性提升1.7%（M3CoT上71.7 vs 70.0）
- **推理长度反直觉发现**：PRM DPO训练初期模型**缩短**推理长度以提高质量，稳定后才慢慢增长；而Outcome DPO则持续增长长度→说明VLM推理中质量 > 长度
- **Step-wise DPO失败**：每步构造preference pair→chosen和rejected太相似，模型拒绝输出两者

## 亮点 / 我学到了什么
- **结构化推理步骤的设计**（Name+Thought+Reflection）非常实用，Reflection组件有效连接视觉内容和前序推理
- **"VLM推理质量>长度"的发现**与LLM中"更长=更好"的趋势相反——视觉推理更依赖视觉信息利用和知识连接
- PRM训练只需一次，可服务多个模型——38B PRM为8B模型提供reward是scale-efficient的
- Step-wise DPO的失败案例很有教育意义——chosen和rejected的差异需要足够大才能形成有效的学习信号
- Inference-time scaling：PRM-BS在N=64时比Self-Consistency高5%+

## 局限性 / 可改进方向
- MC估计和LLM-as-Judge的标注质量无法保证100%正确
- 仅在8B模型上全面验证，更大模型（如72B）的效果未知
- Reflection组件是否真正利用了视觉信息需要更深入的分析
- 38B PRM的inference成本在生产环境中可能偏高
- ShareGPT-Step-300K用GPT-4o生成，对闭源模型有依赖

## 与相关工作的对比
- vs **LLaVA-CoT**：LLaVA-CoT用粗粒度推理（SUMMARY/CAPTION/REASONING/CONCLUSION），CoS用细粒度步骤（Name/Thought/Reflection）+ PRM
- vs **Sherlock (2505.22651)**：Sherlock做response-wise自纠正，CoS做step-wise精细评估——互补
- vs **NoisyRollout (2504.13055)**：NoisyRollout增强exploration diversity，CoS用PRM提供精细reward——可组合
- vs **URSA**：URSA也用PRM但推理链是粗粒度的，CoS的结构化步骤使PRM评估更准确

## 与我的研究方向的关联
- PRM是VLM推理后训练的关键组件——与所有RL-based VLM方法互补
- "质量>长度"的insight与overthinking研究方向一致，可用于指导adaptive inference
- 与Sherlock的trajectory-level纠正互补——CoS提供step-level reward，Sherlock做response-level纠正

## 评分
- 新颖性: ⭐⭐⭐⭐ 结构化步骤+PRM在VLM领域是新的组合，但各组件并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 消融极其全面——step weight、PRM选择、推理长度、reasoning pattern、GRPO验证、step-wise DPO失败分析
- 写作质量: ⭐⭐⭐⭐⭐ 逻辑递进清晰，每个发现都有实验支撑，失败案例也诚实报告
- 对我的价值: ⭐⭐⭐⭐⭐ VLM推理后训练的complete framework，PRM+step-level beam search对inference scaling有直接价值
