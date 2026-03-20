# TempSamp-R1: Effective Temporal Sampling with Reinforcement Fine-Tuning for Video LLMs

## 基本信息
- **arXiv**: 2509.18056
- **会议**: NeurIPS 2025
- **作者**: Yunheng Li, Jing Cheng, Shaoyong Jia, Hangyi Kuang, Shaohui Jiao, Qibin Hou, Ming-Ming Cheng
- **机构**: Nankai University (HVision Lab)
- **代码**: https://github.com/HVision-NKU/TempSamp-R1

## 一句话总结
提出 TempSamp-R1，针对视频时序定位任务改进 GRPO 强化微调框架，通过 off-policy 时间精确引导 + 非线性软优势计算 + 混合 CoT 训练，在 Charades-STA/ActivityNet/QVHighlights 上分别提升 +2.7%/+5.3%/+3.0%。

## 背景与动机
R1-style 强化微调（如 GRPO）在数学推理上效果显著，但在**视频时序定位 (temporal grounding)** 上效果有限：
- 时序定位搜索空间巨大（连续时间轴上的起止时刻对）
- GRPO 的 on-policy 采样在大搜索空间中难以命中高奖励解
- 奖励信号稀疏且噪声大——生成的时间段很少与 GT 高度重合
- 纯 on-policy 更新导致策略探索不足、收敛困难

## 核心问题
如何使强化微调在时序搜索空间巨大的视频理解任务中有效工作？

## 方法详解

### 1. Off-policy Temporal Supervision
- 问题：GRPO on-policy 采样生成的时间段大多与 GT 不匹配（IoU 很低）
- 解决：利用 GT 标注作为 **off-policy 监督**
  - 将 GT 时间段混入采样组
  - 提供时间精确的正例信号
  - 弥补 on-policy 采样在大搜索空间中的稀疏性

### 2. 非线性软优势计算 (Non-linear Soft Advantage)
- 问题：标准 GRPO 的优势函数方差大，训练不稳定
- 解决：对 reward 反馈做**非对称变换**——动态重塑奖励分布
  - 高奖励样本获得更大优势
  - 低奖励样本优势被压缩而非硬截断
  - 减少方差，提高 reward-based update 的稳定性

### 3. 混合 CoT 训练范式 (Hybrid Chain-of-Thought)
- 统一模型支持 CoT 和 non-CoT 两种推理模式
- CoT 模式：先分析视频内容再定位（适合复杂查询）
- Non-CoT 模式：直接输出时间段（适合简单查询）
- 训练时混合两种模式，推理时按查询复杂度选择

### 4. 奖励设计
- 基于 IoU 的分级奖励
- 考虑预测时间段与 GT 的重合度
- 结合格式合规性奖励

## 实验关键数据

### 视频时序定位 SOTA
| 数据集 | 指标 | GRPO baseline | TempSamp-R1 | 提升 |
|---|---|---|---|---|
| Charades-STA | R1@0.7 | 50.2% | **52.9%** | +2.7% |
| ActivityNet Captions | R1@0.5 | 50.7% | **56.0%** | +5.3% |
| QVHighlights | mAP | 27.0% | **30.0%** | +3.0% |

### Few-shot 泛化
- 有限训练数据下仍表现出色
- 展示了强化微调的 data efficiency

## 亮点
1. **将 R1-style RL 推广到视频理解**：首个系统性解决 GRPO 在时序定位中失效的工作
2. **Off-policy + On-policy 混合**：利用 GT 监督弥补 on-policy 在大搜索空间中的不足
3. **非线性优势计算**：优雅解决 reward-based update 的高方差问题
4. **混合 CoT**：单一模型支持多种推理深度，灵活应对不同查询
5. **3 个 SOTA**：在 3 个视频定位 benchmark 上全部超越

## 局限性
1. Off-policy 监督依赖 GT 标注，推理时无法使用
2. 主要针对时序定位任务，对其他视频理解任务（如 VQA）的效果未验证
3. 非线性变换的超参数可能需要任务特定调整
4. IoU-based 奖励设计可能不适合所有时序任务

## 与相关工作的对比
- **vs. GRPO (标准)**：GRPO 在时序定位中因搜索空间太大而效果差，TempSamp-R1 通过 off-policy 信号解决
- **vs. DeepVideo-R1 (之前写过)**：DeepVideo-R1 用 R1 做视频推理，TempSamp-R1 专注时序定位
- **vs. TimeChat/VTimeLLM**：这些方法用 SFT 做时序定位，TempSamp-R1 用 RL 微调更灵活
- **vs. NoisyRollout (之前写过)**：NoisyRollout 在推理中加噪声提升探索，TempSamp-R1 用 off-policy sample

## 启发与关联
- **RL 在视觉任务中的挑战**：数学推理的离散答案空间 vs. 视频定位的连续搜索空间——后者需要更精心的采样策略
- **与 Does Thinking More Help? 的联系**：两者都关注推理过程的效率——前者发现过度思考有害，后者发现 on-policy 采样在大空间中无效
- **Off-policy 的普适性**：在 RL-based LLM 训练中，纯 on-policy 可能不够，适度引入 off-policy 信号值得探索

## 评分
- 新颖性：★★★★☆ — 将 R1-style RL 推广到视频时序定位有价值
- 技术深度：★★★★☆ — Off-policy + 非线性优势 + 混合 CoT 设计完整
- 实验完整度：★★★★☆ — 3 benchmark SOTA + few-shot 评估
- 写作质量：★★★★☆ — 问题分析到位
