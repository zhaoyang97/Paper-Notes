# From Utterance to Vividity: Training Expressive Subtitle Translation LLM via Adaptive Local Preference Optimization

**会议**: ICLR 2026 / **arXiv**: [2602.01068](https://arxiv.org/abs/2602.01068)  
**代码**: 公开多方向字幕平行语料  
**领域**: llm_alignment / 机器翻译  
**关键词**: subtitle translation, preference optimization, LLM-as-Judge, liberal translation, ALPO  

## 一句话总结
提出**ALPO**（自适应局部偏好优化），通过segment-wise采样和细粒度对齐损失训练表达力强的字幕翻译LLM，解决传统DPO在多段局部偏好对齐中梯度稀释的问题。

## 背景与动机
1. LLM翻译在通用领域接近人类水平，但垂直领域（字幕翻译）需要生动意译而非直译
2. 实证发现：字幕/文学翻译的回译BLEU显著低于法律/医学，说明意译程度高
3. Chat LLM倾向直译，reasoning LLM更擅长意译（通过BLEU相似度矩阵验证）
4. LLM-as-Judge在字幕翻译评估中与人类评估高度一致（Spearman $\rho \geq 0.82$）
5. 现有DPO/PPO等outcome-supervised方法对多行字幕的局部偏好对齐粒度不足
6. 自建并公开多方向字幕平行语料MuSC

## 方法
- **SFT阶段**: 在平行语料上微调Qwen2.5-14B作为基线翻译模型
- **ALPO采样策略**: 对每行字幕逐段采样k=15个候选翻译，用Qwen3-14B评估生动性评分，选优质翻译作为下一段的prefix
- **ALPO损失**: 基于评分排序构造细粒度偏好对，逐段计算局部对齐损失，避免全序列级DPO的梯度稀释
- **Evaluator验证**: Bland-Altman图证明14B模型与人类评估偏差极小，LoA在可接受范围内

## 实验
| 设置 | 关键发现 |
|------|---------|
| 回译一致性 (Table 1) | 字幕翻译BLEU最低（en→de仅15.00），证实高度意译 |
| LLM意译能力 (Fig 3) | reasoning模型(GPT-5 Thinking, DeepSeek-R1)翻译间相似度低→意译度高 |
| 评估一致性 (Fig 1) | Qwen3-14B与人类evaluator Spearman ρ≥0.82，可作为低成本reward model |
| ALPO vs DPO/SimPO | ALPO在多维评估中显著优于全局偏好优化方法 |
| 14B模型 | 训练后的14B模型超越SOTA大模型在字幕翻译上的表现 |

## 亮点
- 首次系统研究视觉媒体字幕翻译任务，量化验证"人类翻译更偏意译"的直觉
- ALPO的segment-wise采样+局部对齐设计针对多段翻译任务非常自然
- 14B模型作为evaluator成本低且可靠，形成"小模型评估→偏好数据→训练"的高效闭环

## 局限性
- 主要在字幕翻译场景验证，ALPO是否推广到其他多段局部对齐任务（如对话生成）尚不明确
- 依赖LLM评估可能存在systematic bias（如偏好verbose表达）
- 未与token-level reward的方法（如PRM）做深入对比

## 相关工作
- DPO/SimPO/KTO: 全局偏好优化 → ALPO解决其在局部对齐任务中的梯度稀释
- LLM-as-Judge: 验证其在翻译评估中的可靠性，作为reward model基础
- RLHF: ALPO避免了explicit reward model训练和RL的不稳定性

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
