# Reasoning Boosts Opinion Alignment in LLMs

**会议**: ICLR2026  
**arXiv**: [2603.01214](https://arxiv.org/abs/2603.01214)  
**代码**: [GitHub](https://github.com/ETH-DISCO/reasoning-boosts-llm-alignment)  
**领域**: llm_reasoning  
**关键词**: opinion alignment, GRPO, political reasoning, survey data, digital democracy  

## 一句话总结
用GRPO强化学习训练LLM从政治调查数据中学习推理式观点对齐，在美国/德国/瑞士三个数据集上证明推理能提升个体级政治观点建模的准确性。

## 背景与动机
1. 政治观点建模对数字民主有重要价值，但LLM朴素提示产生带偏见的意见
2. 现有方法主要依赖人口统计提示（年龄/党派等），存在代表性、可控性和一致性三大问题
3. 面试式个体建模（如generative agents）数据收集成本过高，难以规模化
4. 政治调查数据丰富（ANES、VAA），但仅有立场标签无推理链，需要模型自行学习推理
5. GRPO等RL方法在数学推理中大获成功，能否迁移到政治推理？

## 方法
**核心框架**: SFT → GRPO两阶段训练。每个个体/政党单独训练一个模型。

**输出格式**: `<reasoning>[推理文本]</reasoning><answer>[立场]</answer>`

**复合奖励函数**: R = 0.25×R_format + 0.01×R_length + 1.0×R_correct。R_format奖励正确标签(最高4分); R_length惩罚偏离目标长度; R_correct为核心信号——匹配调查答案得1分。

**SFT初始化**: 用Llama-70B生成合成论证（支持/反对每个政策问题），训练模型掌握输出格式和基本推理能力。

**无显式persona表示**: 仅在系统提示中给国家标签，通过正确回答问题隐式对齐个体偏好。

## 实验
| 方法 | smartvote(CH) | WoM(DE) | ANES(US) |
|------|:---:|:---:|:---:|
| SFT+GRPO (Magistral-24B) | **70.73** | **53.21** | **45.43** |
| SFT (Magistral-24B) | 67.63 | 51.86 | 39.15 |
| GRPO only | 60.56 | 51.00 | 43.79 |
| ICL | 66.16 | 26.19 | 19.23 |
| ORPO | 23.31 | 24.73 | 24.25 |
| Random | 50.0 | 33.33 | 33.33 |

**关键发现**: (1) SFT+GRPO在所有数据集上一致最优，推理显著提升观点对齐; (2) 更大模型(Magistral-24B)效果更好; (3) 推理预训练骨干(Qwen3/Magistral)略优于非推理骨干(Llama); (4) 训练后的agents在PCA空间中偏向中右和保守派，与文献报告的左自由偏见相反; (5) ANES上F1仅~45%，说明推理不能完全消除偏见。

## 亮点
- 将政治观点对齐视为推理问题，首次用GRPO进行政治推理训练
- 跨三国三政治体系验证（美/德/瑞），发布公开benchmark
- 发现推理后模型的立场论证可以"翻转"——用类似论点支持相反立场（表1示例有趣）
- 意识形态分析揭示系统性偏差方向

## 局限
- 每个个体需单独训练一个模型，计算成本高，不可扩展
- 测试集较小（12-30题），统计置信度有限
- 三分类简化({Yes,Neutral,No})丢失原始细粒度信息
- F1最高仅70%左右，距"忠实数字双胞胎"仍有差距
- 未探索如何从少量调查数据泛化到全新政策议题

## 相关工作
- 人口统计提示: Santurkar et al. 2023 揭示LLM默认意见分布不代表真实人群; Argyle et al. 2023 "silicon sampling"
- 个体级建模: Park et al. 2024 用面试transcript构建个体persona
- 推理训练: DeepSeek-R1 (2025) GRPO在数学推理中成功; Yu et al. 2025 persona-based CoT提示但未用RL

## 评分
- 新颖性: ⭐⭐⭐⭐ (GRPO用于政治推理的新颖应用)
- 实验充分度: ⭐⭐⭐ (3模型3数据集，但测试集小)
- 写作质量: ⭐⭐⭐⭐ (结构清晰，案例分析有趣)
- 价值: ⭐⭐⭐ (方向有趣但可扩展性存疑)
