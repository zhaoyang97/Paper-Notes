---
title: >-
  [论文解读] Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths
description: >-
  [AAAI2026 Oral][时间序列][Fedspeak] 提出基于 LLM 的 uncertainty-aware 框架解读 Fedspeak（美联储语言）：通过货币政策传导路径的领域推理增强输入，引入 dynamic uncertainty decoding 模块量化预测置信度（Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk），在 FOMC 政策立场分析任务上达到 SOTA。
tags:
  - "AAAI2026 Oral"
  - "时间序列"
  - "Fedspeak"
  - "monetary policy stance"
  - "LLM"
  - "uncertainty quantification"
  - "financial sentiment analysis"
---

# Interpreting Fedspeak with Confidence: A LLM-Based Uncertainty-Aware Framework Guided by Monetary Policy Transmission Paths

**会议**: AAAI2026 Oral  
**arXiv**: [2508.08001](https://arxiv.org/abs/2508.08001)  
**代码**: [yuuki20001/FOMC-sentiment-path](https://github.com/yuuki20001/FOMC-sentiment-path)  
**领域**: 时间序列  
**关键词**: Fedspeak, monetary policy stance, LLM, uncertainty quantification, financial sentiment analysis

## 一句话总结
提出基于 LLM 的 uncertainty-aware 框架解读 Fedspeak（美联储语言）：通过货币政策传导路径的领域推理增强输入，引入 dynamic uncertainty decoding 模块量化预测置信度（Perceptual Uncertainty = Environmental Ambiguity × Cognitive Risk），在 FOMC 政策立场分析任务上达到 SOTA。

## 研究背景与动机
Fedspeak 是美联储用于传达政策信号的特殊语言，具有高度语境依赖性——同一词在不同经济环境下可能指向相反立场（如 "strong" labor market 在弱经济中偏鸽派、过热经济中偏鹰派）。

现有方法的问题：
- **Dictionary-based 方法**：简单可解释但无法理解复杂语境
- **FinBERT 等微调模型**：性能好但黑盒，缺乏透明性
- **GPT-4 等大模型 zero-shot**：能力强但忽略可靠性、偏差和幻觉问题
- 现有 LLM 工作多聚焦性能指标，忽略预测的可靠性评估

核心思路：将 LLM 类比为政策分析师，引入认知风险 (CR) 和环境模糊性 (EA) 两个不确定性维度来量化预测置信度。

## 方法详解

### 数据增强：领域推理
1. **Financial Entity Relations 提取**：从 Fedspeak 中分解原子关系 $r(e_i, e_j) \in \mathcal{R}$，涵盖 CAUSE、COND、EVID、PURP、ACT、COMP 六类
2. **货币政策传导路径推理**：构建四元组 $\Gamma = (\mathbf{X}, \mathbf{Y}, \mathbf{Z}, \mathbf{M})$
    - $\mathbf{X}$：经济冲击向量
    - $\mathbf{Y}$：传导渠道（信贷渠道、资产价格渠道、总需求渠道等）
    - $\mathbf{Z}$：传导路径（状态转移序列）
    - $\mathbf{M}$：最终政策建议
3. 用结构化模板 + human-AI 协作构造 SFT 数据集

### Dynamic Uncertainty Decoding
利用 LLM 输出的 top-$k$ logits 构造 Dirichlet 分布，定义三个不确定性度量：

- **Environmental Ambiguity (EA)**：预测分布的期望熵
$$EA(a_t) = -\sum_{k=1}^{K} \frac{\alpha_k}{\alpha_0}(\psi(\alpha_k+1) - \psi(\alpha_0+1))$$

- **Cognitive Risk (CR)**：与总证据量成反比
$$CR(a_t) = \frac{K}{\sum_{k=1}^{K}(\alpha_k + 1)}$$

- **Perceptual Uncertainty (PU)**：$PU = EA \times CR$

解码策略根据 PU 阈值动态切换：
- 低 PU → aggressive（直接选 top-1 token）
- 高 PU → conservative（从 top-2 中采样）

## 实验关键数据

### 实验设置
- **数据集**：Trillion Dollar Words FOMC dataset（1996–2022），含会议纪要、新闻发布会、演讲三类
- **基线**：10+ 模型，包括 GPT-4.1、Gemini-2.5-Pro、DeepSeek-R1、Phi-4、FinBERT、AICBC 等
- **基座**：Qwen3-14B + LoRA 微调

### 主要结果 (All Categories)

| 方法 | Macro F1 | Weighted F1 |
|---|---|---|
| GPT-4.1 (zero-shot) | 0.6662 | 0.6763 |
| AICBC (zero-shot) | 0.6637 | 0.6802 |
| Qwen3-8B (fine-tuned) | 0.6586 | 0.6745 |
| **Ours** | **0.7327** | **0.7426** |

- 较最强基线 Macro F1 提升 **+6.6%**，Weighted F1 提升 **+6.2%**
- 会议纪要上表现最突出：Macro F1 = 0.7449（+7.4%）
- 演讲类：Macro F1 = 0.7291（+6.7%）

### 消融实验

| 配置 | Macro F1 | Weighted F1 |
|---|---|---|
| Full model | 0.7327 | 0.7426 |
| w/o PU | 0.7291 | 0.7378 |
| w/o Transmission Path | 0.6538 | 0.6699 |
| w/o Entity Relations | 0.6397 | 0.6551 |

传导路径贡献最大（去除后 -7.9%），实体关系次之，PU 模块贡献相对温和但有效。

### Uncertainty 验证
- 低 PU 预测：Macro F1 = 0.7791，高 PU 预测：Macro F1 = 0.2473
- T-test / Mann-Whitney U test / Logistic regression 的 p-value 均远低于 0.001，统计显著性强

## 亮点
- **领域推理创新**：首次将货币政策传导机制形式化为结构化推理模板，模拟人类专家分析流程
- **PU 度量实用**：EA × CR 的分解符合经济学中 risk / ambiguity 的经典划分，在金融场景中直觉自然
- **高 PU 预警机制**：可识别不可靠预测，支持 human-in-the-loop 决策
- **全面超越 GPT-4.1**：在会议纪要和演讲上大幅领先闭源大模型

## 局限与展望
- 新闻发布会表现弱于 GPT-4.1（-1.3%），实时问答的动态上下文依赖捕捉不足
- 依赖手工模板构建传导路径，自动化程度有限
- 仅在 FOMC 英文数据上验证，未扩展到 ECB/BoE 等其他央行或多语言场景
- PU 阈值需在验证集上搜索，不同数据集需重新调参
- 未探索 "拒绝回答" 策略在实际部署中的效果

## 评分
- 新颖性: ⭐⭐⭐⭐ — 货币政策传导路径推理和 PU 度量的结合有明确方法论贡献
- 实验充分度: ⭐⭐⭐⭐ — 10+ 基线、三类文本、消融 + 统计检验，覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 结构清晰，经济学与 NLP 概念衔接流畅
- 价值: ⭐⭐⭐⭐ — 对金融 NLP 可靠性研究有实际推动作用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] HiVid: LLM-Guided Video Saliency For Content-Aware VOD And Live Streaming](../../ICLR2026/time_series/hivid_llm-guided_video_saliency_for_content-aware_vod_and_live_streaming.md)
- [\[AAAI 2026\] ProbFM: Probabilistic Time Series Foundation Model with Uncertainty Decomposition](probfm_probabilistic_time_series_foundation_model_with_uncertainty_decomposition.md)
- [\[ICML 2025\] Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets: A Transparent Framework for Interpretable Quant Trading](../../ICML2025/time_series/event-aware_sentiment_factors_from_llm-augmented_financial_tweets_a_transparent_.md)
- [\[ICLR 2026\] EVEREST: A Transformer for Probabilistic Rare-Event Anomaly Detection with Evidential and Tail-Aware Uncertainty](../../ICLR2026/time_series/everest_a_transformer_for_probabilistic_rare-event_anomaly_detection_with_eviden.md)
- [\[CVPR 2026\] Towards Uncertainty-aware Unsupervised Domain Adaptation for Videos and Time-Series with Causal Optimal Transport](../../CVPR2026/time_series/towards_uncertainty-aware_unsupervised_domain_adaptation_for_videos_and_time-ser.md)

</div>

<!-- RELATED:END -->
