---
title: >-
  [论文解读] Learning to Rewrite: Generalized LLM-Generated Text Detection
description: >-
  [ACL 2025][AIGC检测][AI文本检测] 提出Learning2Rewrite（L2R）框架，通过微调LLM的改写模型来放大人写文本和AI生成文本在改写编辑距离上的差异，从而实现跨领域高度泛化的AI文本检测——在21个独立领域上平均AUROC达0.9009，域外测试超越RAIDAR达4.67%、超越直接分类微调达51.35%。
tags:
  - "ACL 2025"
  - "AIGC检测"
  - "AI文本检测"
  - "改写距离"
  - "域外泛化"
  - "对抗鲁棒性"
  - "LoRA微调"
---

# Learning to Rewrite: Generalized LLM-Generated Text Detection

**会议**: ACL 2025  
**arXiv**: [2408.04237](https://arxiv.org/abs/2408.04237)  
**代码**: 无  
**领域**: AIGC检测  
**关键词**: AI文本检测, 改写距离, 域外泛化, 对抗鲁棒性, LoRA微调

## 一句话总结

提出Learning2Rewrite（L2R）框架，通过微调LLM的改写模型来放大人写文本和AI生成文本在改写编辑距离上的差异，从而实现跨领域高度泛化的AI文本检测——在21个独立领域上平均AUROC达0.9009，域外测试超越RAIDAR达4.67%、超越直接分类微调达51.35%。

## 研究背景与动机

现有LLM生成文本检测方法面临**泛化性**这一核心挑战。基于训练的分类器在训练域内表现优秀但在域外严重过拟合；基于统计度量的零样本方法（如DetectGPT、Fast-DetectGPT）依赖特定的统计特征，这些特征容易被简单攻击破坏且跨域不稳定。

RAIDAR方法观察到LLM改写AI文本时产生的修改量少于改写人写文本，利用编辑距离作为检测信号。但该方法的问题在于，**改写阈值在不同领域间显著变化**——法律文档和产品评论的最优阈值完全不同，这使得单一分类器难以跨域泛化。

L2R的核心洞察是：既然原始改写的LLM对AI文本和人写文本的处理差异不够大且不稳定，那么**训练改写模型本身**来最大化这种差异，就能得到一个跨领域稳定的检测信号。

## 方法详解

### 整体框架

L2R包含三个阶段：
1. 用LLM改写输入文本
2. 计算原文与改写结果的编辑距离（Levenshtein距离归一化后的相似度）
3. 用阈值判定AI生成 vs 人写

其关键创新在于**微调改写模型**，而非使用预训练模型直接改写。

### 关键设计

1. **改写编辑距离作为检测信号**: 对输入文本x，用LLM F重写后计算Levenshtein相似度：$D_k(x, F(p,x)) = 1 - \frac{Levenshtein(F(p,x), x)}{max(len(F(p,x)), len(x))}$。AI文本被改写后变化较小（高相似度），人写文本变化较大（低相似度）。

2. **微调目标函数设计**: 训练改写模型F'，使其对人写文本$x_h$尽可能多改写、对AI文本$x_{ai}$尽可能少改写：$\max\{D(x_h, F'(p, x_h)) - D(x_{ai}, F'(p, x_{ai}))\}$。由于编辑距离不可微，用交叉熵损失作为代理，对人写文本取负号使梯度方向"鼓励更多改写"，对AI文本正常最小化损失"鼓励保持原样"。

3. **Calibration Loss（校准损失）**: 无约束地最大化人写文本的改写量容易导致模型退化（冗长输出、过拟合）。因此引入阈值t：对人写文本，仅当损失L < t时才回传梯度；对AI文本，仅当损失L > t时才回传。这使模型只优化"困难样本"——那些尚未被现有阈值正确分类的文本，类似DPO的偏好学习思想。阈值t通过在微调前对训练集做一次前向传播并训练逻辑回归模型来确定。Calibration loss使AUROC平均提升4.54%（0.8555→0.9009）。

4. **多样化prompt数据集**: 构建了200个不同的改写prompt（从正式到随意、从简单到复杂），训练时随机采样。这使模型捕获更真实的AI文本分布，相比单一prompt，AUROC在Gemini改写上从0.7302提升到0.7566。

### 数据集构建

- 收集21个独立领域的人写文本（学术、法律、体育、美食、宗教等），每个领域200段
- 用GPT-4o、GPT-3.5-Turbo、Gemini 1.5 Pro、Llama-3-70B四个模型生成AI对应文本
- 总计21,000个样本
- 严格保证人写文本的采集时间在ChatGPT发布前（2022.11.30）

## 实验关键数据

### 主实验（21域平均AUROC）

| 检测器 | 平均AUROC | 标准差 |
|--------|-----------|--------|
| Fast-DetectGPT | 0.6705 | 0.1015 |
| Ghostbusters | 0.7053 | 0.1259 |
| RAIDAR (Gemini改写) | 0.7566 | 0.0928 |
| RAIDAR (Llama改写) | 0.7970 | 0.1212 |
| **Llama L2R** | **0.9009** | **0.0634** |

L2R在21个领域中的20个优于Fast-DetectGPT，平均高出23.04%；在20个领域优于Ghostbusters，平均高出19.56%。且标准差最低（0.0634），说明跨域稳定性最好。

### 域外泛化（M4数据集）

| 模型 | In-Distribution | Out-of-Distribution |
|------|----------------|---------------------|
| Ghostbusters | 0.7053 | 0.3888 |
| Llama Logits（直接分类） | 0.9774 | 0.1426 |
| RAIDAR (Llama) | 0.7970 | 0.6931 |
| **Llama L2R（低参数）** | **0.8315** | **0.7398** |

直接微调Llama做分类虽然ID最高（0.9774），但OOD暴跌到0.1426——严重过拟合。L2R低参数版本OOD达0.7398，比RAIDAR高4.67%，比直接分类高51.35%。

### 对抗攻击

| 模型 | 无攻击 | Decoherence攻击 | Rewrite攻击 |
|------|--------|-----------------|-------------|
| Fast-DetectGPT | 0.6705 | 0.4984 | 0.5100 |
| Llama Logits | 0.9774 | 0.7281 | 0.6563 |
| RAIDAR (Llama) | 0.7970 | 0.7681 | 0.7944 |
| **Llama L2R** | **0.9009** | **0.8746** | **0.8927** |

L2R在两种攻击下均保持最高AUROC，Decoherence攻击下仅下降2.63%，Rewrite攻击下仅下降0.82%。

### 消融实验

| 配置 | 关键指标 | 说明 |
|------|---------|------|
| 带Calibration Loss | AUROC 0.9009 | 完整L2R |
| 不带Calibration Loss | AUROC 0.8555 | 下降4.54%，模型有退化风险 |
| 训练在MAGE数据集 vs 自建数据集 | OOD AUROC差距15.98% | 验证多样化数据集的价值 |
| 参数量从1M增到7M | ID↑, OOD↓ | 参数越多过拟合越严重 |

### 关键发现

- **训练目标决定泛化性**：同样参数量的Llama微调，L2R的改写目标比直接分类在OOD上高51.35%
- **过拟合与参数量正相关**：随着LoRA参数增加，ID性能提升但OOD下降
- **编辑距离是域无关特征**：微调后人写文本平均编辑比0.6981，AI文本0.8606，差距足够大且跨域稳定
- **Calibration Loss类似DPO**：只优化困难样本，防止模型退化

## 亮点与洞察

- **优雅的训练目标设计**：不是训练分类器，而是训练改写器——间接获得的检测信号天然更泛化
- **理论与直觉一致**：LLM对自己生成的文本"没什么可改的"，对人写文本"有很多可改的"，微调放大了这种天然倾向
- **Calibration Loss设计精巧**：类似DPO的困难样本挖掘策略，阈值自动确定
- **数据集设计值得借鉴**：21个领域确保训练分布足够多样，200个prompt避免prompt偏差
- **可解释性**：改写后的差异部分天然标注了"哪些是AI可能写的"

## 局限与展望

- 推理开销大：每次检测需要LLM做一次完整改写（~13.5秒/120词），不适合大规模部署
- 改写模型本身也可能被攻击者利用来生成更难检测的文本
- Llama-3-8B作为改写模型可能对某些领域不够强，使用更大模型可能提升但成本更高
- 未测试中文或其他非英语语言的性能
- 微调数据的领域选择对性能有影响，如何最优选择训练领域值得研究
- 改写prompt的具体措辞可能影响检测效果

## 相关工作与启发

- 直接改进RAIDAR，解决了其阈值跨域不稳定的核心问题
- 与Fast-DetectGPT等基于统计特征的方法互补——L2R更鲁棒但更慢
- Calibration Loss的设计思路可迁移到其他需要"分布分离"的任务
- 启发：在AI检测中，"让模型自己暴露"可能比"训练分类器"更有前景

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ 训练改写模型而非分类器做检测，思路独特且直觉清晰
- 实验充分度: ⭐⭐⭐⭐⭐ 21域、4种LLM、ID/OOD/对抗攻击、消融实验、参数敏感性分析，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、方法阐述精准、图示直观、结构合理
- 价值: ⭐⭐⭐⭐ 泛化性突破显著，但推理效率是实际部署的主要瓶颈

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2026\] Enabling Supervised Learning of Generative Signatures for Generalized AI-Generated Images Detection](../../CVPR2026/aigc_detection/enabling_supervised_learning_of_generative_signatures_for_generalized_ai-generat.md)
- [\[ICML 2026\] Black-Box Detection of LLM-Generated Text Using Generalized Jensen-Shannon Divergence](../../ICML2026/aigc_detection/black-box_detection_of_llm-generated_text_using_generalized_jensen-shannon_diver.md)
- [\[ACL 2025\] Who Writes What: Unveiling the Impact of Author Roles on AI-generated Text Detection](who_writes_what_ai_detection.md)
- [\[ACL 2025\] Comparing LLM-generated and human-authored news text using formal syntactic theory](llm_vs_human_formal_syntax.md)
- [\[ACL 2025\] KatFishNet: Detecting LLM-Generated Korean Text through Linguistic Feature Analysis](katfishnet_detecting_llm-generated_korean_text_through_linguistic_feature_analys.md)

</div>

<!-- RELATED:END -->
