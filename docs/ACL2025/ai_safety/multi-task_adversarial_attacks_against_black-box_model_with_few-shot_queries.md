---
title: >-
  [论文解读] Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries
description: >-
  [ACL 2025][AI安全][adversarial attack] 提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。
tags:
  - "ACL 2025"
  - "AI安全"
  - "adversarial attack"
  - "multi-task"
  - "black-box"
  - "text classification"
  - "transfer attack"
  - "few-shot queries"
---

# Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries

**会议**: ACL 2025  
**代码**: -  
**领域**: AI安全  
**关键词**: adversarial attack, multi-task, black-box, text classification, transfer attack, few-shot queries  

## 一句话总结

提出 CEMA（Cluster and Ensemble Multi-task Text Adversarial Attack）方法，通过训练"深层替代模型"将复杂的多任务黑盒攻击转化为单任务文本分类攻击，仅需约 100 次查询即可同时攻击分类、翻译、摘要、文生图等多种任务，并在 ChatGPT-4o、百度翻译、Stable Diffusion 等商用模型上验证了有效性。

## 研究背景与动机

- **文本对抗攻击**已广泛研究，但主要集中在单任务场景（分类、翻译）
- **多任务对抗攻击**在图像领域有少量工作（MTA、MTADV），但在文本领域几乎空白
- **现有方法的局限**：
  1. 多任务白盒攻击需要访问模型内部特征，不适用于黑盒 API
  2. 单任务黑盒攻击需要大量查询（数万次）
  3. 不同任务类型（分类 vs. 翻译 vs. 生成）难以统一处理
- **实际威胁场景**：现代 AI 系统越来越多地采用多任务架构（同一输入给多个下游任务），如何用极少查询同时攻击多个任务是重要的安全问题
- **核心观察**：不同任务输出虽然形式各异，但共享"深层特征"——如分辨猫和鸟的深层特征比判断"是否哺乳动物/能否飞/几条腿"更根本

## 方法详解

### 整体框架 CEMA

CEMA 分三步：
1. **深层替代模型训练**
2. **候选对抗样本生成**
3. **基于迁移性的对抗样本选择**

### 第一步：深层替代模型训练

**核心假设（Deep-level Attack Hypothesis）**：用深层标签训练的替代模型生成的对抗样本可以有效攻击受害模型的多个下游任务。

**具体流程**：
1. 收集少量辅助文本（如 100 条无标签受害文本）
2. 对每条辅助文本 $x_i$ 查询受害模型获取所有任务输出 $\{y_i^1, y_i^2, ..., y_i^N\}$
3. 用预训练编码器（mT5）将文本和所有输出分别编码并拼接为统一向量 $E_i = \text{Concat}(E_{x_i}, E_{y_i^1}, ..., E_{y_i^N})$
4. 对 $E_i$ 做**二聚类**（Spectral Clustering），得到深层标签 $y_i^c \in \{0, 1\}$
5. 用辅助文本-聚类标签对训练**二分类替代模型** $f_s$

**关键设计**：
- 聚类数 = 2：实验证明二聚类捕获最根本的深层标签，多聚类最终会合并为二
- 替代模型是即插即用的，不需要模仿受害模型的具体架构
- 仅需 ~100 次查询获取辅助数据

### 第二步：候选对抗样本生成

- 对替代模型 $f_s$ 应用 $l$ 种文本分类攻击方法（Hotflip、FD、TextBugger），生成 $l$ 个对抗候选
- 筛选条件：对抗样本与原始文本的余弦相似度 ≥ 阈值 $\epsilon = 0.8$
- 多方法策略的数学保证（定理 3.2）：
    - 候选越多，至少一个成功攻击的概率单调递增
    - 候选越多，至少一个超过相似度阈值的概率单调递增

### 第三步：基于迁移性的对抗样本选择

1. 用 80% 辅助数据重新训练 $w$ 个替代模型（$w=6$）
2. 对每个候选对抗样本，计算它成功攻击了多少个替代模型（迁移性得分 $I_{ij}$）
3. 选择攻击最多替代模型的候选作为最终对抗样本
4. 平局时选概率变化最大的（$p_c^j = p_{\hat{y}}(x_i^*) - p_{\hat{y}}(\tilde{x}_i^j)$）

## 实验

### 实验设置

**数据集**：SST5（情感分析，5 类）、Emotion（情绪分类，6 类）

**受害模型设置**：
- LLM 攻击：ChatGPT-4o、Claude 3.5（prompt = 同时翻译为法语/中文 + 预测情绪类别）
- M3TL（Multi-Model Multi-Task Learning）：
    - Victim A：dis-sst5 + dis-emotion + opus-mt（En-Zh）
    - Victim B：ro-sst5 + ro-emotion + T5-small（En-Fr）
    - Victim C：百度翻译 + 阿里翻译（商用 API）
- 图像生成：Stable Diffusion V2

**对比方法**：BAE、FD、Hotflip、SememePSO、TextBugger（分类）；kNN、Morphin、Seq2Sick、TransFool（翻译）

### 主实验：LLM 攻击

| 数据集 | 模型 | 分类 ASR↑ | En-Fr BLEU↓ | En-Zh BLEU↓ | 查询数↓ |
|--------|------|-----------|-------------|-------------|---------|
| Emotion | ChatGPT-4o | **32.05** | **0.39** | **0.33** | **100** |
| Emotion | Claude 3.5 | **36.80** | **0.38** | **0.35** | **100** |
| SST5 | ChatGPT-4o | **38.63** | **0.32** | **0.27** | **100** |
| SST5 | Claude 3.5 | **37.12** | **0.29** | **0.25** | **100** |

仅 100 次查询，CEMA 在分类 ASR 和翻译 BLEU 上均大幅优于 Random-Del 基线。

### M3TL 多任务攻击

**分类任务**（SST5/Emotion，Victim A/B）：
- CEMA 在 Emotion 上 ASR 达 **80.80%**（Victim A），远超所有单任务方法
- 查询数仅 **100**，而对比方法需要 2-7 万次查询

**例如 SST5 + Victim A**：

| 方法 | ASR↑ | Sim.↑ | 查询数↓ |
|------|------|-------|---------|
| BAE | 42.71 | 0.888 | 47,360 |
| PSO | 45.14 | 0.954 | 24,398 |
| HQA | 46.11 | 0.936 | 64,864 |
| **CEMA** | **73.57** | 0.934 | **100** |

ASR 从 46% 提升到 73%，查询数从数万降到 100。

### 商用 API 和图像生成攻击

- 对百度翻译和阿里翻译（Victim C），CEMA 仅用 100 条辅助文本即可达到 BLEU < 0.35
- 可攻击 Stable Diffusion V2 的文生图功能

### 消融实验

- **聚类数**：2 聚类最优，3/4 聚类效果下降
- **辅助数据量**：100 条足够，增加到 200/400 提升有限
- **攻击方法组合**：三方法组合优于单方法
- **替代模型数量 $w$**：6 个替代模型为最优平衡点

## 亮点与洞察

1. **深层攻击假设**极具创新性：绕过任务差异，从数据的内在结构出发攻击
2. **实用性极强**：仅需 100 次查询（可能 1 分钟内完成），查询数比现有单任务方法低 2-3 个数量级
3. **任务通用性**：同一框架攻击分类、翻译、摘要、文生图四类不同任务
4. **打败 ChatGPT-4o 和 Claude 3.5**：在商用 LLM 上验证了威胁的真实性
5. **即插即用**：替代模型训练仅需 4 分钟（24GB 3090），不需要了解受害模型架构

## 局限性

- 分类 ASR 在 LLM 上仅 30-38%，说明 LLM 对此类攻击仍有一定鲁棒性
- 依赖文本分类攻击方法（Hotflip/FD/TextBugger），攻击质量受限于这些基方法
- 聚类质量对最终效果有较大影响，不同数据分布下聚类效果可能不稳定
- 相似度阈值 $\epsilon = 0.8$ 的设定对攻击成功率和隐蔽性有 trade-off
- 未讨论防御措施和对抗鲁棒训练的对策

## 相关工作

- **文本分类攻击**：Hotflip（Ebrahimi et al.）、BAE（Garg et al.）、TextBugger（Ren et al.）、HQA（Liu et al.）
- **NMT 攻击**：Seq2Sick（Cheng et al.）、TransFool（Sadrizadeh et al.）、Morphin（Tan et al.）
- **多任务攻击**：MTA（Guo et al., 2020）图像领域；MTADV（Wang et al., 2024）人脸认证
- **迁移攻击**：替代模型训练 + 辅助数据（Li et al., 2020；Sun et al., 2022）

## 评分 ⭐⭐⭐⭐

填补了文本多任务对抗攻击的空白，核心思想（深层聚类 → 统一攻击）清晰优雅。100 次查询的效率令人印象深刻。但 LLM 上的攻击成功率仍有提升空间，且缺少防御讨论。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICML 2025\] Adaptive Multi-prompt Contrastive Network for Few-shot Out-of-distribution Detection](../../ICML2025/ai_safety/adaptive_multi-prompt_contrastive_network_for_few-shot_out-of-distribution_detec.md)
- [\[CVPR 2025\] Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis](../../CVPR2025/ai_safety/mind_the_gap_detecting_black-box_adversarial_attacks_in_the_making_through_query.md)
- [\[ICCV 2025\] Active Membership Inference Test (aMINT): Enhancing Model Auditability with Multi-Task Learning](../../ICCV2025/ai_safety/active_membership_inference_test_amint_enhancing_model_auditability_with_multi-t.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/ai_safety/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[CVPR 2026\] Robustness Under Data Scarcity: Few-Shot Continual Adversarial Training for Evolving Threats](../../CVPR2026/ai_safety/robustness_under_data_scarcity_few-shot_continual_adversarial_training_for_evolv.md)

</div>

<!-- RELATED:END -->
