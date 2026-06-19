---
title: >-
  [论文解读] PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models
description: >-
  [CVPR 2025][多模态VLM][提示敏感性] 提出PARC框架，通过**11种语言/视觉提示变异**、**可靠性评分**和**指标校准**三大支柱，首次系统量化分析了22个VLM在7个数据集上的提示敏感性，发现VLM继承了LLM的语言敏感性并在视觉域呈现对称表现，InternVL2家族对提示变化最鲁棒。
tags:
  - "CVPR 2025"
  - "多模态VLM"
  - "提示敏感性"
  - "VLM鲁棒性"
  - "可靠性评分"
  - "指标校准"
  - "提示变异"
---

# PARC: A Quantitative Framework Uncovering the Symmetries within Vision Language Models

**会议**: CVPR 2025  
**arXiv**: [2506.14808](https://arxiv.org/abs/2506.14808)  
**代码**: [https://github.com/NVlabs/PARC](https://github.com/NVlabs/PARC)  
**领域**: 多模态VLM  
**关键词**: 提示敏感性, VLM鲁棒性, 可靠性评分, 指标校准, 提示变异

## 一句话总结
提出PARC框架，通过**11种语言/视觉提示变异**、**可靠性评分**和**指标校准**三大支柱，首次系统量化分析了22个VLM在7个数据集上的提示敏感性，发现VLM继承了LLM的语言敏感性并在视觉域呈现对称表现，InternVL2家族对提示变化最鲁棒。

## 研究背景与动机
VLM正在自动驾驶、医疗筛查等安全关键场景中部署，但它们继承了LLM的一个致命弱点——**提示敏感性**。同一语义的不同表述可能导致模型给出截然不同的回答（如问"Which lane has people?"时调换图片顺序就可能导致错误回答）。然而，提示敏感性在VLM中几乎未被研究：现有工作仅关注噪声/损坏的提示（非真实用户场景），缺乏统一的可靠性度量，且不同数据集/提示变异间的分数无法直接比较。核心矛盾是：**我们不知道哪些提示变化对VLM最具破坏性，也不知道哪些VLM对提示变化最鲁棒**。本文的切入角度是构建一套完整的分析框架，覆盖语言和视觉两个模态的提示变异，提出可解释的可靠性评分并引入校准机制。

## 方法详解

### 整体框架
PARC框架有三大支柱：(1) 语言和视觉域共11种现实的提示变异（分为重述型和语义改变型）；(2) 新颖的可靠性评分（结合准确率和确定性）；(3) 基于随机基线的指标校准（使不同数据集和提示变异间的分数可比）。在多选视觉问答（MC-VQA）任务上，对22个VLM在7个数据集上进行系统评估。

### 关键设计
1. **11种提示变异设计（Prompt Variations）**:
    - 功能：系统性生成真实用户可能产生的提示变体，覆盖语言和视觉两个维度
    - 核心思路：分两大类×两个模态 = 四组变异：
        - **语言重述（LR）**：指令化 LR-I（"State which..."）、简洁化 LR-C（减少词数）、冗长化 LR-V（增加词数），不改变正确答案
        - **语言语义（LS）**：否定 LS-N（加"not"）、反义词 LS-A、more/less互换 LS-M，改变正确答案
        - **视觉重述（VR）**：模糊 VR-B、亮度变化 VR-L、旋转90° VR-R，不改变正确答案
        - **视觉语义（VS）**：图像交换 VS-S、图像替换 VS-E，改变正确答案
        使用LLaMA3-70B自动生成语言变异，手动检查结果质量
    - 设计动机：首次在VLM中镜像LLM的语言变异到视觉域，探索两个模态的对称性

2. **可靠性评分（Reliability Score）**:
    - 功能：将准确率和确定性整合为单一、直观、带保证的可靠性指标
    - 核心思路：定义 $\mathit{rel} = (2 \cdot \mathit{acc} - 1) \cdot \mathit{cert}$，其中确定性基于conformal prediction计算：$\mathit{cert}(p) = 1 - \frac{|\mathcal{C}(p)|-1}{|\mathcal{P}(p)|-1}$。可靠性为1表示准确且自信（高度可靠），为-1表示自信但错误（高度不可靠），为0表示不确定。提供两个显式保证：$\mathit{cert} \geq |\mathit{rel}|$ 且 $\mathit{acc}_{\text{calib}} \geq \mathit{rel}$（正值时）
    - 设计动机：现有分析需要同时看准确率、确定性、一致性等多个指标，缺少一个"一看即懂"的综合分数。可靠性评分能在单个数字中传达最关键的信息

3. **指标校准（Score Calibration）**:
    - 功能：消除不同数据集和提示变异间的难度差异，使分数可直接跨数据集、跨提示比较
    - 核心思路：将分数校准为相对于随机基线的提升幅度：
$$s_{\text{calib}} = \begin{cases} \frac{s - s_{\text{rand}}}{1 - s_{\text{rand}}} & s \geq s_{\text{rand}} \\ \frac{s - s_{\text{rand}}}{s_{\text{rand}}} & s < s_{\text{rand}} \end{cases}$$
        校准后分数∈[-1,1]，1为理想，0为随机，-1为最差。对可靠性额外引入幂次校准：将 $\mathit{acc}$ 替换为 $\mathit{acc}^m$，其中 $m = \frac{\log 2}{\log(1/\mathit{acc}_{\text{rand}})}$，将中性可靠性0点移至随机准确率处
    - 设计动机：MMBench有3个选项（随机准确率0.27），NYU-Depth仅2个选项（随机0.5），否定后正确答案变多（更容易猜对）。不校准则无法公平比较

### 损失函数 / 训练策略
本文是评估分析框架，不涉及模型训练。

## 实验关键数据

### 主实验（22个VLM的可靠性排名，校准后，跨数据集平均）

| 模型 | 可靠性 AVG | 准确率C AVG | 确定性C AVG | 一致性 AVG |
|------|-----------|-------------|-------------|-----------|
| InternVL2 40B | **0.40** | **0.65** | 0.57 | **0.71** |
| InternVL2 26B | 0.38 | 0.61 | **0.58** | 0.70 |
| LLaVA-1.6 34B | 0.33 | 0.54 | 0.57 | 0.63 |
| InternVL2 8B | 0.32 | 0.56 | 0.52 | 0.63 |
| Cambrian 34B | 0.30 | 0.52 | 0.52 | 0.64 |
| CogVLM GG | -0.12 | -0.15 | 0.40 | 0.02 |
| Qwen-VL | 0.06 | 0.15 | 0.29 | 0.14 |

### 提示变异破坏性排名（22模型平均，校准后可靠性）

| 提示变异 | 可靠性 AVG | 一致性 AVG | 最具破坏性 |
|----------|-----------|-----------|-----------|
| 原始 O | 0.29 | - | - |
| LR-I 指令化 | 0.26 | 0.78 | |
| LR-V 冗长化 | **0.21** | 0.63 | ✓ 语言重述最差 |
| LS-A 反义词 | **0.10** | 0.09 | ✓ 语言语义最差 |
| VR-L 亮度变化 | **0.21** | 0.63 | ✓ 视觉重述最差 |
| VS-E 图像替换 | **0.13** | 0.00 | ✓ 视觉语义最差 |

### 关键发现
- **VLM继承了LLM的提示敏感性**，且在视觉域表现出对称的脆弱性——语言重述/语义变化的模式完美映射到视觉域
- **改变正确答案的变化最具破坏性**：语义变化（否定、反义词）远比简单重述更难处理，这在两个模态中一致
- **模型族比模型大小更重要**：InternVL2-2B的可靠性与LLaVA-1.5 13B相当，家族内大模型更好但家族间差距更大
- **高质量训练数据是关键**：1B低质量网络数据训练的模型不如0.01B精心策刘数据训练的Cambrian可靠

## 亮点与洞察
- **首次语言-视觉对称性分析**：发现语言重述vs视觉重述、语言语义vs视觉语义的行为模式惊人地"对称"
- **校准的价值**：揭示了未校准分数导致的误判——如MMBench上模型看似在否定问题上表现更好，实际是因为否定后有更多正确答案
- **可靠性评分的优雅**：$\mathit{rel} = (2 \cdot \mathit{acc} - 1) \cdot \mathit{cert}$ 简单公式同时编码准确性和置信度，且自带两个保证

## 局限与展望
- 仅支持白盒VLM（需要softmax分数计算确定性），无法分析API-only模型
- 报告MC-VQA结果，生成式任务需要额外LLM验证答案，引入噪声
- 视觉语义变化种类少于其他类型（制作成本高），可能影响结论泛化性
- 未探索prompting技巧（如CoT）能否缓解敏感性

## 相关工作与启发
- 与CompBench的比较式VQA风格一致，但首次用于系统性提示敏感性分析
- 校准思想可推广到任何benchmark的跨数据集比较
- 启发：选择VLM时不应只看accuracy排行榜，提示鲁棒性（reliability）可能更重要。InternVL2的成功暗示训练数据策略是提升鲁棒性的关键路径

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首创VLM提示敏感性分析框架，视觉-语言对称性发现令人惊艳
- 实验充分度: ⭐⭐⭐⭐⭐ 22个模型×7数据集×11变异，分析极其全面，结论可靠
- 写作质量: ⭐⭐⭐⭐⭐ 数学严谨，图表直观（特别是校准前后对比），逻辑链完整
- 价值: ⭐⭐⭐⭐⭐ 对VLM的可靠性评估和模型选择有重大指导意义，框架可直接被社区采用

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)
- [\[ICML 2025\] Ranked from Within: Ranking Large Multimodal Models Without Labels](../../ICML2025/multimodal_vlm/ranked_from_within_ranking_large_multimodal_models_without_labels.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[CVPR 2025\] UPME: An Unsupervised Peer Review Framework for Multimodal Large Language Model Evaluation](upme_an_unsupervised_peer_review_framework_for_multimodal_large_language_model_e.md)

</div>

<!-- RELATED:END -->
