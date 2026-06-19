---
title: >-
  [论文解读] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage
description: >-
  [ACL 2025][多模态VLM][越狱攻击] 提出多模态链接（MML）攻击框架，通过跨模态加密-解密机制和"邪恶对齐"策略，以极高成功率（GPT-4o上达99%+）越狱当前最先进的视觉语言模型。 随着GPT-4o等大型视觉语言模型（VLM）的快速发展，其潜在的滥用和安全风险引发了广泛关注。现有的越狱攻击方法主要分为三类：…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "越狱攻击"
  - "视觉语言模型"
  - "多模态安全"
  - "加密解密"
  - "对齐绕过"
---

# Jailbreak Large Vision-Language Models Through Multi-Modal Linkage

**会议**: ACL 2025  
**arXiv**: [2412.00473](https://arxiv.org/abs/2412.00473)  
**代码**: [github.com/wangyu-ovo/MML](https://github.com/wangyu-ovo/MML)  
**领域**: 多模态VLM  
**关键词**: 越狱攻击, 视觉语言模型, 多模态安全, 加密解密, 对齐绕过

## 一句话总结

提出多模态链接（MML）攻击框架，通过跨模态加密-解密机制和"邪恶对齐"策略，以极高成功率（GPT-4o上达99%+）越狱当前最先进的视觉语言模型。

## 研究背景与动机

随着GPT-4o等大型视觉语言模型（VLM）的快速发展，其潜在的滥用和安全风险引发了广泛关注。现有的越狱攻击方法主要分为三类：基于扰动的攻击（利用对抗性噪声）、基于结构的攻击（将恶意内容嵌入视觉元素）以及混合攻击。然而，现有方法在面对GPT-4o等最先进模型时效果大幅下降。

作者分析了现有结构化攻击方法失败的两个关键原因：

**恶意内容过度暴露**：直接在输入图像中展示有害内容（如炸弹图片或恶意排版文字），随着VLM图像理解能力和安全对齐的增强，这些显性内容很容易触发拒绝机制。

**中性文本引导**：缺乏隐蔽的恶意引导文本，即使模型没有直接拒绝响应，其输出也往往局限于道德建议或法律提醒——本质上是一种隐式拒绝。

基于这一洞察，作者借鉴密码学思想，提出了跨模态加密-解密的MML攻击框架。

## 方法详解

### 整体框架

MML攻击遵循以下流程：首先将恶意查询转换为排版图像（类似FigStep），然后对图像进行加密以隐藏恶意信息，在推理阶段通过文本提示引导模型解密图像内容，最后利用"邪恶对齐"策略将模型输出与恶意目标对齐。整个攻击在黑盒设定下进行，攻击者无需了解目标模型的参数或架构。

### 关键设计

1. **加密模块（Encryption）**: 为减少恶意内容的直接暴露，采用四种加密策略：

    - **词替换（Word Replacement）**：利用NLTK进行词性标注，将恶意名词替换为食物相关词汇，将恶意形容词替换为正面描述词。例如"illegal drugs"变为"delicious pancakes"。
    - **图像镜像（Image Mirroring）**：对包含排版提示的图像进行水平翻转。
    - **图像旋转（Image Rotation）**：对图像进行几何旋转变换。
    - **Base64编码**：将恶意文本编码为Base64格式并渲染为排版图像，视觉上晦涩但机器可解码。

2. **解密模块（Decryption）**: 在推理阶段引导模型以链式思维（CoT）方式逐步解密：

    - 从图像中提取标题内容
    - 应用提供的替换字典重建原始标题
    - 提供原始恶意查询的打乱词列表作为解密提示（hint），用于校验解密结果
    - 基于重建的标题生成最终输出
   
   打乱词列表的设计巧妙地隐藏了有害信息，同时提供了足够的解密线索。

3. **邪恶对齐（Evil Alignment）**: 受Zeng等人启发，将攻击嵌入虚构的电子游戏开发场景：输入图像被描述为反派巢穴中一块缺失内容的屏幕，模型被要求以符合反派目标的方式补全内容。这种叙事框架将恶意意图伪装为创意任务，有效绕过安全过滤器。邪恶对齐与加密解密过程互补，显著提升了攻击的隐蔽性和成功率。

### 损失函数 / 训练策略

MML是一个纯推理时的攻击框架，不涉及模型训练或梯度计算。其核心优势在于完全黑盒、无需模型访问权限，仅通过精心设计的多模态输入实现攻击。

## 实验关键数据

### 主实验

| 数据集 | 指标 | MML-最佳 | FigStep (SOTA) | 提升 |
|--------|------|----------|----------------|------|
| SafeBench (GPT-4o) | ASR | 97.80% (旋转) | 33.00% | +64.80% |
| MM-SafeBench (GPT-4o) | ASR | 98.81% (旋转) | 6.86% | +91.95% |
| HADES-Dataset (GPT-4o) | ASR | 99.07% (B64) | 4.00% | +95.07% |
| SafeBench (Claude-3.5) | ASR | 69.40% (镜像) | 16.60% | +52.80% |
| MM-SafeBench (Claude-3.5) | ASR | 60.00% (镜像) | 9.32% | +50.68% |

### 消融实验

| 配置 | ASR(%) | DSR(%) | 说明 |
|------|--------|--------|------|
| FigStep基线 | 34.00 | - | 无加密无对齐 |
| +加密解密 | 75.20 | 64.20 | ASR提升41.2% |
| +加密解密+提示 | 79.80 | 59.80 | 提示提升ASR但DSR下降 |
| +加密解密+邪恶对齐 | 96.20 | 65.40 | 邪恶对齐贡献最大 |
| 完整MML | 97.60 | 91.60 | 三组件协同效果最佳 |

### 关键发现

- 图像变换类加密（镜像、旋转）在大多数情况下优于词替换和Base64编码
- Claude-3.5-Sonnet是最鲁棒的模型，可能针对Base64编码进行了专门的防御训练
- 邪恶对齐是提升ASR的最关键组件，单独使用加密解密时模型仍倾向于拒绝响应
- MML在OpenAI o1推理模型上也能保持超越基线29.6%的ASR
- 加解密提示虽提升ASR，但在无邪恶对齐时反而降低解密成功率，因为微小错误（如单复数、标点）不影响恶意内容传达

## 亮点与洞察

- **跨模态弱点利用**：MML将模态间的链接视为VLM的薄弱环节，通过在不同模态间分散恶意信息来绕过安全机制，这一洞察非常深刻
- **灵活可扩展**：框架可集成任意编码策略，只要目标VLM能在推理时解码
- **实用性强**：完全黑盒、单轮对话、无需系统提示修改，具有高度实际威胁性
- **防御启示**：揭示了当前VLM安全对齐在跨模态场景下的根本脆弱性

## 局限与展望

- 攻击对Claude-3.5-Sonnet的成功率相对较低（最高69.4%），说明针对性防御是可能的
- 词替换加密的制作时间较长（500张图片需120秒 vs 镜像仅需2.37秒）
- 论文主要关注结构化攻击，未充分探讨与扰动攻击结合的混合方法
- 防御方案的讨论较为有限，实验显示添加安全提示词后MML效果显著下降（如词替换ASR从96%降至80%）
- 未来可以探索更强的多模态一致性检查机制作为防御措施

## 相关工作与启发

- FigStep（Gong et al., 2023）开创了将恶意文本转为排版图像的方法，但直接暴露有害内容
- HADES（Li et al., 2024c）结合了结构化和扰动攻击，但仍需梯度信息
- 邪恶对齐策略源自DeepInception（Zeng et al., 2024）的虚拟场景方法
- 本文的加密-解密思路可以启发新的防御方法：检测跨模态间的语义不一致性

## 评分

- 新颖性: ⭐⭐⭐⭐ 跨模态加密解密的思路新颖，但邪恶对齐借鉴自前人工作
- 实验充分度: ⭐⭐⭐⭐⭐ 三个数据集、四个模型、详细消融和多维度分析
- 写作质量: ⭐⭐⭐⭐ 结构清晰，图表丰富，可读性好
- 价值: ⭐⭐⭐⭐⭐ 揭示了当前顶级VLM的严重安全漏洞，对安全研究社区具有重要警示意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)
- [\[ACL 2025\] Weaving Context Across Images: Improving Vision-Language Models through Focus-Centric Visual Chains](weaving_context_across_images_improving_vision-language_models_through_focus-cen.md)
- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[ACL 2025\] SPHERE: Unveiling Spatial Blind Spots in Vision-Language Models Through Hierarchical Evaluation](sphere_unveiling_spatial_blind_spots_in.md)
- [\[ACL 2025\] Transferring Textual Preferences to Vision-Language Understanding through Model Merging](transferring_textual_preferences_to_vision-language_understanding_through_model_.md)

</div>

<!-- RELATED:END -->
