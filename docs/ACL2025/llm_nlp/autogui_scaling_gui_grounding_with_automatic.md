---
title: >-
  [论文解读] AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs
description: >-
  [ACL 2025][LLM 其他][GUI grounding] 提出AutoGUI自动标注管线——通过模拟交互比较UI状态变化+LLM推断元素功能+双LLM验证过滤，构建704K高质量UI功能标注数据集，标注正确率96.7%可比人类，显著提升VLM的UI grounding能力且展现数据扩展效应。
tags:
  - "ACL 2025"
  - "LLM 其他"
  - "GUI grounding"
  - "功能标注"
  - "VLM"
  - "自动管线"
  - "UI理解"
---

# AutoGUI: Scaling GUI Grounding with Automatic Functionality Annotations from LLMs

**会议**: ACL 2025  
**arXiv**: [2502.01977](https://arxiv.org/abs/2502.01977)  
**代码**: [项目页面](https://autogui-project.github.io/)  
**领域**: GUI理解 / VLM  
**关键词**: GUI grounding, 功能标注, VLM, 自动管线, UI理解

## 一句话总结

提出AutoGUI自动标注管线——通过模拟交互比较UI状态变化+LLM推断元素功能+双LLM验证过滤，构建704K高质量UI功能标注数据集，标注正确率96.7%可比人类，显著提升VLM的UI grounding能力且展现数据扩展效应。

## 研究背景与动机

**领域现状**: VLM在UI理解上潜力巨大但受限于数据瓶颈——UI数据集规模远小于自然图像数据集。**现有痛点**: 现有标注仅是元素alt-text或简短意图，缺乏上下文化的功能语义描述（如两个相同放大镜图标可能分别代表"搜索"和"缩放"）。**核心矛盾**: 需要大规模高质量的上下文化UI元素功能标注，但人工标注成本禁止性地高。**本文目标**: 设计全自动标注管线，无需人工即可大规模生成高质量UI元素功能描述。**切入角度**: 通过"点击元素后UI发生什么变化"来推断其功能——类似人类探索未知界面的方式。**核心idea**: 用LLM比较交互前后的UI状态变化来自动推断元素功能，结合双重LLM质控确保质量。

## 方法详解

### 整体框架

三阶段管线：(1) 自动爬取Web/Android UI交互轨迹；(2) LLM功能推断+LLM-aided rejection+双LLM verification；(3) 转化为grounding/referring任务数据微调VLM。

### 关键设计

1. **基于UI状态变化的功能推断**:
    - 功能：利用交互前后的UI AXTree差异推断元素功能
    - 核心思路：paper_notes/docs/ACL2025/multilingual_mt/cosmmic_commentsensitive_multimodal_multilingual_indian_corpus.md = \text{LLM}(p_{\text{anno}}, s_t, s_{t+1})$，用difflib生成AXTree前后的行级差异（增/删/移位/属性更新），LLM通过Chain-of-Thought分析变化并总结功能
    - 设计动机：不是看元素外观而是看"点击后发生什么"——一个放大镜如果点击后出现搜索框就是搜索，出现缩放滑块就是缩放

2. **LLM-aided rejection（无效样本过滤）**:
    - 功能：LLM评估交互产生的状态变化是否足以推断功能
    - 核心思路：$\text{score} = \text{LLM}(p_{\text{reject}}, e, s_t, s_{t+1})$，按3个标准打分（变化明确度/相关性/可预测性），丢弃底部30%
    - 设计动机：非所有交互都产生有意义的状态变化——如未完全加载的页面或需要登录的重定向

3. **双LLM验证（标注质量控制）**:
    - 功能：用两个不同LLM交叉验证功能标注是否正确
    - 核心思路：Llama-3-70B和Mistral-7B分别对标注打分，仅双满分才保留
    - 设计动机：单LLM可能有系统性偏差（如对dropdown菜单的错误描述），双LLM交叉验证可互相纠正

### 损失函数 / 训练策略

VLM微调用LoRA，在8×A100上训练1 epoch。Qwen-VL和Qwen2-VL用LoRA，SliME仅冻结视觉编码器。坐标归一化到[0,999]。

## 实验关键数据

### 主实验

微调后的UI grounding准确率（%）：

| 模型 | FuncPred | ScreenSpot | ScreenSpot-v2 | MoTIF | VWB EG |
|------|:---:|:---:|:---:|:---:|:---:|
| Qwen-VL (基线) | 3.0 | 5.2 | 5.6 | 7.8 | 1.7 |
| Qwen-VL + AutoGUI | **48.7**(+45.7) | 41.2(+36.0) | 40.2(+34.6) | 44.0(+36.2) | 42.1(+40.4) |
| Qwen2-VL (基线) | 38.7 | 66.4 | 66.9 | 71.1 | 55.9 |
| Qwen2-VL + AutoGUI | **65.0**(+26.3) | 80.0(+13.6) | 83.2(+16.3) | 72.3(+1.2) | **90.3**(+34.4) |

### 消融实验

标注质量vs人类标注者对比：

| 标注器 | Rejector | Verifier | 正确率 |
|--------|----------|----------|:---:|
| Llama-3-70B | 无 | 无 | 64.5% |
| Llama-3-70B | 规则+LLM | 双LLM | **96.7%** |
| 人类标注者 | - | - | 95.5% |

数据扩展效应：25k→125k→702k，三个VLM的grounding准确率持续上升。

### 关键发现

1. **功能标注显著优于HTML/元数据标注**: 在FuncPred上比HTML标注高4倍
2. **清晰的数据扩展效应**: 数据量增加→性能持续提升
3. **可辅助GUI agent任务**: Qwen2-VL+AutoGUI在AITW上比Gemini原生grounding提升9.73%
4. **标注正确率96.7%可比受训人类标注者（95.5%）**

## 亮点与洞察

- **"交互差异推断功能"的设计思路非常巧妙**——模仿人类探索未知界面的方式
- **双重LLM质控**确保全自动管线质量与人工可比
- **数据扩展效应**表明管线可持续扩大规模
- **功能标注>HTML标注**的发现证明了上下文化语义描述的价值

## 局限与展望

- 仅覆盖Web和Android，iOS和桌面应用未涉及
- 当前管线无法标注修改互联网内容的元素（如发帖、购买）
- 缺乏任务导向的交互轨迹（仅随机交互）
- 部分低流量网站在手机分辨率下渲染失真

## 相关工作与启发

- **vs SeeClick（Cheng et al. 2024）**: 仅用静态页面+HTML标注——本文用交互差异+LLM推断
- **vs Widget Captioning（Li et al. 2020）**: 人工标注163K——本文全自动704K且质量可比
- **vs UGround（Gou et al. 2025）**: 仅简短功能描述——本文提供上下文化详细功能标注
- **启发**: UI理解的数据瓶颈可通过自动交互+LLM推断彻底解决

## 评分

- 新颖性: ⭐⭐⭐⭐ 首个全自动UI功能标注管线，交互差异推断功能思路新颖
- 实验充分度: ⭐⭐⭐⭐ 704K数据集+多VLM+扩展性+agent应用
- 写作质量: ⭐⭐⭐⭐ 管线描述清晰
- 价值: ⭐⭐⭐⭐⭐ 解决UI理解的数据瓶颈，对GUI agent有基础设施价值

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)
- [\[ACL 2025\] AutoExp: Automatic Experiment Design and Execution by LLMs](autoexp_automatic_experiment_design_and_execution_by_llms.md)
- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)
- [\[ACL 2025\] NewsInterview: a Dataset and a Playground to Evaluate LLMs' Grounding Gap via Informational Interviews](newsinterview_a_dataset_and_a_playground_to_evaluate_llms_grounding_gap_via_info.md)
- [\[ACL 2025\] LESA: Learnable LLM Layer Scaling-Up](lesa_learnable_llm_layer_scaling-up.md)

</div>

<!-- RELATED:END -->
