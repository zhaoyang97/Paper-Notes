---
title: >-
  [论文解读] Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs
description: >-
  [NeurIPS 2025][多模态VLM][视频安全] 构建首个面向视频 LVLM 安全评估的综合基准 Video-SafetyBench，包含 2264 个视频-文本对覆盖 48 个细粒度不安全类别，通过可控视频生成管线和基于 LLM 置信度的 RJScore 指标，对 24 个 LVLM 进行大规模安全评测，揭示良性查询下视频攻击平均成功率达 67.2%。
tags:
  - "NeurIPS 2025"
  - "多模态VLM"
  - "视频安全"
  - "LVLM评测"
  - "攻击成功率"
  - "多模态安全基准"
  - "RJScore"
---

# Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs

**会议**: NeurIPS 2025  
**arXiv**: [2505.11842](https://arxiv.org/abs/2505.11842)  
**代码**: [https://liuxuannan.github.io/Video-SafetyBench.github.io/](https://liuxuannan.github.io/Video-SafetyBench.github.io/)  
**领域**: Multimodal / VLM Safety  
**关键词**: 视频安全, LVLM评测, 攻击成功率, 多模态安全基准, RJScore

## 一句话总结

构建首个面向视频 LVLM 安全评估的综合基准 Video-SafetyBench，包含 2264 个视频-文本对覆盖 48 个细粒度不安全类别，通过可控视频生成管线和基于 LLM 置信度的 RJScore 指标，对 24 个 LVLM 进行大规模安全评测，揭示良性查询下视频攻击平均成功率达 67.2%。

## 研究背景与动机

大型视觉语言模型（LVLM）日益广泛部署，其安全性评估至关重要。然而现有多模态安全评估存在关键空白：

**现有基准聚焦于静态图像**：FigStep、MM-SafetyBench、HADES、VLSBench 等均只考虑图像-文本输入，忽略了视频的时序动态特性可能引入的独特安全风险（如随时间演化的有害动作）

**视频输入扩大了攻击面**：相比单帧图像，视频的连续帧序列在安全对齐上面临更大挑战，攻击者可利用时序信息绕过安全机制

**评估指标不足以处理边界情况**：现有自动评判模型对不确定或边界有害输出的判断能力有限，缺乏与人类判断对齐的校准机制

**核心矛盾**：视频 LVLM 的安全风险日益突出，但缺乏系统的视频-文本攻击基准和可靠的安全评估方法。

**切入角度**：(1) 构建组合式视频-文本攻击任务，包含带显式恶意的有害查询和利用视频上下文隐式传达恶意的良性查询；(2) 设计可控视频生成管线确保视频与有害意图语义对齐；(3) 提出基于 LLM 置信度的评判指标 RJScore 并校准决策阈值。

## 方法详解

### 整体框架

Video-SafetyBench 包含三大组成：(1) 涵盖 13 个主类别和 48 个子类别的安全分类体系；(2) 三阶段可控视频生成管线；(3) 基于 LLM 置信度的 RJScore 评判指标。每个视频配对一个有害查询和一个良性查询变体。

### 关键设计

1. **两级安全分类体系**:

    - 功能：定义系统的视频安全风险层级
    - 核心思路：13 个主类别（暴力犯罪、非暴力犯罪、性犯罪、儿童性剥削、诽谤、专业建议、隐私、知识产权、大规模杀伤性武器、仇恨、自杀自残、性内容、选举）和 48 个细粒度子类别
    - 设计动机：借鉴现有 LLM 安全分类体系并针对视频场景扩展，确保评估覆盖面

2. **三阶段可控视频生成管线**:

    - 功能：合成与有害意图语义对齐的视频
    - 核心思路：将视频语义分解为"展示什么"（主体图像）和"如何运动"（运动文本），分三步执行：
        - **Stage 1（文本）**：基于安全策略生成有害查询，并通过 LLM 改写为良性查询（将有害短语替换为视频引用表达，如"高爆炸药" → "视频中展示的装置"）
        - **Stage 2（文本→图像）**：用 LLM（GPT-4o）将抽象查询转化为丰富场景描述，再用 T2I 模型（Midjourney-V6、KLING 1.5）生成主体图像
        - **Stage 3（图像+文本→视频）**：用 LVLM 推断运动轨迹描述，结合主体图像输入 I2V 模型（KLING 1.6、Sora、即梦）生成 10 秒视频
    - 设计动机：直接用 T2V 模型难以精确控制复杂语义，分解为主体+运动两个维度可大幅提升可控性。最终数据集 FID=73（视频质量）、VQAScore=0.522（文本-视频相关性）均优于现有基准

3. **RJScore 评判指标**:

    - 功能：量化模型输出的有害性并与人类判断对齐
    - 核心思路：使用 Qwen2.5-72B 对输出进行 5 级毒性评分，不取最终预测，而是利用 5 个候选 token 的 logit 值计算 softmax 概率，再算期望分数 $RJScore = \sum_{k=1}^{5} k \cdot p(k)$。通过 5 折交叉验证校准决策阈值 τ=2.85
    - 设计动机：传统二分类判断无法处理不确定和边界情况。利用 token 级 logit 分布捕捉评判置信度，校准阈值使与人类标注一致率达 91%（高于 GPT-4o 的 88.2%）

### 攻击模式

- **有害查询攻击**：文本显式表达恶意意图，视频放大有害效果
- **良性查询攻击**：文本本身无害，但通过引用视频内容隐式传达恶意（如"丢下视频中的装置"替代"投掷高爆炸药"）

## 实验关键数据

### 主实验

对 24 个 LVLM（7 个闭源 + 17 个开源）的攻击成功率（ASR）评测：

| 模型 | 有害查询 ASR | 良性查询 ASR | 说明 |
|------|-------------|-------------|------|
| Qwen-VL-Max | 25.4% | **78.3%** | 良性查询 ASR 高 52.9% |
| GPT-4o | 14.8% | 43.3% | 闭源中最安全 |
| Claude 3.5 Sonnet | 7.8% | 19.9% | 整体最安全 |
| Qwen2-VL-72B | 44.6% | 83.3% | 开源 72B 中最脆弱 |
| Qwen2.5-VL-7B | - | 68.7% | 7B 比 72B 反而更安全 |
| Qwen2.5-VL-72B | - | 74.0% | 大模型≠更安全 |
| **平均（所有模型）** | 39.1% | **67.2%** | 良性查询高 28.1% |

### 消融实验

| 评判模型 | 与人类一致率 | F1 | FPR | FNR |
|---------|------------|-----|------|------|
| Rule-based | 76.5% | 75.1% | 46.4% | 0.7% |
| HarmBench | 77.1% | 76.1% | 2.7% | 43.0% |
| Llama Guard 3 | 79.5% | 79.4% | 12.2% | 28.6% |
| GPT-4o | 88.2% | 88.1% | 19.7% | 3.9% |
| Qwen-2.5-72B | 88.4% | 88.3% | 18.4% | 4.7% |
| **RJScore (τ=2.85)** | **91.0%** | **91.0%** | **12.3%** | 5.8% |

### 关键发现

- **良性查询远比有害查询危险**：良性查询平均 ASR 比有害查询高 28.1%，说明模型难以识别通过视频引用隐式传达的恶意
- **模型大小≠更安全**：同系列 Qwen2.5-VL 的 7B/32B/72B 在良性查询下 ASR 分别为 68.7%/73.2%/74.0%，模型越大攻击成功率反而越高
- **视频比图像更危险**：视频输入的 ASR 比静态图像平均高 8.6%，时序信息带来额外风险
- **专业建议（S6-SA）类别最脆弱**：几乎所有模型在该类别上 ASR 都超过 70%，尤其在良性查询下

## 亮点与洞察

- 首次系统地研究视频-文本组合攻击对 LVLM 的安全威胁，填补了视频安全评估的空白
- 可控视频生成管线的"主体 + 运动"分解思路简洁有效，确保了视频与有害意图的语义对齐
- RJScore 的 logit 置信度 + 交叉验证阈值校准方案，为 LLM-as-Judge 提供了更可靠的范式
- 良性查询攻击的高成功率揭示了 LVLM 在隐式恶意推理上的根本薄弱环节

## 局限与展望

- 当前仅用合成视频，真实视频的攻击效果可能不同
- 良性查询的改写策略相对简单（模板化替换），更复杂的语义隐藏手段值得探索
- 未涵盖音频模态引入的安全风险
- 评估仅关注输出有害性，未考虑模型拒绝有用请求的过度防御问题

## 相关工作与启发

- 对现有图像安全基准（MM-SafetyBench、VLSBench 等）的视频维度扩展方向有很强指导意义
- RJScore 的置信度量化方法可迁移到其他 LLM 评判场景
- 良性查询攻击与 VLSBench 的"无害文本+有害图像"思路一脉相承，但在视频领域风险更大

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[NeurIPS 2025\] Learning Skill-Attributes for Transferable Assessment in Video](learning_skill-attributes_for_transferable_assessment_in_video.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](training-free_online_video_step_grounding.md)
- [\[CVPR 2026\] Breaking Multimodal LLM Safety via Video-Driven Prompting](../../CVPR2026/multimodal_vlm/breaking_multimodal_llm_safety_via_video-driven_prompting.md)
- [\[NeurIPS 2025\] MME-VideoOCR: Evaluating OCR-Based Capabilities of Multimodal LLMs in Video Scenarios](mme-videoocr_evaluating_ocr-based_capabilities_of_multimodal_llms_in_video_scena.md)
- [\[ACL 2025\] GODBench: A Benchmark for Multimodal Large Language Models in Video Comment Art](../../ACL2025/multimodal_vlm/godbench_a_benchmark_for_multimodal_large_language_models_in_video_comment_art.md)

</div>

<!-- RELATED:END -->
