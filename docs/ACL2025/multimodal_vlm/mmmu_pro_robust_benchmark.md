---
title: >-
  [论文解读] MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark
description: >-
  [ACL 2025][多模态VLM][多模态] 在 MMMU 基础上通过三步加固（过滤纯文本可解题目、扩展选项至 10 个、引入 Vision-only 输入）构建更鲁棒的 MMMU-Pro 基准，所有模型性能下降 16.8%~26.9%，揭示当前多模态模型远未实现真正的跨模态理解。 领域现状：MMMU 是评估多模态大模型学…
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "多模态"
  - "MMMU"
  - "vision-only evaluation"
  - "shortcut exploitation"
  - "robust evaluation"
---

# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

**会议**: ACL 2025  
**arXiv**: [2409.02813](https://arxiv.org/abs/2409.02813)  
**代码**: [https://mmmu-benchmark.github.io/#leaderboard](https://mmmu-benchmark.github.io/#leaderboard)  
**领域**: 多模态VLM / Benchmark  
**关键词**: multimodal benchmark, MMMU, vision-only evaluation, shortcut exploitation, robust evaluation

## 一句话总结
在 MMMU 基础上通过三步加固（过滤纯文本可解题目、扩展选项至 10 个、引入 Vision-only 输入）构建更鲁棒的 MMMU-Pro 基准，所有模型性能下降 16.8%~26.9%，揭示当前多模态模型远未实现真正的跨模态理解。

## 研究背景与动机

**领域现状**：MMMU 是评估多模态大模型学科理解能力的标杆基准，包含 11.5K 道大学级多学科题目。GPT-4o 已在 MMMU 上达到 69.1% 准确率，似乎接近人类专家水平。

**现有痛点**：深入分析发现，很多 MMMU 题目存在严重的"捷径"问题——纯文本 LLM（无视觉输入）就能答对部分题目。原因有二：(1) 部分题目实际不依赖图像即可解答；(2) 模型利用选项中的统计模式和预训练知识猜答，无需真正理解图像。例如 Llama-3-70B 仅凭文本就能在部分题目上找到选项间的 shortcut。

**核心矛盾**：4 选项的选择题格式给了模型 25% 的随机猜对概率，加上选项中的语义线索，实际猜对概率更高。这意味着 MMMU 的高分可能有很大"水分"——模型的真实多模态理解能力被高估。

**本文目标** (1) 如何过滤掉纯文本可解的"伪多模态"题目？(2) 如何降低选项猜测的成功率？(3) 如何测试模型在文本嵌入图像时的"看+读"综合能力？

**切入角度**：从人类认知能力出发——人类在日常中天然地同时处理嵌入在视觉场景中的文字（如读截图、看海报），这种视觉-文本无缝整合是核心认知能力。Vision-only 设置直接测试模型是否具备这种能力。

**核心 idea**：通过过滤+扩选项+Vision-only 三步加固，将 MMMU 升级为更能反映真实多模态理解能力的 MMMU-Pro。

## 方法详解

### 整体框架
三步构建流程：从 MMMU 出发，依次执行过滤、增强、视觉化，最终得到 3,460 道题（1,730 标准 + 1,730 截图/照片）。MMMU-Pro 总分 = 10 选项标准成绩与 Vision-only 成绩的平均。

### 关键设计

1. **纯文本可解题目过滤（Text-only Filtering）**:

    - 功能：移除不需要图像即可解答的"伪多模态"题目
    - 核心思路：选 4 个强开源 LLM（Llama3-70B, Qwen2-72B, Yi-1.5-34B, Mixtral-8×22B），每道题在无图像条件下各回答 10 次。如果某模型 >5 次答对，标记为"可解"。≥3 个模型均标记可解的题目被排除。从剩余题目中均匀采样 1,800 道（30 学科各 60 道）
    - 设计动机：4 个模型 × 10 次重复的多数投票机制确保过滤的稳健性，避免因单模型偶然猜中而误筛

2. **选项扩展至 10 个（Option Augmentation）**:

    - 功能：降低选择题的随机猜对概率（从 25% 降至 10%）
    - 核心思路：由人类专家借助 GPT-4o 生成额外选项，Claude 3.5 过滤不合理选项，再经两轮人工审核验证。同时审查原始题目与图像的关联性，去除不连贯题目（过滤 70 道，最终 1,730 道）
    - 设计动机：实验证实仅增加选项数就能让纯文本 LLM 准确率大幅下降（图 3），有效抑制了基于选项的猜测策略

3. **Vision-only 输入设置（Vision-only Input Setting）**:

    - 功能：测试模型在文本嵌入图像中时的真正"看+读"能力
    - 核心思路：人工标注员将题目文本和选项嵌入截图/照片中，变换背景、字体样式、字号以模拟真实场景多样性。模型只接收图像输入，不接收任何显式文本
    - 设计动机：模仿用户实际使用习惯（分享截图而非手动输入文本），测试模型是否具备人类"无缝整合视觉与文本信息"的核心认知能力

### 人类专家性能估计
基于原始 MMMU 人工评估数据近似——核心题目内容未变，且人类专家需写出完整解题过程（降低猜测），视觉-文本整合是人类天然能力。Human Expert 三档：75.4%/82.1%/88.6%（Low/Medium/High），远超所有模型。

## 实验关键数据

### 主实验

| 模型 | Standard(4选项) | Standard(10选项) | Vision-only | MMMU(Val) | Δ(10选项-MMMU) |
|------|----------------|-----------------|-------------|-----------|----------------|
| GPT-4o | 64.7% | 54.0% | 49.7% | 69.1% | -15.1% |
| Claude 3.5 Sonnet | 63.7% | 55.0% | 48.0% | 68.3% | -13.3% |
| Gemini 1.5 Pro | 60.6% | 49.4% | 44.4% | 65.8% | -16.4% |
| InternVL2-76B | 55.0% | 41.9% | 38.0% | 58.3% | -16.4% |
| LLaVA-OneVision-72B | 52.3% | 38.0% | 24.0% | 56.8% | -18.8% |
| VILA-1.5-40B | 46.8% | 35.9% | 14.1% | 51.9% | -16.0% |
| Human Expert(High) | 88.6% | 85.4% | 85.4% | 88.6% | -3.2% |

### 消融分析：CoT 与 OCR 影响

| 模型 | Standard w/o CoT | Standard w/ CoT | OCR Acc | Vision w/ OCR | Vision w/o OCR |
|------|-----------------|-----------------|---------|---------------|----------------|
| Claude 3.5 Sonnet | 42.7% | 55.0% | - | - | - |
| GPT-4o | - | - | 92.3% | 49.7% | 49.4% |
| InternVL2-40B | - | - | 85.5% | 32.1% | 28.9% |
| MiniCPM-V2.6 | - | - | 67.0% | 24.2% | 21.1% |

### 关键发现
- 选项从 4 扩展到 10 使 GPT-4o 下降 10.7%（64.7→54.0），Vision-only 再降 4.3%（54.0→49.7），总计下降 19.4%
- LLaVA-OneVision-72B 在 Vision-only 上暴跌 14.0%（38.0→24.0），暴露其文本嵌入图像理解的严重不足
- OCR 准确率普遍很高（GPT-4o 92.3%），但显式 OCR 提示对准确率几乎无影响（49.7% vs 49.4%），说明瓶颈不在文字识别而在深层理解
- CoT 在工程/科学等推理密集学科提升显著（GPT-4o +14.5%），在艺术设计类主观学科效果有限甚至为负
- 人类专家在所有加固步骤中仅下降 ~3%，模型下降 15-27%，差距巨大

## 亮点与洞察
- Vision-only 设置是个 able 且简单的强化手段——成本低（人工截图即可），但直击模型软肋：当文本不再作为显式输入时，"读图中的字"成了前所未有的挑战
- "增加选项数"这个简单操作就能让 benchmark 更 robust，对其他选择题型基准（如 ARC、ScienceQA）也有启发
- OCR 能力与视觉理解能力的分离是个重要发现——模型可以精准提取图中文字，但无法正确理解文字与视觉元素的关系和语境

## 局限与展望
- 人类专家性能为近似估计而非重新评测，可能高估人类在 Vision-only 上的表现
- Vision-only 照片/截图由标注员手动拍摄，规模和多样性受限
- 学科覆盖仍沿用 MMMU 的 30 学科，未新增编程/法律等实用领域
- 未测试 GPT-4o 之后的更新模型（如 o1、o3），结论可能随模型迭代而部分过时

## 相关工作与启发
- **vs MMMU**: MMMU-Pro 是 MMMU 的严格加固版，继承题目内容但通过三步构建消除 shortcut，定位为"同基准、更难版"
- **vs MathVista / ScienceQA**: 同样是多学科视觉推理基准，但 MMMU-Pro 的 Vision-only 设置和 10 选项设计显著更 robust
- **vs MMBench**: MMBench 侧重感知能力，MMMU-Pro 侧重学科知识推理，两者互补
- 启发：benchmark 设计的"三步加固"方法论（过滤伪题→增加难度→换输入模态）可作为通用的基准鲁棒化范式

## 评分
- 新颖性: ⭐⭐⭐⭐ Vision-only 设置和选项扩展的组合带来了真正有区分度的基准
- 实验充分度: ⭐⭐⭐⭐⭐ 20+ 模型、3 种设置、CoT/OCR 消融、学科维度分析，极其全面
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰、构建过程透明、结果呈现直观
- 价值: ⭐⭐⭐⭐⭐ MMMU-Pro 已成为多模态模型发布时的标准评测基准之一

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] AGRI-CM3: A Chinese Massive Multi-Modal Multi-Level Benchmark for Agricultural Understanding](agri-cm3_a_chinese_massive_multi-modal_multi-level_benchmark_for_agricultural_un.md)
- [\[NeurIPS 2025\] DanmakuTPPBench: A Multi-modal Benchmark for Temporal Point Process Modeling and Understanding](../../NeurIPS2025/multimodal_vlm/danmakutppbench_a_multimodal_benchmark_for_temporal_point_pr.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)
- [\[NeurIPS 2025\] Face-Human-Bench: A Comprehensive Benchmark of Face and Human Understanding for Multi-modal Assistants](../../NeurIPS2025/multimodal_vlm/face-human-bench_a_comprehensive_benchmark_of_face_and_human_understanding_for_m.md)
- [\[ACL 2025\] Exploring How Generative MLLMs Perceive More Than CLIP with the Same Vision Encoder](exploring_how_generative_mllms_perceive_more.md)

</div>

<!-- RELATED:END -->
