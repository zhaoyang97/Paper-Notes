---
title: >-
  [论文解读] OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference
description: >-
  [ACL 2025][多模态VLM][MLLM alignment] 构建了 OmniAlign-V（200K 高质量多模态 SFT 数据集）和 MM-AlignBench 评测基准，通过多样化图片来源、开放式问题设计和多样化回答格式，显著提升开源 MLLM 的人类偏好对齐能力，使 LLaVA-Next-32B 经 SFT+DPO 后超越 Qwen2VL-72B。
tags:
  - "ACL 2025"
  - "多模态VLM"
  - "MLLM alignment"
  - "human preference"
  - "instruction tuning"
  - "DPO"
  - "多模态"
  - "benchmark"
---

# OmniAlign-V: Towards Enhanced Alignment of MLLMs with Human Preference

**会议**: ACL 2025  
**代码**: [https://github.com/PhoenixZ810/OmniAlign-V](https://github.com/PhoenixZ810/OmniAlign-V)  
**领域**: 多模态VLM  
**关键词**: MLLM alignment, human preference, instruction tuning, DPO, multi-modal dataset, benchmark  

## 一句话总结

构建了 OmniAlign-V（200K 高质量多模态 SFT 数据集）和 MM-AlignBench 评测基准，通过多样化图片来源、开放式问题设计和多样化回答格式，显著提升开源 MLLM 的人类偏好对齐能力，使 LLaVA-Next-32B 经 SFT+DPO 后超越 Qwen2VL-72B。

## 研究背景与动机

### 问题发现：MLLM 的对齐能力退化

- 开源 MLLM 在标准 VQA 基准上接近商业模型，但在人类偏好对齐方面存在显著差距
- **关键实验**（Table 1）：多模态 SFT 后，MLLM 在纯文本对齐基准上大幅退化
    - InternLM2.5-7B → InternVL2-8B：AlpacaEval-V2 从 27.58 降至 3.35（**-87.9%**）
    - Qwen2-7B → Qwen2VL-7B：ArenaHard 从 32.84 降至 6.46（**-80.3%**）

### 仅加高质量文本数据无济于事

- 将 LLaVA-Next-778K 中的文本数据替换为 Magpie/Condor 高质量数据
- 结果（Table 2）：纯文本对齐提升，但**多模态对齐反而下降**
    - WildVision、MMVet、MMBench 等多模态指标全面恶化
- **结论**：语言对齐能力不能直接迁移到多模态对齐，需要专门的多模态人类对齐数据

### 现有多模态数据的问题

- 以 VQA 格式为主：简短问答、事实性回答
- 缺乏开放式问题、创意任务、多样回答风格
- 不满足人类偏好对齐的需求

## 方法详解

### OmniAlign-V 数据集构建

#### 4.1 任务分类

**自然图像**（3 类任务）：
- Knowledge（知识问答）：需要背景知识理解
- Inferential（推理任务）：需要逻辑推理和分析
- Creation（创作任务）：开放式创意问答

**信息图**（4 类图像）：
- Arts（艺术）、Charts（图表）、Diagrams（图解）、Posters（海报）

#### 4.2 图像筛选策略（自然图像）

两步筛选确保语义丰富度：
1. **IC9600 图像复杂度模型**：过滤低语义内容图像
2. **Recognize Anything Model**：过滤高复杂度但无意义内容的图像（如反复出现的帐篷）

#### 4.3 数据生成流水线

**Knowledge & Inferential**：GPT-4o + 精心设计的 few-shot prompt 直接生成

**Creative**：更复杂的流程（受 Condor 启发）：
1. 创建种子创意问题集 $Q_s = \{Q_1, Q_2, ..., Q_N\}$
2. 用轻量 MLLM 生成图像 caption $C$
3. LLM 根据 caption 从种子集选择相关子集 $Q_s'$
4. 随机选 3 种问题类型作为 few-shot 示例给 GPT-4o

**Infographic**：针对不同图类设计专门 prompt，生成需要全面背景知识的问题

#### 4.4 后精炼

1. **Instruction Augmented Knowledge QAs**：为知识问答加入复杂指令和限制条件
2. **Enriched Inferential QAs**：用知识丰富的 LLM 补充详细解释和推理逻辑
3. **Quality Improved Infographic QAs**：
    - GPT-4o 擅长背景知识解释但 OCR 不准
    - 开源 MLLM OCR 准但解释不够
    - **融合**两者的回答 + 人工审核

#### 数据规模

| 子集 | 数量 |
|------|------|
| Knowledge QAs | 39K |
| Inferential QAs | 37K |
| Creative QAs | 10K |
| Instruction-Following QAs | 38K |
| Infographic QAs | 44K |
| Detail QAs | 35K |
| **总计** | **~205K** |

### DPO 数据生成（OmniAlign-V-DPO）

- OmniAlign-V 的高质量回答作为 positive sample
- 用 LLaVA-Next baseline（generator G）高温采样 N 个回答
- LLM Judger 选出最偏离原始意图的回答作为 negative sample

### MM-AlignBench 评测基准

- 252 个高质量样本，人工标注
- 多样图像来源（SAM-1B、CC-3M、AI2D、ChartQA、InfographicVQA）
- 先 IC 筛选 + RAM 筛选得 2000 张自然图 + 1000 张信息图
- GPT-4o 生成多样问题 → 人工审核精炼
- 评估方式：GPT-4o 判断，对比 Claude3V-Sonnet 参考回答

## 实验

### SFT 阶段评估

将 OmniAlign-V 与 LLaVA-Next-778k（去除文本样本）合并为 OmniAlign-Vmix（946K）。

**InternLM2.5-7B 作为 LLM 的 LLaVA-Next**：

| 指标 | LLaVA-Next-778k | OmniAlign-Vmix | 变化 |
|------|-----------------|----------------|------|
| MM-AlignBench | 20.6 / -42.7 | 57.1 / +11.1 | **+36.5** |
| WildVision | 23.4 / -45.0 | 29.6 / -31.3 | **+6.2** |
| MIA-Bench | 76.9 | 86.7 | +9.8 |
| MMVet | 41.8 | 47.7 | +5.9 |
| MMMU | 44.1 | 46.8 | +2.7 |
| OCRBench | 56.2 | 58.9 | +2.7 |

- 人类偏好对齐大幅提升（MM-AlignBench +36.5 winning rate）
- 标准 VQA 基准不降反升

**Qwen2.5-32B 作为 LLM**：
- MM-AlignBench：26.6 → 62.3（+35.7）
- MMMU：55.2 → 60.7（+5.5）

### 纯文本对齐也改善

即使训练数据不含纯文本样本，OmniAlign-V 也提升了纯文本对齐：
- AlpacaEval-V2（vs GPT-3.5）：29.8 → 50.1
- ArenaHard：21.4 → 30.4
- **洞察**：高质量多模态数据能反哺语言能力

### DPO 阶段评估

| 模型 | 阶段 | MM-AlignBench | WildVision |
|------|------|---------------|------------|
| LLaVANext-778k | SFT | 9.5 / -69.2 | 30.4 / -34.2 |
| LLaVANext-778k | SFT+DPO | 11.1 / -64.5 | 35.5 / -23.4 |
| LLaVANext-OA | SFT | 57.1 / +11.1 | 29.6 / -31.3 |
| LLaVANext-OA | **SFT+DPO** | **64.3 / +22.4** | **41.8 / -10.1** |
| InternVL2-8B | SFT+DPO | 64.7 / +19.4 | 51.4 / +1.9 |

- DPO 在 OmniAlign-V SFT 基础上进一步提升
- 仅用 778k 数据做 SFT 后再 DPO 效果有限——说明 **SFT 阶段的对齐数据质量是 DPO 效果的前提**

### MM-AlignBench 排行榜

| 模型 | Win Rate↑ | Reward↑ |
|------|-----------|---------|
| Claude3.5V-Sonnet | 84.9 | +51.4 |
| GPT-4o | 81.3 | +49.0 |
| **LLaVA-OA-32B-DPO** | **74.2** | **+36.9** |
| Qwen2VL-72B | 61.5 | +21.6 |
| InternVL2-72B | 44.4 | -6.9 |

- LLaVA-OA-32B-DPO（32B）超越 Qwen2VL-72B（72B），仅次于 Claude 和 GPT-4o

### 消融实验

逐步添加 OmniAlign-V 子集的效果：
- +Knowledge/Inferential/Detail：小幅提升
- +**Instruction Following**：MM-AlignBench 从 23.4 跃升至 36.5（关键子集）
- +Creation：MM-AlignBench 继续提升至 43.7
- +Chart/Diagram/Poster：最终达到 57.1

## 亮点与洞察

1. **发现并量化了 MLLM 对齐退化问题**：多模态 SFT 导致语言对齐能力下降 60-90%
2. **揭示反直觉现象**：加高质量文本数据不改善甚至损害多模态对齐——必须用专门的多模态对齐数据
3. **数据工程系统性强**：图像筛选（IC+RAM）→ 任务分类 → 多种生成策略 → 后精炼 → 人工审核
4. **SFT + DPO 的协同效应**：SFT 阶段的对齐质量决定了 DPO 能否生效
5. **32B 模型打败 72B**：证明数据质量 > 模型规模
6. **MM-AlignBench** 填补了多模态偏好对齐评测的空白

## 局限性

- 数据生成严重依赖 GPT-4o，成本较高
- 信息图的 OCR 融合策略需要人工审核把关
- MM-AlignBench 仅 252 样本，规模较小
- 评估使用 GPT-4o as judge，可能存在评判偏差
- 未讨论安全对齐（如拒绝有害请求），主要关注偏好和有用性对齐

## 相关工作

- **LLM 对齐**：Magpie（Xu et al., 2024）、Condor（Cao et al., 2025）高质量 SFT 数据
- **视觉问答数据**：LLaVA（Liu et al., 2023b）将传统 VQA 转换为指令格式；ShareGPT4V 等
- **多模态对齐评测**：WildVision（Lu et al., 2024）、MIA-Bench（Qian et al., 2024），但问题重复且简单
- **DPO**：Rafailov et al., 2024；在视觉领域的应用尚不充分

## 评分 ⭐⭐⭐⭐⭐

研究动机清晰（发现并解决 MLLM 对齐退化）、数据工程极为系统、实验全面且有力（32B 超 72B），提供了完整的数据集 + 基准 + 代码。是多模态对齐方向的标杆工作。

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)
- [\[ICCV 2025\] Spatial Preference Rewarding for MLLMs Spatial Understanding](../../ICCV2025/multimodal_vlm/spatial_preference_rewarding_for_mllms_spatial_understanding.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[ACL 2025\] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension](punchbench_mllm_punchline.md)

</div>

<!-- RELATED:END -->
