---
title: "MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark"
conference: "ACL 2025"
arxiv: "2409.02813"
code: "https://mmmu-benchmark.github.io/#leaderboard"
domain: "multimodal_vlm"
keywords: ["benchmark", "multimodal reasoning", "vision-language model", "robustness evaluation", "MMMU"]
---

# MMMU-Pro: A More Robust Multi-discipline Multimodal Understanding Benchmark

## 一句话总结

提出 MMMU-Pro 基准，通过过滤纯文本可解题目、扩增选项至 10 个、引入 vision-only 输入设置三步法，构建更鲁棒的多学科多模态理解评测，所有模型性能显著下降 16.8%-26.9%。

## 研究背景与动机

1. GPT-4o 等模型在 MMMU 上已达 69.1%，但高分是否真正反映深度多模态理解存疑
2. 发现部分 MMMU 问题可被纯文本 LLM 正确回答——模型可能利用文本中的捷径或先验知识
3. 4 选项的多选题格式限制了评估力度，模型可通过排除法或猜测获得虚高分数
4. 现实中用户常以截图方式向 AI 提问，但现有 benchmark 都将文本和图片分开输入
5. 人类可以无缝地在视觉和文本信息间切换理解，这一核心认知能力未被现有 benchmark 充分测试
6. 需要一个更接近真实场景、能揭示模型真实多模态能力的评测基准

## 方法详解

### 整体框架

基于 MMMU 进行三步构建：过滤 → 扩增选项 → vision-only 设置，最终得到 3,460 道题。

### 关键设计

**Step 1: 过滤纯文本可解题目**
- 使用 4 个强开源 LLM（Llama3-70B, Qwen2-72B, Yi-1.5-34B, Mixtral-8×22B）无图回答
- 每模型重复 10 次，正确率 >5/10 判定为可回答
- 至少 3/4 模型可答的题目被排除，从剩余池中均匀采样 1800 题（30科目×60题）

**Step 2: 扩增候选选项**
- 将原始 4 个选项扩增至 10 个，由 GPT-4o 生成 + Claude 3.5 过滤 + 两轮人工审核
- 人工专家同时检查题目与图片的相关性，剔除 70 道不连贯题目，最终 1730 题

**Step 3: Vision-only 输入设置**
- 人工标注者在不同背景、字体、字号的模拟显示环境中截取题目截图或拍照
- 模型仅接收图片输入、不显式接收文本，测试"同时看和读"的能力
- 共 1730 张截图/照片，与标准版配对生成 3,460 题

### 评估方式

- MMMU-Pro 总分 = Standard(10选项) 与 Vision-only 的平均
- 同时测试 Direct 和 CoT prompt，报告较高分

## 实验关键数据

### 主要性能下降（相比 MMMU Val）

| 模型 | Standard(4选项) | Standard(10选项) | Vision-only | MMMU Val | Δ₁ | Δ₂ |
|------|----------------|-----------------|-------------|---------|-----|------|
| GPT-4o | 64.7 | 54.0 | 49.7 | 69.1 | -15.1 | -19.4 |
| Claude 3.5 Sonnet | 63.7 | 55.0 | 48.0 | 68.3 | -13.3 | -20.3 |
| Gemini 1.5 Pro | 60.6 | 49.4 | 44.4 | 65.8 | -16.4 | -21.4 |
| InternVL2-76B | 55.0 | 41.9 | 38.0 | 58.3 | -16.4 | -20.3 |

### 消融与分析

| 分析维度 | 关键发现 |
|---------|---------|
| 4→10选项 | GPT-4o 降10.7%，有效减少猜测 |
| Vision-only | GPT-4o 再降4.3%，LLaVA-OV-72B降14.0% |
| CoT 效果 | Claude 3.5 在Standard上提升12.3%（42.7→55.0），但部分模型反降 |
| OCR prompt | 对大多数模型影响<1%，说明强模型已内置文字识别能力 |

### 人类专家表现

- 低/中/高水平专家在 MMMU-Pro 上分别约 73.0%/80.8%/85.4%
- 相比 MMMU Val 仅降 1.8%-3.2%，远小于模型的 13-27% 下降

### CoT 在不同学科的效果

- Tech & Engineering 提升最大（GPT-4o +14.49%）
- Art & Design 几乎无效甚至下降（LLaVA-OV-72B -17.12%）
- 提示 CoT 在需结构化推理的领域更有效

## 亮点与洞察

- **三步构建法系统有效**: 每一步都有数据验证，纯文本模型准确率从约 30% 降至约 12%
- **Vision-only 设置的洞察**: 简单 OCR 不够，模型需理解文本与图像的上下文关系，整体视觉复杂度显著增加
- **排名洗牌现象**: MMMU → MMMU-Pro 存在明显排名变化，部分模型在 vision-only 上剧烈下降，暴露视觉理解短板
- **人机差距拉大**: 人类在增强版测试上表现稳健（降幅 <3.2%），而模型降幅 13-27%，gap 被放大

## 局限性

- 仍然基于 MMMU 原始题库，新颖性和多样性受限于源数据集
- OCR accuracy 评估仅用 Levenshtein distance，可能无法捕获语义级别的理解差异
- Vision-only 截图由人工拍摄制作，标注成本高且难以大规模扩展
- 人类专家表现为估算而非重新评测
- 10 选项可能引入部分不合理的干扰项

## 相关工作

- MMMU 原始 benchmark（Yue et al., 2024）
- 多模态 LLM 评测（OpenAI GPT-4o, Claude 3.5 Sonnet, Gemini 1.5）
- VLM 捷径利用与鲁棒性研究（Du et al., 2023; Wu & Xie, 2024）
- CoT 推理（Wei et al., 2022）
- 视觉文本理解与 OCR 能力评测

## 评分

- **新颖性**: ★★★★☆ — Vision-only 设置和系统化三步去偏方法设计新颖
- **技术深度**: ★★★☆☆ — 更偏基准构建和评测分析，方法论创新有限
- **实验充分性**: ★★★★★ — 覆盖 20+ 模型，多维度分析（CoT、OCR、学科、排名变化）
- **实用价值**: ★★★★★ — 成为多模态模型评测的重要标准，已被广泛采用
