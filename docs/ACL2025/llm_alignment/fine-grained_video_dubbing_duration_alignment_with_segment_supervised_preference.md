---
title: >-
  [论文解读] Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization
description: >-
  [ACL 2025][LLM对齐][视频配音] 提出 Segment Supervised Preference Optimization (SSPO)，将视频配音中译文与源语音的时长对齐问题建模为段级偏好优化，通过逐句采样+细粒度 DPO 损失实现每行对话的时长一致性，同时维持翻译质量和输出格式。
tags:
  - "ACL 2025"
  - "LLM对齐"
  - "视频配音"
  - "时长对齐"
  - "偏好优化"
  - "DPO"
  - "段级监督"
  - "可控文本生成"
---

# Fine-grained Video Dubbing Duration Alignment with Segment Supervised Preference Optimization

**会议**: ACL 2025  
**arXiv**: [2508.08550](https://arxiv.org/abs/2508.08550)  
**代码**: [CcQunResearch/SSPO](https://github.com/CcQunResearch/SSPO)  
**机构**: 阿里巴巴数字媒体与娱乐集团 / 华中科技大学 / 北京交通大学
**领域**: LLM对齐  
**关键词**: 视频配音, 时长对齐, 偏好优化, DPO, 段级监督, 可控文本生成

## 一句话总结

提出 Segment Supervised Preference Optimization (SSPO)，将视频配音中译文与源语音的时长对齐问题建模为段级偏好优化，通过逐句采样+细粒度 DPO 损失实现每行对话的时长一致性，同时维持翻译质量和输出格式。

## 研究背景与动机

视频配音系统通常由 ASR→NMT→TTS 三个级联子任务组成。由于不同语言的信息密度差异（如中文高信息密度→英文/泰文低信息密度），翻译文本的语音时长常与原文不匹配，导致音画不同步，严重影响观看体验。

现有方法的局限：

- **传统方法**（AutoDubbing、VideoDubber）：针对 Seq2Seq 模型设计，修改 embedding 或引入长度预测模块，不适用于 LLM
- **Prompt Engineering**（GPT-4o、Claude 3.5）：LLM 缺乏对文本语音时长的直接感知能力，仅靠提示控制效果有限
- **SFT**：人工翻译字幕关注文本质量而非语音时长，SFT 后模型无法有效处理时长对齐
- **标准 DPO/RLHF**：对整个输出做偏好对齐，粒度太粗，无法实现逐行时长控制；且可能破坏输出格式

核心挑战：时长一致性指标 P(s_i, t_i) 不可微，无法直接用梯度下降优化 LLM。

## 方法详解

### 1. 时长一致性度量

定义非对称惩罚函数 P(s_i, t_i)：

- 当译文时长 > 源文时长时：使用指数惩罚 exp(Dur(t_i) - Dur(s_i))，因为超时会导致字幕溢出时间轴，更不可接受
- 当译文时长 < 源文时长时：使用线性惩罚 Dur(s_i) - Dur(t_i)，较为宽松
- 使用 Microsoft Edge TTS (edge-tts) 合成语音并获取时长
- 翻译质量评估采用两个无参考翻译模型：KIWI-XXL 和 XCOMET（均为 10B 参数）

### 2. 段级采样策略（Algorithm 1）

对查询集中每个样本（含 n=35 行对话），逐句采样 k 个翻译候选：

1. 以前文已选译文为前缀，从 SFT 模型采样 k 个候选翻译
2. 去重后，丢弃 KIWI-XXL 和 XCOMET 评分最低的 20%（保证翻译质量下限）
3. 按 P 指标选择 chosen（最小 P）和 rejected（最大 P）
4. 过滤低多样性行：去重后样本数 < e1=4 或 P 差值 < e2=0.08 的行不参与优化

示例：对于源句"历史虽然会重演，但是人类是无法回到过去的。"(2.89s)，从多个英文候选中选择时长最接近的 (2.93s) 为 chosen，时长偏差最大的 (3.19s) 为 rejected。

### 3. 段级 DPO 损失

- 对每行对话 s_i 独立计算标准 DPO 损失，前缀 p_i 包含指令和前面所有行的 chosen 译文
- 每行的 DPO 损失仅控制该行译文的时长，不影响其他行
- 总损失为所有行 DPO 损失之和

### 4. 输出格式控制

两种方案防止训练后输出格式崩坏：

- **Token-level KL Divergence (TKLD)**：在损失函数中加入逐 token 的 KL 散度约束项，防止 policy 偏离 reference 模型
- **LoRA 训练**（推荐）：低秩适配天然限制参数更新幅度，显存需求更低，格式保持率接近 100%，但收敛较慢需更多迭代

## 实验关键数据

### 主实验：时长对齐性能（Table 2）

数据集为自建 PolySC 数据集，测试集保留 4 部电视剧。

| 方法 | zh→en P↓ | zh→en CR↑ | zh→th P↓ | zh→th CR↑ |
|------|------|------|------|------|
| Gold Reference（人工翻译） | 0.501 | 17.9% | 0.489 | 20.3% |
| GPT-3.5-Turbo (PE) | 0.526 | 17.7% | 0.567 | 20.7% |
| GPT-4o (PE) | 0.417 | 19.2% | 0.318 | 27.0% |
| Claude 3.5 Sonnet (PE) | 0.410 | 19.3% | 0.313 | 27.5% |
| AutoDubbing (SFT) | 0.388 | 21.0% | 0.334 | 27.8% |
| VideoDubber (SFT) | 0.344 | 22.2% | 0.314 | 29.0% |
| Qwen2.5-14B SFT | 0.423 | 20.2% | 0.362 | 26.4% |
| **Qwen2.5-14B SSPO** | **0.272** | **24.9%** | **0.198** | **36.1%** |
| Llama3.1-8B SFT | 0.389 | 20.7% | 0.370 | 26.4% |
| **Llama3.1-8B SSPO** | **0.263** | **25.9%** | **0.206** | **36.4%** |
| Alignment Bound（理论上界） | 0.220 | 44.3% | 0.203 | 50.4% |

SSPO 在所有基座模型上均大幅降低 P，Qwen2.5-14B 上 zh→th 的 P 从 0.362 降至 0.198，已接近理论对齐上界 0.203。

### 人工评估：翻译质量（Table 4）

4 位英语/西语翻译专业评估者，对 200 段对话（每段 20 行）做配对比较。

| SSPO (Qwen2.5-14B) vs | 准确性 Win:Tie:Loss | 自然度 | 生动性 | 综合 |
|---|---|---|---|---|
| Gold Reference | 21:63:16 | 24:55:21 | 23:51:26 | 24:50:26 |
| Vanilla Qwen2.5-14B | 24:51:25 | 26:50:24 | 26:53:21 | 32:41:27 |
| GPT-4o | 25:51:24 | 19:58:22 | 23:48:29 | 23:49:28 |
| SFT 模型 | 23:48:29 | 21:54:25 | 24:49:27 | 24:50:26 |

SSPO 在不显著降低翻译质量的前提下大幅提升时长一致性。LLM 方法准确性普遍优于人工翻译，但生动性略逊（缺乏视频/音频多模态信息）。

### 消融实验

- **格式控制（Table 5）**：全参数微调格式合规率骤降（Qwen2.5: 81.2%/73.9%），LoRA 保持 99.8%/99.7%，TKLD 为 96.9%/97.2%
- **beta 敏感度（Figure 4）**：beta 越小时长一致性越好但格式合规率下降；折中选 beta=0.5
- **数据规模（Figure 5）**：约 10,000 行对话（~600 个 prompt-response 对，占 PolySC ~3%）即可取得显著效果，数据过多则格式合规率急降

## 亮点

- **问题建模新颖**：将视频配音时长对齐转化为“局部多段偏好优化”问题，提出一种全新的 CTG 范式
- **细粒度控制**：逐句独立采样+逐句 DPO 损失，实现行级时长对齐而非整体对齐，解决标准 DPO 粒度太粗的问题
- **不对称惩罚设计**：译文超时用指数惩罚、偏短用线性惩罚，完美贴合配音场景的实际需求
- **数据高效**：仅需 3% 数据即可获得接近理论上界的对齐效果
- **通用性强**：跨三个基座模型（Llama3.1-8B、GLM-4-9B、Qwen2.5-14B）和四个语言对（zh→en/th/es, es→zh）均有效

## 局限与展望

- **TTS 时长估计粗糙**：使用 edge-tts 合成时长作为代理指标，未考虑角色情感、语速变化等实际配音因素
- **语言对上界差异**：不同语言对的信息密度差异决定了对齐上限，某些语言对天然难以完全消除时长不一致
- **缺乏多模态信息**：翻译模型无法感知视频画面和音频情感，导致生动性系统性低于人工翻译
- **采样成本高**：逐句采样 k 个候选需 TTS 合成+翻译质量评估（10B 参数模型），训练数据构建开销大
- **仅在配音场景验证**：段级偏好优化思想有泛化潜力（如对话生成、代码生成的局部对齐），但本文未探索

## 相关工作

- **时长可控生成**：Kikuchi et al. (2016) 解码时奖励引导；Lakew et al. (2019) embedding 注入长度信息；Wu et al. (2023) VideoDubber 用长度预测——均为 Seq2Seq 设计，不适用 LLM
- **偏好优化**：DPO (Rafailov et al., 2024) 简化 RLHF 但整体粒度太粗；SimPO、KTO、IPO 等变体同样面临梯度稀释问题
- **视频配音系统**：Federico et al. (2020) AutoDubbing 控制翻译冗余度；Wu et al. (2023) VideoDubber 构建语音感知长度控制 NMT

## 评分

- 新颖性: ⭐⭐⭐⭐ — 将时长对齐建模为段级偏好优化，视角独特，定义了新的“局部多段偏好优化”问题
- 实验充分度: ⭐⭐⭐⭐ — 多语言对、多基座模型、人工评估+自动指标+消融+可视化+案例分析覆盖全面
- 写作质量: ⭐⭐⭐⭐ — 问题定义清晰，符号系统完整，理论推导严谨，附录详尽
- 价值: ⭐⭐⭐⭐ — 解决阿里优酷实际产业问题，段级偏好优化框架有泛化潜力

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ACL 2025\] SDPO: Segment-Level Direct Preference Optimization for Social Agents](sdpo_segment-level_direct_preference_optimization_for_social_agents.md)
- [\[ACL 2025\] ASPO: Adaptive Sentence-Level Preference Optimization for Fine-Grained Multimodal Reasoning](aspo_adaptive_sentence-level_preference_optimization_for_fine-grained_multimodal.md)
- [\[ACL 2025\] Balancing the Budget: Understanding Trade-offs Between Supervised and Preference-Based Finetuning](balancing_the_budget_understanding_trade-offs_between_supervised_and_preference-.md)
- [\[ACL 2025\] Retrieval-Augmented Fine-Tuning With Preference Optimization For Visual Program Generation](retrieval-augmented_fine-tuning_with_preference_optimization_for_visual_program_.md)
- [\[ACL 2025\] Reverse Preference Optimization for Complex Instruction Following](reverse_preference_optimization_for_complex_instruction_following.md)

</div>

<!-- RELATED:END -->
