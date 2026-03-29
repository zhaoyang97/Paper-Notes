# Pre-Training Curriculum for Multi-Token Prediction in Language Models

**会议**: ACL 2025  
**arXiv**: [2505.22757](https://arxiv.org/abs/2505.22757)  
**代码**: https://github.com/aynetdia/mtp_curriculum (有)  
**领域**: LLM NLP  
**关键词**: multi-token prediction, 课程学习, 预训练策略, 推理加速, 小语言模型

## 一句话总结
针对小语言模型（SLM）难以直接受益于多 token 预测（MTP）目标的问题，提出前向/反向课程学习策略——前向课程（NTP→MTP）使 SLM 在保持自推测解码加速的同时提升生成质量，反向课程（MTP→NTP）在 NTP 性能上更优但失去推理加速优势。

## 研究背景与动机

1. **领域现状**：Multi-Token Prediction（MTP）是一种新兴的预训练目标，让模型在每一步预测接下来的 $k$ 个 token（而非传统的 1 个）。Gloeckle et al. (2024) 和 DeepSeek-V3 已证明 MTP 能提升大模型的下游性能、推理速度和训练效率。
2. **现有痛点**：MTP 的收益随模型规模增大而增大——**小语言模型（SLM，1-3B）从 MTP 中获益有限**，甚至性能下降。原因是 SLM 参数量有限，难以从一开始就处理多个 token 之间的形态学和语义依赖。
3. **核心矛盾**：MTP 目标对 SLM 来说"太难了"——直接让 SLM 同时预测 4 个 token 超出了其学习能力，但 MTP 带来的推理加速和更丰富的隐状态表示仍然有价值。
4. **本文要解决什么？** 设计一种预训练课程学习策略，让 SLM 也能有效利用 MTP 目标的优势。
5. **切入角度**：Bengio et al. (2009) 的课程学习——从简单到复杂地安排学习任务。MTP 的复杂度随预测 token 数自然递增，非常适合用课程学习来调控。
6. **核心 idea 一句话**：训练过程中动态调整预测 token 数——前向课程（$k$=1→2→...→$k_{max}$）和反向课程（$k_{max}$→...→2→1），替代静态 MTP。

## 方法详解

### 整体框架
在标准 Transformer 架构上，用多个 LM head 实现 MTP（每个 head 预测未来第 $i$ 个 token）。训练过程中按课程动态调整激活的 head 数量 $k_{current}(e)$，未激活的 head 不参与 loss 计算。总训练步数均匀分配给每个阶段。

### 关键设计

1. **前向课程（Forward Curriculum）**：
   - 做什么：从 NTP（$k$=1）开始，逐步增加预测 token 数到 $k_{max}$
   - 公式：$k_{current}(e) = \min(k_{max}, \lfloor e / (E/k_{max}) \rfloor + 1)$
   - 设计动机：先让模型学好基本的逐 token 预测，建立稳固的语言建模基础，再逐步增加复杂度。类似人类学习中的"先学走再学跑"
   - 预期优势：保持自推测解码能力（最终所有 head 都被训练过）

2. **反向课程（Reverse Curriculum）**：
   - 做什么：从完整 MTP（$k_{max}$）开始，逐步减少到 NTP（$k$=1）
   - 公式：$k_{current}(e) = \max(1, k_{max} - \lfloor e / (E/k_{max}) \rfloor)$
   - 设计动机：基于"MTP 预训练提升 NTP 下游性能"的发现——先用 MTP 学到更丰富的隐状态表示，再逐步聚焦到 NTP，让主 LM head 获得来自 MTP 的"感知收益"
   - 预期优势：主 LM head 的 NTP 性能更强

3. **两种 LM Head 设计**：
   - **线性层（LL）**：每个额外 head 是一个 $d_{hidden} \times d_{vocab}$ 的线性层，可并行预测，但增加内存开销（subword 模型每 head +65-98M 参数）
   - **Transformer 层（TL）**：将模型最后 $k$ 层分别负责不同 token 的预测，共享输出线性层。不增加参数，反而"减少"了 backbone 的有效深度

### 训练策略
- 模型：1.3B 和 3B Llama 架构，subword（32K 词表）和 byte（320 词表）两种 tokenization
- 数据：MiniPile（1.7B subword tokens / 5.9B bytes）
- 1 epoch 训练，batch size 1024，cosine decay 学习率

## 实验关键数据

### 主实验（NTP 性能，仅用主 LM head）

| 模型 | 类型 | Heads | 课程 | MiniPile BPB↓ | LAMBADA BPB↓ | BLiMP Acc↑ |
|---|---|---|---|---|---|---|
| 1.3B subword | NTP baseline | 1 LL | - | 1.08 | 1.34 | 71.80 |
| 1.3B subword | Static MTP | 4 LL | - | 1.19 | 1.61 | 67.48 |
| 1.3B subword | Forward | 4 LL | ✓ | ~1.16 | ~1.52 | ~68.5 |
| 1.3B subword | **Reverse** | 4 LL | ✓ | ~1.14 | ~1.45 | ~69.5 |
| 3B byte | NTP baseline | 1 LL | - | 1.07 | 1.00 | 71.20 |
| 3B byte | Static MTP | 4 LL | - | 1.08 | 1.00 | 71.42 |
| 3B byte | **Reverse** | 4 LL | ✓ | ≤1.07 | ≤1.00 | **≥71.42** |

- 反向课程在 byte-level 模型上达到或超过 NTP baseline，是唯一做到这点的 MTP 策略
- 前向/反向课程在所有配置上**均优于静态 MTP**，证明课程学习对 SLM 的 MTP 有效
- Subword 模型受益少于 byte 模型：因为 subword token 承载更多语义信息，同时预测多个更难

### 消融实验（推理加速）

| 配置 | 相对于 NTP 的前向传递减少 |
|---|---|
| Static MTP 4 heads | 最高（~1.6x-1.8x 加速） |
| **Forward Curriculum** | 接近 static MTP（~1.5x-1.7x） |
| Reverse Curriculum | 基本无加速（收敛到 ~1.0x） |

### 关键发现
- **前向 vs 反向课程的互补性**：前向课程保持推理加速但 NTP 性能稍逊，反向课程 NTP 性能更强但失去推理加速。不存在"两全其美"的方案
- **Byte-level 模型更适合 MTP 课程**：因为单个 byte 承载的语义信息少，预测多个 byte 比预测多个 subword 简单得多，SLM 更容易掌握
- **LL vs TL head 差异很小**：即使 TL 模型在推理时"损失"了部分参数（因为最后几层被分配给其他 head），性能差距也可忽略不计
- **10B token 实验验证了可扩展性**：在 FineWeb-Edu 10B 数据上的额外实验确认了课程学习的优势
- **生成质量（BLEU/ROUGE/SemScore/G-Eval）**：前向和反向课程均提升了生成文本的质量，反向课程在语义相似度上表现最好

## 亮点与洞察
- **极简但有效的思路**：只需在训练过程中按时间表调整激活的 head 数量，不改变模型架构、不增加额外参数、不改变总训练步数，就能让 SLM 更好地利用 MTP
- **前向/反向课程的对称设计**：优雅地分离了"推理加速"和"NTP 性能"两个优化方向，为实际应用提供了清晰的选择指南——需要推理速度就用前向，需要 NTP 性能就用反向
- **Byte vs Subword 的独到分析**：揭示了 MTP 效果差异的根本原因是 token 粒度，而非模型容量本身。这为 byte-level LLM 的发展提供了新动机

## 局限性 / 可改进方向
- **数据规模有限**：MiniPile 仅 1.7B tokens，远小于正常 LLM 预训练数据量，结论的大规模可扩展性需进一步验证
- **均匀分阶段的课程过于简单**：根据训练动态自适应调整 $k_{current}$ 可能效果更好
- **模型规模偏小**：1.3B 和 3B，Gloeckle et al. 发现 MTP 在 7B+ 模型上效果更好，课程学习在大模型上是否仍有价值未知
- **评测 benchmark 不够全面**：知识密集型任务（MMLU 等）由于数据量限制无法充分评测
- 改进方向：自适应课程策略（根据 loss 曲线动态调节 $k$）；在更大模型和数据上验证；探索非均匀阶段划分；结合 Medusa 等多头推测方法

## 相关工作与启发
- **vs Gloeckle et al. (2024)**：他们提出 MTP 并发现大模型受益更多、小模型受益有限；本文通过课程学习让小模型也能受益，是对原工作的直接改进
- **vs DeepSeek-V3 (Liu et al. 2024)**：DeepSeek-V3 在 671B MoE 模型上用静态 MTP，本文的课程策略可以作为小模型的替代方案
- **vs Medusa (Cai et al. 2024)**：Medusa 在训练后添加额外 head 做推测解码；本文的 MTP head 在预训练阶段就参与训练，表示更一致
- 这篇论文为 MTP 的民主化提供了关键一步——让资源有限的研究者也能在小模型上利用 MTP

## 评分
- 新颖性: ⭐⭐⭐⭐ 将课程学习应用于 MTP 训练目标是自然但有效的想法，前向/反向的对称设计简洁优雅
- 实验充分度: ⭐⭐⭐ 两种模型大小 × 两种 tokenization × 两种 head 类型，但数据规模太小
- 写作质量: ⭐⭐⭐⭐ 表述清晰，实验设计合理
- 价值: ⭐⭐⭐⭐ 对 MTP 训练的实用改进，特别是对资源受限场景有参考意义
