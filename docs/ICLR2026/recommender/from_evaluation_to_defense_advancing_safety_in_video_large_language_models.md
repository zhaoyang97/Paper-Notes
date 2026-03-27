# From Evaluation to Defense: Advancing Safety in Video Large Language Models

**会议**: ICLR2026  
**arXiv**: [2505.16643](https://arxiv.org/abs/2505.16643)  
**代码**: 待确认  
**领域**: multimodal_vlm  
**关键词**: video LLM safety, benchmark, alarm token, GRPO, safety alignment

## 一句话总结
构建 VideoSafetyEval（11.4k 视频-查询对覆盖 19 种风险类别）揭示视频模态使安全性能下降 34.2%，提出 VideoSafety-R1 三阶段框架（报警 Token+SFT+Safety-guided GRPO）在 VSE-HH 上提升 71.1% 防御成功率。

## 研究背景与动机

1. **领域现状**：图像 LLM 的安全风险已被广泛研究，但视频 LLM 的安全对齐严重不足。视频的时间动态和语义复杂性引入更微妙的风险。
2. **现有痛点**：21 个视频 LLM 测试发现引入视频模态后防御成功率平均下降 34.2%。
3. **本文要解决什么？** (1) 系统评估视频 LLM 安全；(2) 提出有效的后训练安全对齐方法。
4. **核心idea一句话**：报警 Token 感知危害+GRPO 推理安全=从感知到主动推理的安全对齐。

## 方法详解

### 关键设计

1. **VideoSafetyEval (VSE)**: 11.4k 样本，19 子类，6 大风险类别，3 个子集（HH/SH/SafeQ）
2. **报警 Token (AT-SFT)**: 在视觉和文本序列中注入可学习报警 Token，通过多任务分类训练感知危害
3. **Safety-guided GRPO**: 基于双模态验证（视频有害性+文本有害性）的规则奖励，动态调整 ROUGE 权重

### 训练策略
AT-SFT → Cold-start SFT → Safety-guided GRPO，46k video-query-thinking 三元组。

## 实验关键数据

| 方法 | VSE-HH DSR↑ | MMBench DSR↑ |
|------|------|------|
| VideoLLaMA3-2B 基线 | 低 | 低 |
| **VideoSafety-R1** | **+71.1%** | **+59.1%** |

### 关键发现
- 视频模态引入使文本分支安全性严重退化
- 报警 Token 作为显式安全信号有效
- 推理时思维链提升了安全推理质量

## 亮点与洞察
- 首个大规模视频 LLM 安全基准
- 从感知（AT-SFT）到推理（GRPO）的渐进式安全对齐设计巧妙

## 局限性 / 可改进方向
- 安全分类的二值标签可能过于粗糙
- 过度防御（误拒率）需要权衡

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 首个系统的视频 LLM 安全工作
- 实验充分度: ⭐⭐⭐⭐⭐ 21个模型评估+多基准验证
- 写作质量: ⭐⭐⭐⭐ 结构清晰
- 价值: ⭐⭐⭐⭐⭐ 填补视频安全的关键空白
