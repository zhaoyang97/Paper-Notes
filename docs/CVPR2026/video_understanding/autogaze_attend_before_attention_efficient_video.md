# AutoGaze: Attend Before Attention — Efficient and Scalable Video Understanding via Autoregressive Gazing

**会议**: CVPR 2026  
**arXiv**: [2603.12254](https://arxiv.org/abs/2603.12254)  
**代码**: 有 (项目页: [https://autogaze.github.io/](https://autogaze.github.io/))  
**领域**: 视频理解 / 多模态VLM / 模型压缩  
**关键词**: 视频token压缩, 自回归选择, MLLM加速, 长视频, 高分辨率  

## 一句话总结
提出AutoGaze——在ViT/MLLM处理视频之前，用一个轻量模块自回归地选择最少的多尺度patch，减少4x-100x视觉token，加速最高19x，支持1K帧4K视频并在VideoMME达67.0%。

## 背景与动机
多模态大语言模型（MLLM）在通用视频理解上取得了显著进展，但面对长时、高分辨率视频时遇到严重瓶颈：ViT和LLM都平等地处理每一个像素，而视频中存在大量时空冗余（如静态背景、重复帧）。这导致计算量随帧数和分辨率急剧增长，4K长视频基本无法处理。现有的token减少方法要么依赖固定规则（均匀采样），要么需要重新训练整个模型，都不够灵活。

## 核心问题
如何在不损失关键信息的前提下，大幅减少送入ViT/MLLM的视觉token数量？核心挑战在于：不同视频片段的信息密度差异巨大——静态画面只需极少token，但快速运动或复杂场景需要密集采样。需要一种**内容自适应**的方法来决定"看哪里、看多细"。

## 方法详解

### 整体框架
AutoGaze是一个即插即用的前置模块，放在ViT或MLLM之前。输入一段视频，AutoGaze自回归地逐步选择一组多尺度patch，使得这组patch足以在用户指定的误差阈值内重建原始视频。选出的稀疏patch集再送入下游ViT/MLLM处理。

### 关键设计
1. **自回归patch选择**: AutoGaze像语言模型生成token一样地"生成"patch选择决策。每一步基于已选patch的信息，决定下一个最有价值的patch位置和尺度。这种自回归策略比一次性选择更精确——后续选择能利用前面的上下文。
2. **多尺度patch**: 不是固定分辨率选择，而是支持多个尺度——对整体结构用粗粒度patch，对细节区域用细粒度patch。这模仿人眼"先整体后局部"的注视策略。
3. **误差阈值控制**: 用户可指定视频重建误差的容忍度——容忍度高则少选patch（更快），容忍度低则多选（更精确）。这提供了一个编程接口来控制效率-精度trade-off。

### 损失函数 / 训练策略
用**next-token prediction + 强化学习**联合训练。Next-token prediction让模型学习patch间的依赖结构；RL让模型学会在最少patch数下达到目标重建质量——奖励信号来自"用多少patch达到给定精度"。

## 实验关键数据
| 数据集 | 指标 | AutoGaze | 基线MLLM | 提升 |
|--------|------|----------|----------|------|
| VideoMME | Acc | 67.0% | ~60% | +7% |
| HLVid (新) | Acc | +10.1% vs baseline | - | +4.5% vs prev best |
| 通用 | Token减少 | 4x-100x | 1x | 显著 |
| 通用 | 速度提升 | 最高19x | 1x | 显著 |

### 消融实验要点
- 多尺度选择比单尺度选择提升显著——粗细结合效果远好于纯细粒度
- RL训练比纯监督学习更能找到最优的Token-精度平衡点
- 误差阈值提供了平滑的效率-精度曲线，适配不同部署场景

## 亮点 / 我学到了什么
- 🔥 **"先看再算"的范式** — 不是先提取所有特征再筛选，而是先决定看哪里再提取。这颠覆了传统ViT"均匀计算"的范式
- 自回归选择 = **内容感知的自适应计算分配**，比ToMe/EViT等固定规则方法更智能
- **HLVid benchmark**填补了高分辨率长视频QA评估的空白
- RL训练patch选择策略是巧妙的——把离散选择问题转化为序列决策

## 局限性 / 可改进方向
- AutoGaze模块本身的推理开销——对超短视频(几帧)可能得不偿失
- 误差阈值需要手动指定，能否自动根据任务难度调整？
- 能否把这个思路推广到图像MLLM上？（单图内不同区域的token分配）
- → 直接关联idea: `task_aware_token_compression.md`

## 与相关工作的对比
- **vs ToMe**: ToMe是静态的基于相似度的token合并，AutoGaze是动态的自回归选择——后者能利用上下文做更精准的决策
- **vs FastV/LLaVA-PruMerge**: 这些方法在LLM内部做token减少，AutoGaze在ViT之前就减少了——更早减少=更大加速
- **vs EVATok**: EVATok针对视频生成的tokenizer优化，AutoGaze针对视频理解的token选择——互补而非竞争

## 与我的研究方向的关联
- 🔥🔥 直接启发 `task_aware_token_compression.md` — 自回归patch选择可以扩展为任务感知的
- 与BiGain（频域token压缩）、TrajTok（轨迹token）共同构成视频效率方法族
- 19x加速意味着5080也能跑长视频理解 → 对我的"无需大规模计算"需求非常有价值

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ 自回归patch选择+RL训练是全新的范式
- 实验充分度: ⭐⭐⭐⭐⭐ 多个benchmark+消融+新benchmark HLVid
- 写作质量: ⭐⭐⭐⭐⭐ 故事讲得清楚，"Attend Before Attention"标题极好
- 对我的价值: ⭐⭐⭐⭐⭐ 直接相关+实际可用+启发idea
