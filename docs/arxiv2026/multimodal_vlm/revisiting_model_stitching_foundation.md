# Revisiting Model Stitching In the Foundation Model Era

**会议**: arXiv 2026  
**arXiv**: [2603.12433](https://arxiv.org/abs/2603.12433)  
**作者**: Zheda Mai, Ke Zhang, Fu-En Wang, Zixiao Ken Wang, Albert Y. C. Chen
**代码**: 待确认  
**领域**: 多模态/VLM  
**关键词**: revisiting, model, stitching, foundation, era  

## 一句话总结
模型拼接，通过轻缝合层将一个模型（源）的早期层连接到另一个模型（目标）的后期层，已成为表征兼容性的探针。
## 背景与动机
Model stitching, connecting early layers of one model (source) to later layers of another (target) via a light stitch layer, has served as a probe of representational compatibility.. Prior work finds that models trained on the same dataset remain stitchable (negligible accuracy drop) despite different initializations or objectives.

## 核心问题
之前的工作发现，尽管初始化或目标不同，但在同一数据集上训练的模型仍然是可缝合的（精度下降可以忽略不计）。
## 方法详解

### 整体框架
- Model stitching, connecting early layers of one model (source) to later layers of another (target) via a light stitch layer, has served as a probe of representational compatibility.
- We introduce a systematic protocol spanning the stitch points, stitch layer families, training losses, and downstream tasks.
- Building on these findings, we further propose the VFM Stitch Tree (VST), which shares early layers across VFMs while retaining their later layers, yielding a controllable accuracy-latency trade-off for multimodal LLMs that often leverage multiple VFMs.
- Taken together, our study elevates stitching from a diagnostic probe to a practical recipe for integrating complementary VFM strengths and pinpointing where their representations align or diverge.

### 关键设计
待深读后补充。

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- (3) 对于深缝合点，缝合模型只需很小的推理开销（对于缝合层）就可以超越任一组成模型。
## 局限性 / 可改进方向
- 待深读后补充

## 与相关工作的对比
待深读后补充。

## 与我的研究方向的关联
- 待分析

## 评分
- 新颖性: ⭐⭐⭐
- 实验充分度: ⭐⭐⭐
- 写作质量: ⭐⭐⭐
- 对我的价值: ⭐⭐⭐⭐
