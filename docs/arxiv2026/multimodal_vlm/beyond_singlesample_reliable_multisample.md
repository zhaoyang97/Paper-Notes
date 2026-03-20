# Beyond Single-Sample: Reliable Multi-Sample Distillation for Video Understanding

**会议**: arXiv 2026  
**arXiv**: [2603.11423](https://arxiv.org/abs/2603.11423)  
**作者**: Songlin Li, Xin Zhu, Zechao Guan, Peipeng Chen, Jian Yao
**代码**: 待确认  
**领域**: 多模态/VLM / 视频理解  
**关键词**: single-sample, reliable, multi-sample, distillation, video  

## 一句话总结
大型视觉语言模型 (LVLM) 的传统黑盒蒸馏通常依赖于每个输入的单个教师响应，这通常会在多模式或时间场景中产生高方差响应和格式不一致。
## 背景与动机
Traditional black-box distillation for Large Vision-Language Models (LVLMs) typically relies on a single teacher response per input, which often yields high-variance responses and format inconsistencies in multimodal or temporal scenarios.. To mitigate this unreliable supervision, we propose R-MSD (Reliable Multi-Sample Distillation), a framework that explicitly models teacher sampling variance to enhance distillation stability.

## 核心问题
大型视觉语言模型 (LVLM) 的传统黑盒蒸馏通常依赖于每个输入的单个教师响应，这通常会在多模式或时间场景中产生高方差响应和格式不一致。
## 方法详解

### 整体框架
- To mitigate this unreliable supervision, we propose R-MSD (Reliable Multi-Sample Distillation), a framework that explicitly models teacher sampling variance to enhance distillation stability.
- Rather than relying on a single teacher response, our approach leverages a task-adaptive teacher pool to provide robust supervision tailored to both closed-ended and open-ended reasoning.
- By integrating quality-aware signal matching with an adversarial distillation objective, our approach effectively filters teacher noise while maximizing knowledge transfer.
- We additionally include an original SFT+RL 4B baseline under the same training budget, which shows only marginal gains, while our method achieves significant improvements.

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
- 对综合视频理解基准的广泛评估表明，R-MSD 始终优于单样本蒸馏方法。
- 我们还在相同的训练预算下添加了原始的 SFT+RL 4B 基线，这仅显示出边际收益，而我们的方法实现了显着改进。
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
