# Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**会议**: arXiv 2026  
**arXiv**: [2603.12751](https://arxiv.org/abs/2603.12751)  
**作者**: James Akl, Jose Nicolas Avendano Arbelaez, James Barabas, Jennifer L. Barry, Kalie Ching
**代码**: 待确认  
**领域**: 目标检测  
**关键词**: show, don't, tell, detecting, novel  

## 一句话总结
机器人如何快速识别和识别在人类演示中向其展示的新物体？

## 背景与动机
How can a robot quickly identify and recognize new objects shown to it during a human demonstration?. Existing closed-set object detectors frequently fail at this because the objects are out-of-distribution.

## 核心问题
机器人如何快速识别和识别在人类演示期间向其展示的新物体？现有的闭集物体检测器经常会失败，因为物体不分布。

## 方法详解

### 整体框架
- While open-set detectors (e.g., VLMs) sometimes succeed, they often require expensive and tedious human-in-the-loop prompt engineering to uniquely recognize novel object instances.
- In this paper, we present a self-supervised system that eliminates the need for tedious language descriptions and expensive prompt engineering by training a bespoke object detector on an automatically created dataset, supervised by the human demonstration itself.
- In our approach, "Show, Don't Tell," we show the detector the specific objects of interest during the demonstration, rather than telling the detector about these objects via complex language descriptions.
- We develop an integrated on-robot system to deploy our "Show, Don't Tell" paradigm of automatic dataset creation and novel object-detection on a real-world robot.

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
- 经验结果表明，我们的流程明显优于最先进的被操纵物体检测和识别方法，从而提高了机器人的任务完成率。

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
