# Composing Driving Worlds through Disentangled Control for Adversarial Scenario Generation

**会议**: arXiv 2026  
**arXiv**: [2603.12864](https://arxiv.org/abs/2603.12864)  
**作者**: Yifan Zhan, Zhengqing Chen, Qingjie Wang, Zhuo He, Muyao Niu
**代码**: 待确认  
**领域**: 扩散模型/生成 / 自动驾驶/机器人  
**关键词**: composing, driving, worlds, through, disentangled  

## 一句话总结
自动驾驶的一个主要挑战是安全关键边缘情况的“长尾”，这些边缘情况通常是由常见交通元素的不寻常组合产生的。
## 背景与动机
A major challenge in autonomous driving is the "long tail" of safety-critical edge cases, which often emerge from unusual combinations of common traffic elements.. Synthesizing these scenarios is crucial, yet current controllable generative models provide incomplete or entangled guidance, preventing the independent manipulation of scene structure, object identity, and ego actions.

## 核心问题
自动驾驶的一个主要挑战是安全关键边缘情况的“长尾”，这些边缘情况通常是由常见交通元素的不寻常组合产生的。
## 方法详解

### 整体框架
- We introduce CompoSIA, a compositional driving video simulator that disentangles these traffic factors, enabling fine-grained control over diverse adversarial driving scenarios.
- To support controllable identity replacement of scene elements, we propose a noise-level identity injection, allowing pose-agnostic identity generation across diverse element poses, all from a single reference image.
- Furthermore, a hierarchical dual-branch action control mechanism is introduced to improve action controllability.

### 关键设计
1. **关键组件1**: Furthermore, a hierarchical dual-branch action control mechanism is introduced to improve action controllability.

### 损失函数 / 训练策略
待深读后补充。

## 实验关键数据
| 数据集 | 指标 | 本文 | 之前SOTA | 提升 |
|--------|------|------|----------|------|
| 待补充 | - | - | - | - |

### 消融实验要点
- 待深读后补充

## 亮点 / 我学到了什么
- 此外，引入分层双分支动作控制机制，提高动作可控性。
- 广泛的比较表明，与最先进的基线相比，其可控生成质量更高，身份编辑的 FVD 提高了 17%，动作控制的旋转和平移错误分别减少了 30% 和 47%。
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
