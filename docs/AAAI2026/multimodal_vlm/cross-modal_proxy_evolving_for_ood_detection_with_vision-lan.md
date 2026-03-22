# Cross-modal Proxy Evolving for OOD Detection with Vision-Language Models

**会议**: AAAI 2026  
**arXiv**: [2601.08476](https://arxiv.org/abs/2601.08476)  
**代码**: 未公开  
**领域**: 多模态VLM  
**关键词**: OOD检测, VLM, 跨模态代理演化, zero-shot, test-time adaptation  

## 一句话总结
提出CoEvo，一个training-free和annotation-free的test-time框架，通过双向样本条件化的文本/视觉proxy协同演化来增强VLM的zero-shot OOD检测，在ImageNet-1K上比强baseline提升AUROC 1.33%并降低FPR95达45.98%。

## 背景与动机
VLM（如CLIP）在开放世界部署时需要可靠的OOD检测。Zero-shot OOD检测的难点是没有标注的负样本。现有方法依赖固定的文本proxy（文本负标签），但存在两个问题：(1) 稀疏采样ID类别之外的语义空间；(2) 静态文本proxy无法适应视觉特征的分布偏移，导致跨模态不对齐和预测不稳定。

## 核心问题
如何在没有训练和标注的情况下，动态调整用于OOD检测的跨模态proxy，使其在test-time适应分布偏移？

## 方法详解

### 整体框架
CoEvo在test-time维护两个演化的proxy缓存（文本proxy缓存+视觉proxy缓存），通过双向co-evolution机制持续更新。

### 关键设计
1. **Proxy-Aligned Co-Evolution**: 利用测试图像引导动态挖掘上下文文本负样本——不使用固定的负标签集，而是根据当前测试样本的视觉语义动态生成更相关的负标签
2. **视觉Proxy迭代精化**: 基于文本proxy的更新反过来精化视觉proxy，逐步重新对齐跨模态相似度并扩大局部OOD边际
3. **双模态Proxy动态加权**: 动态调整文本和视觉proxy对最终OOD分数的贡献权重，获得对分布偏移鲁棒的校准分数

### 损失函数 / 训练策略
完全Training-free和Annotation-free，仅在test-time在线演化proxy。

## 实验关键数据
| 基准 | 指标 | CoEvo | vs Baseline |
|--------|------|-------|-------------|
| ImageNet-1K | AUROC | SOTA | +1.33% |
| ImageNet-1K | FPR95 | SOTA | -45.98% (相对降低) |

### 消融实验要点
- 文本proxy演化和视觉proxy演化都有独立贡献
- 双向演化优于单向
- 动态加权优于固定权重

## 亮点
- **双向co-evolution是优雅的设计** — 文本指导视觉、视觉反馈文本，形成闭环
- **完全training/annotation-free** — 即插即用到任何VLM
- FPR95降低45.98%是非常显著的改进
- 解决了固定proxy的两个根本局限（稀疏性和静态性）

## 局限性 / 可改进方向
- Test-time online演化有额外计算开销
- 仅基于abstract，具体proxy缓存大小和更新策略细节待补充
- 是否适用于fine-grained OOD（如ID子类的near-OOD）未验证

## 与相关工作的对比
- **vs MCM/NegLabel等固定proxy方法**: CoEvo通过动态演化proxy解决固定proxy的稀疏性和静态性
- **vs DiVE(AAAI2026)**: DiVE解决VLM fine-tuning的OOD泛化，CoEvo解决zero-shot OOD检测——不同层面的问题

## 启发与关联
- Co-evolution的思路可以用到VLM的其他test-time adaptation场景（如域适应、continual learning）
- Proxy演化机制与AStar的thought cards有相似的"动态检索+适配"思想

## 评分
- 新颖性: ⭐⭐⭐⭐ 双向proxy co-evolution是新颖有效的框架
- 实验充分度: ⭐⭐⭐⭐ 标准OOD benchmark，显著改进
- 写作质量: ⭐⭐⭐⭐ 问题分析到位
- 价值: ⭐⭐⭐⭐ 对VLM安全部署有直接价值
