<!-- 由 src/gen_stubs.py 自动生成 -->
# GraphFaaS: Serverless GNN Inference for Burst-Resilient, Real-Time Intrusion Detection

**会议**: NEURIPS2025  
**arXiv**: [2511.10554](https://arxiv.org/abs/2511.10554)  
**代码**: 待确认  
**领域**: graph_learning / systems  
**关键词**: GNN推理, Serverless, 入侵检测, 突发负载, 图分区  

## 一句话总结
提出GraphFaaS，基于Serverless的GNN推理架构用于突发负载下的实时入侵检测：时间局部性图构建+频率过滤+贪心图分区实现延迟降低85%、变异系数降低64%同时保持准确率。

## 背景与动机
来源图基于GNN的入侵检测在突发负载下延迟火爆。Serverless动态扩缩解决。

## 方法详解
- 时间局部性图构建 + 频率过滤
- 特征长度感知节点嵌入
- 贪心best-fit图分区

## 实验关键数据
- 延迟 2.1s→14.16s（降85%）
- CV降64%，准确率保持

## 亮点
1. 首个Serverless GNN推理系统
2. 实时入侵检测实用价值

## 局限性
超级节点依赖爆炸；需垂直扩展回退

## 评分
- 新颖性: ⭐⭐⭐⭐ Serverless+GNN
- 实验充分度: ⭐⭐⭐ 系统评估
- 写作质量: ⭐⭐⭐ 系统导向
- 价值: ⭐⭐⭐⭐ 全负载GNN推理实用
