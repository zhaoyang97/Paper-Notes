<!-- 由 src/gen_stubs.py 自动生成 -->
# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**会议**: NEURIPS2025  
**arXiv**: [2504.02922](https://arxiv.org/abs/2504.02922)  
**代码**: 待确认  
**领域**: recommender / interpretability  
**关键词**: Crosscoder, 稀疏性伪影, BatchTopK, chat调优解释  

## 一句话总结
识别Crosscoder L1训练中的稀疏性伪影导致虚假模型特定潜变量归因，提出BatchTopK损失+Latent Scaling揭示真正的chat特定概念。

## 背景与动机
Crosscoder用于解释base/chat模型差异，但L1稀疏化引入伪影。

## 方法详解
Latent Scaling标记存在测量误差；BatchTopK代替L1；因果效应测试。

## 实验关键数据
Gemma 2 2B base/chat；揭示chat特定潜变量。

## 亮点
识别L1引入的伪影；BatchTopK缓解。

## 局限性
BatchTopK仅部分缓解；单模型对。

## 评分
- 新颖性: ⭐⭐⭐⭐ 稀疏伪影识别
- 实验充分度: ⭐⭐⭐ 单模型对
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐ 可解释性方法论
