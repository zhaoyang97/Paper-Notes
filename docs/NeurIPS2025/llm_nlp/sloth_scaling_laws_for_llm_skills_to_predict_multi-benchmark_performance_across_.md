# Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families

**会议**: NeurIPS 2025  
**arXiv**: [2412.06540](https://arxiv.org/abs/2412.06540)  
**代码**: https://github.com/felipemaiapolo/sloth  
**领域**: LLM效率 / 缩放定律  
**关键词**: scaling laws, LLM benchmarks, latent skills, factor analysis, performance prediction

## 一句话总结
提出Skills Scaling Laws (Sloth)，通过假设LLM性能由低维潜在技能（如推理、指令遵循）驱动，利用benchmark间的相关性构建跨模型家族的缩放定律，用少量家族数据即可预测大模型在多个benchmark上的表现。

## 研究背景与动机
1. **领域现状**：传统缩放定律（如Chinchilla）预测loss而非benchmark准确率，且在不同模型家族间泛化性差。
2. **现有痛点**：家族内缩放定律需要训练不同大小的模型（3-5个），成本高；跨家族缩放定律准确度差。
3. **核心矛盾**：不考虑家族信息→不准确；考虑家族信息→参数太多，需要大量训练数据。
4. **本文要解决什么**：用最少的家族数据（甚至1个模型），预测该家族更大模型在多个benchmark上的表现。
5. **切入角度**：不同benchmark的分数是相关的（因为都反映了底层的"技能"），利用这种相关性可以减少参数。
6. **核心idea一句话**：将缩放定律建立在低维"技能空间"上而非直接在benchmark空间上，通过因子分析共享参数。

## 方法详解

### 整体框架
输入：各家族LLM在多个benchmark上的分数 → 用因子分析提取d维潜在技能 → 对每个技能建立关于模型大小s和训练token数t的缩放定律 → 新家族只需估计效率参数α → 预测大模型表现。

### 关键设计

1. **潜在技能分解**:
   - 做什么：将J个benchmark的分数分解为d个低维潜在技能的线性组合
   - 核心思路：$\eta_i(s,t) = \Lambda \theta_i(s,t) + b$，其中 $\Lambda \in \mathbb{R}^{J \times d}$ 是因子载荷矩阵，$\theta_i$ 是第i个家族的技能向量
   - 设计动机：利用benchmark间的相关性减少参数量，避免过拟合

2. **跨家族技能缩放模型**:
   - 做什么：建模技能如何随计算资源缩放
   - 核心思路：$\theta_{ik}(s,t) = \alpha_{ik} + \beta_k^\top x(s,t)$，其中 $x = (\log s, \log t, \log s \cdot \log t)$，斜率 $\beta_k$ **跨家族共享**，截距 $\alpha_{ik}$ **家族特定**
   - 设计动机：$\alpha_{ik}$ 吸收家族特有因素（数据质量、后训练等），$\beta_k$ 捕捉计算→技能的通用规律

3. **可学习激活函数**:
   - 做什么：用单调神经网络替代固定的sigmoid函数
   - 核心思路：$\sigma_j$ 是每个benchmark特定的单调递增函数，权重约束为非负
   - 设计动机：不同benchmark的难度曲线形状不同，固定sigmoid可能不适配

### 损失函数 / 训练策略
- 使用Huber损失最小化，估计条件中位数
- 通过约束优化确保 $\gamma_j \in [0,1]$（猜测正确的概率）和 $\sigma_j$ 单调
- 整个模型是简单神经网络，在笔记本电脑上秒级拟合

## 实验关键数据

### 主实验 - 12个benchmark预测

| 方法 | MAE↓ | 说明 |
|------|------|------|
| Owen et al. (无家族信息) | 较高 | 不区分家族 |
| Ruan et al. (家族特定) | 中等 | 需要已有大模型 |
| Sloth (d=2技能) | **最低** | 仅需1个小模型 |

### 消融实验

| 配置 | 效果 |
|------|------|
| d=1 (单维技能) | 好于baseline |
| d=2 (二维技能) | 最佳平衡 |
| d=3+ | 过拟合，无额外增益 |
| 固定sigmoid | 略差于学习sigmoid |
| 无交互项 | 预测准确度下降 |

### 关键发现
- 2个潜在技能足以捕捉12个benchmark的变异（类似IQ的g-因子 + 第二因子）
- 家族效率参数 $\alpha$ 差异巨大——解释了为什么相同FLOPs的不同家族表现差异大
- 交互项 $\log s \cdot \log t$ 是重要的——说明参数量和数据量对技能不是独立作用的
- 可预测test-time compute scaling的效果
- 可推导compute-optimal技能缩放规则

## 亮点与洞察
- **技能空间的优雅抽象**：将"模型在不同benchmark上的表现"归结为"模型掌握了哪些技能"，这不仅提高了预测准确度，还提供了可解释的洞察。例如可以看到哪些benchmark测量相似的技能
- **家族效率参数的实用价值**：仅需一个已评估的小模型即可估计新家族的效率，然后预测大模型表现——这对决定是否值得训练大模型非常有价值
- **与经济学的巧妙连接**：借用随机前沿分析中的translog生产函数，将LLM训练视为"将计算转化为技能的生产过程"

## 局限性 / 可改进方向
- 技能维度d的选择缺乏理论指导
- 假设家族间技能斜率相同（仅截距不同）可能过强
- 仅在Open LLM Leaderboard的benchmark上验证
- 外推到极大模型时准确度未知

## 相关工作与启发
- **vs Chinchilla (Hoffmann et al.)**：他们预测loss，本文预测benchmark表现；他们需要训练多个模型，本文利用公开数据
- **vs Ruan et al.**：他们假设大模型已训练好来预测新benchmark，本文假设小模型已有来预测大模型
- **核心启发**：LLM评估的低秩结构意味着我们可能只需要很少的精心设计的benchmark来全面评估模型

## 评分
- 新颖性: ⭐⭐⭐⭐ 潜在技能视角的缩放定律是新颖的框架
- 实验充分度: ⭐⭐⭐⭐ 12个benchmark，多家族验证，多基线对比
- 写作质量: ⭐⭐⭐⭐⭐ 动机清晰，与经济学理论的连接优雅
- 价值: ⭐⭐⭐⭐⭐ 对LLM训练决策和评估有重要实用价值
