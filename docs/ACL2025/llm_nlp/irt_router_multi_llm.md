# IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory

**会议**: ACL 2025  
**arXiv**: [2506.01048](https://arxiv.org/abs/2506.01048)  
**代码**: [https://github.com/Mercidaiha/IRT-Router](https://github.com/Mercidaiha/IRT-Router)  
**领域**: LLM/NLP  
**关键词**: LLM routing, item response theory, multi-model selection, interpretability, cost optimization

## 一句话总结
IRT-Router 借鉴心理测量学的项目反应理论（IRT），将 LLM 视为"考生"、query 视为"考题"，学习多维能力向量和难度/区分度参数实现可解释的多 LLM 路由，在 OOD 场景下达 87%+ 准确率且成本仅为 GPT-4o 的 1/30。

## 研究背景与动机
1. **领域现状**：使用多个 LLM 时需要根据 query 特点自动选择最合适的模型，平衡性能和成本
2. **现有痛点**：现有路由方法（RouteLLM、RouterBench）用简单启发式或黑箱分类器，缺乏可解释性，无法说明"为什么路由到这个模型"
3. **核心矛盾**：需要同时解决可解释性、cold-start（新 query 如何路由）、性能-成本权衡三个问题
4. **核心idea一句话**：IRT 天然建模"能力-难度"关系，将其迁移到 LLM 路由可同时获得可解释性和效果

## 方法详解

### 整体框架
两个实现版本：(1) **MIRT-Router**（多维IRT）：$\hat{P}(q_i, M_j) = 1/(1 + \exp(-a_i^T \theta_{M_j} + b_i))$，$\theta_{M_j}$ 为 LLM 能力向量，$a_i$ 为区分度，$b_i$ 为难度；(2) **NIRT-Router**（神经IRT）：引入 relevance vector 和神经网络交互函数。

### 关键设计

1. **IRT 建模**：每个 LLM 有多维能力向量 $\theta_{M_j}$，每个 query 有难度 $b_i$ 和区分度 $a_i$，参数通过 embedding + 线性变换学习
2. **Cold-start Warm-up**：对未见 query，用邻近已知 query 的嵌入插值：$e_{q_i}' = (1-\lambda) e_{q_i} + \lambda \cdot \text{mean(neighbors)}$，$\lambda=0.3\text{-}0.4$ 最优
3. **评分函数**：$S(q_i, M_j) = \alpha \hat{P}(q_i, M_j) - \beta C(M_j)$，$\alpha+\beta=1$ 平衡性能和成本

## 实验关键数据

### 主实验
| 方法 | 准确率 | 成本 | Reward |
|------|-------|------|--------|
| MIRT-Router | 80.67% | $0.42 | 63.89 |
| RouterBench | 80.01% | $1.15 | 62.23 |
| RouteLLM | 77.25% | $12.80 | 42.00 |
| GPT-4o only | 77.53% | $12.93 | 42.02 |

### OOD 场景（20 个候选 LLM，12 个数据集）
| 方法 | 准确率 | 成本 |
|------|-------|------|
| MIRT-Router | 87.12% | $0.14 |
| NIRT-Router | 87.37% | $0.15 |

### 关键发现
- **性能-成本最优**：准确率接近最强单模型，成本仅 1/30
- **可解释性**：能力向量和难度分数有明确语义（DeepSeek-Chat 能力最强=81%，GPT-4o=78%）
- **Cold-start 有效**：warm-up 机制显著提升 OOD 表现

## 亮点与洞察
- **IRT→LLM 路由的跨领域迁移**很优雅：心理测量学的成熟理论直接适用于 LLM 能力评估
- **可解释性是核心卖点**：不仅路由效果好，还能解释每个 LLM 擅长什么、每个 query 难在哪里

## 局限性 / 可改进方向
- Top-1 路由准确率较低（2-3%），因为多个模型能力相似
- 对全新 LLM（训练时未见过的模型）泛化有限
- 路由器对成本参数变化不够敏感

## 评分
- 新颖性: ⭐⭐⭐⭐ IRT 用于 LLM 路由是巧妙的跨领域迁移
- 实验充分度: ⭐⭐⭐⭐⭐ 20 个 LLM × 12 个数据集 × ID+OOD 场景
- 写作质量: ⭐⭐⭐⭐ IRT 理论介绍清晰，可解释性分析充分
- 价值: ⭐⭐⭐⭐⭐ 对多 LLM 部署场景有直接实用价值
