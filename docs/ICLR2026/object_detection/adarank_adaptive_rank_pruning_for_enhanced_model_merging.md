# AdaRank: Adaptive Rank Pruning for Enhanced Model Merging

**会议**: ICLR2026  
**arXiv**: [2503.22178](https://arxiv.org/abs/2503.22178)  
**代码**: 待确认  
**领域**: object_detection  
**关键词**: model merging, SVD, task vector, test-time adaptation, multi-task learning  

## 一句话总结
提出 AdaRank，通过可学习二值掩码自适应选择 task vector 的奇异分量（而非启发式 top-k），结合测试时熵最小化优化，大幅缓解多任务模型合并中的任务间干扰。

## 背景与动机
1. 模型合并（Model Merging）将多个独立微调模型整合为一个统一模型，避免多模型部署开销
2. Task Arithmetic 通过加权求和 task vector（微调与预训练权重之差）实现合并，但存在严重的任务间干扰
3. 近期 SVD 方法利用低秩结构截断 task vector，但依赖启发式固定 top-k 选择
4. 作者发现两个关键问题：(i) top 奇异分量虽然对本任务有益，但可能对其他任务造成更大干扰；(ii) 不同任务和层的内禀秩差异很大，固定秩截断不合理
5. 例如 SUN397（397类）的 task vector 需要更高的秩，而 MNIST 等简单任务秩更低
6. 早期层捕获任务无关特征（秩高、方差低），后期层编码任务特定表示（秩低、变异大）

## 方法详解
### 框架
- 对每个层 l 的每个任务 i 的 task vector 做 SVD 分解，引入二值掩码 B_i^l ∈ {0,1}^m 决定保留/剪枝每个奇异分量
- 合并公式：θ_m^l = θ_0^l + λ^l Σ_i U_i^l (diag(B_i^l) ⊙ Σ_i^l) V_i^l⊤

### 核心设计
- **自适应掩码**：不同于固定 top-k，每个奇异分量独立决定是否保留，允许任务间和层间不同秩
- **测试时优化**：使用 Shannon 熵最小化作为无监督代理目标，在无标签测试数据上优化掩码
- **STE 优化**：前向传播用二值掩码，反向传播连续化传梯度（Straight-Through Estimator）
- 可与 λ^l（层级系数）联合优化
- 兼容多种基线：Task Arithmetic、CART、TSV-M、Iso-CTS

## 实验关键数据
| 设置 | 方法 | ViT-B/32 (8任务) | ViT-L/14 (8任务) |
|------|------|---------|---------|
| 静态 | CART | 84.7 | 92.6 |
| 静态 | Iso-CTS | 84.9 | 93.0 |
| 自适应 | TA+AdaMerging | 80.1 | 90.8 |
| 自适应 | **TA+AdaRank** | **87.9** | **93.0** |
| 自适应 | **CART+AdaRank** | **89.2** | **93.5** |
| 自适应 | **Iso-CTS+AdaRank** | **89.4** | **95.5** |
| 路由 | WEMoE | 89.5 | - |

- NLP 任务（RoBERTa/GPT-2）：CART+AdaRank 分别达 0.7547/0.6587，显著优于 AdaMerging
- 20 任务场景下增益更大：TSV-M+AdaRank 达 86.9%（ViT-B/32），远超 WEMoE 的 80.2%
- 额外参数仅占总量 0.032%，TTA 时间与 AdaMerging 相当

## 亮点
- 揭示了 top-k 奇异分量在多任务场景下并非最优的反直觉现象
- 方法通用，可即插即用到多种静态/自适应模型合并框架
- 参数量恒定（不随任务数增长），优于路由方法的线性增长
- 跨视觉/NLP、跨架构（双向/自回归）均有效

## 局限性 / 可改进方向
- 需要无标签测试数据进行测试时适应，不适用于完全无数据场景
- SVD 分解有一定额外计算开销
- 熵最小化作为代理目标并非总与多任务损失完美相关

## 相关工作
- Task Arithmetic / TIES-Merging / DARE：逐元素稀疏化 task vector
- CART / TSV-M / STAR：SVD 低秩截断
- AdaMerging：测试时适应层级系数
- WEMoE / Twin-Merging：路由方法，参数随任务数线性增长

## 评分
- 新颖性: ⭐⭐⭐⭐ (自适应奇异分量选择替代启发式 top-k)
- 实验充分度: ⭐⭐⭐⭐⭐ (视觉+NLP，多backbone，多任务数，消融充分)
- 写作质量: ⭐⭐⭐⭐ (分析清晰，动机充分)
- 价值: ⭐⭐⭐⭐ (模型合并领域实用方法)
