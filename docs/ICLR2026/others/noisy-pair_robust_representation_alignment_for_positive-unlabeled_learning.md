# Noisy-Pair Robust Representation Alignment for Positive-Unlabeled Learning

**会议**: ICLR 2026 / **arXiv**: [2510.01278](https://arxiv.org/abs/2510.01278)  
**代码**: [GitHub](https://github.com/Hengwei-Zhao96/NcPU)  
**领域**: llm_alignment / 半监督学习  
**关键词**: positive-unlabeled learning, non-contrastive learning, noisy pairs, representation alignment, EM framework  

## 一句话总结
提出**NcPU**框架，通过**噪声对鲁棒的非对比损失(NoiSNCL)** 在不可靠监督下对齐类内表示，配合**幻影标签消歧(PLD)** 迭代优化伪标签，无需辅助负样本或预估类先验即可在PU学习中逼近甚至超越有监督性能。

## 背景与动机
1. PU学习仅有少量正样本标注+大量无标注数据（含正负混合），广泛应用于推荐/医疗/遥感等场景
2. SOTA PU方法与全监督存在巨大差距（CIFAR-100上14.26%），核心瓶颈是**在不可靠监督下学习判别性表示**
3. t-SNE显示现有PU方法（LaGAM/HolisticPU）学到的特征正负分布严重重叠，远不如监督学习清晰
4. 将表示学习引入PU面临新挑战：伪标签噪声导致错误的类内配对（noisy pairs），干扰对齐
5. 梯度分析表明：标准非对比损失下，**noisy pair的梯度反而大于clean pair**，噪声对主导训练

## 方法
- **NoiSNCL**: 将标准非对比损失 $2(1-\langle\tilde{q},\tilde{k}\rangle)$ 改为 $2\sqrt{1-\langle\tilde{q},\tilde{k}\rangle}$，使得clean pair梯度 > noisy pair梯度（梯度反转），对齐过程由正确配对主导
- **PLD (幻影标签消歧)**: 基于类原型(momentum更新)的伪标签分配 + PhantomGate机制：超过自适应阈值 $\tau$ 的样本赋予负标签，低于阈值的使用prototype-based soft label
- **自适应阈值(SAT)**: $\tau$ 从低开始（早期多给负监督），逐渐升高（后期过滤不确定负样本）
- **EM理论解释**: E-step=伪标签分配，M-step=最小化NoiSNCL（cluster tightening）；NoiSNCL是ℒr的上界，最小化它等价于最大化数据对数似然的下界
- **总损失**: $\mathcal{L} = \mathcal{L}_c^P + \mathcal{L}_c^U + w_r \tilde{\mathcal{L}}_r$，分类损失+NoiSNCL表示损失

## 实验
| 数据集 | NcPU (OA/F1) | 次优方法 | 监督上界 |
|--------|-------------|---------|---------|
| CIFAR-10 | **97.36/96.67** | LaGAM 95.78/94.90 (需负样本) | 96.96/96.24 |
| CIFAR-100 | **88.28/88.14** | LaGAM 84.82/84.42 | 89.65/89.78 |
| STL-10 | **91.40/90.82** | LaGAM 88.64/88.50 | — |
| ABCD (灾损建筑) | **91.10/91.21** | DistPU 86.25/87.36 | 92.00/91.96 |
| xBD (灾损建筑) | **87.60/64.84** | DistPU 82.94/57.58 | 88.47/73.32 |

| 消融 | 发现 |
|------|------|
| 仅NoiSNCL + 简单PU | 已达到有竞争力的性能，证明表示对齐是关键 |
| NoiSNCL vs 标准NCL | 标准NCL在PU中失败（noisy pair主导），NoiSNCL有效 |
| 去掉PhantomGate | 退化为trivial解（全预测为正） |
| NoiSNCL + PLD协作 | 两者迭代互相增强，符合EM框架理论预测 |

## 亮点
- 精准定位PU学习瓶颈为**表示质量**，并从梯度分析出发设计优雅的损失修改（仅加sqrt）
- 理论扎实：梯度反转证明→NoiSNCL设计→EM框架解释→似然下界保证，逻辑链完整
- NcPU比需要辅助负样本的LaGAM还好，且在CIFAR-10上**超越了全监督**（97.36 vs 96.96）
- 在遥感灾损评估任务(ABCD/xBD)上验证了实际应用价值

## 局限性
- 仅验证二分类场景，多类PU学习的扩展未讨论
- NoiSNCL的sqrt操作对噪声比例的鲁棒性边界未理论分析
- 类原型更新依赖momentum超参和batch采样，大规模不平衡数据可能不稳定
- wrw_r=50对所有数据集固定，但不同噪声率下可能需要调整

## 相关工作
- nnPU/DistPU: 基于unbiased risk的PU方法 → NcPU从表示学习角度切入
- BYOL(Grill,2020)/SupCon(Khosla,2020): 非对比/对比表示学习 → NoiSNCL拓展到噪声标签场景
- LaGAM(Long,2024): 需辅助负样本的SOTA → NcPU无需额外信息即超越

## 评分
- 新颖性: ⭐⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐⭐
