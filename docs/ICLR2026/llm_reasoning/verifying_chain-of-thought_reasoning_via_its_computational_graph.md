# Verifying Chain-of-Thought Reasoning via Its Computational Graph

**会议**: ICLR2026  
**arXiv**: [2510.09312](https://arxiv.org/abs/2510.09312)  
**代码**: [GitHub](https://github.com/facebookresearch/CRV)  
**领域**: llm_reasoning  
**关键词**: mechanistic interpretability, CoT verification, attribution graph, transcoder, circuit analysis  

## 一句话总结
提出CRV白盒方法，通过分析LLM推理步骤的归因图（计算图）结构特征来验证CoT正确性，在Arithmetic任务上AUROC达92.47，远超黑盒(76.45)和灰盒方法，并通过因果干预成功纠正错误推理。

## 背景与动机
1. CoT推理虽强大但过程本身可能有缺陷，需要自动验证
2. 黑盒方法（logit分布）和灰盒方法（hidden state探针）只能检测错误相关性，无法解释为何计算出错
3. 成熟的机制可解释性理论认为模型通过"circuits"（特化子图）实现推理，错误是潜在算法的执行缺陷
4. 归因图可视为推理的"执行轨迹"，类似软件调试中检查execution trace
5. 需要让模型中间计算可解释——通过transcoder替换MLP实现

## 方法
**CRV四阶段pipeline**:

1. **可解释化**: 将LLM每层MLP替换为训练好的transcoder(稀疏过完备表示)，保持功能等价但中间表示可解释

2. **构建归因图**: 对每个推理步骤，用贪心路径查找算法从最终logit反向追踪高归因因果路径，得到稀疏加权有向图 $G_i=(\mathcal{V}, \mathcal{E})$，节点=输入token+transcoder特征+输出logit

3. **提取结构指纹**: 从归因图提取固定维度特征向量(全局图统计+节点影响力统计+拓扑/路径特征)，包括图密度、中心性、连通性等

4. **诊断分类器**: 用Gradient Boosting Classifier在结构特征上训练，预测推理步骤正确/错误

## 实验
| 方法类别 | 方法 | Boolean AUROC | Arithmetic AUROC | GSM8K AUROC |
|---------|------|:---:|:---:|:---:|
| Black-box | MaxProb | 58.81 | 61.87 | 54.91 |
| Black-box | Energy | 51.08 | 76.45 | 62.55 |
| Gray-box | CoE-C | 51.03 | 69.39 | 53.57 |
| Gray-box | MLP Probe | 53.63 | 54.41 | 56.02 |
| **White-box** | **CRV** | **75.87** | **92.47** | **70.17** |

**关键发现**: (1) CRV在所有数据集和指标上大幅超越所有baseline，Arithmetic上FPR@95从63%降至37%; (2) 错误的结构签名高度领域特定——Boolean/Arithmetic/GSM8K的失败模式计算图结构各异; (3) 跨域泛化有限(GSM8K→Boolean仅45.77 AUROC)，但Combined训练有所改善; (4) 通过分析可解释特征和因果干预(修改单个transcoder特征)，成功纠正模型的错误推理; (5) 结构化推理任务(合成数据)的错误签名更一致、更可检测。

## 亮点
- 从"检测错误"走向"理解错误计算"，white-box范式极具科学意义
- 归因图结构特征作为推理正确性信号是全新视角
- 因果干预实验：修改transcoder特征→纠正推理，建立了结构签名与错误的因果关系
- 领域特异性发现揭示不同推理任务的失败源于不同计算模式

## 局限
- 计算密集：需trainscoder替换+归因图构建+特征提取，不适合实际部署
- 仅在Llama-3.1-8B上验证，对更大模型/推理模型的适用性未知
- 跨域泛化差，说明结构签名不够通用
- 依赖transcoder质量——替换MLP后模型行为可能有微妙偏移
- 聚焦单步验证，未扩展到完整CoT链级验证

## 相关工作
- 黑盒验证: PRM800K (Lightman et al. 2024), REVEAL (Jacovi et al. 2024)
- 灰盒探针: CoT-Kinetics (Bi et al. 2025), Chain-of-Embedding (Wang et al. 2025)
- 机制可解释性: Olah et al. 2020 circuits; transcoders (Dunefsky et al. 2024); Ameisen et al. 2025 circuit analysis
- SAE/Transcoder: Cunningham et al. 2023 sparse autoencoders

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (计算图结构验证推理正确性，全新方向)
- 实验充分度: ⭐⭐⭐⭐ (3数据集+多baseline+因果干预)
- 写作质量: ⭐⭐⭐⭐⭐ (问题定义精准，RQ驱动的实验设计)
- 价值: ⭐⭐⭐⭐⭐ (对理解LLM推理机制有深远意义)
