# Human Knowledge Integrated Multi-modal Learning for Single Source Domain Generalization

**会议**: CVPR 2026  
**arXiv**: [2603.12369](https://arxiv.org/abs/2603.12369)  
**代码**: [github.com/IMPACTLabASU/GenEval](https://github.com/IMPACTLabASU/GenEval)  
**领域**: 医学图像 / 域泛化 / 视觉语言模型  
**关键词**: 单源域泛化, 域保形界, 因果因子, 人类知识整合, 糖尿病视网膜病变, MedGemma-4B, LoRA  

## 一句话总结
提出域保形界(DCB)理论框架量化域间因果因子差异，并据此设计GenEval——通过知识精炼+MedGemma-4B LoRA微调，将人类专家领域知识整合到VLM中实现单源域泛化，在8个DR和2个SOZ数据集上显著超越SOTA。

## 背景与动机
医学图像分类（如DR分级、SOZ检测）跨域泛化是核心挑战。关键瓶颈：域间存在未知因果因子差异——如新生血管仅出现在EyePACS而Messidor中无此指标，形成因果鸿沟。现有DG方法在DR上未能一致超越ERM。更实际的单源域泛化(SDG)——仅用一个域训练跨域部署——挑战更大。

## 核心问题
(1) 如何无分布假设地量化两域因果因子差异？(2) 如何将专家定性知识转化为可优化信号？(3) 如何高效整合到VLM中实现跨域泛化？

## 方法详解
### 整体框架
Step 1 DCB理论量化因果覆盖 → Step 2 SDCD指标评估域一致性 → Step 3 知识精炼+GenEval多模态分类。

### 关键设计
1. **域保形界(DCB)**: 基于保形推断的分布无关框架。用SINDy/Koopman将因果因子建模为稀疏系数矩阵K，通过Mahalanobis距离构造置信区间C。目标域样本的鲁棒性度量落入C内则以概率>=1-alpha共享源域因果模式
2. **源域一致度(SDCD)**: 计算目标域中多少比例样本落入源域DCB内。证明SDCD与SDG精度正相关（Pearson r=0.692, p<0.02），可预测泛化效果
3. **知识量化与精炼**: YOLOv12检测眼底病征转为14维IoU向量，SDCD引导消融找最优知识子集
4. **GenEval**: 精炼知识+图像构造多模态prompt，MedGemma-4B LoRA微调（rank=16, alpha=16, 95M/4B=2.4%可训练）

### 损失函数 / 训练策略
CAUSAL_LM标准损失，LoRA作用于全部attention和MLP投影层。单域训练1-10小时，推理约424ms/样本。

## 实验关键数据
**MDG**: GenEval 79.21% vs SPSD-ViT 73.3%（+5.71%）

**SDG关键对比**:
| 源域 | 目标域 | Baseline | GenEval | 提升 |
|------|--------|----------|---------|------|
| EyePACS | Messidor | DRGen 54.6% | 69.5% | +14.9% |
| EyePACS | Messidor2 | DRGen 65.4% | 80.5% | +15.1% |
| Messidor | EyePACS | SPSD-ViT 57.4% | 80.0% | +22.6% |

- 扩展SDG（EyePACS训练，6外部目标）：GenEval均66.2% vs DECO 50.68%（+15.5%）
- VLM对比：GenEval F1=75.1% vs CLIP-DR 46.8%（+28.3%）
- SOZ检测：GenEval F1=90.0% vs CuPKL GPT-4o 88.1%，跨站点更稳定

### 消融实验要点
- 知识精炼：从no-ablation SDCD 59%逐步移除成分到82.81%，精度65.01%→73.23%，SDCD与精度正相关
- 零样本MedGemma-4B：均71.73%但域间差异大（APTOS 61.8% vs EyePACS 79.66%），需微调
- SDCD噪声敏感性：PSNR>15dB时稳定，<10dB相关性塌缩
- YOLOv11 vs v12：SDCD略变但精度差异不显著

## 亮点
- 理论-实践紧密结合：DCB/SDCD从理论解释现有DG方法失败原因并给出改进路径
- 知识精炼机制巧妙：SDCD指标自动筛选有用知识成分
- SDCD有独立价值——无需训练即可预测源-目标域SDG可行性
- 8个DR+2个SOZ数据集大规模验证，覆盖不同设备/协议/人群

## 局限性
- DCB假设连续可微数据生成过程，尖锐阈值效应下有误差
- 人类知识获取依赖领域专家，可扩展性受限
- 知识精炼贪心消融非全局最优
- 仅医学图像场景验证

## 与相关工作的对比
- **SPSD-ViT**: DR域泛化SOTA，假设目标域可交换，无法判断新域是否在训练支持外。GenEval通过DCB提供部署前安全评估
- **BiomedCLIP/CLIP-DR**: 预训练VLM迁移，GenEval通过LoRA+知识注入大幅超越（F1 75.1% vs 46.8%）
- **CuPKL**: SOZ的GPT-4o零样本方法，单站点强但跨站点不稳，GenEval更一致

## 启发与关联
- 领域专家知识参数化注入VLM的范式可推广到任何垂直领域
- DCB/SDCD提供不依赖目标域标签的泛化能力预测工具

## 评分
- 新颖性: ⭐⭐⭐⭐
- 实验充分度: ⭐⭐⭐⭐⭐
- 写作质量: ⭐⭐⭐⭐
- 价值: ⭐⭐⭐⭐
