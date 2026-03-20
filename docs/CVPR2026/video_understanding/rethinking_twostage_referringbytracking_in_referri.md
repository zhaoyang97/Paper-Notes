# FlexHook: Rethinking Two-Stage Referring-by-Tracking in RMOT

**会议**: CVPR 2026  
**arXiv**: [2503.07516](https://arxiv.org/abs/2503.07516)  
**代码**: [GitHub](https://github.com/buptLwz/FlexHook)  
**领域**: 目标检测/追踪 / 视觉-语言  
**关键词**: referring multi-object tracking, two-stage RBT, language-conditioned sampling, pairwise correspondence  

## 一句话总结
FlexHook重新激活了两阶段RBT(Referring-by-Tracking)范式：用C-Hook从backbone直接采样目标特征(替代双编码)并注入语言条件线索，用PCD(成对对应解码器)替代CLIP余弦相似度做主动对应建模，首次让两阶段方法全面超越一阶段RMOT的SOTA——Refer-KITTI-V2上HOTA从10.32(iKUN)提升到42.53，训练仅1.91小时(2×4090)。

## 背景与动机
Referring Multi-Object Tracking (RMOT)用自然语言追踪多目标。三种范式：(1) TBR(GroundingDINO定位+关联)；(2) 一阶段RBT(MOTR轨迹query端到端)；(3) 两阶段RBT(先离线跟踪再做referring匹配)。两阶段的优点：训练成本低、可增量部署(升级tracker不影响referring模块)。但iKUN(CVPR24)仅达10.32 HOTA(Refer-KITTI-V2)，远落后一阶段(35+)。原因：(1) 启发式特征构建——双编码全图+裁剪patch，浪费计算且忽略预训练backbone的上下文能力；(2) 脆弱的CLIP余弦相似度匹配——受限于CLIP对齐空间，加额外模块后容易失效。

## 核心问题
如何在保持两阶段RBT的训练效率和部署灵活性的同时，大幅提升其referring匹配精度？

## 方法详解

### 整体框架
输入：离线tracker输出的轨迹bbox序列 + 视频帧 + 语言表达式集。FlexHook用backbone编码全图(仅一次)，然后通过C-Hook从多尺度特征图直接采样目标特征和语言条件参考特征，经时间整合后用PCD解码各表达式-轨迹对的匹配分数。

### 关键设计
1. **C-Hook (Conditioning Hook)**: 两个子组件——(a) Neighboring Grid Sampling：根据轨迹bbox构建采样网格，从backbone特征图双线性插值采样目标特征。引入3种扰动(轨迹片段随机遮断/高斯噪声/batch内ID交换)模拟跟踪噪声，增强训练-推理一致性。(b) Conditioning Enhancement：用Transformer decoder从语言特征解码M个参考点坐标(learnable query + 交叉注意力 + MLP + sigmoid)，在特征图上额外采样语言条件特征。不同语义表达(如"红色的人" vs "左边的人")会关注不同区域。

2. **Temporal Integration**: 将多帧目标特征与帧间网格坐标位移(显式光流)拼接后用MLP压缩，捕获运动信息(处理"向左转弯的车"等运动表达)。

3. **PCD (Pairwise Correspondence Decoder)**: learnable query向量通过masked cross-attention同时访问轨迹特征(共享)和对应表达式的语言特征+参考特征(private)。用attention mask确保每个query只看自己对应的语言分支，同时共享轨迹特征实现隐式对比学习。多尺度解码(通过FPN)。输出N^个匹配分数。

### 损失函数 / 训练策略
Focal Loss做匹配分数监督(处理正负样本不平衡)。参考点边界正则L_r(softplus barrier，防止学习到的参考坐标退化到边界)。AdamW lr=3e-5，20 epochs，2×4090。

## 实验关键数据

### Refer-KITTI-V2 (最主要benchmark)
| 方法 | 范式 | HOTA | DetA | AssA |
|------|------|------|------|------|
| TransRMOT | 一阶段 | 31.00 | 19.40 | 49.68 |
| HFF-Tracker | 一阶段 | 36.18 | 24.64 | 53.27 |
| iKUN (CVPR24) | 两阶段 | 10.32 | 2.17 | 49.77 |
| **FlexHook-best** | **两阶段** | **42.53** | **30.63** | **59.19** |

### 训练效率 (Refer-KITTI-V2)
| 方法 | 训练时间 | HOTA |
|------|---------|------|
| TempRMOT(一阶段) | 51.68h (60ep) | 35.04 |
| iKUN(两阶段) | 2.46h (100ep) | 10.32 |
| **FlexHook** | **1.91h** (20ep) | **42.53** |

### LaMOT (大规模多场景)
FlexHook-best: HOTA 56.77 vs LaMOTer 48.45(+8.32)

### 消融实验要点
- **C-Hook贡献最大**: iKUN→+C-Hook=34.49 HOTA(+24.17!)，说明特征构建是之前两阶段方法的核心瓶颈
- **PCD额外提升**: +PCD=38.62(+4.13)，替代CLIP余弦相似度
- **Conditioning Enhancement(M=10)**: 一致提升0.5-1.5 HOTA
- **Neighboring噪声扰动**: 去掉后降1.3 HOTA，说明模拟跟踪噪声对训练-推理对齐很重要
- **冻结编码器**: 冻结所有编码器仅降~1.7 HOTA(42.53→40.86)，可用于极低资源部署
- **不依赖CLIP**: 用RoBERTa+Swin-T(非对齐空间)反而比CLIP更好(42.53 vs 41.42)

## 亮点 / 我学到了什么
- **"采样替代重编码"的轻量设计**: C-Hook直接从backbone特征图grid sampling，避免了重复编码全图和裁剪patch，同时保留了预训练backbone的上下文梯度流，设计极其简洁
- **跟踪噪声的数据增强**: 训练用GT轨迹、推理用tracker输出→引入合成噪声(片段删除/位置扰动/ID交换)显著缩小train-test gap，这个思路对所有两阶段pipeline都有参考价值
- **PCD的masked attention**: 让所有表达式-轨迹对共享轨迹特征做cross-attention，通过mask隔离语言分支——既实现了成对判别又获得了跨对对比学习的效果
- **两阶段方法的回归**: 首次证明正确设计的两阶段RBT可以全面超越一阶段，且训练更快，对"是否需要端到端"的讨论有启发

## 局限性 / 可改进方向
- 性能仍部分依赖tracker质量(用弱tracker D-DETR+StrongSORT时HOTA从42.53降至40.73)
- 参考点数M=10是手动选择的，自适应确定M可能更好
- 仅在多目标追踪场景验证，单目标referring tracking未测试
- PCD的多尺度解码增加一定推理开销

## 与相关工作的对比
- **vs iKUN**: 同为两阶段，iKUN用双编码+CLIP相似度得10.32 HOTA，FlexHook用C-Hook+PCD得42.53——差距来自特征构建和匹配策略的根本性改变
- **vs TransRMOT/TempRMOT(一阶段)**: 需要60epoch端到端训练(51.68h)，FlexHook只需1.91h且性能更好
- **vs LaMOTer(TBR)**: 用GroundingDINO做开集定位，开集泛化强但RMOT性能弱于FlexHook

## 与我的研究方向的关联
- "语言条件特征采样"思路可迁移到其他视觉-语言定位任务
- 两阶段vs一阶段的权衡分析对多模态系统设计有参考
- PCD的masked pairwise attention可用于其他需要一对多匹配的场景

## 评分
- 新颖性: ⭐⭐⭐⭐ C-Hook的采样替代重编码思路简洁有效，PCD替代CLIP解耦了模型选择
- 实验充分度: ⭐⭐⭐⭐⭐ 4个benchmark、多种encoder组合、详细消融(C-Hook/PCD/噪声/参考点/冻结/效率)
- 写作质量: ⭐⭐⭐⭐⭐ 动机论证有力(两个fundamental limitations)，图表清晰，"Make it Strong Again"的叙事完整
- 对我的价值: ⭐⭐⭐⭐ 视觉-语言目标追踪方向，C-Hook和PCD的设计思路有较高迁移价值
