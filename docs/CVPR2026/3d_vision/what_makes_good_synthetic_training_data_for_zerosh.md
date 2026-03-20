# WMGStereo: What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?

**会议**: CVPR 2026  
**arXiv**: [2504.16930](https://arxiv.org/abs/2504.16930)  
**代码**: [GitHub](https://github.com/princeton-vl/InfinigenStereo)  
**领域**: 3D视觉 / 立体匹配  
**关键词**: stereo matching, synthetic data, procedural generation, zero-shot, dataset design  

## 一句话总结
系统研究合成立体数据集的设计空间——变换Infinigen过程化生成参数(浮动物体密度/背景/材质/相机baseline/光照等)分析其对零样本立体匹配的影响，发现"真实室内场景+浮动物体"的组合最有效；据此构建WMGStereo-150k数据集，仅用此单一数据集训练超越SceneFlow+CREStereo+TartanAir+IRS四合一(Middlebury降28%，Booster降25%)，与FoundationStereo竞争力相当。

## 背景与动机
合成数据是训练立体匹配网络的关键，但"什么使一个数据集有效？"这个问题缺乏系统研究。现有数据集各自引入不同设计(SceneFlow飞行物体/TartanAir写实室内/FoundationStereo混合)，但同时改变多个因素，无法确定哪个因素真正重要。而且大多数数据集不开源生成代码，无法复现和修改。

## 核心问题
合成立体数据集的哪些设计参数对零样本泛化最重要？如何用这些洞察构建更有效的训练数据集？

## 方法详解

### 整体框架
基于Infinigen(过程化生成器)构建可控stereo数据生成系统。支持3种场景类型：室内+浮动物体、密集浮动物体(空场景)、自然场景。对每个参数单独变化生成5000对stereo对→训练RAFT-Stereo→评估7个benchmark的零样本性能→找最优参数组合→生成WMGStereo-150k大规模数据集。

### 关键发现(Tab.1)
1. **浮动物体密度**：最关键因素。无浮动物体→加0-10个→加10-30个，Middlebury(H)从12.52→7.78→6.60持续下降。更多浮动物体=更多几何多样性
2. **背景物体**：有背景家具 > 无背景(空房间)。所有benchmark一致提升。说明场景真实性确实有帮助(与optical flow中"realism is overrated"的结论不同)
3. **物体类型**：只用椅子→室内benchmark好但驾驶差；只用灌木→KITTI好但室内差。使用全部生成器跨benchmark最鲁棒
4. **材质**：全材质 > 单一漫反射 > 纯金属玻璃 > 无材质。材质多样性重要但非朗伯材质过多会伤害漫反射区域性能
5. **相机baseline**：宽范围[0.04, 0.4]m大幅优于窄范围。基线多样性对泛化至关重要
6. **光照增强**：效果微弱但略有帮助
7. **场景比例**：室内+浮动 > 密集浮动 > 自然(单独)。33-33-33混合最优

### 成本优化
- 降低室内求解器步数(550→60步)：速度4倍但家具摆放不那么真实——在固定计算budget下反而更好(更多数据>更真实)
- 降低光线追踪采样+去噪(1024样本+OptiX denoise vs 8192样本)：渲染时间27秒/帧
- 场景复用：每个室内场景放20个相机位/每个密集飞行场景随机化200次→减少CPU成本

### 损失函数 / 训练策略
RAFT-Stereo标准训练75k步(参数研究)/200k步(最终)。DLNR和Selective-IGEV也验证跨架构泛化。所有使用默认超参和增强。

## 实验关键数据

### WMGStereo-150k vs 现有数据集(DLNR, 200k步)
| 训练数据 | Midd-14(H) | Midd-21 | ETH3D | KITTI-12 | KITTI-15 | Booster |
|---------|-----------|---------|-------|---------|---------|---------|
| SceneFlow | 6.20 | 8.44 | 23.01 | 9.08 | 16.05 | 18.15 |
| CREStereo | 11.53 | 10.60 | 5.18 | 4.95 | 5.90 | 14.61 |
| IRS | 6.13 | 8.49 | 3.91 | 4.56 | 5.60 | 10.32 |
| FSD(FoundationStereo) | 3.27 | 6.93 | 2.13 | 3.56 | 4.18 | 7.51 |
| **WMGStereo-150k** | **3.76** | **6.72** | 2.50 | **3.30** | 4.54 | **9.09** |
| FSD+WMGStereo | 3.24 | 6.88 | **2.08** | 3.59 | **4.26** | 7.42 |

### 跨架构验证(vs Mixed=SF+CRE+Tartan+IRS 600k)
| 模型-数据 | Midd-14(H) | Booster |
|----------|-----------|---------|
| DLNR-Mixed | 5.21 | 12.17 |
| DLNR-WMGStereo | **3.76** (-28%) | **9.09** (-25%) |
| RAFT-Mixed | 5.50 | 11.46 |
| RAFT-WMGStereo | **4.48** | **9.17** |

### 采样效率(Fig.5)
仅500个WMGStereo样本 < 100K个CREStereo样本的Middlebury EPE

### 消融要点
- 室内+浮动物体是最佳单一场景类型
- 混合三种场景类型最鲁棒
- 高error物体(仙人掌、海胆等针状结构)和problemmatic材质(全透明玻璃、全反射金属)需移除
- 降低渲染质量+增加数据量(固定计算)通常更好

## 亮点 / 我学到了什么
- **"真实场景+随机物体"的组合超越两者单独使用**: 推翻了optical flow领域"realism is overrated"的结论——对stereo来说，背景的真实几何确实提供有用的训练信号
- **相机baseline多样性极其重要**: 单一baseline范围→严重偏向特定disparity分布。这对数据生成是简单但高效的改进
- **数据质量 vs 数据量的tradeoff**: 降低渲染精度→虽每张略差但固定计算budget下总量更多→零样本性能更好。量>质(在足够的质的基础上)
- **500样本WMGStereo > 100K CREStereo**: 数据集的设计比规模更重要
- **开源生成代码的价值**: 与FSD(静态数据集)不同，WMGStereo提供生成代码，允许针对特定领域定制数据

## 局限性 / 可改进方向
- 非朗伯表面(玻璃/金属)仍是瓶颈——当前移除了high-error材质作为折衷
- 自然场景类型(Nature)独立表现最差，可能需要更好的自然物体生成和相机放置
- 未涉及动态/temporal stereo的数据生成
- 与FoundationStereo的差距主要在architecture+data的联合设计上

## 与相关工作的对比
- **vs FoundationStereo数据集(FSD)**: FSD同时引入很多新特性+新架构，本文分离并分析每个因素的贡献。WMGStereo与FSD互补(两者合用更好)
- **vs SceneFlow/FlyingThings3D**: 经典飞行物体数据集，但缺乏场景真实性和材质多样性。WMGStereo用约1/4的数据量大幅超越
- **vs Mayer et al.(2018)的flow数据研究**: 那个研究聚焦2D warp+光学流，结论"realism overrated"。本文在3D stereo中发现背景真实性确实有帮助

## 与我的研究方向的关联
- "什么使合成数据有效？"的方法论对任何使用合成数据的任务都有参考价值
- 过程化生成+系统消融的实验范式值得学习
- 3D视觉领域但与核心关注方向距离适中

## 评分
- 新颖性: ⭐⭐⭐⭐ 系统的参数分析是该领域首次，洞察有实用价值(尤其background+floating objects的组合)
- 实验充分度: ⭐⭐⭐⭐⭐ 7个benchmark、多种架构、极其详细的参数消融(Tab.1每行一个实验)、采样效率分析、held-out benchmark验证
- 写作质量: ⭐⭐⭐⭐⭐ 结构清晰、发现描述直观、开源贡献完整
- 对我的价值: ⭐⭐⭐ stereo matching非核心方向，但合成数据设计的方法论有借鉴意义
