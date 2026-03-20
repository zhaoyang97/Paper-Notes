# A Simple Baseline for Spoken Language to Sign Language Translation with 3D Avatars

**会议**: ECCV 2024  
**arXiv**: [2401.04730](https://arxiv.org/abs/2401.04730)  
**代码**: [https://github.com/FangyunWei/SLRT](https://github.com/FangyunWei/SLRT)  
**领域**: 3D视觉 / 手语翻译 / 人体姿态估计  
**关键词**: Spoken2Sign, 手语生产, SMPLSign-X, 3D Avatar, co-articulation  

## 一句话总结
提出首个基于3D Avatar输出的Spoken2Sign翻译基线系统，通过三步流程（字典构建→SMPLSign-X 3D手语估计→检索-连接-渲染翻译）将口语文本翻译为3D手语动画，在Phoenix-2014T上back-translation BLEU-4达25.46，同时其3D手语副产品（关键点增强和多视角理解）显著提升了手语理解任务性能。

## 背景与动机
手语翻译领域长期聚焦于Sign2Spoken（手语→口语）方向，而反向的Spoken2Sign（口语→手语）研究严重不足。已有的Spoken2Sign工作主要输出2D关键点序列或用生成模型合成2D视频，但关键点序列难以被聋人理解，2D视频则存在模糊和视觉失真问题。随着SMPL-X等参数化3D人体模型的成熟，利用3D Avatar展示手语成为可能——它不仅避免了2D表示的失真问题，还允许从任意视角观看手语，更接近真实沟通场景。

## 核心问题
如何构建一个端到端的Spoken2Sign系统，将输入文本翻译为高质量的3D手语动画？这涉及三个子问题：（1）现有手语数据集缺乏现成的手语词典，如何自动构建？（2）如何从单目手语视频中准确估计出时间一致的3D手语表示？（3）如何将检索到的孤立3D手语自然地拼接起来，模拟真实的co-articulation过渡？

## 方法详解

### 整体框架
系统分三个阶段：**字典构建**→**3D手语估计**→**Spoken2Sign翻译**。输入是一段口语文本，首先通过Text2Gloss翻译器（mBART）将文本翻译为gloss序列（手语标注序列），然后从预构建的gloss-3D手语词典中检索每个gloss对应的3D手语，再用sign connector预测相邻手语间co-articulation的时长并在3D空间插值过渡帧，最后用Blender渲染为3D Avatar动画。

### 关键设计
1. **字典构建（CTC强制对齐分割）**：利用训练好的连续手语识别模型TwoStream-SLR，通过CTC强制对齐算法将连续手语视频切割为孤立手语片段，自动构建gloss-video词典。这比使用外部词典更好，因为分割出的孤立手语不包含起始/结束时的无意义动作（如举手放手），更适合后续手语拼接。

2. **SMPLSign-X（手语专用3D估计器）**：在SMPLify-X基础上针对手语特性做了三项改进——（a）**不可见关节正则化 $\mathcal{L}_{unseen}$**：将HRNet检测置信度低（<0.65）的关节推向rest pose，避免对不可见的下半身或遮挡手部产生错误估计；（b）**上身直立约束 $\mathcal{L}_{upright}$**：强制颈部和骨盆等关节保持深度一致性，确保签手上半身直立不倾斜；（c）**时序平滑损失 $\mathcal{L}_{smooth}$**：约束相邻帧的姿态参数差异，解决逐帧独立拟合导致的时序抖动问题。

3. **Sign Connector（手语连接器）**：一个4层MLP，输入前一手语末帧的3D关键点、后一手语首帧的3D关键点以及二者的欧氏距离差，输出co-articulation的持续帧数。训练目标使用L1损失。推理时，按预测帧数在3D关节空间均匀插值生成过渡动画。与固定长度插值相比，动态预测时长更贴合实际co-articulation的变化性。

4. **Sign Retrieval（手语检索）**：每个gloss可能对应多个视频实例，训练一个孤立手语识别模型（NLA-SLR），检索时选择对目标gloss置信度最高的实例，保证选取质量最优的3D手语。

### 损失函数 / 训练策略
- SMPLSign-X总损失：$\mathcal{L} = \mathcal{L}_{joint} + \mathcal{L}_{prior} + \mathcal{L}_{penetration} + \lambda_1\mathcal{L}_{unseen} + \lambda_2\mathcal{L}_{upright} + \lambda_3\mathcal{L}_{smooth}$，默认 $\lambda_1=3e5, \lambda_2=7e5, \lambda_3=1e3$
- 使用L-BFGS优化器多阶段优化，每帧300 epochs拟合SMPL-X参数
- Text2Gloss用mBART，训练80 epochs，lr=1e-5，dropout=0.3，label smoothing=0.2
- Sign Connector用Adam优化器，lr=1e-5，过滤极端长co-articulation

## 实验关键数据

**Spoken2Sign Back-Translation（P-2014T）**：

| 方法 | 指标 | Dev BLEU-4 | Dev ROUGE | Test BLEU-4 | Test ROUGE |
|------|------|------------|-----------|-------------|------------|
| Progressive Transformer | 2D关键点 | 11.82 | 33.18 | 10.51 | 32.46 |
| FS-Net | 2D关键点+GAN | 16.92 | 35.74 | 21.10 | 42.57 |
| SignDiff | 2D关键点+Diffusion | 18.26 | 39.62 | 22.15 | 46.82 |
| **本文** | **3D Avatar** | **24.16** | **49.12** | **25.46** | **49.68** |

**3D手语估计器对比（P-2014T Back-Translation）**：

| 方法 | Dev BLEU-4 | Dev ROUGE | 2D KL↓ | TC↑ |
|------|------------|-----------|--------|-----|
| HRNet（2D pseudo GT） | 22.94 | 48.81 | 0.00 | 0.961 |
| SMPLify-X | 19.21 | 44.28 | 31.56 | 0.945 |
| OSX | 22.31 | 47.71 | 26.87 | 0.969 |
| **SMPLSign-X** | **24.16** | **49.12** | **22.09** | **0.982** |

**聋人用户评分（1-5分）**：

| 数据集 | 方法 | 自然度 | 平滑度 | 相似度 |
|--------|------|--------|--------|--------|
| P-2014T | SMPLify-X | 1.52 | 1.98 | 2.41 |
| P-2014T | **本文** | **3.58** | **4.04** | **3.94** |
| CSL | SMPLify-X | 1.27 | 1.75 | 1.69 |
| CSL | **本文** | **3.78** | **4.14** | **3.78** |

### 消融实验要点
- **三个损失函数贡献**：去掉$\mathcal{L}_{unseen}$掉点最多（Dev BLEU-4从24.16降到22.57），$\mathcal{L}_{upright}$和$\mathcal{L}_{smooth}$各贡献约1个BLEU点
- **Sign Connector关键**：不用connector直接拼接→Dev BLEU-4从24.16降到20.69，掉了3.5个点
- **Sign Retrieval重要**：随机选取替代最优检索→Dev BLEU-4从24.16降到22.25
- **3D空间建模co-articulation优于2D**：默认3D connector的L1预测误差为1.04，固定时长/仅手部/无坐标距离分别为1.83/1.22/1.34
- **3D关键点增强**：在WLASL/MSASL上提升约1.2-1.5%的top-1准确率
- **多视角理解**：正面+侧面双流输入比单正面在P-2014T上提升约1个BLEU点

## 亮点
- **检索-连接范式的简洁性**：不需要端到端生成手语序列，而是将问题分解为"翻译→检索→拼接→渲染"，每一步都可以独立优化和替换，工程友好
- **SMPLSign-X的手语专用先验**：三个简单的正则化损失（unseen/upright/smooth）就让估计质量大幅超越通用3D估计器，说明领域先验的重要性
- **Sign Connector动态预测时长**：用4层MLP预测co-articulation持续帧数，比固定插值效果好很多，且几乎无额外开销
- **副产品思路有启发性**：3D表示天然支持旋转增强和多视角输入，轻松为手语理解任务提供免费的数据增强

## 局限性 / 可改进方向
- **数据稀缺性**：text-gloss对训练数据不足可能导致Text2Gloss翻译质量受限，大规模手语数据集的构建是瓶颈
- **2D关键点伪标签质量**：3D估计依赖HRNet的2D关键点，若检测不准则3D结果也会退化；手语专用的2D检测器可能更好
- **深度歧义未完全解决**：单目到3D的深度估计仍是开放问题，当前方法主要靠先验约束缓解
- **co-articulation建模粗糙**：简单的线性插值不能完全捕捉真实co-articulation的非线性动态，可考虑学习型过渡生成
- **表情和非手工信号不足**：手语中面部表情承载重要语义信息，当前系统对此建模有限
- **仅支持已见gloss**：基于检索的方式无法处理词典外的新gloss，泛化性受限

## 与相关工作的对比
- **vs FS-Net**：FS-Net使用外部词典+固定长度2D关键点插值+GAN生成2D视频。本文用CTC分割构建更干净的词典+动态3D空间插值+Avatar渲染，避免了生成模型的失真，back-translation大幅领先（24.16 vs 16.92 Dev BLEU-4）
- **vs SignDiff**：SignDiff引入扩散模型生成2D手语视频，效果优于FS-Net但仍局限在2D。本文在同一back translator下仍超过SignDiff 4个BLEU点以上，且输出3D格式更灵活
- **vs SMPLify-X/SMPLer-X/OSX**：通用3D人体估计器未考虑手语特性（签手直立、下半身不动、时序连贯性），本文针对性加入三个约束后在back-translation和视觉质量上均显著优于它们

## 启发与关联
- 检索-连接的模块化范式适用于其他需要生成长序列动作的任务（如舞蹈生成、手势合成），每个动作原子可以独立估计后拼接
- 领域专用先验可以自然地融入通用参数化模型（如SMPL-X），不需要改架构，只需添加正则化损失
- 3D表示的旋转增强思路可以迁移到其他基于2D关键点的人体理解任务

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次将3D Avatar引入Spoken2Sign，但整体流程是已有技术的组合
- 实验充分度: ⭐⭐⭐⭐⭐ back-translation、3D估计对比、消融、用户研究、副产品验证，非常全面
- 写作质量: ⭐⭐⭐⭐⭐ 概念清晰、流程图直观、背景知识铺垫到位，论文结构非常好
- 价值: ⭐⭐⭐⭐ 建立了Spoken2Sign 3D基线并释放词典数据，对后续研究有较好推动作用
