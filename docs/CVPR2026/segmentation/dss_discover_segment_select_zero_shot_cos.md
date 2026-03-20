# DSS: Discover, Segment, and Select - A Progressive Mechanism for Zero-shot Camouflaged Object Segmentation

**会议**: CVPR 2026  
**arXiv**: [2602.19944](https://arxiv.org/abs/2602.19944)  
**代码**: 无  
**领域**: 语义分割 / 伪装目标检测 / 零样本  
**关键词**: 伪装目标分割, 零样本, DINOv2聚类, MLLM选择, SAM, 无训练  

## 一句话总结
提出三阶段零样本伪装目标分割框架DSS：先用DINOv2特征聚类+部件组合发现候选区域（Discover），再用SAM分割（Segment），最后用MLLM逐对比较选最优mask（Select），无需任何训练即在四个COD基准上全面超越先前零样本方法，尤其在多实例场景中优势显著。

## 背景与动机
零样本伪装目标分割(COS)的主流方法是"发现-分割"两阶段：先用MLLM定位伪装目标生成bbox/point prompt，再用SAM分割。但MLLM在伪装场景下常产生三种失败：(1) 定位不准（高级语义vs低对比度前景之间的gap）；(2) 漏检（多实例场景中只发现最显著的一个）；(3) 误检（把背景误判为伪装目标）。核心原因是**仅依赖MLLM的语言先验做发现**，忽视了图像内在的视觉判别特征。

## 核心问题
如何在零样本、无训练的设置下，结合视觉特征先验和语言推理能力，实现鲁棒的伪装目标发现和高质量mask选择？

## 方法详解

### 整体框架
三阶段：(1) **Discover (FOD)**：用DINOv2提取patch级特征→Leiden聚类→部件组合(PC)迭代精炼每个聚类→相似性驱动的bbox生成(SBG)→多个异质bbox prompt；同时也用QWen2.5-VL做VLOS获取MLLM bbox作为补充。(2) **Segment**：所有bbox送入SAM2(ViT-L)生成候选mask集合。(3) **Select (SMS)**：启发式评分（相关性+边界接触率）筛选top-K→MLLM逐对升序比较→最终选mask。

### 关键设计
1. **部件组合(PC)**：聚类可能将一个伪装物体过分割为多个部件。PC通过迭代软分配解决：每个patch的前景概率$y_i^{(t)} = \sigma(\|x_i - \mu_b\|^2 - \|x_i - \mu_f\|^2)$基于到前景/背景中心的相对距离更新，直到特征一致性能量收敛（类内紧凑+类间分离）。5次迭代内通常收敛。

2. **相似性驱动bbox生成(SBG)**：不是直接从二值mask提bbox（容易不完整），而是用每个聚类的前景中心$\mu_f$与所有patch计算余弦相似度生成连续的相似性图$\text{sim}_c$，然后阈值化+连通域提取bbox。去重通过Pearson相关系数（$\tau=0.95$）合并高度相关的相似性图。SBG比直接提bbox在多实例场景下完整性大幅提升。

3. **语义驱动mask选择(SMS)**：面对多个候选mask，先用启发式评分$s_i = \text{corr}(m_i, \text{sim}_i) + (1 - \text{BC}(m_i))$（mask与相似性图的相关性 + 低边界接触率）筛选top-K=5。关键设计是pairwise升序比较——从低分mask开始两两对比，让MLLM判断"哪个更像伪装目标"，赢家进入下一轮。消融显示升序(74.4%)远优于降序(50.8%)和单次全选(OOM)。

### 损失函数 / 训练策略
完全training-free，无需任何训练。整个流程是推理时的pipeline。QWen2.5-VL-Instruct作为MLLM，SAM2(ViT-L)做分割，DINOv2做特征提取→PCA降至16维→Leiden聚类（自动确定聚类数无需指定K）。单卡RTX 3090 24GB可运行。

## 实验关键数据
| 方法 | 设置 | CAMO $S_\alpha$↑ | NC4K $S_\alpha$↑ | COD10K $S_\alpha$↑ | CHAMELEON $S_\alpha$↑ |
|--------|------|------|----------|------|------|
| ProMaC (NeurIPS24) | ZS | .725 | .777 | .716 | .790 |
| RDVP-MSD (MM25) | ZS | .785 | .795 | .775 | .814 |
| IAPF | ZS | .768 | .828 | .799 | — |
| QWen+SAM2 | ZS | .741 | .846 | .827 | .785 |
| **DSS** | **ZS** | **.766** | **.870** | **.849** | **.848** |

关键优势在**多实例场景**：3+实例时DSS性能下降最小（NC4K上3+实例比单实例仅差约5%，而其他方法差15-25%）。

效率比较：DSS 42s/image（FOD 7.7s + SAM 3.6s + SMS 30.6s）vs ProMaC 130.5s，GPU仅需17.9GB vs ProMaC 32.9GB。

### 消融实验要点
- **FOD+VLOS互补**：单用聚类(Leiden/PC)在COD10K上0.82，单用VLOS 0.85，联合0.89——视觉发现和语言发现互补
- **SBG的多实例优势**：3+实例场景中sim-map-based bbox比直接mask-based bbox在NC4K和COD10K上全面领先
- **SMS选择有效但有提升空间**：DSS的SMS(0.87 score) vs Ideal Seg(0.89) vs VLOS baseline(0.85)，说明mask选择有效但距离上界还有差距
- **pairwise升序最优**：升序pairwise(74.4%) >> 降序(50.8%) >> 随机(54.0%) >> 单次全选(OOM)
- **Leiden优于K-means**：自动确定聚类数避免了K-means需要指定K的问题

## 亮点
- "发现-分割-选择"三阶段范式比"发现-分割"更鲁棒——选择阶段容错能力强，利用MLLM做推理而非定位（MLLM更擅长判断而非发现）
- FOD的视觉特征聚类与MLLM的语言定位**正交互补**——视觉先验在低对比度场景下比语言先验更可靠
- pairwise升序比较的MLLM选择策略是一个实用的trick——将N选1分解为多次二选一更稳定
- Leiden聚类+PCA降维的组合巧妙——自动确定聚类数且计算效率高，整个FOD仅7.7s
- 在多实例场景的优势显著——这正是现有方法的最大痛点

## 局限性 / 可改进方向
- SMS模块耗时占72.9%（每图30.6s），主要因为多次MLLM推理调用，速度瓶颈在选择阶段
- Ideal Seg消融显示mask质量距离理想还有差距（COD10K：0.892 vs 0.871），更好的选择策略或端到端优化可能帮助
- 微小伪装目标检测未专门处理——结论中提到未来需要多尺度特征聚合
- 对DINOv2特征质量有依赖——在DINOv2表征不好的domain可能效果下降
- pairwise比较中MLLM的判断并非完全可靠，但已经是现有最优方案

## 与相关工作的对比
- **vs GenSAM (AAAI24)**：GenSAM用BLIP2生成类别+CLIP attention做prompt，DSS用DINOv2特征聚类做发现，在CAMO上.766 vs .659
- **vs ProMaC (NeurIPS24)**：ProMaC利用MLLM幻觉先验降低文本prompt不准确性，但仍依赖MLLM做发现。DSS将MLLM职责从"发现"转为"选择"
- **vs RDVP-MSD (MM25)**：RDVP用步进分解缓解语义歧义，但在多实例场景下仍受限。DSS的FOD模块天然适合多实例（聚类可以发现多个区域）
- **vs IAPF**：IAPF用Grounding DINO做实例级bbox+单前景多背景点prompt，DSS用无监督聚类做发现，不依赖有监督定位器

## 启发与关联
- 与`ideas/segmentation/20260316_unified_freq_prompt_sam.md`相关——DSS展示了用DINOv2特征做SAM prompt的新范式，可以考虑频域特征+聚类的组合
- "发现-分割-选择"范式可推广到其他视觉理解任务——如开放世界检测中的候选区域生成+筛选
- MLLM作为**评判者而非发现者**的角色转变是关键洞察——让模型做它最擅长的事

## 评分
- 新颖性: ⭐⭐⭐⭐ 三阶段范式和FOD的视觉聚类发现是新的视角，但各组件不算全新
- 实验充分度: ⭐⭐⭐⭐⭐ 4个基准、与fully-supervised/unsupervised/zero-shot全面对比、多实例分析、详细消融
- 写作质量: ⭐⭐⭐⭐ 方法描述清晰，但符号和模块多，阅读门槛略高
- 价值: ⭐⭐⭐⭐ 零样本伪装分割的新SOTA，且在多实例场景优势最为突出
