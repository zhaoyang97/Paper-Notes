<!-- 由 src/gen_stubs.py 自动生成 -->
# CAPT: Confusion-Aware Prompt Tuning for Reducing Vision-Language Misalignment

**会议**: CVPR2026  
**arXiv**: [2603.02557](https://arxiv.org/abs/2603.02557)  
**代码**: [github.com/greatest-gourmet/CAPT](https://github.com/greatest-gourmet/CAPT)  
**领域**: 多模态VLM  
**关键词**: prompt tuning, 视觉语言对齐, 混淆感知, CLIP, few-shot, 细粒度分类

## 一句话总结
提出 CAPT 混淆感知 prompt tuning 框架，通过语义混淆挖掘器（SEM）和样本混淆挖掘器（SAM）显式建模 VLM 的系统性误对齐模式，配合多粒度差异专家（MGDE）融合不同层次的混淆信息，在 11 个基准上取得 HM 83.90% 的最优表现。

## 研究背景与动机
1. CLIP 等视觉语言模型存在系统性误对齐：特定类别对之间的混淆不是随机的，而是持续发生的固定模式
2. 例如 OxfordPets 数据集中，terrier 被错分为 bulldog 30 次，而几乎不被误认为其他类
3. 现有 prompt tuning 方法（MaPLe、PromptSRC）优化全局图像-文本特征对齐，忽略了这种固定混淆模式
4. 模型误对齐源于视觉和语义上高度相似类别之间的模糊语义边界和局部表示相似性
5. 需要让模型从自身的误对齐中学习——显式建模混淆关系并修正
6. 现有方法未从混淆样本中挖掘可区分的细粒度线索

## 方法详解

### 整体框架
CAPT 在 CLIP prompt tuning 基础上，引入：(1) 混淆银行记录误分类样本；(2) SEM 挖掘语义层面混淆模式；(3) SAM 挖掘样本层面混淆线索；(4) MGDE 融合多粒度信息。

### 关键设计

**混淆银行（Confusion Bank）**：记录每个样本在其被误分类到的类别下，形成类间混淆关系索引。用 pseudo-GT（模型最高置信度预测类）代替标注 GT 更好反映模型真实混淆倾向。

**SEM（Semantic Confusion Miner）**：
- 用混淆统计 $n_i$ 和当前样本置信度 $C_i$ 计算混淆分数：$S_i = (1 + \frac{n_i}{\sum n_i}) C_i$
- 选 top-k 生成语义混淆对
- 用 LLM（CoT 风格）为混淆对生成共性（commonality）和差异（difference）prompt

**SAM（Sample Confusion Miner）**：
- 从混淆银行检索混淆样本集 $U \in \mathbb{R}^{c \times l}$
- 选择与当前实例特征最相似的代表性混淆样本：$I_c^* = \arg\max \cos(E_I(I), E_I(U_j^i))$
- Diff-Manner Adapter：融合 ViT 全局注意力和 2D 深度可分离卷积局部细节
  $$[X] \leftarrow [X] + \alpha \cdot DWConv2D(\hat{[X]})$$

**MGDE（Multi-Granularity Discrepancy Expert）**：
- MoE 架构，包含语义专家（由文本差异/共性 prompt 初始化）和样本专家（由 CLIP FFN 初始化）
- K-means 聚类压缩 prompt token，去除低区分度 token
- 路由网络自适应决定各专家的输出权重

### 损失函数
$$\mathcal{L} = \mathcal{L}_{ori} + \mathcal{L}_{confuse}$$
- $\mathcal{L}_{ori}$：标准交叉熵对齐损失
- $\mathcal{L}_{confuse}$：InfoNCE 风格对比损失，作用于混淆样本特征和prompt

## 实验关键数据

### 主实验：Base-to-New 泛化（16-shot）

| 方法 | Base | Novel | HM |
|------|------|-------|-----|
| CoOp (IJCV'22) | 82.69 | 63.22 | 71.66 |
| MaPLe (CVPR'23) | 82.28 | 75.14 | 78.55 |
| PromptKD (CVPR'24) | 86.96 | 80.73 | 83.73 |
| TAC (CVPR'25) | 85.42 | 77.60 | 81.24 |
| 2SFS (CVPR'25) | 85.55 | 75.48 | 80.20 |
| **CAPT (本文)** | **87.41** | **80.90** | **83.90** |

### 混淆样本修复率

| 指标 | 数值 |
|------|------|
| 混淆样本对修复率 | 50.72% |
| Base 类准确率 | 87.41% |
| Novel 类准确率 | 80.90% |

### 关键发现
- CAPT 在 HM 上超越所有先前方法，Base 和 Novel 类均有显著提升
- 50.72% 的混淆样本对被成功修复，验证了混淆感知学习的有效性
- SEM 和 SAM 各自贡献互补——语义层面捕获全局混淆模式，样本层面捕获细粒度差异

## 亮点与洞察
- 独特视角：从模型自身误对齐中挖掘改进信号，变"bug"为"feature"
- 混淆银行的设计简洁有效，为后续研究提供了可复用的工具
- Diff-Manner Adapter 融合 ViT 全局性和 CNN 局部性的思路有普适价值

## 局限性
- pseudo-GT 的质量依赖模型初始预测能力，弱模型可能产生噪声过大的混淆银行
- LLM 生成语义 prompt 的质量和一致性未充分分析
- 模型复杂度增加（SEM+SAM+MGDE），训练成本可能显著高于简单 prompt tuning

## 相关工作与启发
- 与 PromptSRC/MaPLe 的本质区别：关注的不是如何更好地对齐，而是如何从误对齐中学习
- 混淆银行思想可迁移到对比学习中的难负例挖掘
- 启发：模型的系统性错误模式本身是有价值的信号，值得在更多任务中被利用

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (混淆感知的视角非常新颖，从误对齐中学习)
- 实验充分度: ⭐⭐⭐⭐ (11 个数据集，充分的消融和可视化)
- 写作质量: ⭐⭐⭐⭐ (框架图清晰，方法描述系统)
- 价值: ⭐⭐⭐⭐ (prompt tuning 的新方向，混淆修复率指标有启发性)
