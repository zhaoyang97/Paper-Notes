# SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion

**会议**: CVPR 2026  
**arXiv**: [2603.12764](https://arxiv.org/abs/2603.12764)  
**代码**: [github.com/jack1ee/SAVAX](https://github.com/jack1ee/SAVAX)  
**领域**: 视频理解 / 跨视角错误检测  
**关键词**: Ego-Exo, 模仿错误检测, 跨视角融合, 自适应采样, 视角嵌入  

## 一句话总结
提出Align-Fuse-Detect框架SAVA-X，通过Gumbel Top-K自适应采样去冗余、场景自适应视角嵌入缩小域差距、双向交叉注意力融合互补语义，在EgoMe数据集上Mean AUPRC达22.36，超越最强baseline +13.56%。

## 背景与动机
工业装配、医疗训练等场景需检测操作者的模仿错误——操作者看第三人称(exo)示范视频进行第一人称(ego)模仿。现有方法主要假设单视角输入，无法处理跨视角的域差异(ego关注手-物交互/exo关注全身布局)、时间不对齐(节奏差异)和严重冗余(大量无信息帧)的耦合挑战。

## 核心问题
给定异步录制的exo示范和ego模仿视频，在ego时间线上定位操作步骤并判断每步是否为错误模仿。

## 方法详解

### 整体框架
冻结TSP编码器提取逐帧特征 → 三阶段：(1)Gated Adaptive Sampling选关键帧 → (2)Scene-Adaptive View Embeddings注入视角条件 → (3)Bidirectional Cross-Attention Fusion双向融合 → Deformable Transformer预测动作段+错误分类。

### 关键设计
1. **Gated Adaptive Sampling (AS)**: Gumbel Top-K离散帧选择+残差门控(soft indices缩放原始特征)提供可微梯度路径。Exo端用自注意力评分选关键帧；Ego端用以exo摘要为K/V的交叉注意力评分。附加选择熵正则+VICReg正则防止坍塌。

2. **Scene-Adaptive View Embeddings (SVE)**: 共享视角-场景字典$D \in \mathbb{R}^{M \times d}$，通过带温度τ的交叉注意力根据当前帧自适应查询字典生成视角嵌入。比固定可学习视角token更灵活——可跨场景自适应。用注意力熵正则+字典多样性正则确保覆盖广泛。

3. **Bidirectional Cross-Attention Fusion (BiX)**: 对称双向——Ego查询Exo获取全局线索，Exo查询Ego获取手-物细节。可学习sigmoid门控残差混合。双向优于单向(21.06 vs 20.73/19.48)。

### 损失函数 / 训练策略
匈牙利匹配集合预测损失(时间GIoU+focal分类+事件计数)+模仿错误BCE损失(query-level+video-level)+多项正则。AdamW, lr=1e-4, batch 16。

## 实验关键数据
| 数据集 | 指标 | SAVA-X | Exo2EgoDVC | PDVC | 提升 |
|--------|------|--------|-----------|------|------|
| EgoMe Val | Mean AUPRC | **22.36** | 19.69 | 18.88 | +13.56% |
| EgoMe Val | AUPRC@0.5 | **24.04** | - | 20.48 | +17.4% |
| EgoMe Test | Mean AUPRC | **18.50** | 15.99 | 16.20 | +14.2% |

ActionFormer等TAL方法表现更差(16.47)。仅用ego输入PDVC降至12.79，说明exo信息至关重要。

### 消融实验要点
- AS单独贡献+10.70%，SVE+12.76%（最大），BiX+11.55%——三者互补
- SVE+BiX是最强两两组合(22.33)，接近完整模型(22.36)
- 双向融合(21.06)优于Exo→Ego单向(20.73)和Ego→Exo单向(19.48)
- SVE优于固定视角嵌入，且对字典大小M稳健

## 亮点
- 首次形式化Ego→Exo模仿错误检测任务，建立统一评估协议
- 模块设计与挑战一一对应：AS→冗余、SVE→域差距、BiX→跨视角融合
- Gumbel Top-K + 残差门控的采样策略可迁移到其他帧选择任务

## 局限性 / 可改进方向
- 仅在EgoMe一个数据集上验证
- 使用冻结TSP backbone，未探索更强视频基础模型
- 假设exo示范本身正确，未处理步骤顺序错误

## 与相关工作的对比
- **Exo2EgoDVC**: 用视角不变对抗学习做exo→ego迁移，Mean AUPRC 19.69 vs 本文22.36
- **PDVC**: 经典端到端DVC+简单拼接融合，18.88 vs 22.36

## 启发与关联
- 跨视角自适应采样和场景自适应嵌入思路可迁移到多视角视频问答/检索

## 评分
- 新颖性: ⭐⭐⭐⭐ 任务新颖，模块设计合理
- 实验充分度: ⭐⭐⭐⭐ 消融详尽但仅单数据集
- 写作质量: ⭐⭐⭐⭐ 问题-方法-验证对应清晰
- 价值: ⭐⭐⭐⭐ 对跨视角视频理解有参考价值
