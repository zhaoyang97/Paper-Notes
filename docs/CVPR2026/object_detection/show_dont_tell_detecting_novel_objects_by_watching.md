# Show, Don't Tell: Detecting Novel Objects by Watching Human Videos

**会议**: CVPR 2026  
**arXiv**: [2603.12751](https://arxiv.org/abs/2603.12751)  
**代码**: 无  
**领域**: 目标检测 / 机器人视觉  
**关键词**: novel object detection, self-supervised, robot learning, human demonstration, object manipulation  

## 一句话总结
提出"Show, Don't Tell"范式：通过观看人类演示视频，自动构建新物体标注数据集（SODC），训练轻量级定制检测器（MOD），完全绕过语言描述和prompt engineering，在真实机器人分拣任务上成功部署。

## 背景与动机
机器人在执行任务时需要识别人类演示中出现的新物体。闭集检测器（YOLO、DETR等）对分布外物体失效；开集VLM（GroundingDINO、YoloWorld、RexOmni）虽然能力更广泛，但对于难以用语言精确描述的新物体（如STEM组装零件），需要繁琐的人工prompt工程来区分视觉相似的不同实例。核心矛盾在于：语言是模糊的，但视觉是精确的——用文字描述"一个红白色的建筑玩具"可能匹配多个不同零件。

## 核心问题
如何让机器人仅通过观看一次人类演示视频，就能快速学会识别演示中出现的所有新交互物体，且无需语言描述或人工标注？

## 方法详解

### 整体框架
三步Pipeline：(1) SODC自动从演示视频中创建带标注的新物体数据集，(2) MOD在该数据集上训练轻量级检测器，(3) 集成到机器人系统实现从观察到执行的闭环。

### 关键设计
1. **SODC (Salient Objects Dataset Creation)**：自动从人类演示视频中创建标注数据集，分三步：
   - **检测抓取实体**：用HOIST-Former检测每帧中人手抓取的物体，输出分割mask
   - **跨帧跟踪**：以HOIST-Former输出的mask为种子，用SAMURAI在整个视频中前后追踪物体（包括未被操作的帧），得到大量bounding box轨迹
   - **时空聚类合并**：先用DBSCAN在每帧内按IoU距离做空间聚类（将重叠的box归为同一物体），再按时间维度将经过相同空间聚类序列的轨迹合并为同一物体的统一轨迹，丢弃噪声短轨迹。最终输出带标签的图像+bounding box数据集

2. **MOD (Manipulated Objects Detector)**：在SODC生成的数据集上微调预训练F-RCNN (ResNet50)，标准RCNN Loss（分类+objectness），训练3-4分钟（4×T4 GPU），配合随机翻转、畸变、亮度、对比度等数据增强。核心优势：针对演示中具体物体的定制检测器，擅长实例级识别和消歧。

3. **机器人端到端部署**：用ChatGPT-4o从视频生成plan skeleton（[Pick(MOD_ID0), Place("basket")]格式），MOD负责检测pick物体，VLM识别place目标（如"绿色托盘"）。机器人先定位所有新物体，再按plan顺序执行pick-and-place。

### 训练策略
- 训练数据完全来自单次人类演示视频，无需人工标注
- 演示者在放置物体时旋转两次以覆盖更多视角
- 整体pipeline（视频处理+训练+部署）在分钟级完成

## 实验关键数据

| 数据集 | 方法 | Prompt | mAP₀.₅₋₀.₉₅ | mAR₁ | F1₀.₅₋₀.₉₅ |
|---|---|---|---|---|---|
| In-House #1 | RexOmni | GPT | 0.06 | 0.09 | 0.98 |
| In-House #1 | GroundingDINO | Human | 0.04 | 0.08 | 0.87 |
| In-House #1 | **MOD (Ours)** | – | **0.10** | **0.17** | 0.92 |
| In-House #2 | RexOmni | GPT | 0.09 | 0.12 | 0.99 |
| In-House #2 | GroundingDINO | GPT | 0.08 | 0.10 | 0.98 |
| In-House #2 | **MOD (Ours)** | – | **0.15** | **0.19** | 0.95 |
| Meccano | GroundingDINO | Human | 0.19 | 0.26 | 0.24 |
| Meccano | **MOD (Ours)** | – | 0.06 | 0.10 | 0.18 |

- MOD在自有数据集上mAP/mAR显著超越所有VLM基线（含人工精心设计的prompt）
- Meccano数据集上MOD总体低于GroundingDINO+Human Prompt，作者归因于Meccano的物体更"可描述"（标准组装零件）
- VLM基线的关键问题：能做类别级检测但在实例消歧上失败（如把不同的红白色零件混淆）

### 消融实验要点
- 人工prompt engineering虽然效果更好但需要大量人力，GPT自动prompt的效果不稳定
- MOD训练只需3-4分钟，推理速度远快于大型VLM
- 机器人实机部署中MOD的missed detection和物体混淆率明显低于VLM方案

## 亮点
- 范式创新：用视觉展示替代文字描述来教检测器，这个理念既简洁又直觉——"a demonstration can be worth a thousand words"
- SODC的时空聚类trick很巧妙：先空间DBSCAN再时间轨迹一致性匹配，无监督地解决了多轨迹→多物体的对应问题
- 整个pipeline从演示到部署只需几分钟，实际可用性极强
- 在"难以用语言描述"的新物体上优势尤其明显

## 局限性 / 可改进方向
- 需要演示者刻意旋转物体以覆盖视角——不够自然
- MOD是实例级检测器而非开放词汇检测器，无法泛化到未被演示的新物体
- Meccano数据集上表现不如VLM+Human Prompt，说明当物体可被语言清晰描述时，此方法没有优势
- 依赖HOIST-Former和SAMURAI的质量——如果手物交互检测/跟踪失败，后续全链路受影响
- 数据集规模较小（12-17个新物体），大规模复杂场景的验证不足

## 与相关工作的对比
- **vs VLM检测器 (GroundingDINO, YoloWorld, RexOmni)**：VLM依赖语言prompt，对"难以描述"的新物体实例消歧能力弱；MOD直接从视觉训练，擅长实例级识别
- **vs 行为克隆/端到端模仿学习 (π-0.5, InstantPolicy)**：这些方法将检测折叠进端到端模型，但需要机器人训练数据；MOD只需人类演示视频
- **vs 闭集检测器微调**：MOD同样是微调F-RCNN，但数据集创建完全自动化，无需人工标注

## 启发与关联
- "Show, Don't Tell"思想可扩展到实例分割、动作参数估计等任务
- SODC pipeline可作为通用的零标注数据集创建工具，服务于其他学习方法

## 评分
- 新颖性: ⭐⭐⭐⭐ 范式新颖（视觉替代语言），但具体组件（HOI检测+跟踪+聚类+微调F-RCNN）都是现有方法的组合
- 实验充分度: ⭐⭐⭐ 三个数据集但规模较小，物体类别有限，缺少大规模验证
- 写作质量: ⭐⭐⭐⭐ 故事讲得清楚，pipeline图示直观，但实验分析可以更深入
- 价值: ⭐⭐⭐⭐⭐ 直接解决机器人部署中的新物体识别痛点，分钟级从演示到部署，实用性极强
