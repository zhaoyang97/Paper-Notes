# VIR-Bench: Evaluating Geospatial and Temporal Understanding of MLLMs via Travel Video Itinerary Reconstruction

**会议**: AAAI 2026  
**arXiv**: [2509.19002](https://arxiv.org/abs/2509.19002)  
**代码**: https://github.com/nlp-waseda/VIR-Bench  
**领域**: 视频理解  
**关键词**: 地理空间理解, 时间推理, 旅行视频, 行程重建, MLLM评估

## 一句话总结
提出VIR-Bench——一个基于200个日本旅行vlog视频的benchmark，通过行程重建任务（visiting order graph构建）评估MLLM的地理空间和时间理解能力，发现SOTA模型（包括GPT-4.1和Gemini-2.5）在POI识别和时间转移推理上仍困难重重。

## 研究背景与动机
1. **领域现状**：MLLM的视频理解能力提升迅速，但现有benchmark（Ego4D、HourVideo、VSI-Bench等）主要关注室内场景或短距离户外活动，缺乏对长距离旅行（跨城市/跨地区）的地理空间-时间理解评估。
2. **现有痛点**：长距离地理空间-时间推理对embodied AI规划、导航等应用至关重要，但无相关benchmark。
3. **核心矛盾**：现有benchmark的空间尺度太小（室内/短距离），无法评估模型在宏观地理（城市间导航）和长时间跨度（多日行程）上的理解能力。
4. **本文要解决什么？** 构建一个评估MLLM长距离地理空间-时间理解的视频benchmark。
5. **切入角度**：用旅行vlog视频的行程重建（itinerary reconstruction）作为测试任务——模型需要从视频中识别所有访问的地点及其层次关系和时间顺序。
6. **核心idea一句话**：从旅行视频中重建visiting order graph（节点=地点，边=包含/转移关系）来评估MLLM的地理空间和时间智能。

## 方法详解

### 整体框架
VIR-Bench定义了一个层次化的visiting order graph：
- **节点类型**：Root → Prefecture（都道府县）→ City（市区町村）→ POI（景点/车站/餐厅等）
- **边类型**：Inclusion（包含关系，如东京→新宿）和Transition（时间转移，如浅草寺→东京塔）
- **两个子任务**：Node Prediction（识别所有访问地点）和Edge Prediction（推断包含和转移关系）

### 关键设计

1. **数据集构建**:
   - 200个YouTube旅行vlog（100英文+100日文），覆盖日本47都道府县中的43个
   - 10名日本本地标注员，每人收集并标注20个视频
   - 每个POI标注时间戳+Google Maps URL，通过Google Places API获取详细信息
   - 总计3,689个POI，手工构建visiting order graph + 质量检查

2. **Node Prediction任务**:
   - 给定视频，模型需输出所有访问的prefecture/city/POI的JSON列表
   - 评估地理空间理解能力（类似GeoGuessr游戏）
   - 用macro-averaged Precision/Recall/F1评估

3. **Edge Prediction任务**:
   - 给定视频+所有访问地点（gold labels，打乱顺序），预测inclusion和transition边
   - Inclusion评估地理知识（A属于B），Transition评估时间理解（先A后B）

### 损失函数 / 训练策略
纯评估benchmark，无训练。Zero-shot评估主流MLLM。

## 实验关键数据

### 主实验
| 模型 | Node-Prefecture F1 | Node-City F1 | Node-POI F1 | Edge-Inclusion F1 | Edge-Transition F1 |
|------|----|----|----|----|----|
| VideoLLaMA3 | 低 | 低 | 极低 | 低 | 极低 |
| InternVL3 | 中 | 中 | 低 | 中 | 低 |
| Qwen2.5-VL | 中 | 中 | 低 | 中 | 低 |
| GPT-4.1 | 较高 | 中高 | 中 | 中高 | 中 |
| Gemini-2.5-Pro | 最高 | 最高 | 中 | 最高 | 中 |

### 消融实验
| 配置 | 效果 | 说明 |
|------|------|------|
| 更多视频帧 | 稳步提升 | 更多视觉context有帮助 |
| 更多推理effort | 提升 | o4-mini > GPT-4.1 |
| 音频输入 | 提升 | Gemini可用音频，提供额外线索 |
| 开源 vs 闭源 | 闭源大幅领先 | 开源模型地理知识不足 |

### 关键发现
- **POI node prediction是最大瓶颈**：所有模型在POI粒度上表现远不如prefecture/city——从视频画面推断具体景点名称非常困难
- **Transition edge prediction（时间顺序推理）很难**：需要理解整个视频的时间线，当前模型能力不足
- **开源模型缺乏地理知识**：InternVL3等模型在prefecture识别上就挣扎，说明地理知识的预训练不足
- **Travel-planning agent案例验证了benchmark价值**：结合视频+行程的agent生成的旅行计划最可行且吸引人

## 亮点与洞察
- **任务设计精巧**：visiting order graph同时评估地理知识（node + inclusion）和时间推理（transition），一个任务覆盖两个维度
- **宏观尺度的补位**：填补了micro（室内/短距离）到macro（跨城市/跨地区）的benchmark空白
- **从评估到应用的闭环**：不只是评测，还开发了travel-planning agent验证benchmark的实用价值

## 局限性 / 可改进方向
- **地理覆盖有限**：只包含日本，其他国家/地区的旅行视频可能有不同特征
- **POI标注依赖视频内容质量**：有些vlog拍摄角度有限，导致POI难以标注
- **End-to-end generation太难**：实际把node和edge prediction分开了，失去了端到端的挑战性

## 相关工作与启发
- **vs VSI-Bench**: VSI-Bench评估3D场景理解（室内），VIR-Bench评估跨城市地理理解（宏观），尺度完全不同
- **vs CityGuessr**: CityGuessr做地理定位但基于驾驶视频和静态定位，VIR-Bench涉及时间序列和行程重建
- **对embodied AI的启示**：未来的embodied AI需要在城市/区域尺度做规划，VIR-Bench评估的正是这种能力

## 评分
- 新颖性: ⭐⭐⭐⭐ 首个评估宏观地理空间-时间理解的视频benchmark
- 实验充分度: ⭐⭐⭐⭐ 8种模型、消融分析、agent案例验证
- 写作质量: ⭐⭐⭐⭐ 任务定义清晰，数据集构建规范
- 价值: ⭐⭐⭐⭐ 填补了macro-scale视频理解评估的空白
