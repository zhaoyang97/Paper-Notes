# AdaVideoRAG: Omni-Contextual Adaptive Retrieval-Augmented Efficient Long Video Understanding

**会议**: NeurIPS 2025  
**arXiv**: [2506.13589](https://arxiv.org/abs/2506.13589)  
**代码**: [https://github.com/xzc-zju/AdaVideoRAG](https://github.com/xzc-zju/AdaVideoRAG)  
**领域**: 多模态VLM / 视频理解  
**关键词**: long video understanding, retrieval-augmented generation, adaptive retrieval, knowledge graph, intent classification  

## 一句话总结
提出 AdaVideoRAG，通过轻量级意图分类器将查询按难度路由到三级检索路径（无检索/朴素检索/图检索），结合全知识索引模块（caption+ASR+OCR+视觉+知识图谱）实现长视频理解的效率-精度最优平衡，在 MLVU 上为 Qwen2.5-VL-7B 带来 39.8% 提升。

## 背景与动机
MLLM 在长视频理解中面临三个问题：(1) 固定上下文窗口导致长视频信息丢失，(2) 知识固化无法动态更新，(3) 多跳推理能力不足。现有 VideoRAG 方案存在固定检索范式缺陷：
- 朴素检索（VideoRAG [Luo]）：caption+ASR+OCR 向量检索，无法处理需要全局理解的多跳问题
- 图检索（VideoRAG [Ren]）：构建层次知识图谱，精度高但计算开销大（复杂图遍历），对简单问题造成不必要延迟

关键洞察：不同难度的问题应该用不同复杂度的检索策略。

## 核心问题
如何**自适应地**为不同复杂度的视频理解查询分配合适的检索策略，在简单问题上节省计算、在困难问题上保证深度推理？

## 方法详解

### 整体框架
四阶段流程：(1) 查询意图分类 → (2) 全知识索引构建 → (3) 自适应检索 → (4) 多模态信息整合与生成。系统作为即插即用的 API 与现有 MLLM 集成。

### 关键设计
1. **查询意图分类器**: 用轻量 LLM（Qwen2.5-7B + CoT）将查询分为三级：
   - L1（直接事实）：如"第5秒出现什么物体"→ 直接送 MLLM，无需检索
   - L2（简单推理）：如"为什么下雨前女人哭了"→ 朴素向量检索（caption/ASR/OCR + 视觉检索）
   - L3（复杂推理）：如"这部电影传达了什么人生道理"→ 图检索 + 多跳推理
   
   分类器耗时仅占总推理的 ≤5%。

2. **全知识索引模块 Omni-Knowledge Indexing**: 从视频中提取多模态信息构建四个知识库：
   - **Caption 库**：每 30s 采样 5 帧，用 MiniCPM-V 生成细粒度描述
   - **ASR 库**：FastWhisper 提取语音转文本
   - **OCR 库**：EasyOCR 提取场景文字
   - **视觉库**：ImageBind 提取帧级视觉特征，映射到统一语义空间
   - **知识图谱**：从文本 chunk 中用 BGE-M3 提取实体和关系（时空/因果/功能关系）

3. **自适应检索范式**:
   - L1：直接 MLLM 推理
   - L2：查询改写（针对 caption/ASR/OCR 分别改写）→ 向量检索 + 视觉 grounding → 过滤排序
   - L3：在 L2 基础上加入 LightRAG 图检索，提取实体关系和关联信息，构建以查询为中心的思维图谱
   
4. **证据过滤与排序**: 去重 → 用小模型（Qwen2.5-7B）精细过滤无关结果 → 按视频时间顺序重排保持因果关系

### 损失函数 / 训练策略
无训练框架，全部基于推理时的 API 调用。意图分类器通过 prompt engineering 实现。

## 实验关键数据

| 模型 | MLVU AVG | 提升 | Video-MME Overall | 提升 |
|------|----------|------|-------------------|------|
| Qwen2.5-VL-7B | 29.0 | - | 47.2 | - |
| + VideoRAG | - | - | 55.0 | +7.9 |
| + **AdaVideoRAG** | **40.5** | **+39.8%** | **59.9** | **+12.7** |
| VideoLLaMA3-7B | 47.7 | - | 64.2 | - |
| + VideoRAG | - | - | 67.3 | +3.1 |
| + **AdaVideoRAG** | **53.2** | **+11.6%** | **68.5** | **+4.3** |
| GPT-4o | 54.9 | - | 71.9 | - |

VideoLLaMA3 + AdaVideoRAG（7B）可与 GPT-4o 媲美（53.2 vs 54.9 MLVU）。

HiVU 基准：在 L3（困难推理）上 Overall Winner 77.13% vs baseline 22.87%，优势极为显著。

### 消融实验要点
- **分类器选择**：Qwen2.5-7B 精度 0.81，明显优于 1.5B（0.41），且 overall 68.5 最高
- **去掉分类器**：全走 L1 得 64.2，全走 L2 得 67.5，全走 L3 得 67.1，自适应得 68.5——验证了按需路由的价值
- **去掉图检索**：Overall Winner 54.18%（vs 完整 69.42%），说明图检索对复杂问题至关重要
- **去掉文本检索**：影响最大（68.75% → 31.25%），辅助文本是最核心的知识来源
- **采样频率**：5帧/30s 与 30帧/30s 仅差 ~1 点，5帧即够用

## 亮点
- 自适应路由是非常实用的设计——简单问题不浪费资源，复杂问题不遗漏信息
- 即插即用架构，不修改 MLLM 本身，通过 API 调用即可增强任何视频 MLLM
- 提出 HiVU 基准：首个分层难度的长视频理解评估集（L1/L2/L3），120 个视频 60 小时
- 7B 模型加上 AdaVideoRAG 可超越 72B 模型甚至媲美 GPT-4o

## 局限性 / 可改进方向
- 仅测试了三级路由，实际应用可能需要更细粒度的难度划分
- 知识库构建耗时较长（L3 约 412s），虽然可并行加速但仍是部署瓶颈
- 意图分类器的准确率 0.81 存在误分类风险，L2 误分为 L1 会导致信息不足
- HiVU 基准规模较小（120 视频），评估可能不够充分

## 与相关工作的对比
- **vs VideoRAG [Luo]**: 仅用朴素检索、不支持多跳推理；AdaVideoRAG 在长视频上优势显著（+4.8 Video-MME）
- **vs VideoRAG [Ren]**: 图检索对所有查询一视同仁，效率低；AdaVideoRAG 在 HiVU L3 上优于 VideoRAG（57.77 vs 42.23 Overall Winner）同时在简单查询上更高效
- **vs Adaptive-RAG [文本]**: 将自适应检索概念从文本扩展到视频多模态场景，增加了视觉 grounding 和知识图谱

## 启发与关联
- 自适应路由思想可迁移到图像理解：简单问题直接 VLM 回答，复杂问题才启用 RAG
- 全知识索引的多模态信息提取流程（caption+ASR+OCR+vision+graph）可作为通用视频知识库构建范式
- 与 Balanced Token Pruning 形成有趣互补：BTP 压缩输入 token，AdaVideoRAG 扩展外部知识

## 评分
- 新颖性: ⭐⭐⭐⭐ 自适应路由 + 全知识索引的整合方案系统性强
- 实验充分度: ⭐⭐⭐⭐⭐ 多个基准、多个MLLM、消融分析全面
- 写作质量: ⭐⭐⭐⭐ 系统架构描述清楚、动机分析到位
- 价值: ⭐⭐⭐⭐ 实用性强，即插即用方案对工业部署有价值
