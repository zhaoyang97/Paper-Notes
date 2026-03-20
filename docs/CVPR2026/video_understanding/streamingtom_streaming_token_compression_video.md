# StreamingTOM: Streaming Token Compression for Efficient Video Understanding

**会议**: CVPR 2026  
**arXiv**: [2510.18269](https://arxiv.org/abs/2510.18269)  
**代码**: 有项目页面  
**领域**: 多模态VLM / 视频理解  
**关键词**: 流式视频理解, token压缩, KV-cache优化, 因果时序缩减, 4-bit量化记忆  

## 一句话总结
针对流式视频 VLM 面临的因果性（无法访问未来帧）和累积性（token 无界增长）两个约束，提出 StreamingTOM——一个免训练、即插即用的两阶段框架，通过因果时序缩减（减少 pre-LLM prefill）和在线量化记忆（4-bit KV-cache 存储+按需检索反量化），实现 15.7× KV-cache 压缩比、较 SOTA LiveVLM 降低 1.2× 峰值内存和 2× 更快 TTFT，在离线基准平均 63.8% 和流式基准 RVS 55.8% 达到免训练方法 SOTA。

## 背景与动机
流式视频理解与离线处理根本不同：(1) **因果性约束**——只能看到已有帧，不能利用未来帧信息；(2) **累积性约束**——随着时间推移，token 数量无界增长导致内存和延迟不断恶化。现有方法主要在 LLM 后端控制 KV-cache（如 eviction 策略），但忽略了 LLM 前端的 prefill 开销——每一帧都需要处理大量视觉 token 的前向传播，这是延迟的主要来源。

## 核心问题
如何在因果约束下同时解决 pre-LLM prefill 和 post-LLM KV-cache 两个效率瓶颈，实现有界活跃内存的实时流式视频理解？

## 方法详解

### 整体框架
两阶段免训练框架：Stage 1（Causal Temporal Reduction）处理 pre-LLM 瓶颈，Stage 2（Online Quantized Memory）处理 post-LLM 瓶颈。

### 关键设计

1. **因果时序缩减（Causal Temporal Reduction）**: 对每帧施加固定的 token 预算上限。Token 选择基于两个信号：(a) 相邻帧间的变化量——只保留有显著变化的区域对应的 token；(b) token 显著性——保留高信息量的 token。通过只处理每帧的紧凑 token 子集，大幅降低 per-frame prefill 成本，确保可预测的延迟。

2. **在线量化记忆（Online Quantized Memory）**: 将 KV-cache 中的 token 以 4-bit 格式存储，按需检索相关 token 组并反量化。关键特性：(a) 活跃 KV-cache 大小有上界，不随视频流长度无限增长；(b) 4-bit 量化大幅减少内存占用但保持足够精度；(c) 按需检索避免一次性加载全部历史记忆。

3. **即插即用、免训练**: 不需要重新训练模型，可以直接应用于现有 VLM 上。

### 损失函数 / 训练策略
完全免训练方法，无需任何额外训练或微调。

## 实验关键数据

| 指标 | StreamingTOM | 对比 |
|------|-------------|------|
| KV-cache 压缩比 | **15.7×** | - |
| 峰值内存 vs LiveVLM | **1.2× 更低** | LiveVLM 是之前 SOTA |
| TTFT (首token时延) vs LiveVLM | **2× 更快** | - |
| 离线基准平均准确率 | **63.8%** | 免训练方法 SOTA |
| RVS 流式基准准确率 | **55.8%** | 免训练方法 SOTA |
| RVS 流式基准得分 | **3.7** | 免训练方法 SOTA |

## 亮点
- **同时解决 pre-LLM 和 post-LLM 瓶颈**：之前方法只管 KV-cache eviction（post-LLM），StreamingTOM 首次在 pre-LLM prefill 层面也做优化
- **有界活跃内存**：活跃 KV-cache 大小不随视频长度增长，理论上可以处理无限长视频流
- **免训练即插即用**：不需要重训练，可直接应用于开源 VLM
- **实际效率提升显著**：15.7× 压缩比和 2× TTFT 加速，对实时部署有重要意义

## 局限性 / 可改进方向
- 4-bit 量化可能在极端精度要求场景下引入质量损失
- 基于相邻帧变化的 token 选择可能在快速运动场景下遗漏重要信息
- 仅基于摘要分析，具体的两阶段交互细节需参阅原文

## 与相关工作的对比
- **vs LiveVLM**: LiveVLM 只做 KV-cache 管理（post-LLM），StreamingTOM 同时优化 pre-LLM 和 post-LLM，内存更低速度更快
- **vs FastV / TokenPacker**: 这些方法关注单张图像的 token 压缩，StreamingTOM 专注于流式视频场景的时序累积问题
- **vs Video-LLM token pruning**: 大多数方法是离线的（可以看全部帧），StreamingTOM 是因果的（只看已有帧）

## 启发与关联
- 因果时序缩减的思想可以推广到其他流式多模态任务（如实时对话、直播分析）
- 4-bit 量化记忆 + 按需检索的设计可以与 RAG 类似的 VLM 长文本/长视频处理方法结合
- 对视频 VLM 的部署落地有直接指导意义

## 评分
- 新颖性: ⭐⭐⭐⭐ 首次同时地处两个层面的效率瓶颈，4-bit 量化记忆设计新颖
- 实验充分度: ⭐⭐⭐⭐ 离线和流式基准都达到免训练 SOTA，效率指标全面
- 写作质量: ⭐⭐⭐⭐ 摘要清晰，问题定义明确
- 价值: ⭐⭐⭐⭐⭐ 解决了流式视频 VLM 的实际部署痛点，实用价值极高
