# Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval

**会议**: ICLR 2026  
**arXiv**: [2602.08224](https://arxiv.org/abs/2602.08224)  
**代码**: [GitHub](https://github.com/jingjing0419/Efficient-SAM2)  
**领域**: segmentation / efficient inference  
**关键词**: SAM2, video object segmentation, post-training acceleration, sparse attention, memory compression  

## 一句话总结
提出 Efficient-SAM2，通过对象感知的稀疏窗口路由(SWR)和稀疏记忆检索(SMR)两个后训练加速方案，利用 SAM2 本身的稀疏感知模式消除冗余计算，在 SAM2.1-L 上实现 1.68× 加速且仅损失 1.0% 精度。

## 背景与动机
1. SAM2 在视频分割中表现优异，但计算开销大，限制实时部署
2. 主要瓶颈在图像编码器和记忆注意力模块
3. 现有方法(EdgeTAM)需要昂贵的端到端重训练
4. Token merge 方法(ToMe)与 SAM2 的窗口注意力架构不兼容
5. **关键观察1**: 解码器注意力集中在前景目标，但编码器注意力分布广泛→编码冗余
6. **关键观察2**: 记忆帧中仅少量 token 贡献显著，且显著性模式时间一致→记忆冗余

## 方法详解
**Sparse Window Routing (SWR)** — 图像编码器加速:
- 窗口级计算分配：将窗口分为目标相关 vs 背景
- 目标相关窗口 = 上一帧预测掩码覆盖区域 + 高注意力显著性窗口
- 背景窗口路由到轻量 shortcut 分支（两层线性层，参数仅 $d^2+2d$）
- shortcut 用 30 个无标签样本的重建损失训练，~1h 完成

**Sparse Memory Retrieval (SMR)** — 记忆注意力加速:
- 每个记忆帧首次参与注意力时缓存其显著性模式(Top-K token)
- 后续帧复用该模式，仅保留显著 token 参与计算
- 最新帧保持完整，提示帧不变
- 稀疏率 $s=0.95$，整体稀疏率约 0.68

## 实验关键数据
| 方法 | SA-V test J&F | 编码器加速 | 记忆加速 |
|------|---------------|-----------|---------|
| SAM2.1-L 原始 | 79.2 | - | - |
| ToMe | 56.4 | 1.36× | - |
| SWR (ours) | 76.9 | **1.83×** | - |
| MemPool | 73.8 | - | 2.14× |
| SMR (ours) | 79.0 | - | **1.78×** |
| **SWR+SMR** | **78.2** | **1.68× (端到端)** | - |

- SAM2.1-B+ 上：SWR 1.69× 加速 (75.0 vs 77.7)，SMR 1.82× 加速 (77.8 vs 77.7，几乎无损)
- DAVIS 2017：SWR+SMR 几乎无性能下降 (89.4 vs 89.7)
- 对比 EdgeTAM(蒸馏法)：Efficient-SAM2 无需重训练，性能更高

## 亮点
- **后训练加速**: 无需端到端重训练，轻量 shortcut 仅需 30 样本 1 小时训练
- **与 SAM2 架构天然匹配**: SWR 在窗口级操作，与窗口注意力完美兼容
- **SMR 几乎无损**: 稀疏率 95% 下记忆检索精度几乎不降
- **两模块独立且可叠加**: SWR 和 SMR 可分别使用也可组合

## 局限性
- 依赖上一帧预测质量来估计目标窗口，跟踪失败时可能级联恶化
- 仅在半监督 VOS 设置下评估，未验证交互式/推理式分割
- shortcut 分支设计较简单，可能在复杂背景下信息丢失
- 稀疏率是固定超参，未探索自适应稀疏

## 相关工作
- **SAM2/SAM2.1**: 本文加速对象
- **EdgeTAM/EfficientTAM**: 蒸馏重训练的 SAM2 加速方案
- **ToMe/ALGM**: 通用 ViT token merge 方法，不适用 SAM2 的窗口注意力

## 评分
- 新颖性: ⭐⭐⭐⭐ (从 SAM2 稀疏感知出发设计加速方案)
- 实验充分度: ⭐⭐⭐⭐ (4 个 benchmark + 两种模型规模 + 消融)
- 写作质量: ⭐⭐⭐⭐ (观察-方案对应清晰)
- 价值: ⭐⭐⭐⭐⭐ (实用性极强，后训练加速 SAM2 有广泛需求)
