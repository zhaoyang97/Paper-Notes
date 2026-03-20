# VINCIE: Unlocking In-context Image Editing from Video

**会议**: ICLR 2026  
**arXiv**: [2506.10941](https://arxiv.org/abs/2506.10941)  
**代码**: [项目页](https://vincie2025.github.io/)  
**领域**: image editing / generation（被分到 segmentation 分区）  
**关键词**: in-context editing, video learning, multi-turn editing, diffusion transformer, interleaved sequence, segmentation mask  

## 一句话总结
提出 VINCIE，首次仅从视频数据学习上下文图像编辑能力——将视频标注为交错多模态序列，设计三个代理任务(次帧预测/当前分割/次帧分割预测)，在多轮编辑 benchmark 上达到 SOTA，展现了视频数据作为编辑训练源的可扩展性。

## 背景与动机
1. 上下文图像编辑需要基于文本+图像的上下文序列生成编辑结果，支持多轮交互
2. 现有方法依赖任务特定流水线(分割+修复模型)构造训练数据，难以大规模扩展
3. 单轮编辑数据(InstructPix2Pix等)无法捕捉多步编辑中的依赖和演变意图
4. 视频天然包含长时间视觉动态（物体进出、相机运动、动作变化）
5. 已有视频利用方法仅用两帧，忽略丰富的长程上下文信息
6. 缺乏评估多轮编辑能力的高质量 benchmark

## 方法详解
**数据构造 — 交错多模态序列**:
- 从视频稀疏采样 K 帧，用 VLM 标注帧间视觉转换描述
- 用 GroundingDINO + SAM2 提取编辑区域(RoE)分割掩码
- 构建序列: $(I_0, T_0, M_{00}, M_{01}, I_1, \ldots, I_K)$
- 混合采样策略：等间距 + 固定帧数

**模型架构**:
- MM-DiT (3B/7B)，从视频基础模型初始化
- 两种注意力变体：完全注意力 vs 块级因果注意力
- 1D RoPE(文本) + 3D RoPE(图像)，分离位置编码
- 可学习 `<TURN>` token 标记轮次边界

**三个代理任务**:
1. **NIP (Next Image Prediction)**: 主任务，预测下一帧编辑结果
2. **CSP (Current Segmentation Prediction)**: 理解哪些区域需要编辑
3. **NSP (Next Segmentation Prediction)**: 预测变化将发生在哪里

**上下文组合学习**: 对上下文随机 dropout 增强鲁棒性

## 实验关键数据
**MSE-Bench (5轮编辑成功率, GPT-4o 评估)**:
| 方法 | Turn-1 | Turn-5 |
|------|--------|--------|
| InstructPix2Pix | 33% | <2% |
| UltraEdit | 38% | <2% |
| **VINCIE-7B** | **54%** | **25.0%** |
| GPT-4o | 82% | 62.7% |

**MagicBrush (Turn-3)**:
| 方法 | DINO | CLIP-I | CLIP-T |
|------|------|--------|--------|
| UltraEdit | 0.683 | 0.810 | 0.266 |
| OmniGen | 0.586 | 0.786 | 0.261 |
| **VINCIE** | 与 SOTA 可比，SFT 后超越 |

- 数据从 0.25M→10M sessions 扩展时，5轮成功率从 5%→22%
- 256×H100 训练 ~150h (7B)
- 还展现多概念组合、故事生成、链式编辑能力

## 亮点
- **视频即编辑数据**: 首次证明仅从原生视频可学习上下文编辑能力
- **极致可扩展**: 海量视频数据直接可用，无需手工配对
- **三代理任务设计**: 巧妙地将分割理解整合到生成框架中
- **MSE-Bench**: 5轮实用编辑评估，揭示现有方法的多轮弱点
- **从视频学到解耦表示**: 物体出/入、姿态变化等编辑操作自然习得

## 局限性
- 计算成本极高(256×H100, 150h)
- 与 GPT-4o 等商业模型仍有较大差距(25% vs 62.7%)
- 视频数据的视觉变化可能与用户期望的编辑操作存在分布偏差
- RoE 分割质量依赖 GroundingDINO+SAM2 的准确性

## 相关工作
- **InstructPix2Pix/UltraEdit**: 配对数据方法，单轮编辑
- **OmniGen/ICEdit/Step1X-Edit**: 通用编辑模型
- **GPT Image/Nano Banana**: 商业级上下文编辑模型
- **RealGeneral/UES**: 从视频学编辑但仅用两帧

## 评分
- 新颖性: ⭐⭐⭐⭐⭐ (视频→编辑的全新范式)
- 实验充分度: ⭐⭐⭐⭐⭐ (两个benchmark + 扩展性实验 + 全面对比)
- 写作质量: ⭐⭐⭐⭐ (结构清晰)
- 价值: ⭐⭐⭐⭐⭐ (可扩展数据源 + 新benchmark)
