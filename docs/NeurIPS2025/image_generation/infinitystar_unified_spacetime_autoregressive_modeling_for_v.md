# InfinityStar: Unified Spacetime AutoRegressive Modeling for Visual Generation

## 基本信息
- **arXiv**: 2511.04675
- **会议**: NeurIPS 2025 **Oral**
- **作者**: Jinlai Liu, Jian Han, Bin Yan, Hui Wu, et al.
- **机构**: ByteDance
- **代码**: https://github.com/FoundationVision/InfinityStar

## 一句话总结
提出 InfinityStar，首个能生成工业级 720p 视频的纯离散自回归模型，通过时空金字塔建模统一 T2I/T2V/I2V/交互式长视频生成，VBench 83.74 超越 HunyuanVideo，推理速度比扩散模型快 10-32×。

## 背景与动机
视频生成两大范式各有缺陷：
- **扩散模型**（Sora, HunyuanVideo, Wan）：质量好但推理慢（数十步去噪），且难以自然扩展到视频外推
- **自回归模型**（Emu3, Nova）：支持流式生成，但 next-token 预测需要数千步推理且质量不及扩散模型

VAR（Visual AutoRegressive）和 Infinity 证明 next-scale prediction（从粗到细）在图像生成上可匹配扩散模型，且速度快得多。InfinityStar 的核心目标：将这一范式扩展到视频生成。

## 核心问题
如何将空间 next-scale prediction 扩展到时空维度，实现高质量、高效率、统一的视觉生成？

## 方法详解

### 1. Spacetime Pyramid Modeling
核心设计：将视频分解为 **Image Pyramid + Clip Pyramids**
- 第一帧作为 $c_1$（$T=1$），编码静态外观
- 后续 clip 共享相同时长 $T > 1$，编码动态运动
- 每个 clip 内按空间维度做多尺度金字塔（$K$ 个 scale），时间维度保持不变

autoregressive likelihood：
$$p(r_1^1, \ldots, r_K^N) = \prod_{c=1}^N \prod_{k=1}^K p(r_k^c | r_1^1, \ldots, r_{k-1}^c, \psi(t))$$

这一设计：(1) 解耦外观和运动；(2) T2I 模型知识可直接继承；(3) 天然支持 I2V 和视频外推。

### 2. Visual Tokenizer 创新
**Knowledge Inheritance from Continuous VAE**：
- 复用 Wan 2.1 VAE 的架构和权重，在 encoder-decoder 间插入无参数量化器（BSQ）
- 不引入 codebook，直接二值球形量化
- 零微调即可重建视频，微调后显著提升质量
- 收敛速度比从头训练快数倍

**Stochastic Quantizer Depth (SQD)**：
- 问题：多尺度量化中信息严重偏向最后几个 scale，早期 scale 学不到有用表示
- 解决：训练时随机丢弃最后 $N$ 个 scale（概率 $p$），强迫模型在早期 scale 编码更多信息
- 效果：早期 scale 重建质量大幅提升，VBench +0.21

### 3. Spacetime Autoregressive Transformer
**Semantic Scale Repetition (SSR)**：
- 早期 scale 决定整体布局和运动方向（"语义 scale"）
- 重复前 $K_s=12$ 个 scale $N=3$ 次，让模型反复精炼语义表示
- 计算开销几乎可忽略（早期 scale token 数占比极小）
- VBench 提升：75.72 → 81.28（+5.56 分巨大提升）

**Spacetime Sparse Attention (SSA)**：
- 每个 clip 内部只关注前面 scale 的输入（非所有历史 token）
- 跨 clip 仅关注前一个 clip 的最大 scale（而非全部历史）
- 比 full attention 快 1.5×（192p）到显著更多（480p 从 OOM 到可运行）
- 性能反而更好（81.28 vs 80.77），因为减少了曝光偏差和误差累积

**Spacetime RoPE**：分解为 scale、time、height、width 四个分量。

### 4. Long Interactive Video Generation (InfinityStar-Interact)
- 滑动窗口方法：将长视频分解为 10s chunk（重叠 5s）
- **Semantic-Detail Conditions**：
  - Detail features：前一 clip 最后 $K$ 帧的全分辨率特征
  - Semantic features：对前一 clip 空间下采样获得的语义特征
  - 将条件 token 从 33.6K 压缩到 5.8K

## 实验关键数据

### T2I 生成
| 模型 | 参数量 | GenEval Overall | DPG Overall |
|---|---|---|---|
| FLUX-dev | 12B | 0.67 | 84.0 |
| Infinity | 2B | 0.73† | 83.46 |
| **InfinityStar-T2I** | **8B** | **0.79†** | **86.55** |

### T2V 生成 (VBench)
| 模型 | 类型 | VBench Overall |
|---|---|---|
| HunyuanVideo | Diffusion (13B) | 83.24 |
| Wan 2.1 | Diffusion (14B) | 84.70 |
| Emu3 | AR (8B) | 80.96 |
| Nova | AR (0.6B) | 80.12 |
| **InfinityStar** | **AR (8B)** | **83.74** |

### 推理延迟 (5s 720p)
| 模型 | 延迟 | 加速比 |
|---|---|---|
| Wan 2.1 (14B) | 1864s | 1× |
| Nova (0.6B) | 354s | 5× |
| **InfinityStar (8B)** | **58s** | **32×** |

### 消融实验 (192p)
| 配置 | VBench Total |
|---|---|
| 完整模型 | 81.28 |
| w/o SSR | 75.72 (-5.56) |
| w/o Spacetime Pyramid | 80.30 (-0.98) |
| w/o SQD | 81.07 (-0.21) |
| Full Attention | 80.77 (-0.51) |

## 亮点
1. **里程碑式工作 (Oral)**：首个产出工业级 720p 视频的离散 AR 模型
2. **速度优势碾压**：比 Wan 2.1 快 32×，单 GPU 58s 生成 5s 720p 视频
3. **统一框架**：T2I/T2V/I2V/视频外推/交互式长视频，一个模型全搞定
4. **知识继承策略**：从连续 VAE 到离散 tokenizer 的权重迁移极其优雅
5. **SSR 贡献巨大**：仅重复早期 scale 就带来 VBench +5.56 的飞跃

## 局限性
1. 高运动场景中图像质量和运动保真度存在 trade-off
2. 受限于计算资源，模型参数和训练规模未达到顶级扩散模型水平
3. 推理管线未完全优化（仍有加速空间）
4. 长交互式生成受累积误差影响，多轮后质量下降

## 与相关工作的对比
- **vs. Emu3**：Emu3 用 next-token prediction（逐 token），InfinityStar 用 next-scale prediction（逐尺度），后者推理步数少几个数量级
- **vs. Nova**：Nova 空间 set-by-set + 时间 frame-by-frame，InfinityStar 空间多尺度 + 时间 clip-by-clip
- **vs. HunyuanVideo/Wan**：扩散模型需数十步去噪，InfinityStar 一次前向 per scale，总步数 $K \times N$
- **vs. Infinity (T2I)**：InfinityStar 在 Infinity 基础上扩展到视频，引入时空金字塔和多项视频特有优化
- **vs. SANA-Sprint**：都关注 AR 式高效生成，但 SANA-Sprint 用于图像（连续 token + 扩散混合），InfinityStar 用于视频（纯离散）

## 启发与关联
- **AR vs. Diffusion 的竞争格局**：InfinityStar 证明离散 AR 可以在视频质量上匹敌甚至超越扩散模型，同时保持 10-30× 速度优势。这可能是视频生成范式转变的信号
- **知识继承范式的普适性**：从连续 VAE 到离散 tokenizer 的迁移策略可推广到其他 VQ-based 系统
- **与 FramePack 的关联**：InfinityStar-Interact 借鉴了 FramePack 的条件压缩思路，进一步验证了"语义+细节"双路条件的有效性

## 评分
- 新颖性：★★★★★ — 时空金字塔 + 知识继承 + SQD + SSR 多项创新
- 技术深度：★★★★★ — 从 tokenizer 到 transformer 到训练策略全栈优化
- 实验完整度：★★★★★ — T2I/T2V/I2V/外推/交互/人类评估/消融全覆盖
- 写作质量：★★★★★ — Oral 论文，结构清晰，insight 深刻
