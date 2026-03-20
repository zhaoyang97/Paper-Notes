# Autoregressive Speech Synthesis without Vector Quantization

**会议**: ACL 2025
**arXiv**: [2407.08551](https://arxiv.org/abs/2407.08551)
**代码**: https://aka.ms/melle (Demo)
**领域**: 语音合成 / TTS
**关键词**: 语音合成, 连续值token, mel-spectrogram, 变分推断, 自回归语言模型

## 一句话总结
MELLE 提出了一种基于连续 mel-spectrogram 帧的自回归语言模型 TTS 方法，通过回归损失 + 变分推断采样模块 + spectrogram flux loss 直接预测连续频谱帧，避免了向量量化带来的保真度损失和采样鲁棒性问题，单阶段模型即可达到与人类水平相当的语音合成质量。

## 研究背景与动机

1. **领域现状**：当前零样本 TTS 的主流方法是 codec 语言模型（如 VALL-E），先将音频通过神经编解码器（EnCodec 等）量化为离散 token，再用自回归语言模型预测这些离散 token。
2. **现有痛点**：
   - 向量量化本质是为音频压缩设计的，量化后的离散码相比连续表示会丢失保真度
   - 离散 codec 码之间高度相似，随机采样策略容易导致长时间静音或持续噪音等鲁棒性问题
   - 需要两阶段解码：AR 模型生成粗糙一级 token → NAR 模型迭代预测多层 codebook 码进行精细化，推理效率低
3. **核心矛盾**：离散 token 范式天然不适合音频这种连续信号——离散化本身就是信息有损压缩，而连续表示（如 mel-spectrogram）保留更丰富的声学细节。但连续值 token 面临两大难题：如何定义训练目标（无法用交叉熵）？如何在连续空间实现采样（无法用 top-p）？
4. **本文要解决什么？** 在自回归语音合成中用连续值 mel-spectrogram 帧完全替代离散量化 token，解决训练目标和采样机制两大挑战。
5. **切入角度**：从 mel-spectrogram 重建的语音在 WER 和说话人相似度上都优于 EnCodec 重建的语音，证实连续表示保真度更高。作者借鉴 VAE 的变分推断来实现连续空间的采样。
6. **核心 idea 一句话**：用回归损失替代交叉熵、用 VAE 式变分推断替代 top-p 采样，实现单阶段自回归连续 mel-spectrogram 预测的 TTS 模型。

## 方法详解

### 整体框架
MELLE 是一个 decoder-only 的自回归语言模型，输入是 BPE 文本 token 和 mel-spectrogram 帧的拼接，输出逐帧预测下一个连续 mel-spectrogram 帧。整体 pipeline：文本 → BPE embedding + mel-spectrogram → pre-net 投影 → Transformer decoder → Latent Sampling Module → 连续 mel 帧 → Post-Net 精细化 → vocoder 恢复波形。关键区别于 VALL-E：无需量化编解码器，无需 NAR 第二阶段，单阶段即完成。

### 关键设计

1. **自回归语言模型 (Transformer Decoder)**:
   - 做什么：作为核心骨干，自回归地生成连续声学 token
   - 核心思路：12 层 Transformer block（16 head, 1024 dim），输入文本通过 embedding layer、mel-spectrogram 通过 3 层 MLP pre-net 投影到模型维度，拼接后建模语义-声学依赖。每步输出 $\boldsymbol{e}_t$ 传给后续模块
   - 设计动机：利用 LM 的 in-context learning 能力实现零样本 TTS，decoder-only 结构简洁高效

2. **Latent Sampling Module（变分采样模块）**:
   - 做什么：在连续空间实现采样机制，增强输出多样性和鲁棒性
   - 核心思路：基于 VAE 的重参数化技巧。对 LM 输出 $\boldsymbol{e}_t$，用线性层预测高斯分布的均值 $\boldsymbol{\mu}_t$ 和对数方差 $\log \boldsymbol{\sigma}_t^2$，然后用 $\boldsymbol{z}_t = \boldsymbol{\mu}_t + \boldsymbol{\sigma}_t \odot \boldsymbol{\epsilon}$（$\boldsymbol{\epsilon} \sim \mathcal{N}(0, \boldsymbol{I})$）采样潜变量，再通过带残差连接的 3 层 MLP 映射回 mel-spectrogram 空间
   - 设计动机：离散模型可以用 top-p 采样引入多样性，连续模型无法使用。VAE 式采样自动学习每个输入对应的分布，比手动设计采样策略更自适应。消融实验表明该模块对保持说话人相似度（SIM）贡献显著

3. **Spectrogram Flux Loss（频谱变化损失）**:
   - 做什么：鼓励生成帧间动态变化，防止重复/静音
   - 核心思路：$\mathcal{L}_{\text{flux}} = -\sum_{t=1}^{T-1} \|\boldsymbol{\mu}_t - \boldsymbol{y}_{t-1}\|_1$，即最大化预测均值与前一帧真值的 L1 差异。这是一个负的度量，奖励帧间变化，惩罚过于静态的预测
   - 设计动机：纯回归损失容易让模型预测过于平滑/重复的帧，导致合成语音单调。flux loss 直接解决这个问题，消融显示对跨句任务 WER 从 10.87 降至 2.10（WERH）

4. **Reduction Factor（缩减因子）**:
   - 做什么：每步预测多帧 mel-spectrogram，加速推理
   - 核心思路：将序列按因子 $r$ 分组，每步预测 $r$ 帧而非 1 帧，训练和推理速度提高约 $r$ 倍
   - 设计动机：长序列建模的效率与鲁棒性权衡。$r=4$ 时推理时间仅 1.40s（VALL-E 需 7.32s），性能仍可接受

### 损失函数 / 训练策略

总损失 $\mathcal{L} = \mathcal{L}_{\text{reg}} + \lambda \mathcal{L}_{\text{KL}} + \beta \mathcal{L}_{\text{flux}} + \gamma \mathcal{L}_{\text{stop}}$

- **回归损失** $\mathcal{L}_{\text{reg}}$：L1 + L2 损失，同时应用于中间预测 $\boldsymbol{y}'$ 和 post-net 精细化后的 $\boldsymbol{y}''$
- **KL 散度损失** $\mathcal{L}_{\text{KL}}$：约束潜变量分布 $p_\theta(\boldsymbol{z}_t|\boldsymbol{e}_t)$ 向 $\mathcal{N}(\boldsymbol{y}_t, \boldsymbol{I})$（以真值为中心）靠近，而非标准 VAE 的 $\mathcal{N}(0, I)$，相当于优化路径上的捷径
- **Stop 预测损失**：BCE 损失 + 100 倍正样本权重解决正负帧极端不平衡

训练数据：Libriheavy 50K 小时，6736 说话人；小规模版本用 LibriSpeech 960 小时。

## 实验关键数据

### 主实验

| 系统 | 训练数据 | Continuation WERH↓ | Continuation SIM↑ | Cross-Sentence WERH↓ | Cross-Sentence SIM↑ |
|------|---------|-------------------|-------------------|---------------------|---------------------|
| Ground Truth | - | 2.15 | 0.668 | 2.15 | 0.779 |
| VALL-E | 60K h | 3.8 | 0.508 | 5.9 | 0.580 |
| VALL-E 2 | 50K h | 2.32 | 0.504 | 2.44 | 0.643 |
| Voicebox | 60K h | 2.0 | 0.593 | 1.9 | 0.662 |
| **MELLE** | **50K h** | **1.98** | **0.508** | **2.10** | **0.625** |
| MELLE-R4 | 50K h | 2.10 | 0.437 | 2.30 | 0.532 |

主观评测（Cross-sentence）:

| 系统 | MOS↑ | SMOS↑ | CMOS↑ |
|------|------|-------|-------|
| Ground Truth | 4.29 | 3.94 | 0.000 |
| VALL-E | 3.18 | 3.50 | -0.912 |
| VALL-E 2 | 4.08 | 3.88 | -0.085 |
| **MELLE** | **4.20** | **4.40** | **-0.032** |

### 消融实验

| 配置 | Cont. WERH↓ | Cont. SIM↑ | Cross WERH↓ | Cross SIM↑ |
|------|------------|-----------|------------|-----------|
| w/o LS + w/o SFL | 6.91 | 0.483 | 23.65 | 0.518 |
| w/ LS only | 4.07 | 0.486 | 10.87 | 0.584 |
| w/ SFL only | 2.61 | 0.506 | 5.90 | 0.602 |
| LS训练only + SFL | 2.13 | 0.506 | 2.72 | 0.615 |
| **Full (LS + SFL)** | **1.98** | **0.508** | **2.10** | **0.625** |

### 关键发现
- Spectrogram Flux Loss 对鲁棒性贡献最大：去掉后跨句 WERH 从 2.10 飙升至 10.87
- Latent Sampling 对说话人相似度贡献显著：在 WER 改善不如 SFL 的情况下，SIM 提升与 SFL 相当
- Reduction factor 可到 R4（推理加速 4×），性能仍优于大多数基线；R5 开始明显下降
- MELLE 的 SMOS（4.40）甚至超过了真实语音（3.94），说明其说话人特征捕获能力极强
- CMOS 与真实语音无统计显著差异（p > 0.1），达到人类水平

## 亮点与洞察
- **连续 token 替代离散 token 的范式转变**：在 codec LM 大行其道的背景下，证明了连续 mel-spectrogram 可以完全替代离散码并取得更好效果。这个思路可迁移到音乐生成、音频编辑等其他音频生成任务。
- **Spectrogram Flux Loss 的设计巧妙**：通过一个简单的负 L1 正则项解决了连续帧预测的帧间重复问题，原理直觉清晰（最大化帧间差异），实现极简但效果显著。类似思路可用于视频生成、动作序列预测等需要时序多样性的场景。
- **KL 散度以真值为中心**：不同于标准 VAE 使用 $\mathcal{N}(0, I)$ 作为先验，MELLE 用 $\mathcal{N}(\boldsymbol{y}_t, I)$，既提供正则化又加速收敛，算是优化捷径。
- **单阶段 vs 两阶段**：省去 NAR 第二阶段的复杂性，模型更简洁、推理更快，存储需求更低。

## 局限性 / 可改进方向
- **vocoder 质量限制**：使用开源 HiFi-GAN（仅 LibriTTS 585h 训练），如果用更强的 vocoder（大规模数据训练），效果还能进一步提升
- **仅英文评测**：实验只在 LibriSpeech 上做，多语言能力未验证
- **仅 mel-spectrogram**：没探索 VAE latent states 等其他连续表示，潜在的更好表示空间可能存在
- **安全风险**：零样本语音克隆的伦理问题——可用于语音冒充，需要配套合成检测模型和说话人授权机制

## 相关工作与启发
- **vs VALL-E/VALL-E 2**: VALL-E 用离散 codec 码 + 两阶段（AR+NAR），MELLE 用连续 mel + 单阶段。MELLE 在鲁棒性上完胜（WERH 1.98 vs 2.32），主观质量更优（MOS 4.20 vs 4.08）
- **vs Voicebox**: Voicebox 是非自回归 flow-matching 方法，SIM 更高但依赖私有 vocoder + 音素输入，MELLE 仅用 BPE + 开源 vocoder 已非常接近
- **vs 扩散/flow 方法（CosyVoice, SEED-TTS）**: 这些方法仍需两阶段（AR 生成离散码 + 扩散生成连续特征），MELLE 直接端到端生成连续帧，更简洁

## 评分
- 新颖性: ⭐⭐⭐⭐ 连续值自回归TTS是重要范式探索，但核心技术（VAE, 回归损失）并非全新
- 实验充分度: ⭐⭐⭐⭐⭐ 主客观评测全面，消融细致，reduction factor探索充分
- 写作质量: ⭐⭐⭐⭐⭐ 问题动机清晰，方法描述详尽，实验分析到位
- 价值: ⭐⭐⭐⭐ 对TTS领域有重要参考意义，但ACL会议中语音合成关注度相对有限
