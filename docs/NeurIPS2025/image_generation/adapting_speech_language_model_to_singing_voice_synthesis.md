# Adapting Speech Language Model to Singing Voice Synthesis

**会议**: NeurIPS 2025 (Workshop)  
**arXiv**: [2512.14657](https://arxiv.org/abs/2512.14657)  
**代码**: https://tsukasane.github.io/SLMSVS/  
**领域**: 语音生成 / 歌声合成 / 语言模型  
**关键词**: Speech Language Model, SVS, Flow Matching, Codec, 歌声合成

## 一句话总结
将 1.7B 参数的 TTS 预训练 Speech Language Model 适配到歌声合成（SVS）任务，通过乐谱 tokenization + multi-stream LM 预测 + conditional flow matching 精修 + vocoder，仅用 135 小时合成歌声数据达到与专用 SVS 系统可比的性能。

## 研究背景与动机
1. **领域现状**：Speech Language Model (SLM) 成为统一处理 TTS/ASR/SE 等语音任务的范式，但在歌声合成上的泛化能力未被探索
2. **现有痛点**：
   - SVS 公开数据集极少（版权限制+标注昂贵），无法从头训练大模型
   - SVS 输入是结构化乐谱（音素+音高+时值），比 TTS 的文本输入复杂得多
   - 预训练在语音上的 codec 解码器无法忠实重合成歌声，设置了性能上限
3. **核心矛盾**：大规模 SLM 的泛化潜力 vs SVS 数据稀缺
4. **本文要解决什么？** 探索 TTS 预训练 SLM 能否低成本适配到 SVS
5. **切入角度**：将乐谱条件 tokenize 后加入 SLM 词表，微调后用 flow matching 精修
6. **核心idea一句话**：用 TTS 预训练 SLM + flow matching 精修解决歌声合成的低资源问题

## 方法详解

### 整体框架
输入：乐谱（音素+MIDI音高+时值）+ 说话人提示。(1) 乐谱 tokenization 为 50FPS 离散 token；(2) 音频用 codec encoder + SSL 模型提取 multi-stream token；(3) LM 预测目标 token 序列；(4) Flow matching 将 LM 预测的 codec token → mel 频谱；(5) HiFi-GAN vocoder → 波形。

### 关键设计

1. **乐谱 Tokenization (svs_lb)**:
   - 做什么：将音素、MIDI 音高和持续时间编码为帧级离散 token
   - 核心思路：每帧由 (phoneme_token, pitch_token) 元组表示，通过重复次数隐式编码持续时值：repeat = (end - start) × fps。新增 svs_lb 模态扩展 TTS 词表
   - 设计动机：与 SLM 的 token 预测范式一致，复用 TTS 预训练编码器

2. **Multi-stream LM Token 预测**:
   - 做什么：用 1.7B SLM 预测拼接的 SSL + 8层 codec token
   - 核心思路：基于 ESPNet-SpeechLM，输入乐谱条件 + 说话人提示，目标是帧级 SSL+codec token 的交叉熵损失
   - 设计动机：SSL token 编码高层语义，codec token 编码声学细节，拼接融合两者优势

3. **Flow Matching 精修**:
   - 做什么：将 LM 预测的嘈杂 codec token 精修为干净的 mel 频谱
   - 核心思路：Conditional Flow Matching (CFM) 从高斯噪声出发，以 codec token 和 pitch 信号为条件，学习速度场将样本传输到目标 mel 分布。线性插值路径 ψ_t(x|x_1) = (1-t)x + tx_1
   - 设计动机：LM 直接预测的 token 有噪声导致时域不连续和感知毛刺；codec 解码器在语音上预训练、无法忠实重合成歌声。Flow matching 绕过了这两个瓶颈

### 损失函数 / 训练策略
- LM 微调：交叉熵损失，最大化 P(s|m,p)
- Flow matching：条件速度场的 MSE 损失
- 额外训练与 codec STFT 参数一致的 HiFi-GAN vocoder

## 实验关键数据

### 主实验
ACE-Opencpop 数据集（135小时合成歌声）

| 方法 | F0_RMSE↓ | F0_CORR↑ | MCD↓ | PER↓ | SingMOS↑ |
|------|---------|---------|------|------|----------|
| XiaoiceSing | 71.67 | 0.62 | 11.47 | **0.09** | 3.88 |
| TokSing | 55.83 | 0.67 | **6.77** | 0.19 | **4.08** |
| LM + Flow + Voc (ours) | 62.79 | 0.60 | 7.86 | 0.36 | 4.09 |

SingMOS (感知质量) 与最佳专用系统 TokSing 持平

### 消融实验
| 配置 | MCD↓ | PER↓ | SingMOS↑ | 说明 |
|------|------|------|----------|------|
| LM + CD (codec decoder) | 8.26 | 0.56 | 3.65 | 直接用 codec 解码，质量差 |
| LM + Flow1 + CD | 8.44 | 0.45 | 3.64 | Flow 精修但仍用 codec 解码 |
| LM + Flow1 + Voc | **7.86** | **0.36** | **4.09** | Flow + 专用 vocoder，最佳 |
| CD Resynthesis (upper bound) | 5.84 | 0.19 | 3.95 | Codec 解码器的上限 |

### 关键发现
- Codec decoder 是最大瓶颈（在语音上预训练的解码器不适合歌声）
- Flow matching 精修 + 专用 vocoder 显著提升质量（SingMOS 3.65→4.09）
- LM + Flow + Voc 甚至超过了 codec 重合成的 SingMOS 上限（4.09 vs 3.95），说明 flow matching 能弥补 codec 的缺陷
- PER（音素错误率）仍高于专用系统，歌词清晰度有改进空间

## 亮点与洞察
- **SLM 的跨任务泛化**：仅用 135 小时数据就将 TTS SLM 适配到 SVS，验证了大模型的泛化潜力
- **Flow matching 作为解码桥梁**：优雅解决了预训练 codec 解码器领域不匹配问题，是一个通用的"domain gap bridging"策略
- **两阶段的互补设计**：LM 负责序列建模（时间结构），Flow 负责声学质量（频谱细节），分工明确

## 局限性 / 可改进方向
- PER 偏高（0.36 vs TokSing 0.19），歌词发音清晰度不够
- F0 相关指标偏低（0.60 vs 0.67），音高追踪精度还需提升
- 仅在中文歌声（Opencpop）上测试，多语言泛化待验证
- 135 小时是合成数据，真实演唱数据上的效果未知

## 相关工作与启发
- **vs TokSing**: 专用 SVS 系统，在客观指标上更好但感知质量相当。A-Mem 的优势是复用 TTS 预训练
- **vs XiaoiceSing**: 传统端到端 SVS，PER 最低但 MCD 最高，音色建模弱
- **vs ESPNet-SpeechLM**: 本文基于其框架，证明了其多任务扩展能力

## 评分
- 新颖性: ⭐⭐⭐ SLM→SVS 的适配思路有趣但技术贡献增量有限
- 实验充分度: ⭐⭐⭐ 消融充分但仅一个数据集
- 写作质量: ⭐⭐⭐⭐ 简洁清晰（workshop paper）
- 价值: ⭐⭐⭐ 验证了 SLM 跨任务泛化，对低资源语音生成有启发
