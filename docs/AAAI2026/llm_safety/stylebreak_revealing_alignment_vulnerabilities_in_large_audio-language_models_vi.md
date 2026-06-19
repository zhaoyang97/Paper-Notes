---
title: >-
  [论文解读] StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak
description: >-
  [AAAI 2026][LLM安全][音频越狱] 提出 StyleBreak，首个基于语音风格的音频越狱框架，通过两阶段风格感知变换管道和查询自适应策略网络，系统研究语言学、副语言学和超语言学属性对 LAM 对齐鲁棒性的影响，在多种攻击范式下将 ASR 提升 7.1%-22.3%。 LAM 的安全威胁 大型音频语言模型（LA…
tags:
  - "AAAI 2026"
  - "LLM安全"
  - "音频越狱"
  - "大型音频语言模型"
  - "对齐鲁棒性"
  - "语音风格攻击"
  - "自适应策略"
---

# StyleBreak: Revealing Alignment Vulnerabilities in Large Audio-Language Models via Style-Aware Audio Jailbreak

**会议**: AAAI 2026  
**arXiv**: [2511.10692](https://arxiv.org/abs/2511.10692)  
**代码**: 无  
**领域**: AI安全  
**关键词**: 音频越狱, 大型音频语言模型, 对齐鲁棒性, 语音风格攻击, 自适应策略

## 一句话总结

提出 StyleBreak，首个基于语音风格的音频越狱框架，通过两阶段风格感知变换管道和查询自适应策略网络，系统研究语言学、副语言学和超语言学属性对 LAM 对齐鲁棒性的影响，在多种攻击范式下将 ASR 提升 7.1%-22.3%。

## 研究背景与动机

### LAM 的安全威胁

大型音频语言模型（LAM）通过将音频编码器与 LLM 耦合，实现了基于语音的自然交互。然而，LAM 面临**音频越狱（Audio Jailbreak）**威胁——攻击者构造恶意音频提示绕过对齐机制，诱导模型生成有害输出。

### 现有攻击方法的局限

现有音频越狱研究极为有限，且方法简单：

**文本语义级**：直接将文本越狱转换为语音（如 Vanilla），忽略了文本与语音之间的语义和感知差异

**信号级**：添加噪声注入（AdvWave）、音高变换、口音转换等浅层扰动，缺乏语义意图

**关键盲区**：人类语音携带三类信息——语言学（说了什么内容）、副语言学（情感/语调）、超语言学（说话者特征如年龄、性别）。这些丰富的表达性属性如何影响 LAM 对齐鲁棒性，此前完全未被探索。

### 核心动机

既有方法要么忽视语音语义（仅做TTS转换），要么使用浅层扰动（噪声、口音），均未能捕捉人类语音的丰富表达变化。StyleBreak 旨在系统性地回答：**不同的人类语音属性如何影响 LAM 的对齐鲁棒性？**

## 方法详解

### 整体框架

StyleBreak 由三个核心组件构成：
1. **两阶段风格感知变换管道**：生成具有多样语音属性的对抗性音频
2. **查询自适应策略网络**：自动为每个查询搜索最有效的风格配置
3. **目标 LAM 查询与评估**：提交风格化音频并评估越狱效果

### 关键设计

#### 1. **情感驱动的提示变换（Emotion-Driven Prompt Transformation）**

在自然对话中，说话者的情感影响问题的措辞方式。本模块将有害查询 $q$ 改写为情感化版本 $q_e$：
- 使用 GPT-4 根据情感特定指令注入表达性线索（感叹词、情感修饰语）
- 保留原始恶意意图，同时改变语言学表达

**设计动机**：情感化改写可以更好地伪装恶意意图（ARR 比原始查询高 3.9 倍），利用模型对情感表达的宽容性。

#### 2. **风格控制的音频攻击生成（Style-Controlled Audio Attack Generation）**

使用 CosyVoice2-0.5B（可控 TTS 模型）将情感化文本合成为具有特定副语言学和超语言学属性的音频：

$$a_p = C(q_e, x_{ins})$$

其中 $x_{ins} = (t_{ins}, a_{ref})$ 包含风格自然语言描述和参考音频片段。

**风格配置空间**：$\mathcal{S} = \mathcal{E} \times \mathcal{G} \times \mathcal{A}_g$
- 情感 $|\mathcal{E}| = 7$（如愤怒、惊讶、悲伤等）
- 性别 $|\mathcal{G}| = 2$
- 年龄组 $|\mathcal{A}_g| = 5$
- 总计 $|\mathcal{S}| = 70$ 种配置

风格参考集从 GigaSpeech 数据集构建，每种配置随机采样 5 个多样化实例。

#### 3. **查询自适应策略网络**

**关键观察**：不同查询在不同风格配置下的越狱成功率差异巨大——越狱效果是**查询特异性**的而非均匀的。穷举所有 70 种配置计算昂贵且受 API 限制。

**策略网络设计**：多头策略网络 $\pi_\theta: \mathcal{Q} \to \Delta(\mathcal{S})$

- 共享前馈编码器（两层 MLP）处理查询表示向量 $d_q$
- 三个独立分类头分别预测情感、年龄、性别的选择分布

**训练目标**：奖励加权的多任务分类，最大化期望奖励：

$$\max_\theta \mathbb{E}_{q \sim \mathcal{Q}, s \sim \pi_\theta(q)} [J(M(a_p^s, t_i))]$$

其中 $J(\cdot) = \frac{1}{4}(\text{ARR} + \text{PV} + \text{TS} + \text{ASR})$ 为四指标均值的综合评价函数。

### 损失函数 / 训练策略

- 策略网络使用 200 个 AdvBench 查询训练，50 个不重叠的查询用于测试
- TTS 统一使用 CosyVoice2-0.5B
- 每次测试重复 5 次以消除随机性
- 对于黑盒转移攻击（GCG\*、AutoDAN\*），先在 LLaMA-2-7B 上优化再转移

## 实验关键数据

### 主实验

评估模型：Qwen2-Audio、Qwen-Omni、MERaLiON、Ultravox

**StyleBreak 在 Vanilla 基线上的提升（3 次查询迭代）**：

| 模型 | 基线 ASR | +StyleBreak ASR | 提升 |
|------|---------|----------------|------|
| Qwen2-Audio | 10.0% | **30.5%** | +20.5% |
| Qwen-Omni | 0.0% | **22.2%** | +22.2% |
| MERaLiON | 4.0% | **37.8%** | +33.8% |
| Ultravox | 4.0% | **16.9%** | +12.9% |

**跨攻击范式提升（Qwen2-Audio）**：

| 攻击方法 | 原始 ASR | +StyleBreak ASR | 提升 |
|---------|---------|----------------|------|
| Vanilla | 10.0% | 30.5% | +20.5% |
| GCG* | 6.9% | 33.3% | +26.4% |
| AutoDAN* | 11.8% | 16.7% | +4.9% |
| SSJ | 8.0% | **41.7%** | +33.7% |

### 消融实验

**各模块对 ASR（%）的贡献**：

| 配置 | Qwen2-Audio | Qwen-Omni | MERaLiON | Ultravox |
|------|------------|-----------|----------|----------|
| 文本原始查询 | 1.1 | 0.0 | 1.5 | 1.0 |
| +EPT（情感提示变换） | 8.9 | 4.1 | 12.1 | 9.6 |
| Vanilla 音频 | 10.0 | 0.0 | 4.0 | 4.0 |
| +EPT | 15.3 | 7.0 | 20.5 | 5.4 |
| +EPT, EAG（风格音频） | 17.2 | 9.6 | 35.1 | 14.8 |
| **+EPT, EAG, QP（完整）** | **30.5** | **22.2** | **37.8** | **16.9** |

每个模块都贡献了独特的提升，完整 StyleBreak 始终优于所有变体。

**语音属性单因素影响**：
- **情感（语言学）**：即使最鲁棒的 Qwen-Omni 也被提升 ASR 0→9.1%
- **情感（副语言学）**：Ultravox 特别敏感，ASR 提升 4.6-6.8 倍
- **年龄（超语言学）**：老年声音的 ASR 比儿童声音平均高 13.3%
- **性别（超语言学）**：男性声音比女性声音 ASR 平均高 8.3%

### 关键发现

1. **LAM 对低沉声音更脆弱**：男性和老年声音一致地产生更高的攻击成功率——推测 LAM 对高音调声音（儿童、女性）有更强的保护偏好
2. **音频模态本质上比文本更脆弱**：t-SNE 可视化揭示 LAM 在音频模态下区分良性/恶意输入的能力显著弱于文本模态
3. **MERaLiON 在复合攻击下最脆弱**：虽然其在单属性扰动下鲁棒，但其多文化上下文泛化能力反而使其对复杂风格音频更敏感
4. **策略可跨模型迁移**：在 Qwen2-Audio 上训练的策略直接迁移到 GPT-4o 和 Gemini-2.5-flash 仍有效

## 亮点与洞察

1. **首次系统研究语音属性对 LAM 对齐的影响**：填补了音频安全研究的重要空白，揭示了被忽视的攻击面
2. **生理特征成为攻击向量**：年龄、性别等说话者特征竟然影响模型的安全对齐——这暗示 LAM 的对齐训练存在系统性偏差
3. **自适应策略的高效性**：仅需 3 次查询迭代即可达到显著攻击效果（ASR 提升 7.1%-22.3%），远优于穷举搜索
4. **t-SNE 可视化的深刻洞察**：音频查询在模型表示空间中的良性/恶意重叠度远高于文本，解释了音频越狱为何天然更有效

## 局限与展望

1. **仅使用 CosyVoice2-0.5B 作为 TTS**：其他 TTS 系统可能产生不同效果
2. **AdvBench 查询集有限**：可扩展到更多样化的有害查询类型
3. **策略网络架构较简单**：更复杂的模型可能发现更有效的风格组合
4. **防御研究缺失**：论文聚焦攻击，未提出相应防御方案
5. **未来方向**：开发基于语音属性感知的对齐训练方法，使 LAM 对不同声音特征表现出一致的安全行为

## 相关工作与启发

- **Vanilla 方法**直接 TTS 转换 → 忽略模态差异
- **GCG / AutoDAN**文本语义级优化 → 长音频可能损失语义
- **SSJ**拼写式音频扰动 → LAM 倾向复述而非回答
- **SpeechTripleNet**语音信息三分类（语言、副语言、超语言）→ 为 StyleBreak 的分类体系提供理论基础
- 对 LAM 安全的启示：对齐训练不应仅关注文本内容，还需考虑语音的全维度属性

## 评分

- 新颖性: ⭐⭐⭐⭐⭐ — 首次系统探索语音属性对 LAM 对齐的影响
- 实验充分度: ⭐⭐⭐⭐⭐ — 4个LAM×4种攻击范式×3类属性，含消融、迁移和可视化
- 写作质量: ⭐⭐⭐⭐ — 结构清晰、图表丰富，但方法细节分散在正文和附录
- 价值: ⭐⭐⭐⭐⭐ — 揭示了 LAM 安全的关键盲区，对对齐训练有重要指导意义

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## 相关论文

- [\[ICLR 2026\] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models](../../ICLR2026/llm_safety/audiotrust_benchmarking_the_multifaceted_trustworthiness_of_audio_large_language.md)
- [\[NeurIPS 2025\] ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models](../../NeurIPS2025/llm_safety/almguard_safety_shortcuts_and_where_to_find_them_as_guardrails_for_audio-languag.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)

</div>

<!-- RELATED:END -->
