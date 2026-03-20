<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🛡️ AI 安全

**💬 ACL2025** · 共 **16** 篇

**[CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference](centaur_bridging_the_impossible_trinity_of.md)**

:   提出 Centaur 框架，融合随机置换矩阵和安全多方计算（SMPC）来打破隐私保护 Transformer 推理（PPTI）中的"不可能三角"——同时实现强隐私保护、5-30x 加速和明文级别推理精度。

**[Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](dialect_fairness_robustness.md)**

:   本文提出首个系统评估LLM在非标准方言（AAVE）推理任务中公平性与鲁棒性的研究，构建了包含1.2K+平行查询对的ReDial基准，发现几乎所有主流LLM在AAVE输入上表现出显著的性能下降和不公平。

**[Ensemble Watermarks for Large Language Models](ensemble_watermarks_llm.md)**

:   提出集成水印方法，将文体特征（藏头词 acrostic + 感觉运动词 sensorimotor norms）与已有红绿水印组合，在 paraphrasing 攻击后三特征集成检测率达 95%，而单独红绿水印仅 49%。

**[Fairness through Difference Awareness: Measuring Desired Group Discrimination in LLMs](fairness_difference_awareness.md)**

:   本文挑战了主流公平性研究中"对所有群体一视同仁即为公平"的假设，提出"差异意识"(Difference Awareness)概念，构建了包含8个基准共16k问题的评测套件，发现现有"最公平"的LLM在该维度上表现不佳，且现有去偏方法会适得其反。

**[From Trade-off to Synergy: A Versatile Symbiotic Watermarking Framework for Large Language Models](from_tradeoff_to_synergy_a_versatile.md)**

:   提出SymMark共生水印框架，融合logits-based和sampling-based两类水印方法（串行/并行/混合三种策略），通过token熵和语义熵自适应选择水印策略，在可检测性、鲁棒性、文本质量和安全性上实现SOTA。

**[Gender Inclusivity Fairness Index (GIFI): A Multilevel Framework](gifi_gender_fairness.md)**

:   提出 GIFI（Gender Inclusivity Fairness Index），一个多层次综合评估框架，涵盖代词识别、情感中立性、毒性、反事实公平性、刻板印象关联、职业公平性和推理性能一致性七个维度，在 22 个 LLM 上系统评估二元与非二元性别的公平性。

**[Improved Unbiased Watermark for Large Language Models](improved_unbiased_watermark_for_large_language.md)**

:   提出 MCmark，一族基于多通道（Multi-Channel）的无偏水印算法，通过将词表分割为 $l$ 个段并在选中段内提升 token 概率来嵌入统计信号，在保持 LLM 原始输出分布的同时，可检测性比现有无偏水印提升超 10%。

**[Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?](llm_watermark_distillation_robustness.md)**

:   本文首次系统研究 LLM 水印在防止未授权知识蒸馏中的鲁棒性，提出三种水印去除攻击（无目标/有目标释义 + 推理时水印中和），发现有目标释义和水印中和可以彻底去除继承的水印，其中水印中和在保持知识迁移效率的同时实现零额外训练开销的水印去除。

**[MorphMark: Flexible Adaptive Watermarking for Large Language Models](morphmark_adaptive_watermarking.md)**

:   MorphMark 通过多目标权衡分析框架揭示了绿表概率 P_G 在水印效果与文本质量之间的关键作用，并据此提出自适应调整水印强度 r 的方法——当 P_G 高时增强水印、P_G 低时减弱水印，实现了在不依赖额外模型训练的前提下同时提升水印可检测性和文本质量。

**[PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](privacibench_evaluating_privacy_with_contextual_integrity.md)**

:   提出 PrivaCI-Bench，基于 Contextual Integrity 理论构建了目前最大的上下文隐私评估基准（154K 实例），涵盖真实法院案例、隐私政策和 EU AI Act 合规检查器合成数据，评估 LLM 在 HIPAA/GDPR/AI Act 下的法律合规能力。

**[Sandcastles in the Storm: Revisiting Watermarking Impossibility](sandcastles_watermarking_impossibility.md)**

:   本文通过大规模实验和人类评估挑战了 "Watermarks in the Sand" (WITS) 的理论不可能性结论：证明随机游走攻击的两个关键假设在实践中不成立——混合(mixing)速度极慢（100% 的攻击文本仍可追溯原始来源）且质量预言机(quality oracle)不可靠（仅 77% 准确率），自动攻击仅 26% 成功率，人类质量审核后降至 10%。

**[SpeechFake: A Large-Scale Multilingual Speech Deepfake Dataset Incorporating Cutting-Edge Generation Methods](speechfake_a_largescale_multilingual_speech_deepfake.md)**

:   构建 SpeechFake，目前最大的语音深度伪造检测数据集——超 300 万样本、3000+ 小时、40 种生成工具（含最新 TTS/VC/Neural Vocoder）、46 种语言，在自有及未见测试集上展现强基线性能。

**[TIP of the Iceberg: Task-in-Prompt Adversarial Attacks on LLMs](tip_iceberg_adversarial_attacks.md)**

:   本文提出 Task-in-Prompt (TIP) 攻击——一类通过在 prompt 中嵌入序列到序列任务（如密码解码、谜语、代码执行）来间接生成违禁内容的新型越狱攻击类别，并构建 PHRYGE benchmark 系统评估，证明该攻击可成功绕过 GPT-4o、LLaMA 3.2 等六种 SOTA LLM 的安全防护。

**[The Tug of War Within: Mitigating the Fairness-Privacy Conflicts in Large Language Models](tug_of_war_fairness_privacy.md)**

:   发现 LLM 通过 SFT 增强隐私意识会显著降低公平性意识（trade-off），提出无训练方法 SPIN（抑制公平-隐私耦合神经元），基于信息论解耦两种意识，在 Qwen2-7B 上同时提升公平性 12.2% 和隐私意识 14.0%。

**[Efficiently Identifying Watermarked Segments in Mixed-Source Texts](watermark_segment_detection.md)**

:   提出两种高效方法（Geometric Cover Detector 和 Adaptive Online Locator）用于在长文混合来源文本中检测和精确定位水印片段，时间复杂度从 O(n²) 降至 O(n log n)，在三种主流水印技术上均显著优于 baseline。

**[WET: Overcoming Paraphrasing Vulnerabilities in Embeddings-as-a-Service with Linear Transformation Watermark](wet_eaas_watermark.md)**

:   揭示了现有 EaaS 嵌入水印（EmbMarker/WARDEN）可被改写攻击绕过，提出 WET（线性变换水印），通过秘密循环矩阵对嵌入做线性变换注入水印，理论和实验证明其对改写攻击具有鲁棒性，验证 AUC 接近 100%。
