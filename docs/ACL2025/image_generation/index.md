---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎨 图像生成

**💬 ACL2025** · 共 **3** 篇

**[Planning with Diffusion Models for Target-Oriented Dialogue Systems](difftod_diffusion_dialogue_planning.md)**

:   DiffTOD 将对话规划建模为轨迹生成问题，利用掩码扩散语言模型实现非顺序对话规划，并设计三种引导机制（词级/语义级/搜索级）灵活控制对话朝目标推进，在谈判/推荐/闲聊三种场景上显著超越基线。

**[FlashAudio: Rectified Flows for Fast and High-Fidelity Text-to-Audio Generation](flashaudio_rectified_flow_tta.md)**

:   将整流流（Rectified Flow）引入文本转音频生成，通过双焦采样器优化时间步分布、不混溶流减少数据-噪声总距离、锚定优化修正 CFG 引导误差，实现单步生成 FAD=1.49 超越百步扩散模型，生成速度达实时 400 倍。

**[R-VC: Rhythm Controllable and Efficient Zero-Shot Voice Conversion via Shortcut Flow Matching](rvc_rhythm_voice_conversion.md)**

:   R-VC 是首个实现节奏可控的零样本语音转换系统，通过 Mask Transformer 时长模型建模目标说话人的节奏风格，结合 Shortcut Flow Matching 的 DiT 解码器实现仅 2 步采样的高效高质量语音生成，在 LibriSpeech 上 WER 3.51、说话人相似度 0.930。
