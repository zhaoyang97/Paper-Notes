---
search:
  exclude: true
---

<!-- 由 src/gen_blog_index.py 自动生成 -->
# 💬 LLM / NLP

**📷 CVPR2026** · 共 **4** 篇

**[EVATok: 自适应长度视频Tokenization用于高效视觉自回归生成](evatok_adaptive_length_video_tokenization_for_eff.md)**

:   提出EVATok框架——通过最优token分配估计+轻量路由器+自适应tokenizer训练的三步流程，让视频tokenizer按片段复杂度自适应分配token长度，在UCF-101上节省24.4%+ token同时达到SOTA生成质量。

**[Hier-COS: Making Deep Features Hierarchy-aware via Composition of Orthogonal Subspaces](hiercos_making_deep_features_hierarchyaware_via_co.md)**

:   提出Hier-COS框架，为层次标签树中的每个节点分配正交基向量，通过子空间组合（祖先基+自身基+后代基）构建层次感知向量空间（HAVS），理论保证特征空间的距离结构与层次树一致，同时提出HOPS评估指标解决现有层次化评估指标的排列不变性缺陷。

**[IAPL: Generalizable AI-Generated Image Detection via Image-Adaptive Prompt Learning](iapl_aigenerated_image_detection_adaptive_prompt.md)**

:   针对 AI 生成图像检测中现有方法难以泛化到未见生成器的问题，提出图像自适应提示学习（IAPL），在推理时根据每张测试图像动态调整输入到视觉编码器的 prompt——通过条件信息学习器提取伪造特征条件和测试时自适应 token 优化，在 UniversalFakeDetect 和 GenImage 数据集上分别达到 95.61% 和 96.7% 的 SOTA 平均准确率。

**[WeaveTime: 流式视频LLM的帧级逐步记忆](weavetime_streaming_video_llm_memory.md)**

:   诊断出Video-LLM的核心缺陷"时间无感"——把视频当无序图像集处理，产生时序模糊和历史/当前混淆两类失效，提出WeaveTime通过轻量时序重建目标获得顺序感知能力+Past-Current动态焦点缓存实现高效流式推理，在流式基准上一致提升。
