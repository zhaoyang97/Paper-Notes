<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🔄 自监督/表示学习

**💬 ACL2025** · 共 **3** 篇

**[Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing](llm_back_gen_treebank.md)**

:   提出 LLM 反向生成方法自动构建跨领域成分句法树库——给定只有领域关键词叶节点的不完整句法树，用 LLM 填充缺失词汇生成完整的跨领域句法树库，结合 span 级对比学习预训练，在 MCTB 五个目标领域上达到跨领域成分句法分析 SOTA。

**[QAEncoder: Towards Aligned Representation Learning in Question Answering Systems](qaencoder_aligned_representation.md)**

:   提出 QAEncoder，一种免训练方法通过蒙特卡洛采样估计文档对应查询的期望嵌入作为文档表示的代理，配合文档指纹保持区分性，在 BEIR 上将 bge-large 从 58.5 提升到 61.8 NDCG@10，零额外存储和延迟开销。

**[SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction](shubert_self-supervised_sign_language_representation_learning_via_multi-stream_c.md)**

:   提出 SHuBERT（Sign Hidden-Unit BERT），将语音自监督学习模型 HuBERT 的 masked cluster prediction 范式迁移到手语视频——对手部、面部、身体姿态四个流分别聚类并同时预测 masked 帧的聚类标签，在约 984 小时 ASL 视频上预训练后，在翻译/孤立识别/指拼检测多任务上达到公开数据 SOTA。
