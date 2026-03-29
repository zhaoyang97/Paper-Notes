<!-- 由 src/gen_blog_index.py 自动生成 -->
# 🎁 推荐系统

**💬 ACL2025** · 共 **3** 篇

**[GRAM: Generative Recommendation via Semantic-aware Multi-granular Late Fusion](gram_generative_recommendation.md)**

:   提出 GRAM 生成式推荐模型，通过语义到词汇的翻译（将隐式物品关系编码到 LLM 词汇空间）和多粒度迟融合（分别编码不同粒度提示后在解码时融合），在四个基准上比八个 SOTA 方法在 Recall@5 上提升 11.5-16.0%。

**[KERL: Knowledge-Enhanced Personalized Recipe Recommendation using Large Language Models](kerl_knowledge-enhanced_personalized_recipe_recommendation_using_large_language_.md)**

:   提出 KERL 统一食品推荐系统，结合 FoodKG 知识图谱和 Phi-3-mini 多 LoRA 微调，实现个性化食谱推荐（F1=0.973）、食谱生成和微量营养素估算三个功能，大幅超越基线 LLM 和传统嵌入方法。

**[RecLM: Recommendation Instruction Tuning](reclm_recommendation_instruction_tuning.md)**

:   提出 RecLM，一个模型无关的推荐指令微调框架，通过两轮对话式指令微调将协同过滤信号注入 LLM 生成的用户/商品画像，再用 RLHF（PPO）精炼画像质量，在 MIND/Netflix/工业数据集上作为即插即用组件为 BiasMF/NCF/LightGCN/SGL/SimGCL 一致带来提升，尤其在冷启动场景效果显著。
