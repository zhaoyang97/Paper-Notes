<!-- 由 src/gen_blog_index.py 自动生成 -->
# ✂️ 语义分割

**🎞️ ECCV2024** · 共 **4** 篇

**[A Semantic Space is Worth 256 Language Descriptions: Make Stronger Segmentation Models with Descriptive Properties](a_semantic_space_is_worth_256_language_descriptions_make_str.md)**

:   ProLab 用 LLM 生成类别的常识性描述，通过句子嵌入和 K-Means 聚类将其压缩为 256 个可解释的描述性属性，构建属性级多热标签空间替代传统 one-hot 类别标签来监督分割模型，在五个经典基准上一致超越类别级监督且涌现出域外泛化能力。

**[A Simple Latent Diffusion Approach for Panoptic Segmentation and Mask Inpainting](a_simple_latent_diffusion_approach_for_panoptic_segmentation_and_mask_inpainting.md)**

:   基于Stable Diffusion构建了一个极简的潜在扩散分割框架LDMSeg，通过浅层自编码器将分割mask压缩到潜空间、再训练图像条件扩散模型来生成全景分割结果，避免了传统方法中的目标检测模块、匈牙利匹配和复杂后处理，并天然支持mask inpainting和多任务扩展。

**[ActionVOS: Actions as Prompts for Video Object Segmentation](actionvos_actions_as_prompts_for_video_object_segmentation.md)**

:   提出ActionVOS——一种以人类动作叙述作为额外语言提示的Referring Video Object Segmentation新设定，通过无参数的动作感知标注模块生成伪标签，并设计动作引导的focal loss来抑制假阳性，在VISOR上将非活跃物体的误分割降低35.6% mIoU，同时在VOST/VSCOS上对状态变化物体的分割提升3.0% mIoU。

**[Active Coarse-to-Fine Segmentation of Moveable Parts from Real Images](active_coarsetofine_segmentation_of_moveable_parts_from_real.md)**

:   提出首个面向真实室内场景RGB图像中可运动部件实例分割的主动学习框架，通过姿态感知masked attention网络实现由粗到细的分割，仅需人工标注11.45%的图像即可获得全量验证的高质量分割结果，相比最优非AL方法节省60%人工时间。
