##### m_incontext

- Hijacking Context in Large Multi-modal Models
    - prompt GPT4-v to remedy in-context pair

- Towards More Unified In-context Visual Understanding
    - model task as Image -> Image/Text(combination of SegGPT img->img and Flamingo img -> text)
    - arch: encode (vqgan tokenizer, gpt2 tokenizer)- decode (moe)
    - dataset: MSCOCO, Visual Genome dataset

- MMICL: EMPOWERING VISION-LANGUAGE MODEL WITH MULTI-MODAL IN-CONTEXT LEARNING
    - present MMICL, inter-leaved ICL formmat
    - construct a multi-modal in-context learning dataset

- Link-Context Learning for Multimodal LLMs
    - a new setting require mllm use unseen img in context, to infer qs.
    - propose a Link-Context learning dataset: ISEKAI Dataset

- OpenFlamingo: An Open-Source Framework for Training Large Autoregressive Vision-Language Models
    - (img, text img, text, img) -> text
    - training dataset: LAION-2B, Multimodal C4, ChatGPT-generated data

- Generative Pretraining in Multimodality
    - text/img -> text/img
    - training strategy: text token -> classification; visual token -> regression (then put into SDM producing img)
    - dataset: LAION-2B, LAION-COCO, WebVid-10M, MMC4, YT-Storyboard
    - 128 A100 2 days(150B tokens)

- MIMIC-IT: Multi-Modal In-Context Instruction Tuning
    - propose a dataset(MIMIC-IT) consisting of perception, reasoning, planning

- Exploring Diverse In-Context Configurations for Image Captioning
    - extensive work in image-caption in context task, indicate that caption is more important than img, if caption is good enough, the more shot the better, img similarity don't influence ans too much.

- Prophet: Prompting Large Language Models with Complementary Answer Heuristics for Knowledge-based Visual Question Answering
    - 