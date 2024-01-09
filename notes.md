##### hallucination

- BEYOND REVERSE KL: GENERALIZING DIRECT PREFERENCE OPTIMIZATION WITH DIVERSE DIVERGENCE CONSTRAINTS
	- problem: reward hack, more memory-demanding
	- introduce other divergence

- Evaluating Object Hallucination in Large Vision-Language Models
	- hallu highly affected by object hallucination
	- LVLMs tend to generate frequently appearing or co-occurring objects in the instruction corpora
	- object hallucination evaluation approach called POPE(yes or no)

- Mitigating Hallucination in Large Multi-Modal Models via Robust Instruction Tuning
	- introduce LRV-Instruction dataset 400k visual instructions(both pos and neg/manipulation)
	- GAVIE evaluation(GPT4-AIDED)

- MOCHa: Multi-Objective Reinforcement Mitigating Caption Hallucinations
	- multi-task PPO(NLI4fidelity,bertscore4adequacy,kl4reg) c<-->c^groundtruth
	- OpenCHAIR benchmark(80 objects -> more)

- Mitigating Fine-Grained Hallucination by Fine-Tuning Large Vision-Language Models with Caption Rewrites
	- use chatgpt rewrite caption mul times and minimize the margin bewteen these(agnostic to sentence structure, sensitive to verbs, adj, nouns)
	- produce fine-grained than pope dataset(FGHE)

- RLHF-V: Towards Trustworthy MLLMs via Behavior Alignment from Fine-grained Correctional Human Feedback
	- decompose yw into three factors, optimize yp to avoid reward hack.
	- introduce [RLHF-V-Dataset](https://huggingface.co/datasets/HaoyeZhang/RLHF-V-Dataset) (human curated)

- OPERA: Alleviating Hallucination in Multi-Modal Large Language Models via Over-Trust Penalty and Retrospection-Allocation
	- use decode to alleviate hallu(column attention pattern)

- Mitigating Object Hallucinations in Large Vision-Language Models through Visual Contrastive Decoding
	- substract distorted image recognition distribution to alleviate statistical bias and unimodal priors.(1 + a)logits(y| x, v) - alogits(y, x, v1)

- Beyond Hallucinations: Enhancing LVLMs through Hallucination-Aware Direct Preference Optimization
	- indicate the importance of distribution consistence i.e yw and yl in DPO training stability.
	- introduce “Sentence-level Hallucination Ratio” (SHR) to evaluate.(GPT-4 aided)

- Mitigating Hallucination in Visual Language Models with Visual Supervision
	- introduce Relation-Associated Instruction Dataset(more diverse and mis-leading mul-turn dialogue)
	- introduce object CELoss and Mask Loss(disentangle test generation and object recognition) p.s. LVLM 3 output: test response, f_sub, f_obj, the last two -> SAM <-> ground_truth mask

- HalluciDoctor: Mitigating Hallucinatory Toxicity in Visual Instruction Data
	- remediate the toxicity in LLAVA、minigpt4 instruction-tuning dataset
	- decompose steps -> multi-expert consistence -> LLAVA+ -> seesaw aug -> LLAVA++

- FAITHSCORE: Evaluating Hallucinations in Large Vision-Language Models
	- ans -> recognizer -> descriptive content -> decomposer -> atom fact -> verifier(VEM model) -> yes or no 4 each atom fact

- Detecting and Preventing Hallucinations in Large Vision Language Models
	- trained a reward model that can detect hallucination in a sentence level and segment level.
	- use fdpo to optimize performance, preference one +, hallucination one -, neural one dismiss.

- ALIGNING LARGE MULTIMODAL MODELS WITH FACTUALLY AUGMENTED RLHF
	- crowd force to train reward model without hallucination in a ans-level with both ground-truth label and reward model signal as rl supervision

##### 