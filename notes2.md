##### code
- tokenizer.add_tokens(list)
    - register token requiring updating vocab size
    - model.resize_token_embeddings(len(tokenizer))
    - tokenizer.save_pretrained()
    - 执行完tokenizer.save_pretrained(model_dir) 后，你可以在原保存model目录下，查看vocab.txt中[unused]被替换掉，同时也会更新生成新的added_tokens.json，special_tokens_map文件。

- tokenizer()
    - param:
        - one dim list
        - add_special_tokens: False -> do not add [cls] and [sep]
        
- tokenizer.batch_decode()
    - param:
        - skip_special_tokens: True -> skip special tokens when decoding

- method of slice
    - [:, 2], pick all first dimension, only choose index 2 position in second dimension
    - [..., 2], pick all previous dimension(maybe more than one), ...

- casualllmgeneration
    - attention: tuple(torch.floattensor): 32 blocks 1 batch_size 32 heads 625 * 625 dims then 1 * 626...
    - past_key_values: tuple(k, v) (bs, num_seq, hidden_size)
    - *different decoding method has params of input_ids for global and of model_inputs.input_ids for local size=(1, 1)*

- LlamaAttention.forward()
    - hidden_state: bs, seq_len, num_hiddens -> first (1, 40, 4096) then (1, 1, 4096) for instructblip
    - query_states/key_states/value_states: bs, num_heads, seq_len, num_hiddens -> first (1, 32, 40, 128) then (1, 32, 1, 128)
    - past_key_value[0]: bs, num_heads, seq_len, num_hiddens -> first None then (1, 32, 40, 128) then (1, 32, 41, 128)
    - concat(past_key_value[0], key_states, dim=-2) **for next past key value**
    - attn_weights: first bs, num_heads, query_seq_len, key_seq_len -> (1, 32, 40, 40) then (1, 32, 1, 41)
    - attn_output: bs, seq_len, vocab_size -> (1, 40, 4096) then (1, 1, 4096)

- convert_ids_to_tokens
    - only accept 1 dim numpy -> [1, 23, 4, 5, ...]

- model.llm_tokenizer.bos_token_id to get bos_token_id, then get token by model.llm_tokenizer.convert_ids_to_tokens(....to_list())(to_list -> convert numpy array to list | .numpy() -> convert torch array to numpy)

- @decorate execution logitism
    - execute decorate function(use function wrapping the real function)
    - in decorate function call real function
    - **all the params in real function pass to the decorate function**

- exception catch method 
    - raise ValueError(f'Unknown vision tower: {vision_tower}')

- patch_embed
    - bs, patch, hidden_size

- ```pip install -e``` and ```pyproject.toml```
    - when run former, execute the latter.

- an absolute pyproject.toml
```
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "llava"
version = "1.1.1"
description = "Towards GPT-4 like large language and visual assistant."
readme = "README.md"
requires-python = ">=3.8"
classifiers = [
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: Apache Software License",
]
dependencies = [
    "einops", "fastapi", "gradio==3.35.2", "markdown2[all]", "numpy",
    "requests", "sentencepiece", "tokenizers>=0.12.1",
    "torch==2.0.1", "torchvision==0.15.2", "uvicorn", "wandb",
    "shortuuid", "httpx==0.24.0",
    "deepspeed==0.9.5",
    "peft==0.4.0",
    "transformers==4.31.0",
    "accelerate==0.21.0",
    "bitsandbytes==0.41.0",
    "scikit-learn==1.2.2",
    "sentencepiece==0.1.99",
    "einops==0.6.1", "einops-exts==0.0.4", "timm==0.6.13",
    "gradio_client==0.2.9"
]

[project.urls]
"Homepage" = "https://llava-vl.github.io"
"Bug Tracker" = "https://github.com/haotian-liu/LLaVA/issues"

# expose module
[tool.setuptools.packages.find]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]

[tool.wheel]
exclude = ["assets*", "benchmark*", "docs", "dist*", "playground*", "scripts*", "tests*"]
```

- llava -> vision tower + vicuna

- generation_config.json derive from config.json if there is no generation_config.json in model folder
```
{
  "bos_token_id": 1,
  "eos_token_id": 2,
  "max_length": 4096,
  "pad_token_id": 0,
  "transformers_version": "4.31.0"
}
```

- python class init sample
    - ```
    class G:
        def test(self):
            print('123')
    class E:
        def __init__(self, config):
                super().__init__(config)
                print('class E')
            def test(self):
                print('hello')
        class D(E):
            t = 3
            def __init__(self, config):
                super().__init__(config)
                print('class D~')
        class A(D):
            def __init__(self, config):
                super(A, self).__init__(config)
                self.a = config.get('a')
                print('class A')
        class B(G):
            def __init__(self, config):
                super(B, self).__init__()
                self.b = 32
                print('class B')
        class C(A, B):
            def __init__(self, config):
                super().__init__(config)
                print('class C')
        config = {'a': 'test'}
        t = C(config)
        ```
    - output: 
        class B
        class E
        class D~
        class A
        class C
    - sequence:
        - CADEBG

- __init__: LlavaLlamaForCausalLM -> PreTrainedModel -> Module(/mnt/petrelfs/weixilin/miniconda3/envs/minigpt4/lib/python3.9/site-packages/torch/nn/modules/module.py)

- torch.numel(): count the amount of number 
- 