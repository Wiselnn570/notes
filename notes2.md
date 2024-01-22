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

- model.llm_tokenizer.bos_token_id to get bos_token_id, then get token by model.llm_tokenizer.convert_ids_to_tokens(....to_list())(to_list -> convert numpy array to list)
