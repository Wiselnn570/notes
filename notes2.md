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