# How to test - 2.5 section BPE
```sh
uv run cs336_basics/test_bpe_tokenizer.py

uv run pytest tests/test_train_bpe.py
```

# How to test 2.6.1 Encoding Text
```sh
uv run pytest tests/test_tokenizer.py
```


# How to test 3.4
```sh
uv run pytest -k test_linear
uv run pytest -k test_embedding
```

# Test 3.5
```sh
uv run pytest -k test_rmsnorm
uv run pytest -k test_swiglu
uv run pytest -k test_rope
uv run pytest -k test_softmax_matches_pytorch

uv run pytest -k test_scaled_dot_product_attention
uv run pytest -k test_4d_scaled_dot_product_attention
uv run pytest -k test_multihead_self_attention
```

# Test 3.6
```sh
uv run pytest -k test_transformer_block
uv run pytest -k test_transformer_lm
```

# Test 4.1
```sh
uv run pytest -k test_cross_entropy
```