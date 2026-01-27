# KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference

Official implementation of the ICML25 paper: KVTuner: Sensitivity-Aware Layer-wise Mixed Precision KV Cache Quantization for Efficient and Nearly Lossless LLM Inference

## Installation
```sh
cd flexible_quant
pip install -e .
```

## Run Example codes
```sh
python3 flexible_quant_example.py
```
Then you will run a simple example from `GSM8K` with `meta-llama/Meta-Llama-3-8B` and `KV4` quantization.

Change line 17 in `flexible_quant_example.py` to run different quantization methods.

### Run GSM8K
```bash
cd benchmarks
# GSM8K K8V4 with KiVi quantization scheme
python3 example_gsm8k_cot_manyshot.py --model_name="mistralai/Mistral-7B-Instruct-v0.2" --k_bits=8 --v_bits=4 --residual_length=32 --group_size=32 --axis_key=1 --axis_value=0
# GSM8K K8V4 with Per-Token quantization scheme
python3 example_gsm8k_cot_manyshot.py --model_name="mistralai/Mistral-7B-Instruct-v0.2" --k_bits=8 --v_bits=4 --residual_length=0 --group_size=-1 --axis_key=0 --axis_value=0
```

##### Parameters

- `model_name`: the model name from Hugging Face model hub.
- `nshots`: the number of shots for the few-shot inference.
- `k_bits`: the precision for the key.
- `v_bits`: the precision for the value.
- `asym`: whether to use asymmetric quantization.
- `residual_length`: the length of the residual tokens which are not quantized. must be a multiple of `group_size`, use 0 for per-token quantization.
- `group_size`: the size of the group for quantization, use -1 for per-token quantization.
- `axis_key`: the axis for key quantization, 0 for per-token quantization, 1 for per-channel quantization.
- `axis_value`: the axis for value quantization, 0 for per-token quantization, 1 for per-channel quantization.


#### Run LongBench
```sh
cd benchmarks
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 pred_longbench.py
```

Same parameters as GSM8K.

#### Run lm-eval

This repo provides a modified version of `lm-eval` to support the quantization evaluation.

Refer to `lm-evaluation-harness-X/lm_eval/models/huggingface_quant.py`

## Use `FlexibleQuantizedCache` in your code
```python
# Define your model
from transformers import AutoTokenizer, AutoModelForCausalLM, 
model_name = 'meta-llama/Meta-Llama-3-8B'
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, trust_remote_code=True)

# Define the cache
from flexible_quant.flexible_quantized_cache import FlexibleQuantizedCacheConfig, FlexibleVanillaQuantizedCache
cache_config = FlexibleQuantizedCacheConfig(nbits_key=4, nbits_value=4, asym=True, axis_key=0, axis_value=0, device='cuda', q_group_size=-1)
# By default we use FlexibleVanillaQuantizedCache, you can switch to FlexibleHQQQuantizedCache and FlexibleQuantoQuantizedCache
past_key_values = FlexibleVanillaQuantizedCache(cache_config=cache_config)

# Prompt and generate
prompt = '''The quick brown fox jumps over the lazy dog.'''
inputs = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
outputs = model.generate(inputs, past_key_values=past_key_values, use_cache=True, max_new_tokens=256)
```


## FlexibleQuantizedCacheConfig

```python
"""
Configuration for flexible quantized cache.

Attributes:
    backend (str): Backend for quantization. Options: "quanto", "hqq", "vanilla".
    nbits (Optional[int]): Precision for both key and value. Used if `nbits_key` and `nbits_value` are not set.
                            For per-layer or per-head quantization, set `nbits` to -1.
    nbits_key (Optional[int]): Precision for key quantization. For per-layer or per-head quantization, set to -1.
    nbits_value (Optional[int]): Precision for value quantization. For per-layer or per-head quantization, set to -1.
    axis_key (Optional[int]): Axis for key quantization. In Vanilla mode:
                                - 0: Per-token quantization
                                - 1: Per-channel quantization
    axis_value (Optional[int]): Axis for value quantization. In Vanilla mode:
                                - 0: Per-token quantization
                                - 1: Per-channel quantization
    asym (Optional[bool]): Whether to use asymmetric quantization. Works only for Vanilla mode.
    q_group_size (Optional[int]): Group size for quantization. Use -1 for per-token quantization.
    residual_length (Optional[int]): Length of residual tokens that are not quantized.
                                        Must be a multiple of `q_group_size`. Use 0 for per-token quantization.
    compute_dtype (Optional[torch.dtype]): Compute dtype for the model. Default: `torch.float16`.
    device (Optional[str]): Device for the cache. Default: `"cpu"`.
    force_quant (Optional[bool]): Whether to quantize the cache during the pre-filling stage.
    per_layer_quant (Optional[bool]): Whether to use per-layer quantization.
    per_layer_config (Optional[Dict[str, Any]]): If `per_layer_quant` is True, provides the quantization config
                                                    for each layer. Alternatively, use `per_layer_config_path`.
    per_layer_config_path (Optional[str]): Path to the quantization config for each layer.
                                            Used if `per_layer_quant` is True.
    per_head_quant (Optional[bool]): Whether to use per-head quantization.
    per_head_config (Optional[Dict[str, Any]]): If `per_head_quant` is True, provides the quantization config
                                                for each head. Alternatively, use `per_head_config_path`.
    per_head_config_path (Optional[str]): Path to the quantization config for each head.
                                            Used if `per_head_quant` is True.
"""
```

### Example for per_layer_config
```python
per_layer_config = {
    {n_layer}: {
        'nbits_key': 4,
        'nbits_value': 4,
    },
    # ...
```

### Example for per_head_config
```python
per_head_config = {
    {n_layer}: {
        {head_idx}: {
            'nbits_key': 4,
            'nbits_value': 4,
        },
        # ...
    },
    # ...
```

## Citation
```bibtex
@inproceedings{
li2025kvtuner,
title={{KVT}uner: Sensitivity-Aware Layer-Wise Mixed-Precision {KV} Cache Quantization for Efficient and Nearly Lossless {LLM} Inference},
author={Xing Li and Zeyu XING and Yiming Li and Linping Qu and Hui-Ling Zhen and Yiwu Yao and Wulong Liu and Sinno Jialin Pan and Mingxuan Yuan},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=zDwipF6h06}
}
```
