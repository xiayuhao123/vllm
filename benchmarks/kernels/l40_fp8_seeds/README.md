# L40S search seeds, not installed L40 configurations

Source: [vLLM PR #58005](https://github.com/vllm-project/vllm/pull/58005),
head `b35784d5c567741992799aaae386d140f0ffc6ca`.
Original files are under `vllm/model_executor/layers/quantization/utils/configs/`.

These Apache-2.0 upstream JSON files were measured on **L40S**. Their filenames
and contents are preserved. The L40 tuner uses the nearest M entry only as an
additional search candidate. They are not placed in the runtime config directory.
Passing L40 correctness checks and collecting L40 timings are required before
using any exported configuration.

| Original filename | SHA-256 |
| --- | --- |
| `N=17408,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json` | `1337fbe3c13269ecb488707ae7d3ec8c0c46263104bad91ac78589cd4abadaae` |
| `N=5120,K=3072,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json` | `4e3650eb584a40ab44b5e0d12c66b00c9168f8c1ab4a8b94c47a63dae458f809` |
| `N=5120,K=8704,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json` | `c7759a29420d7fc074f00c77a716894259e542d5ca6eb69df39be0eee70b2286` |
| `N=7168,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json` | `808ae6910b419276102644cfaf701a9dd0edb075918484bf927f2a7b23344100` |
| `N=8192,K=5120,device_name=NVIDIA_L40S,dtype=fp8_w8a8,block_shape=[128,128].json` | `89687ddf2977b6cb3051140e88fc32896eda24d0ca73070ffdb3853cd892038c` |
