# TeaCache+xDiT

This repository implements multi-GPU parallelization of TeaCache within the xDiT framework. The supported models include:

- [**CogVideoX**](https://github.com/THUDM/CogVideo)
- [**ConsisID**](https://github.com/PKU-YuanGroup/ConsisID)
- [**HunyuanVideo**](https://github.com/Tencent/HunyuanVideo)
- [**Flux.1 [dev]**](https://github.com/black-forest-labs/flux)
- [**Wan 2.1**](https://github.com/Wan-Video/Wan2.1)

Please run the appropriate script (`run_[your_model].sh`) to get started.

For env, same as FastCache-xDiT, but we need follows to complete

```bash
huggingface-cli login
pip install protobuf
chmod +x run_flux.sh
./run_flux.sh
```


### Note: All following experiments are conducted on the A800

### Performance on Wan 2.1
|Model | Method | 8x |
|-------|-----|-----|
| Wan 2.1-T2V-14B | USP | 473 |
||TeaCache(0.2) | 241 |


### Performance on CogVideoX
| Model | Method | 1x | 2x | 4x | 8x |
|-------|--------|-----|-----|-----|-----|
| CogvideoX | Ulysses | 133.5 | 77.7 | 59.41 | 36.68 |
| | TeaCache(slow) | 103.3 | 60.9 | 46.8 | 29.6 |
| | TeaCache(fast) | 78 | 46.7 | 36.5 | 23.7 |

###  Performance on HunyuanVideo
| Model | Method | 1x | 2x | 4x | 8x |
|-------|--------|-----|-----|-----|-----|
| HunyuanVideo | Ulysses | 3086.5 | 1624.4 | 856.6 | 480.4 |
| | TeaCache(slow) | 1884.5 | 1001 | 540.8 | 312.9 |
| | TeaCache(fast) | 1395.2 | 753.9 | 418.4 | 254.1 |

###  Performance on Flux
| Model | Method | 1x | 2x | 4x | 8x |
|-------|--------|-----|-----|-----|-----|
| Flux | Ulysses | 12.9 | 7.91 | 4.4 | 3.61 |
| | TeaCache(slow) | 12.1 | 4.61 | 2.68 | 2.24 |
| | TeaCache(fast) | 7.9 | 2.45 | 1.53 | 1.3 |

###  Performance on ConsistID
| Model | Method | 1x | 2x | 3x | 6x |
|-------|--------|-----|-----|-----|-----|
| ConsistID | Ulysses | 218.64 | 123.84 | 88.6 | 54.09 |
| | TeaCache(slow)-0.1 | 142.18 | 81.39 | 60.71 | 39.77 |
| | TeaCache(fast)-0.2 | 90.34 | 53.82 | 41.34 | 27.18 |


# Acknowledgement
This respotory is built based on [TeaCache](https://github.com/ali-vilab/TeaCache/tree/main) and [xDiT](https://github.com/xdit-project/xDiT?tab=readme-ov-file), thanks for their contributors!
