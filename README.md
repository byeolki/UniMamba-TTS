# UniMamba-TTS

Non-autoregressive text-to-speech with unidirectional Mamba state space models.

## Description

This repository explores replacing Transformer feed-forward blocks in FastSpeech2 with unidirectional Mamba blocks for efficient speech synthesis. Mamba, based on state space models (SSMs), offers linear-time sequence modeling as an alternative to the quadratic complexity of self-attention mechanisms.

UniMamba-TTS maintains the FastSpeech2 architecture while substituting the decoder's Transformer blocks with Mamba layers, enabling more memory-efficient processing of long speech sequences. This implementation is developed for research on applying state space models to text-to-speech synthesis.

## Citation

FastSpeech2:
```bibtex
@inproceedings{ren2021fastspeech2,
  title={FastSpeech 2: Fast and High-Quality End-to-End Text to Speech},
  author={Ren, Yi and Hu, Chenxu and Tan, Xu and Qin, Tao and Zhao, Sheng and Zhao, Zhou and Liu, Tie-Yan},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```

Mamba:
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
