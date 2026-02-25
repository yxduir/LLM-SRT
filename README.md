# LLM-SRT
LLM-SRT is is a variant of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM), primarily focusing on Speech Translation tasks.
- **License**: CC BY-NC-SA-4.0

# News
- [Update Jan. 26, 2026] The paper "Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion" was accepted by ICLR 2026.
- [Update Jan. 15, 2026] The paper "SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing" was accept by JSTSP.
- [Update May 16, 2025] The paper "Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning" was accepted by ACL 2025.

---

| Model | Paper | Code |
| --- | --- | --- |
| **LLM-SRT** | [Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning](https://arxiv.org/pdf/2409.19510) | [Code](https://github.com/yxduir/LLM-SRT/blob/main/readme/LLM_SRT.md) |
| **SMMT** | [Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion](https://arxiv.org/pdf/2409.19510) | [Code](https://github.com/yxduir/LLM-SRT/blob/main/readme/SMT.md) |
| **MCAT** | [MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages](https://arxiv.org/abs/2512.01512) | [Code](https://github.com/yxduir/m2m-70) |

---

## Installation
```
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

git clone https://github.com/yxduir/LLM-SRT
cd LLM-SRT
uv venv --python 3.10
source .venv/bin/activate

cd SLAM-LLM
uv pip install -r requirements.txt
uv pip install -e .
cd ..
```




##  Citation
```
@article{du2025mcat,
  title={MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages},
  author={Du, Yexing and Liu, Kaiyuan and Pan, Youcheng and Yang, Bo and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bin and Wang, YaoWei},
  journal={arXiv preprint arXiv:2512.01512},
  year={2025}
}

@inproceedings{du2025making,
  title={Making llms better many-to-many speech-to-text translators with curriculum learning},
  author={Du, Yexing and Pan, Youcheng and Ma, Ziyang and Yang, Bo and Yang, Yifan and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={12466--12478},
  year={2025}
}

@article{ma2026slam,
  title={SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing},
  author={Ma, Ziyang and Yang, Guanrou and Chen, Wenxi and Gao, Zhifu and Du, Yexing and Li, Xiquan and Zheng, Zhisheng and Zhu, Haina and Zhuo, Jianheng and Song, Zheshu and others},
  journal={IEEE Journal of Selected Topics in Signal Processing},
  year={2026},
  publisher={IEEE}
}
```
