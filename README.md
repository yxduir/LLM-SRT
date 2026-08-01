<div align="center">

# LLM-SRT

**Speech-to-Text Translation with Multimodal Large Language Models**

[![License](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey)](LICENSE)
[![GitHub repo](https://img.shields.io/badge/GitHub-yxduir%2FLLM--SRT-blue?logo=github)](https://github.com/yxduir/LLM-SRT)

[English](README.md)

</div>

---

## 📋 Overview

LLM-SRT is a speech translation toolkit built upon [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM), featuring a series of models for scalable many-to-many speech-to-text translation.

## 📰 News

- **2026-05-28**: [ESRT-4B](https://huggingface.co/yxdu/ESRT-4B) released on Hugging Face.
- **2026-04-15**: MCAT paper published in IEEE Transactions on Audio, Speech and Language Processing (TASLP).
- **2026-01-26**: SMT paper accepted by ICLR 2026.
- **2026-01-15**: SLAM-LLM paper accepted by IEEE Journal of Selected Topics in Signal Processing (JSTSP).
- **2025-05-16**: LLM-SRT paper accepted by ACL 2025.

## 🚀 Models

| Model             | Paper · Models & Code                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        | Features                                      |
| :---------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------- |
| **ESRT** 🆕 | [**Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation**](https://arxiv.org/abs/2605.28642) `<br>` Y. Du, K. Liu, Y. Pan, B. Yang, M. Liu, B. Qin, Y. Xiang `<br>` *arXiv, 2026* `<br>` [![Code](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/ESRT) [![Models](https://img.shields.io/badge/%F0%9F%A4%97%20-Models-FFD21E?style=flat)](https://huggingface.co/yxdu/ESRT-4B)                                               | edge device<br />1B/4B/12B · 45 languages  |
| **MCAT**    | [**MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages**](https://arxiv.org/abs/2512.01512) `<br>` Y. Du, K. Liu, Y. Pan, B. Yang, K. Deng, X. Chen, Y. Xiang, M. Liu, B. Qin, Y. Wang `<br>` *IEEE TASLP, 2026* `<br>` [![Code](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/m2m-70)                                                                                                                                    | 9B 28 languages<br />27B · 70 languages      |
| **SMT**     | [**Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion**](https://arxiv.org/abs/2602.21646) `<br>` Y. Du, Y. Pan, Z. Wang, Z. Chu, Y. Huang, K. Liu, B. Yang, Y. Xiang, M. Liu, B. Qin `<br>` *ICLR, 2026* `<br>` [![Code](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/LLM-SRT/blob/main/readme/SMT.md) [![Models](https://img.shields.io/badge/%F0%9F%A4%97%20-Models-FFD21E?style=flat)](https://huggingface.co/yxdu/smt-9b-hf) | 9B 28 languages                              |
| **LLM-SRT** | [**Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning**](https://arxiv.org/abs/2409.19510) `<br>` Y. Du, Y. Pan, Z. Ma, B. Yang, Y. Yang, K. Deng, X. Chen, Y. Xiang, M. Liu, B. Qin `<br>` *ACL, 2025* `<br>` [![Code](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/LLM-SRT/blob/main/readme/LLM_SRT.md)                                                                                                            | 3B/7B/32B 15 languages                        |

## 🔧 Installation

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/yxduir/LLM-SRT
cd LLM-SRT

# Create and activate virtual environment
uv venv --python 3.10
source .venv/bin/activate

# Install dependencies
cd SLAM-LLM
uv pip install -r requirements.txt
uv pip install -e .
cd ..
```

## 📖 Citation

If you find this project helpful for your research, please consider citing the relevant papers:

```bibtex
@misc{du2026bandwidthefficientprivacypreservingedgecloudmanytomany,
      title={Bandwidth-Efficient and Privacy-Preserving Edge-Cloud Many-to-Many Speech Translation}, 
      author={Yexing Du and Kaiyuan Liu and Youcheng Pan and Bo Yang and Ming Liu and Bing Qin and Yang Xiang},
      year={2026},
      eprint={2605.28642},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.28642}, 
}

@ARTICLE{11481964,
  author={Du, Yexing and Liu, Kaiyuan and Pan, Youcheng and Yang, Bo and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing and Wang, YaoWei},
  journal={IEEE Transactions on Audio, Speech and Language Processing},
  title={MCAT: Scaling Many-to-Many Speech-to-Text Translation With MLLMs to 70 Languages},
  year={2026},
  volume={34},
  pages={2876-2887},
  doi={10.1109/TASLPRO.2026.3684396}
}

@inproceedings{duscalable,
  title={Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion},
  author={Du, Yexing and Pan, Youcheng and Wang, Zekun and Chu, Zheng and Huang, Yichong and Liu, Kaiyuan and Yang, Bo and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}

@inproceedings{du2025making,
  title={Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning},
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

## 📄 License

This project is licensed under the **CC BY-NC-SA 4.0** license. See the [LICENSE](LICENSE) file for details.
