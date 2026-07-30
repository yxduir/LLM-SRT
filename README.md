# LLM-SRT
LLM-SRT is a variant of [SLAM-LLM](https://github.com/X-LANCE/SLAM-LLM), primarily focusing on Speech Translation tasks.
- **License**: CC BY-NC-SA-4.0

# News
- [Update Jul. 30, 2026] The paper "MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages" was published in IEEE/ACM TASLP.
- [Update Jan. 26, 2026] The paper "Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion" was accepted by ICLR 2026.
- [Update Jan. 15, 2026] The paper "SLAM-LLM: A Modular, Open-Source Multimodal Large Language Model Framework and Best Practice for Speech, Language, Audio and Music Processing" was accepted by JSTSP.
- [Update May 16, 2025] The paper "Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning" was accepted by ACL 2025.

---

| Model | Paper | Code |
| :--- | :--- | :--- |
| **LLM-SRT** | Making LLMs Better Many-to-Many Speech-to-Text Translators with Curriculum Learning | [![GitHub](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/LLM-SRT/blob/main/readme/LLM_SRT.md)|
| **SMT** | <nobr>Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion</nobr> | [![GitHub](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/LLM-SRT/blob/main/readme/SMT.md) [![HF](https://img.shields.io/badge/%F0%9F%A4%97%20-HF-FFD21E?style=flat)](https://huggingface.co/yxdu/smt-9b-hf) |
| **MCAT** | MCAT: Scaling Many-to-Many Speech-to-Text Translation with MLLMs to 70 Languages | [![GitHub](https://img.shields.io/badge/Github-181717?style=flat&logo=github&logoColor=white)](https://github.com/yxduir/m2m-70) |


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
@ARTICLE{11481964,
  author={Du, Yexing and Liu, Kaiyuan and Pan, Youcheng and Yang, Bo and Deng, Keqi and Chen, Xie and Xiang, Yang and Liu, Ming and Qin, Bing and Wang, YaoWei},
  journal={IEEE Transactions on Audio, Speech and Language Processing}, 
  title={MCAT: Scaling Many-to-Many Speech-to-Text Translation With MLLMs to 70 Languages}, 
  year={2026},
  volume={34},
  number={},
  pages={2876-2887},
  keywords={Feeds;Radio broadcasting;Frequency modulation;LoRa;Electronic mail;Video games;Videos;Internet;Video equipment;Modulation;Speech-to-text translation;multimodal large language models;curriculum learning},
  doi={10.1109/TASLPRO.2026.3684396}
}

@inproceedings{duscalable,
  title={Scalable Multilingual Multimodal Machine Translation with Speech-Text Fusion},
  author={Du, Yexing and Pan, Youcheng and Wang, Zekun and Chu, Zheng and Huang, Yichong and Liu, Kaiyuan and Yang, Bo and Xiang, Yang and Liu, Ming and Qin, Bing},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
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
