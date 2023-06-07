# <p align="center">DecompX</p>

<h3 align="center">
  <p><a href="https://2023.aclweb.org/">[ACL 2023]</a> DecompX:<br>Explaining Transformers Decisions by Propagating Token Decomposition</p>
</h3>

<p align="center">
<!--   <a href="https://arxiv.org/abs/2306.02873"><img alt="Paper" src="https://img.shields.io/badge/ACL2023-Paper-2C4F7C?logo=sqlite&logoColor=red&style=for-the-badge"></a> -->
  <a href="https://arxiv.org/abs/2306.02873"><img alt="Paper" src="https://img.shields.io/badge/ACL2023-Paper-2C4F7C?logo=data:image/png;base64,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&style=flat"></a>
  
  <br>
  <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX"><img alt="Gradio Demo" src="https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue?style=flat"></a>
  <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb"><img alt="Colab Demo" src="https://img.shields.io/badge/​-Open%20in%20Colab-blue?logo=googlecolab&logoColor=F9AB00&style=flat"></a>
  <br>
  <a href="https://youtu.be/kQqjp-Dfb-s"><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000&style=flat"></a>
  <a href="https://github.com/mohsenfayyaz/DecompX/blob/main/DecompX_2023_Slides.pdf"><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=airplayvideo&logoColor=FFBB00&style=flat"></a>
  <a href="https://github.com/mohsenfayyaz/DecompX/blob/main/DecompX_2023_Poster_A0.pdf"><img alt="Poster" src="https://img.shields.io/badge/​-Poster-A493E7?logo=shotcut&logoColor=A493E7&style=flat"></a>
</p>

<!-- <h4 align="center">
  <a href="https://arxiv.org/pdf/">📃 Paper</a> |
  <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX">🤗 Gradio Demo</a> |
  <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb">🖥 Colab Demo</a>
</h3> -->

## Online Demos

| Demo | Link |
|-|-|
| Gradio Demo | Check out our online `Gradio` demo on <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX">HuggingFace Spaces</a> |
| Colab Demo | Check out our `Colab` demo on <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb">Google Colab</a> |

## Abstract

> <div align="justify">An emerging solution for explaining Transformer-based models is to use vector-based analysis on how the representations are formed. However, providing a faithful vector-based explanation for a multi-layer model could be challenging in three aspects: (1) Incorporating all components into the analysis, (2) Aggregating the layer dynamics to determine the information flow and mixture throughout the entire model, and (3) Identifying the connection between the vector-based analysis and the model's predictions. In this paper, we present DecompX to tackle these challenges. DecompX is based on the construction of decomposed token representations and their successive propagation throughout the model without mixing them in between layers. Additionally, our proposal provides multiple advantages over existing solutions for its inclusion of all encoder components (especially nonlinear feed-forward networks) and the classification head. The former allows acquiring precise vectors while the latter transforms the decomposition into meaningful prediction-based values, eliminating the need for norm- or summation-based vector aggregation. According to the standard faithfulness evaluations, DecompX consistently outperforms existing gradient-based and vector-based approaches on various datasets.
</div>

## Citation
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{modarressi-etal-2023-DecompX,
    title = "DecompX: Explaining Transformers Decisions by Propagating Token Decomposition",
    author = "Modarressi, Ali and Fayyaz, Mohsen and Aghazadeh, Ehsan and Yaghoobzadeh, Yadollah and Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics.",
    publisher = "Association for Computational Linguistics",
    year = "2023"
}
```
