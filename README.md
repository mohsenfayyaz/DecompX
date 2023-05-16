# <p align="center">DecompX</p>

<h3 align="center">
  <p><a href="https://2023.aclweb.org/">[ACL 2023]</a> DecompX:<br>Explaining Transformers Decisions by Propagating Token Decomposition</p>
</h3>

<p align="center">
  <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX"><img alt="Gradio Demo" src="https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue"></a>
  <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb"><img alt="Colab Demo" src="https://img.shields.io/badge/​-Open%20in%20Colab-blue?logo=googlecolab&logoColor=F9AB00"></a>
  <br>
  <a href=""><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-808080"></a>
  <a href=""><img alt="Video" src="https://img.shields.io/badge/​-Video-red?logo=youtube&logoColor=FF0000"></a>
  <a href=""><img alt="Slides" src="https://img.shields.io/badge/​-Slides-FFBB00?logo=googlesheets&logoColor=FFBB00"></a>
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
@inproceedings{}
```
