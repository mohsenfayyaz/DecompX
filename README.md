# <p align="center">DecompX</p>

<h3 align="center">
  <p><a href="https://2023.aclweb.org/">[ACL 2023]</a> DecompX: Explaining Transformers Decisions by Propagating Token Decomposition</p>
</h3>

<p align="center">
  <a href="https://arxiv.org/pdf/"><img alt="Paper" src="https://img.shields.io/badge/📃-Paper-blue"></a>
  <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX"><img alt="Gradio Demo" src="https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue"></a>
  <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb"><img alt="Colab Demo" data-canonical-src="https://colab.research.google.com/assets/colab-badge.svg" src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667"></a>
</p>

<!-- <h4 align="center">
  <a href="https://arxiv.org/pdf/">📃 Paper</a> |
  <a href="https://huggingface.co/spaces/mohsenfayyaz/DecompX">🤗 Gradio Demo</a> |
  <a href="https://colab.research.google.com/github/mohsenfayyaz/DecompX/blob/main/DecompX_Colab_Demo.ipynb">🖥 Colab Demo</a>
</h3> -->

## Abstract
> An emerging solution for explaining Transformer-based models is to use vector-based analysis on how the representations are formed. However, providing a faithful vector-based explanation for a multi-layer model could be challenging in three aspects: (1) Incorporating all components into the analysis, (2) Aggregating the layer dynamics to determine the information flow and mixture throughout the entire model, and (3) Identifying the connection between the vector-based analysis and the model's predictions. 
In this paper, we present \emph{DecompX} to tackle these challenges. 
DecompX is based on the construction of decomposed token representations and their successive propagation throughout the model without mixing them in between layers.
Additionally, our proposal provides multiple advantages over existing solutions for its inclusion of all encoder components (especially nonlinear feed-forward networks) and the classification head. The former allows acquiring precise vectors while the latter transforms the decomposition into meaningful prediction-based values, eliminating the need for norm- or summation-based vector aggregation.
According to the standard faithfulness evaluations, DecompX consistently outperforms existing gradient-based and vector-based approaches on various datasets.

## Citation
If you found this work useful, please consider citing our paper:
```bibtex
@inproceedings{}
```
