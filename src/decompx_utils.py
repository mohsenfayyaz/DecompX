import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


@dataclass
class DecompXConfig():
    include_biases: Optional[bool] = True
    bias_decomp_type: Optional[str] = "absdot"  # "absdot": Based on the absolute value of dot products | "norm": Based on the norm of the attribution vectors | "equal": equal decomposition | "abssim": Based on the absolute value of cosine similarites | "cls": add to cls token
    include_bias_token: Optional[bool] = False  # Adds an extra input token as a bias in the attribution vectors
    # If the bias_decomp_type is None and include_bias_token is True then the final token in the input tokens of the attr. vectors will be the summation of the biases
    # Otherwise the bias token will be decomposed with the specified decomp type

    include_LN1: Optional[bool] = True

    include_FFN: Optional[bool] = True
    FFN_approx_type: Optional[str] = "GeLU_ZO"  # "GeLU_LA": GeLU-based linear approximation | "ReLU": Using ReLU as an approximation | "GeLU_ZO": Zero-origin slope approximation
    FFN_fast_mode: Optional[bool] = False

    include_LN2: Optional[bool] = True

    aggregation: Optional[str] = None  # None: No aggregation | vector: Vector-based aggregation | rollout: Norm-based rollout aggregation

    include_classifier_w_pooler: Optional[bool] = True
    tanh_approx_type: Optional[str] = "ZO"  # "ZO": Zero-origin slope approximation | "LA": Linear approximation

    output_all_layers: Optional[bool] = False  # True: Output all layers | False: Output only last layer
    output_attention: Optional[str] = None  # None | norm | vector | both
    output_res1: Optional[str] = None  # None | norm | vector | both
    output_LN1: Optional[str] = None  # None | norm | vector | both
    output_FFN: Optional[str] = None  # None | norm | vector | both
    output_res2: Optional[str] = None  # None | norm | vector | both
    output_encoder: Optional[str] = None  # None | norm | vector | both
    output_aggregated: Optional[str] = None  # None | norm | vector | both
    output_pooler: Optional[str] = None  # None | norm | vector | both

    output_classifier: Optional[bool] = True


@dataclass
class DecompXOutput():
    attention: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    res1: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    LN1: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    FFN: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    res2: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    encoder: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    aggregated: Optional[Union[Tuple[torch.Tensor, Tuple[torch.Tensor]], Tuple[torch.Tensor], torch.Tensor]] = None
    pooler: Optional[Union[Tuple[torch.Tensor], torch.Tensor]] = None
    classifier: Optional[torch.Tensor] = None
