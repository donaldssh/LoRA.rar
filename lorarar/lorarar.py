from typing import Optional, Union
import torch
from torch import nn


class MergedLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_merger_value: Optional[float] = 1.0,
        init_merger_value_2: Optional[float] = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        hyper_nn = None,
        args = None,
        part = None
    ):
        super().__init__()
        self.part = part 
        self.weight_1 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.weight_2 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.args = args
        if args.lora_merge_strategy == "lorarar":
            self.merger_1 = torch.ones_like(self.weight_1[0, :], device='cuda:0', dtype=dtype)
            self.merger_2 = torch.ones_like(self.weight_1[0, :], device='cuda:0', dtype=dtype)
        else:
            self.merger_1 = nn.Parameter(
                torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value
            )
            self.merger_2 = nn.Parameter(
                torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value_2
            )
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"
        self.hyper_nn = hyper_nn

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2"]
        self.forward_type = type

    def compute_mergers_similarity(self):
        return nn.functional.cosine_similarity(
            self.merger_1, self.merger_2, dim=0
        ).abs()

    def get_merged_lora_weight(self):
        if self.args.lora_merge_strategy == "ziplora":
            return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2

        elif  self.args.lora_merge_strategy == "lorarar":
            if self.part in ["to_k", "to_v"]:
                return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2
            else:
                batch_weight = torch.concatenate((self.weight_1.T, self.weight_2.T), dim=1)
                merge_coeff = self.hyper_nn(batch_weight)
                self.merger_1 = merge_coeff[:, 0] 
                self.merger_2 = merge_coeff[:, 1]
            return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2

        return self.merger_1.to(self.weight_1.device) * self.weight_1 + self.merger_2.to(self.weight_2.device) * self.weight_2


    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1.dtype
        if self.forward_type == "merge":
            weight = self.get_merged_lora_weight()
        elif self.forward_type == "weight_1":
            weight = self.weight_1
        elif self.forward_type == "weight_2":
            weight = self.weight_2
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class MergedLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        part=None
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
