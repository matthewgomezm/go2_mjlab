from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from rsl_rl.models.cnn_model import CNNModel
from rsl_rl.modules import RNN, HiddenState
from tensordict import TensorDict


# class for CNN encoding, passed thru RNN, fed 1D into MLP
class CNNRNNModel(CNNModel):
  is_recurrent: bool = True

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    cnn_cfg: dict[str, dict] | dict[str, Any] | None = None,
    cnns: nn.ModuleDict | dict[str, nn.Module] | None = None,
    rnn_type: str = "lstm",
    rnn_hidden_dim: int = 256,
    rnn_num_layers: int = 1,
  ) -> None:
    self.latent_dim = rnn_hidden_dim

    super().__init__(
      obs,
      obs_groups,
      obs_set,
      output_dim,
      hidden_dims,
      activation,
      obs_normalization,
      distribution_cfg,
      cnn_cfg,
      cnns,
    )

    # self.obs_dim (1D groups) and self.cnn_latent_dim (CNN output) are set above
    self.rnn = RNN(self.obs_dim + self.cnn_latent_dim, rnn_hidden_dim, rnn_num_layers, rnn_type)

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
  ) -> torch.Tensor:
    """CNN+1D latent (CNNModel), then through the RNN."""
    latent = super().get_latent(obs)
    latent = self.rnn(latent, masks, hidden_state).squeeze(0)
    return latent

  def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
    self.rnn.reset(dones, hidden_state)

  def get_hidden_state(self) -> HiddenState:
    return self.rnn.hidden_state  # type: ignore

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    self.rnn.detach_hidden_state(dones)

  def _get_latent_dim(self) -> int:
    """Return the latent dimensionality consumed by the MLP head."""
    return self.latent_dim

  def as_jit(self) -> nn.Module:
    """Return a version of the model compatible with Torch JIT export."""
    if isinstance(self.rnn.rnn, nn.LSTM):
      return _TorchCNNLSTMModel(self)
    elif isinstance(self.rnn.rnn, nn.GRU):
      return _TorchCNNGRUModel(self)
    raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn.rnn)}")

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    """Return a version of the model compatible with ONNX export."""
    return _OnnxCNNRNNModel(self, verbose)


class _TorchCNNGRUModel(nn.Module):
  """Exportable CNN+GRU model for JIT."""

  def __init__(self, model: CNNRNNModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.rnn.cpu()
    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

  def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
    latent_1d = self.obs_normalizer(obs_1d)
    latent_cnn = torch.cat([cnn(x) for cnn, x in zip(self.cnns, obs_2d)], dim=-1)
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)
    latent, h = self.rnn(latent.unsqueeze(0), self.hidden_state)
    self.hidden_state[:] = h  # type: ignore
    latent = latent.squeeze(0)
    out = self.mlp(latent)
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    self.hidden_state[:] = 0.0  # type: ignore


class _TorchCNNLSTMModel(nn.Module):
  """Exportable CNN+LSTM model for JIT."""

  def __init__(self, model: CNNRNNModel) -> None:
    super().__init__()
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.register_buffer("hidden_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))
    self.register_buffer("cell_state", torch.zeros(self.rnn.num_layers, 1, self.rnn.hidden_size))

  def forward(self, obs_1d: torch.Tensor, obs_2d: list[torch.Tensor]) -> torch.Tensor:
    latent_1d = self.obs_normalizer(obs_1d)
    latent_cnn = torch.cat([cnn(x) for cnn, x in zip(self.cnns, obs_2d)], dim=-1)
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)
    latent, (h, c) = self.rnn(latent.unsqueeze(0), (self.hidden_state, self.cell_state))
    self.hidden_state[:] = h  # type: ignore
    self.cell_state[:] = c  # type: ignore
    latent = latent.squeeze(0)
    out = self.mlp(latent)
    return self.deterministic_output(out)

  @torch.jit.export
  def reset(self) -> None:
    self.hidden_state[:] = 0.0  # type: ignore
    self.cell_state[:] = 0.0  # type: ignore


class _OnnxCNNRNNModel(nn.Module):
  is_recurrent: bool = True

  def __init__(self, model: CNNRNNModel, verbose: bool) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.cnns = nn.ModuleList([copy.deepcopy(model.cnns[g]) for g in model.obs_groups_2d])
    self.rnn = copy.deepcopy(model.rnn.rnn)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()

    if isinstance(self.rnn, nn.LSTM):
      self.rnn_type = "lstm"
    elif isinstance(self.rnn, nn.GRU):
      self.rnn_type = "gru"
    else:
      raise NotImplementedError(f"Unsupported RNN type: {type(self.rnn)}")

    self.obs_groups_2d = model.obs_groups_2d
    self.obs_dims_2d = model.obs_dims_2d
    self.obs_channels_2d = model.obs_channels_2d
    self.obs_dim_1d = model.obs_dim
    self.hidden_size = self.rnn.hidden_size
    self.num_layers = self.rnn.num_layers

  def forward(self, obs_1d: torch.Tensor, *rest: torch.Tensor):
    """rest = (*obs_2d, h_in[, c_in])."""
    n2d = len(self.obs_groups_2d)
    obs_2d = rest[:n2d]
    h_in = rest[n2d]
    c_in = rest[n2d + 1] if self.rnn_type == "lstm" else None

    latent_1d = self.obs_normalizer(obs_1d)
    latent_cnn = torch.cat([cnn(x) for cnn, x in zip(self.cnns, obs_2d)], dim=-1)
    latent = torch.cat([latent_1d, latent_cnn], dim=-1)

    if self.rnn_type == "lstm":
      x, (h, c) = self.rnn(latent.unsqueeze(0), (h_in, c_in))
    else:
      x, h = self.rnn(latent.unsqueeze(0), h_in)
      c = None

    x = x.squeeze(0)
    out = self.mlp(x)
    out = self.deterministic_output(out)
    if self.rnn_type == "lstm":
      return out, h, c
    return out, h

  def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
    dummy_1d = torch.zeros(1, self.obs_dim_1d)
    dummy_2d = []
    for i in range(len(self.obs_groups_2d)):
      h, w = self.obs_dims_2d[i]
      c = self.obs_channels_2d[i]
      dummy_2d.append(torch.zeros(1, c, h, w))
    h_in = torch.zeros(self.num_layers, 1, self.hidden_size)
    if self.rnn_type == "lstm":
      c_in = torch.zeros(self.num_layers, 1, self.hidden_size)
      return (dummy_1d, *dummy_2d, h_in, c_in)
    return (dummy_1d, *dummy_2d, h_in)

  @property
  def input_names(self) -> list[str]:
    names = ["obs", *self.obs_groups_2d, "h_in"]
    if self.rnn_type == "lstm":
      names.append("c_in")
    return names

  @property
  def output_names(self) -> list[str]:
    names = ["actions", "h_out"]
    if self.rnn_type == "lstm":
      names.append("c_out")
    return names
