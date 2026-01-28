import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


class MambaBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            bias=True,
            groups=self.d_inner,
            padding=d_conv - 1,
        )

        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)

        A = repeat(
            torch.arange(1, d_state + 1),
            "n -> d n",
            d=self.d_inner,
        ).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        residual = x

        x_and_res = self.in_proj(x)
        x, res = x_and_res.split(split_size=self.d_inner, dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, : x.shape[-1]]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x, mask)

        y = y * F.silu(res)

        output = self.out_proj(y)
        output = self.dropout(output)
        output = self.norm(residual + output)

        return output

    def ssm(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        A = -torch.exp(self.A_log.float())
        D = self.D.float()

        deltaBC = self.x_proj(x)
        delta, B = deltaBC.split(split_size=self.d_state, dim=-1)

        delta = F.softplus(self.dt_proj(x))

        if mask is not None:
            delta = delta * mask.unsqueeze(-1)
            B = B * mask.unsqueeze(-1)

        y = self.selective_scan(x, delta, A, B, D)

        return y

    def selective_scan(
        self,
        u: torch.Tensor,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        D: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, d_inner = u.shape
        n = A.shape[1]

        deltaA = torch.exp(delta.unsqueeze(-1) * A)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)

        x = torch.zeros(batch, d_inner, n, device=u.device, dtype=u.dtype)
        ys = []

        for i in range(length):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = x @ B[:, i].unsqueeze(-1)
            ys.append(y.squeeze(-1))

        y = torch.stack(ys, dim=1)
        y = y + u * D

        return y
