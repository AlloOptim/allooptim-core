import logging
import time
from abc import ABC, abstractmethod

import numpy as np
from pydantic import BaseModel
from tinygrad import Tensor, nn
from tinygrad.nn.optim import Adam
from tinygrad.nn.state import get_parameters

logger = logging.getLogger(__name__)

# Constants for technical analysis and training parameters
PRICE_DATA_DIMENSIONS_2D = 2  # (batch, features) format
VOLATILITY_WINDOW_DAYS = 10
RSI_WINDOW_DAYS = 14
MIN_REPLAY_BUFFER_SIZE = 32


class FractionalDifferentiator:
    """
    Fractional differentiation for preserving memory while achieving stationarity.
    Based on "Advances in Financial Machine Learning" by Marcos López de Prado.
    """

    def __init__(self, d=0.5, threshold=1e-5):
        """
        Args:
            d: Differentiation order (0 < d < 1)
               d=0: no differencing (original series)
               d=0.5: balance between memory and stationarity
               d=1: standard first difference (loses all memory)
            threshold: Weight threshold for truncation
        """
        self.d = d
        self.threshold = threshold
        self.weights = None

    def _get_weights(self, n_terms):
        """Compute weights for fractional differentiation"""
        if self.weights is not None and len(self.weights) >= n_terms:
            return self.weights[:n_terms]

        # Calculate weights using the formula
        weights = np.zeros(n_terms)
        weights[0] = 1.0

        for k in range(1, n_terms):
            weights[k] = -weights[k - 1] * (self.d - k + 1) / k

            # Stop if weight becomes too small
            if abs(weights[k]) < self.threshold:
                weights = weights[:k]
                break

        # Normalize weights to preserve scale
        weights = weights / np.sum(np.abs(weights))

        self.weights = weights
        return weights

    def transform(self, series):
        """
        Apply fractional differentiation to a time series

        Args:
            series: (n_samples,) array
        Returns:
            frac_diff: (n_samples,) fractionally differentiated series
        """
        n = len(series)
        weights = self._get_weights(n)
        n_weights = len(weights)

        # Apply weights via convolution
        frac_diff = np.zeros(n)
        for i in range(n_weights, n):
            frac_diff[i] = np.dot(weights, series[i - n_weights + 1 : i + 1][::-1])

        # Fill initial values with NaN or zeros
        frac_diff[:n_weights] = 0  # or np.nan

        return frac_diff

    def fit_transform(self, series):
        """Convenience method"""
        return self.transform(series)


# ============================================================================
# PORTFOLIO NETWORK INTERFACE
# ============================================================================


class PortfolioNetInterface(ABC):
    """
    Abstract base class for portfolio optimization networks.
    All implementations must provide the same interface for seamless swapping.
    """

    @abstractmethod
    def __call__(self, x, train=True):
        """
        Forward pass of the network.

        Args:
            x: (batch, n_assets, seq_len, n_features) - Input features
            train: bool - Whether to use dropout (True for training and MC dropout)

        Returns:
            weights: (batch, n_assets) - Portfolio weights (long-only, 0 to 1 per asset)
            invested_ratio: (batch,) - Fraction of capital invested (0 to 1)
            pred_returns: (batch, n_assets) - Predicted returns for each asset
            pred_vols: (batch, n_assets) - Predicted volatility for each asset
        """
        pass


# ============================================================================
# OPTION 1: LSTM + TRANSFORMER (Original)
# ============================================================================


class MultiHeadAttention:
    """Multi-head self-attention for temporal patterns"""

    def __init__(self, embed_dim, num_heads):
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def __call__(self, x):
        batch, seq_len, embed_dim = x.shape

        # Project and reshape for multi-head
        q = self.q_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim**0.5)
        attn = scores.softmax(axis=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, embed_dim)

        return self.out_proj(out)


class LSTMCell:
    """Custom LSTM cell implementation for tinygrad"""

    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim

        # Input gate
        self.W_i = Tensor.glorot_uniform(input_dim, hidden_dim)
        self.U_i = Tensor.glorot_uniform(hidden_dim, hidden_dim)
        self.b_i = Tensor.zeros(hidden_dim)

        # Forget gate
        self.W_f = Tensor.glorot_uniform(input_dim, hidden_dim)
        self.U_f = Tensor.glorot_uniform(hidden_dim, hidden_dim)
        self.b_f = Tensor.zeros(hidden_dim)

        # Cell gate
        self.W_c = Tensor.glorot_uniform(input_dim, hidden_dim)
        self.U_c = Tensor.glorot_uniform(hidden_dim, hidden_dim)
        self.b_c = Tensor.zeros(hidden_dim)

        # Output gate
        self.W_o = Tensor.glorot_uniform(input_dim, hidden_dim)
        self.U_o = Tensor.glorot_uniform(hidden_dim, hidden_dim)
        self.b_o = Tensor.zeros(hidden_dim)

    def __call__(self, x, h_prev, c_prev):
        """
        Args:
            x: (batch, input_dim)
            h_prev: (batch, hidden_dim)
            c_prev: (batch, hidden_dim)
        """
        # Input gate
        i = (x @ self.W_i + h_prev @ self.U_i + self.b_i).sigmoid()

        # Forget gate
        f = (x @ self.W_f + h_prev @ self.U_f + self.b_f).sigmoid()

        # Cell gate
        c_tilde = (x @ self.W_c + h_prev @ self.U_c + self.b_c).tanh()

        # Output gate
        o = (x @ self.W_o + h_prev @ self.U_o + self.b_o).sigmoid()

        # New cell state
        c = f * c_prev + i * c_tilde

        # New hidden state
        h = o * c.tanh()

        return h, c


class LSTM:
    """Custom LSTM layer for tinygrad"""

    def __init__(self, input_dim, hidden_dim):
        self.hidden_dim = hidden_dim
        self.cell = LSTMCell(input_dim, hidden_dim)

    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        batch, seq_len, input_dim = x.shape

        # Initialize hidden and cell states
        h = Tensor.zeros(batch, self.hidden_dim)
        c = Tensor.zeros(batch, self.hidden_dim)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_dim)
            h, c = self.cell(x_t, h, c)
            outputs.append(h)

        # Stack outputs
        output = Tensor.stack(*outputs, dim=1)  # (batch, seq_len, hidden_dim)

        return output, (h, c)


class TemporalBlock:
    """LSTM + Attention block with residual connections"""

    def __init__(self, input_dim, hidden_dim, num_heads=4, dropout=0.1):
        self.lstm = LSTM(input_dim, hidden_dim)
        self.attention = MultiHeadAttention(hidden_dim, num_heads)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = [nn.Linear(hidden_dim, hidden_dim * 4), nn.Linear(hidden_dim * 4, hidden_dim)]
        self.dropout = dropout

    def __call__(self, x, train=True):
        # LSTM with residual
        lstm_out, _ = self.lstm(x)
        x = x + lstm_out if x.shape == lstm_out.shape else lstm_out
        x = self.norm1(x)

        # Attention with residual
        attn_out = self.attention(x)
        if train and self.dropout > 0:
            attn_out = attn_out.dropout(self.dropout)
        x = x + attn_out
        x = self.norm2(x)

        # Feed-forward with residual
        ffn_out = self.ffn[0](x).relu()
        if train and self.dropout > 0:
            ffn_out = ffn_out.dropout(self.dropout)
        ffn_out = self.ffn[1](ffn_out)
        x = x + ffn_out

        return x


class LSTMAttentionPortfolioNet(PortfolioNetInterface):
    """
    Advanced portfolio optimizer using:
    - Stacked LSTM with attention mechanisms
    - Cross-sectional features (relative strength across assets)
    - Multi-task learning (returns + volatility prediction)
    - Uncertainty estimation via MC dropout
    """

    def __init__(self, n_assets, n_features, hidden_dim=128, num_layers=2, num_heads=4, dropout=0.15):  # noqa: PLR0913
        self.n_assets = n_assets
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Temporal blocks (LSTM + Attention)
        self.temporal_blocks = [TemporalBlock(hidden_dim, hidden_dim, num_heads, dropout) for _ in range(num_layers)]

        # Cross-sectional attention (across assets)
        self.cross_asset_attn = MultiHeadAttention(hidden_dim, num_heads)

        # Multi-task heads
        self.return_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        self.vol_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        # Portfolio weight generator with cash option
        self.weight_net = [
            nn.Linear(hidden_dim + 2, hidden_dim),  # +2 for predicted return and vol
            nn.Linear(hidden_dim, 1),
        ]

        # Cash position network (decides overall market exposure)
        self.cash_net = [
            nn.Linear(hidden_dim * n_assets, hidden_dim),
            nn.Linear(hidden_dim, 1),  # Sigmoid output for cash ratio
        ]

    def __call__(self, x, train=True):
        """
        Args:
            x: (batch, n_assets, seq_len, n_features)
        Returns:
            weights: (batch, n_assets) - Long-only weights (0 to 1 per asset)
            invested_ratio: (batch,) - Fraction of capital invested (0 to 1)
            returns: (batch, n_assets)
            vols: (batch, n_assets)
        """
        batch, n_assets, seq_len, n_features = x.shape

        # Process each asset's time series
        x_flat = x.reshape(batch * n_assets, seq_len, n_features)
        h = self.input_proj(x_flat)

        # Apply temporal blocks
        for block in self.temporal_blocks:
            h = block(h, train=train)

        # Take last timestep
        h = h[:, -1, :]  # (batch * n_assets, hidden_dim)
        h = h.reshape(batch, n_assets, self.hidden_dim)

        # Cross-asset attention
        h_cross = self.cross_asset_attn(h)
        h = h + h_cross

        # Predict returns and volatility
        ret = self.return_head[0](h).relu()
        if train:
            ret = ret.dropout(self.dropout)
        pred_returns = self.return_head[1](ret).squeeze(-1)  # (batch, n_assets)

        vol = self.vol_head[0](h).relu()
        if train:
            vol = vol.dropout(self.dropout)
        pred_vols = self.vol_head[1](vol).squeeze(-1).abs() + 1e-6  # (batch, n_assets)

        # Generate relative asset weights (before cash adjustment)
        weight_input = Tensor.cat(h, pred_returns.unsqueeze(-1), pred_vols.unsqueeze(-1), dim=-1)
        w = self.weight_net[0](weight_input).relu()
        if train:
            w = w.dropout(self.dropout)
        logits = self.weight_net[1](w).squeeze(-1)  # (batch, n_assets)

        # Softmax for relative allocation (ensures they sum to 1)
        relative_weights = logits.softmax(axis=-1)  # (batch, n_assets)

        # Decide overall market exposure (cash position)
        h_flat = h.reshape(batch, -1)  # (batch, n_assets * hidden_dim)
        cash_hidden = self.cash_net[0](h_flat).relu()
        if train:
            cash_hidden = cash_hidden.dropout(self.dropout)
        cash_logit = self.cash_net[1](cash_hidden).squeeze(-1)  # (batch,)
        invested_ratio = cash_logit.sigmoid()  # 0 to 1

        # Final weights: relative_weights scaled by invested_ratio
        # Each weight is in [0, 1], sum is in [0, 1]
        weights = relative_weights * invested_ratio.unsqueeze(-1)

        return weights, invested_ratio, pred_returns, pred_vols


# ============================================================================
# OPTION 2: MAMBA (Selective State Space Model)
# ============================================================================


class MambaBlock:
    """
    Simplified Mamba/S4 block - Selective State Space Model
    Based on "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)

        # Convolutional layer for local context
        self.conv1d = nn.Linear(self.d_inner, self.d_inner)  # Simplified as linear

        # SSM parameters (selective)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)

        # SSM state matrices (learned)
        self.A_log = Tensor.randn(d_state, self.d_inner) * 0.01
        self.D = Tensor.ones(self.d_inner)

    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape

        # Input projection - split into two paths
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        # Manual chunk since tinygrad may not have it
        x_inner = xz[:, :, : self.d_inner]
        z = xz[:, :, self.d_inner :]

        # Apply selective SSM
        # Compute delta (time step) based on input - this makes it "selective"
        delta = self.dt_proj(x_inner).softplus()  # (batch, seq_len, d_inner)

        # Get B and C matrices (input-dependent - key to selective mechanism)
        BC = self.x_proj(x_inner)  # (batch, seq_len, d_state * 2)
        # Manual chunk
        B = BC[:, :, : self.d_state]  # (batch, seq_len, d_state)
        C = BC[:, :, self.d_state :]  # (batch, seq_len, d_state)

        # Discretize A matrix (continuous to discrete time)
        A = -self.A_log.exp()  # (d_state, d_inner) - negative ensures stability

        # Simplified SSM state evolution (sequential scan)
        # In a full implementation, this would use parallel scan for efficiency
        # Initialize hidden state
        h = Tensor.zeros(batch, self.d_state, self.d_inner)

        outputs = []
        for t in range(seq_len):
            # Get current timestep inputs
            x_t = x_inner[:, t, :]  # (batch, d_inner)
            B_t = B[:, t, :].unsqueeze(-1)  # (batch, d_state, 1)
            C_t = C[:, t, :].unsqueeze(1)  # (batch, 1, d_state)
            delta_t = delta[:, t, :].unsqueeze(1)  # (batch, 1, d_inner)

            # Discretize: h_new = (I + delta * A) @ h + delta * B @ x
            # Simplified: h_new = exp(delta * A) * h + delta * B * x
            dA = (delta_t * A.unsqueeze(0)).exp()  # (batch, d_state, d_inner)
            dB = delta_t * B_t * x_t.unsqueeze(1)  # (batch, d_state, d_inner)

            # Update state
            h = h * dA + dB  # (batch, d_state, d_inner)

            # Compute output: y = C @ h + D @ x
            y_t = (C_t @ h).squeeze(1)  # (batch, d_inner)
            y_t = y_t + self.D * x_t  # Add skip connection

            outputs.append(y_t)

        # Stack outputs
        y = Tensor.stack(*outputs, dim=1)  # (batch, seq_len, d_inner)

        # Gating with z (GLU-style gating)
        y = y * z.sigmoid()

        # Output projection
        output = self.out_proj(y)

        return output


class MambaPortfolioNet(PortfolioNetInterface):
    """
    Portfolio network using Mamba (Selective State Space Model)
    - Linear time complexity
    - Strong long-range dependency modeling
    - Input-dependent dynamics (selective mechanism)
    """

    def __init__(self, n_assets, n_features, hidden_dim=64, num_layers=2, d_state=16, dropout=0.15):  # noqa: PLR0913
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # Mamba blocks
        self.mamba_blocks = [MambaBlock(hidden_dim, d_state=d_state, expand=2) for _ in range(num_layers)]

        self.norms = [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]

        # Cross-asset attention
        self.cross_asset_attn = MultiHeadAttention(hidden_dim, num_heads=2)

        # Multi-task heads
        self.return_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        self.vol_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        # Portfolio weight generator
        self.weight_net = [nn.Linear(hidden_dim + 2, hidden_dim), nn.Linear(hidden_dim, 1)]

        # Cash position network
        self.cash_net = [nn.Linear(hidden_dim * n_assets, hidden_dim), nn.Linear(hidden_dim, 1)]

    def __call__(self, x, train=True):
        batch, n_assets, seq_len, n_features = x.shape

        # Process each asset's time series
        x_flat = x.reshape(batch * n_assets, seq_len, n_features)
        h = self.input_proj(x_flat)

        # Apply Mamba blocks with residual connections
        for mamba_block, norm in zip(self.mamba_blocks, self.norms):
            h_new = mamba_block(h)
            if train and self.dropout > 0:
                h_new = h_new.dropout(self.dropout)
            h = norm(h + h_new)

        # Take last timestep
        h = h[:, -1, :]  # (batch * n_assets, hidden_dim)
        h = h.reshape(batch, n_assets, self.hidden_dim)

        # Cross-asset attention
        h_cross = self.cross_asset_attn(h)
        h = h + h_cross

        # Predict returns and volatility
        ret = self.return_head[0](h).relu()
        if train:
            ret = ret.dropout(self.dropout)
        pred_returns = self.return_head[1](ret).squeeze(-1)

        vol = self.vol_head[0](h).relu()
        if train:
            vol = vol.dropout(self.dropout)
        pred_vols = self.vol_head[1](vol).squeeze(-1).abs() + 1e-6

        # Generate relative asset weights
        weight_input = Tensor.cat(h, pred_returns.unsqueeze(-1), pred_vols.unsqueeze(-1), dim=-1)
        w = self.weight_net[0](weight_input).relu()
        if train:
            w = w.dropout(self.dropout)
        logits = self.weight_net[1](w).squeeze(-1)
        relative_weights = logits.softmax(axis=-1)

        # Decide overall market exposure
        h_flat = h.reshape(batch, -1)
        cash_hidden = self.cash_net[0](h_flat).relu()
        if train:
            cash_hidden = cash_hidden.dropout(self.dropout)
        cash_logit = self.cash_net[1](cash_hidden).squeeze(-1)
        invested_ratio = cash_logit.sigmoid()

        # Final weights
        weights = relative_weights * invested_ratio.unsqueeze(-1)

        return weights, invested_ratio, pred_returns, pred_vols


# ============================================================================
# OPTION 3: TEMPORAL CONVOLUTIONAL NETWORK (TCN)
# ============================================================================


class CausalConv1d:
    """Causal 1D convolution - only looks at past"""

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation

        self.weight = Tensor.glorot_uniform(out_channels, in_channels, kernel_size)
        self.bias = Tensor.zeros(out_channels)

    def __call__(self, x):
        """
        Args:
            x: (batch, seq_len, in_channels)
        Returns:
            output: (batch, seq_len, out_channels)
        """
        batch, seq_len, in_channels = x.shape

        # Pad left side only (causal)
        if self.padding > 0:
            x_pad = Tensor.zeros(batch, self.padding, in_channels)
            x = Tensor.cat(x_pad, x, dim=1)

        # Manual convolution
        outputs = []
        for i in range(seq_len):
            # Get window
            start = i
            end = i + self.kernel_size * self.dilation
            if end > x.shape[1]:
                break

            window = x[:, start : end : self.dilation, :]  # (batch, kernel_size, in_channels)

            # Convolution: sum over kernel_size and in_channels
            out = (window.unsqueeze(-1) * self.weight.reshape(1, self.kernel_size, in_channels, -1)).sum(axis=(1, 2))
            out = out + self.bias
            outputs.append(out)

        if len(outputs) < seq_len:
            # Pad if needed
            remaining = seq_len - len(outputs)
            for _ in range(remaining):
                outputs.append(outputs[-1])

        return Tensor.stack(*outputs, dim=1)


class TemporalBlock_TCN:
    """TCN residual block with dilated causal convolutions"""

    def __init__(self, n_inputs, n_outputs, kernel_size, dilation, dropout=0.15):
        self.conv1 = CausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.conv2 = CausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.dropout = dropout

        # Residual connection
        self.downsample = nn.Linear(n_inputs, n_outputs) if n_inputs != n_outputs else None

    def __call__(self, x, train=True):
        """
        Args:
            x: (batch, seq_len, n_inputs)
        """
        out = self.conv1(x).relu()
        if train and self.dropout > 0:
            out = out.dropout(self.dropout)

        out = self.conv2(out).relu()
        if train and self.dropout > 0:
            out = out.dropout(self.dropout)

        # Residual
        res = x if self.downsample is None else self.downsample(x)
        return (out + res).relu()


class TCNPortfolioNet(PortfolioNetInterface):
    """
    Portfolio network using Temporal Convolutional Network (TCN)
    - Parallelizable (faster than RNNs)
    - Dilated convolutions for large receptive field
    - Causal (no future leakage)
    - Excellent for financial time series
    """

    def __init__(self, n_assets, n_features, hidden_dim=64, num_layers=3, kernel_size=3, dropout=0.15):  # noqa: PLR0913
        self.n_assets = n_assets
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        # Input projection
        self.input_proj = nn.Linear(n_features, hidden_dim)

        # TCN blocks with increasing dilation
        self.tcn_blocks = []
        for i in range(num_layers):
            dilation = 2**i  # Exponential dilation: 1, 2, 4, 8, ...
            self.tcn_blocks.append(TemporalBlock_TCN(hidden_dim, hidden_dim, kernel_size, dilation, dropout))

        # Cross-asset attention
        self.cross_asset_attn = MultiHeadAttention(hidden_dim, num_heads=2)

        # Multi-task heads
        self.return_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        self.vol_head = [nn.Linear(hidden_dim, hidden_dim // 2), nn.Linear(hidden_dim // 2, 1)]

        # Portfolio weight generator
        self.weight_net = [nn.Linear(hidden_dim + 2, hidden_dim), nn.Linear(hidden_dim, 1)]

        # Cash position network
        self.cash_net = [nn.Linear(hidden_dim * n_assets, hidden_dim), nn.Linear(hidden_dim, 1)]

    def __call__(self, x, train=True):
        batch, n_assets, seq_len, n_features = x.shape

        # Process each asset's time series
        x_flat = x.reshape(batch * n_assets, seq_len, n_features)
        h = self.input_proj(x_flat)

        # Apply TCN blocks
        for tcn_block in self.tcn_blocks:
            h = tcn_block(h, train=train)

        # Take last timestep
        h = h[:, -1, :]  # (batch * n_assets, hidden_dim)
        h = h.reshape(batch, n_assets, self.hidden_dim)

        # Cross-asset attention
        h_cross = self.cross_asset_attn(h)
        h = h + h_cross

        # Predict returns and volatility
        ret = self.return_head[0](h).relu()
        if train:
            ret = ret.dropout(self.dropout)
        pred_returns = self.return_head[1](ret).squeeze(-1)

        vol = self.vol_head[0](h).relu()
        if train:
            vol = vol.dropout(self.dropout)
        pred_vols = self.vol_head[1](vol).squeeze(-1).abs() + 1e-6

        # Generate relative asset weights
        weight_input = Tensor.cat(h, pred_returns.unsqueeze(-1), pred_vols.unsqueeze(-1), dim=-1)
        w = self.weight_net[0](weight_input).relu()
        if train:
            w = w.dropout(self.dropout)
        logits = self.weight_net[1](w).squeeze(-1)
        relative_weights = logits.softmax(axis=-1)

        # Decide overall market exposure
        h_flat = h.reshape(batch, -1)
        cash_hidden = self.cash_net[0](h_flat).relu()
        if train:
            cash_hidden = cash_hidden.dropout(self.dropout)
        cash_logit = self.cash_net[1](cash_hidden).squeeze(-1)
        invested_ratio = cash_logit.sigmoid()

        # Final weights
        weights = relative_weights * invested_ratio.unsqueeze(-1)

        return weights, invested_ratio, pred_returns, pred_vols


# ============================================================================
# PORTFOLIO OPTIMIZER (Model-Agnostic)
# ============================================================================


class LSTMOptimizerConfig(BaseModel):
    n_assets: int
    lookback: int = 30
    hidden_dim: int = 64
    num_layers: int = 1
    risk_aversion: float = 2.0
    lr: float = 0.001
    device: str = "CPU"


class DeepLearningOptimizer:
    """
    State-of-the-art portfolio optimizer with online learning.
    Supports multiple neural network architectures.
    """

    model_type = "lstm"

    def __init__(
        self,
        n_assets: int,
    ) -> None:
        """
        Args:
            n_assets: Number of assets in the portfolio
            model_type: str, one of ['lstm', 'mamba', 'tcn']
                - 'lstm': LSTM + Transformer (original, best for complex patterns)
                - 'mamba': Selective State Space Model (linear time, great for long sequences)
                - 'tcn': Temporal Convolutional Network (fastest, parallelizable)
        """
        self.config = LSTMOptimizerConfig(n_assets=n_assets)

        # Fractional differentiator
        self.frac_diff = FractionalDifferentiator(d=0.5, threshold=1e-5)

        # Feature engineering parameters
        self.feature_dim = 15

        # Track if models have been successfully trained
        self.trained = False

        # Initialize network based on type
        if self.model_type.lower() == "lstm":
            self.net = LSTMAttentionPortfolioNet(
                n_assets=self.config.n_assets,
                n_features=self.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                num_heads=2,
                dropout=0.15,
            )
            logger.debug("✓ Using LSTM + Transformer architecture")
        elif self.model_type.lower() == "mamba":
            self.net = MambaPortfolioNet(
                n_assets=self.config.n_assets,
                n_features=self.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                d_state=16,
                dropout=0.15,
            )
            logger.debug("✓ Using MAMBA (Selective State Space) architecture")
        elif self.model_type.lower() == "tcn":
            self.net = TCNPortfolioNet(
                n_assets=self.config.n_assets,
                n_features=self.feature_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                kernel_size=3,
                dropout=0.15,
            )
            logger.debug("✓ Using TCN (Temporal Convolutional Network) architecture")
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}. Choose from ['lstm', 'mamba', 'tcn']")

        self.optimizer = Adam(get_parameters(self.net), lr=self.config.lr)

        # Online learning buffers
        self.replay_buffer = []
        self.max_buffer_size = 500

        # Track statistics
        self.feature_mean = None
        self.feature_std = None

    def _engineer_features(self, prices: np.ndarray) -> np.ndarray:
        """
        Create advanced features from price data
        Args:
            prices: (batch, n_assets, seq_len) or (n_assets, seq_len)
        Returns:
            features: (batch, n_assets, seq_len, n_features)
        """
        if len(prices.shape) == PRICE_DATA_DIMENSIONS_2D:
            prices = prices[None, ...]  # Add batch dimension

        batch, n_assets, seq_len = prices.shape
        features_list = []

        for b in range(batch):
            batch_features = []
            for i in range(n_assets):
                asset_prices = prices[b, i, :]

                # Returns at multiple horizons
                log_prices = np.log(asset_prices + 1e-8)
                ret_1 = np.diff(log_prices)
                ret_5 = (log_prices[5:] - log_prices[:-5]) / 5
                ret_20 = (log_prices[20:] - log_prices[:-20]) / 20

                # Volatility (rolling std of returns)
                vol_10 = np.array(
                    [
                        ret_1[max(0, j - VOLATILITY_WINDOW_DAYS) : j].std() if j >= VOLATILITY_WINDOW_DAYS else 0
                        for j in range(1, len(ret_1) + 1)
                    ]
                )

                # Momentum indicators
                mom_10 = asset_prices[10:] / asset_prices[:-10] - 1
                mom_20 = asset_prices[20:] / asset_prices[:-20] - 1

                # RSI
                gains = np.maximum(ret_1, 0)
                losses = np.maximum(-ret_1, 0)
                avg_gain = np.array(
                    [
                        gains[max(0, j - RSI_WINDOW_DAYS) : j].mean() if j >= RSI_WINDOW_DAYS else 0
                        for j in range(1, len(gains) + 1)
                    ]
                )
                avg_loss = np.array(
                    [
                        losses[max(0, j - RSI_WINDOW_DAYS) : j].mean() if j >= RSI_WINDOW_DAYS else 0
                        for j in range(1, len(losses) + 1)
                    ]
                )
                rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))

                # MACD
                ema_12 = self._ema(asset_prices, 12)
                ema_26 = self._ema(asset_prices, 26)
                macd = ema_12 - ema_26
                signal = self._ema(macd, 9)
                macd_hist = macd - signal

                # Bollinger Bands position
                sma_20 = np.array([asset_prices[max(0, j - 20) : j + 1].mean() for j in range(len(asset_prices))])
                std_20 = np.array([asset_prices[max(0, j - 20) : j + 1].std() for j in range(len(asset_prices))])
                bb_position = (asset_prices - sma_20) / (2 * std_20 + 1e-8)

                # Fractional differentiation features (NEW)
                # Apply to log prices for different d values
                frac_diff_05 = self.frac_diff.transform(log_prices)  # d=0.5
                frac_diff_03 = FractionalDifferentiator(d=0.3).transform(log_prices)  # d=0.3 (more memory)
                frac_diff_07 = FractionalDifferentiator(d=0.7).transform(log_prices)  # d=0.7 (less memory)

                # Align all features to same length
                min_len = min(
                    len(ret_1),
                    len(ret_5),
                    len(ret_20),
                    len(vol_10),
                    len(mom_10),
                    len(mom_20),
                    len(rsi),
                    len(macd_hist),
                    len(bb_position),
                    len(frac_diff_05),
                    len(frac_diff_03),
                    len(frac_diff_07),
                )

                asset_features = np.stack(
                    [
                        ret_1[-min_len:],
                        ret_5[-min_len:],
                        ret_20[-min_len:],
                        vol_10[-min_len:],
                        mom_10[-min_len:],
                        mom_20[-min_len:],
                        rsi[-min_len:],
                        macd_hist[-min_len:],
                        bb_position[-min_len:],
                        frac_diff_05[-min_len:],  # Fractional diff d=0.5
                        frac_diff_03[-min_len:],  # Fractional diff d=0.3
                        frac_diff_07[-min_len:],  # Fractional diff d=0.7
                        # Cross-sectional features (computed later)
                        np.zeros(min_len),  # Relative strength
                        np.zeros(min_len),  # Z-score vs other assets
                        np.zeros(min_len),  # Correlation rank
                    ],
                    axis=1,
                )  # (seq_len, n_features)

                batch_features.append(asset_features)

            features_list.append(np.stack(batch_features))  # (n_assets, seq_len, n_features)

        features = np.stack(features_list)  # (batch, n_assets, seq_len, n_features)

        # Add cross-sectional features
        features = self._add_cross_sectional_features(features)

        # Normalize
        if self.feature_mean is None:
            self.feature_mean = features.mean(axis=(0, 1, 2), keepdims=True)
            self.feature_std = features.std(axis=(0, 1, 2), keepdims=True) + 1e-8

        features = (features - self.feature_mean) / self.feature_std

        return features

    def _ema(self, data, period):
        """Exponential moving average"""
        alpha = 2 / (period + 1)
        ema = np.zeros_like(data)
        ema[0] = data[0]
        for i in range(1, len(data)):
            ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1]
        return ema

    def _add_cross_sectional_features(self, features):
        """Add features comparing assets to each other"""
        batch, n_assets, seq_len, n_features = features.shape

        for b in range(batch):
            for t in range(seq_len):
                # Relative momentum (compared to median)
                mom_20_idx = 5  # momentum_20 feature index
                all_mom = features[b, :, t, mom_20_idx]
                median_mom = np.median(all_mom)
                features[b, :, t, 12] = all_mom - median_mom  # Updated index

                # Z-score of returns
                ret_1_idx = 0
                all_ret = features[b, :, t, ret_1_idx]
                mean_ret = all_ret.mean()
                std_ret = all_ret.std() + 1e-8
                features[b, :, t, 13] = (all_ret - mean_ret) / std_ret  # Updated index

                # Volatility rank
                vol_idx = 3
                all_vol = features[b, :, t, vol_idx]
                vol_ranks = np.argsort(np.argsort(all_vol)) / len(all_vol)
                features[b, :, t, 14] = vol_ranks  # Updated index

        return features

    def _portfolio_loss(  # noqa: PLR0913
        self,
        weights,
        invested_ratio,
        returns,
        vols,
        actual_returns,
        prev_weights,
        transaction_cost=0.001,
    ):
        """
        Multi-objective loss for long-only with cash:
        1. Return prediction accuracy
        2. Volatility prediction accuracy
        3. Portfolio return (negative)
        4. Portfolio risk (variance)
        5. Transaction costs
        6. Cash position penalty (discourage excessive cash holding in bull markets)
        """
        # Prediction losses
        return_loss = ((returns - actual_returns) ** 2).mean()
        actual_vols = actual_returns.abs()
        vol_loss = ((vols - actual_vols) ** 2).mean()

        # Portfolio metrics (accounting for cash position)
        port_return = (weights * actual_returns).sum(axis=1).mean()
        port_vol = (weights * actual_returns).sum(axis=1).std()

        # Transaction cost (including cash movements)
        turnover = (weights - prev_weights).abs().sum(axis=1).mean()

        # Cash penalty: penalize holding cash when market returns are positive
        # This encourages full investment in bull markets, defensive in bear markets
        market_return = actual_returns.mean(axis=1)  # Average market return
        cash_ratio = 1.0 - invested_ratio
        cash_penalty = (cash_ratio * market_return.clip(0, None)).mean() * 0.5

        # Combined loss (maximize risk-adjusted return)
        loss = (
            return_loss
            + vol_loss
            - port_return
            + self.config.risk_aversion * port_vol
            + transaction_cost * turnover
            + cash_penalty
        )

        return loss

    def train(self, prices, returns, n_epochs=50, batch_size=128):
        """
        Train the network (optimized for speed)
        Args:
            prices: (n_days, n_assets) historical prices
            returns: (n_days, n_assets) historical returns
        """
        # Enable training mode for tinygrad
        Tensor.training = True

        n_days = len(prices)
        n_returns = len(returns)

        # Handle case where returns has one less day than prices
        if n_returns == n_days - 1:
            # Pad returns with the last value or use the available returns
            last_return = returns[-1] if n_returns > 0 else np.zeros(self.config.n_assets)
            returns = np.vstack([returns, last_return])
            n_returns = len(returns)

        logger.debug(f"Training on {n_days} days of data, {n_returns} returns...")

        # Create training sequences
        X_list, y_list = [], []
        for i in range(self.config.lookback, n_days):
            if i < n_returns:
                window_prices = prices[i - self.config.lookback : i]
                X_list.append(window_prices.T)  # (n_assets, lookback)
                y_list.append(returns[i])  # (n_assets,)

        X = np.stack(X_list)  # (n_samples, n_assets, lookback)
        y = np.stack(y_list)  # (n_samples, n_assets)

        logger.debug("Engineering features...")
        X_features = self._engineer_features(X)  # (n_samples, n_assets, lookback, n_features)

        n_samples = len(X_features)
        prev_weights = np.ones((batch_size, self.config.n_assets)) / self.config.n_assets

        # Adjust batch size if we have fewer samples than batch_size
        effective_batch_size = min(batch_size, n_samples)
        if effective_batch_size != batch_size:
            logger.debug(f"Adjusting batch size from {batch_size} to {effective_batch_size} due to limited data")
            prev_weights = np.ones((effective_batch_size, self.config.n_assets)) / self.config.n_assets

        logger.debug(f"Starting training for {n_epochs} epochs...")
        logger.debug(f"Total batches per epoch: {max(1, n_samples // effective_batch_size)}")
        start_time = time.time()

        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0

            # Shuffle data
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, effective_batch_size):
                batch_idx = indices[i : i + effective_batch_size]
                X_batch = Tensor(X_features[batch_idx].astype(np.float32), requires_grad=False)
                y_batch = Tensor(y[batch_idx].astype(np.float32), requires_grad=False)
                prev_w = Tensor(prev_weights[: len(batch_idx)].astype(np.float32), requires_grad=False)

                # Forward pass
                weights, invested_ratio, pred_returns, pred_vols = self.net(X_batch, train=True)

                # Compute loss
                loss = self._portfolio_loss(weights, invested_ratio, pred_returns, pred_vols, y_batch, prev_w)

                # Add small regularization to ensure all parameters get gradients
                reg_loss = Tensor(0.0)
                for param in get_parameters(self.net):
                    reg_loss = reg_loss + (param * param).sum() * 1e-8
                loss = loss + reg_loss

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()

                # Realize the computation graph before optimizer step
                loss.realize()

                self.optimizer.step()

                batch_loss = loss.numpy()
                epoch_loss += batch_loss
                n_batches += 1
                prev_weights = weights.numpy()

                # logger.debug progress every 10 batches
                if n_batches % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_loss = epoch_loss / n_batches
                    logger.debug(
                        f"  Epoch {epoch+1}/{n_epochs}, Batch {n_batches}, Loss: {batch_loss:.6f}, "
                        f"Avg Loss: {avg_loss:.6f}, Time: {elapsed:.1f}s"
                    )

            avg_loss = epoch_loss / n_batches
            elapsed = time.time() - start_time
            logger.debug(f"Epoch {epoch+1}/{n_epochs} complete, Avg Loss: {avg_loss:.6f}, Time: {elapsed:.2f}s")

        total_time = time.time() - start_time
        logger.debug(f"\nTraining complete in {total_time:.2f}s")

        self.trained = True

    def _online_update(self, new_prices, new_returns, n_steps=10):
        """
        Fast online update with new data
        Args:
            new_prices: (lookback+1, n_assets) recent prices
            new_returns: (1, n_assets) latest returns
        """
        # Enable training mode
        Tensor.training = True

        # Add to replay buffer
        if len(new_prices) >= self.config.lookback:
            X = new_prices[-self.config.lookback :].T  # (n_assets, lookback)
            self.replay_buffer.append((X, new_returns[0]))

            if len(self.replay_buffer) > self.max_buffer_size:
                self.replay_buffer.pop(0)

        # Train on buffer
        if len(self.replay_buffer) >= MIN_REPLAY_BUFFER_SIZE:
            buffer_X = np.stack([x for x, _ in self.replay_buffer[-64:]])
            buffer_y = np.stack([y for _, y in self.replay_buffer[-64:]])

            X_features = self._engineer_features(buffer_X)

            prev_weights = np.ones((1, self.config.n_assets)) / self.config.n_assets

            for _ in range(n_steps):
                X_batch = Tensor(X_features, requires_grad=False)
                y_batch = Tensor(buffer_y, requires_grad=False)
                prev_w = Tensor(np.tile(prev_weights, (len(buffer_X), 1)), requires_grad=False)

                weights, invested_ratio, pred_returns, pred_vols = self.net(X_batch, train=True)
                loss = self._portfolio_loss(
                    weights, invested_ratio, pred_returns, pred_vols, y_batch, prev_w, transaction_cost=0.001
                )

                # Add small regularization to ensure all parameters get gradients
                reg_loss = Tensor(0.0)
                for param in get_parameters(self.net):
                    reg_loss = reg_loss + (param * param).sum() * 1e-8
                loss = loss + reg_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def _predict(self, prices, n_samples=10):
        """
        Predict portfolio weights with uncertainty estimation
        Args:
            prices: (lookback, n_assets) recent prices
            n_samples: Number of MC dropout samples
        Returns:
            mean_weights: (n_assets,) mean predicted weights (long-only)
            std_weights: (n_assets,) uncertainty in weights
            cash_ratio: float, fraction held in cash (0 to 1)
            pred_returns: (n_assets,) predicted returns
        """
        # Disable training mode for prediction (but enable for MC dropout)
        Tensor.training = True  # Keep True for MC dropout

        X = prices.T[None, ...]  # (1, n_assets, lookback)
        X_features = self._engineer_features(X)
        X_tensor = Tensor(X_features, requires_grad=False)

        # MC Dropout for uncertainty
        all_weights = []
        all_returns = []
        all_invested = []

        for _ in range(n_samples):
            weights, invested_ratio, returns, _ = self.net(X_tensor, train=True)  # Keep dropout on
            all_weights.append(weights.numpy()[0])
            all_returns.append(returns.numpy()[0])
            all_invested.append(invested_ratio.numpy()[0])

        mean_weights = np.mean(all_weights, axis=0)
        std_weights = np.std(all_weights, axis=0)
        mean_returns = np.mean(all_returns, axis=0)
        mean_invested = np.mean(all_invested)
        cash_ratio = 1.0 - mean_invested

        return mean_weights, std_weights, cash_ratio, mean_returns

    def predict(self, current_prices: np.ndarray) -> np.ndarray:
        """
        Predict portfolio weights

        Args:
            current_prices: (lookback, n_assets) recent prices

        Returns:
            optimal_weights: (n_assets,) portfolio weights
        """
        mean_weights, _, _, _ = self._predict(current_prices)
        return mean_weights

    def incremental_update(self, new_prices: np.ndarray, new_returns: np.ndarray) -> None:
        """
        Incremental update with new data

        Args:
            new_prices: Recent prices
            new_returns: Latest returns
        """
        self._online_update(new_prices, new_returns)
