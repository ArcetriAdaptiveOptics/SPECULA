"""
U-Net regressor used by Conv2dNetTrainer / Conv2dNetTester / Conv2dNetRec:
maps a stack of 2D maps (e.g. x/y slope maps) to a vector of modes.

    input -> encoder (depth levels, each halving the map) -> bottleneck
          -> decoder (depth levels, each doubling it back, with skip
             connections from the encoder) -> regression head -> outputs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================
#   Convolution blocks (selected by conv_block_type)
# ==============================

class ConvBlock(nn.Module):
    """conv_block_type 0: two 3x3 convolutions, residual connection, SE attention."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.residual = _residual(in_ch, out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        return self.se(self.block(x) + self.residual(x))


class DepthwiseSeparableConvBlock(nn.Module):
    """conv_block_type 1: like ConvBlock, with each 3x3 convolution split into
    a per-channel 3x3 and a 1x1 across channels (far fewer parameters)."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.dw_conv1 = nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False)
        self.pw_conv1 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.norm1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout2d(dropout)

        self.dw_conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch, bias=False)
        self.pw_conv2 = nn.Conv2d(out_ch, out_ch, 1, bias=False)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout2d(dropout)

        self.residual = _residual(in_ch, out_ch)
        self.se = SEBlock(out_ch)

    def forward(self, x):
        out = self.drop1(self.act1(self.norm1(self.pw_conv1(self.dw_conv1(x)))))
        out = self.drop2(self.act2(self.norm2(self.pw_conv2(self.dw_conv2(out)))))
        return self.se(out + self.residual(x))


class CoordAttentionConvBlock(nn.Module):
    """conv_block_type 3: two 3x3 convolutions, residual connection,
    coordinate attention (reweights rows and columns of the map)."""
    def __init__(self, in_ch, out_ch, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, out_ch), out_ch),
            nn.GELU(),
            nn.Dropout2d(dropout)
        )
        self.coord_att = CoordinateAttention(out_ch)
        self.residual = _residual(in_ch, out_ch)

    def forward(self, x):
        return self.coord_att(self.conv2(self.conv1(x))) + self.residual(x)


CONV_BLOCKS = {
    0: ConvBlock,
    1: DepthwiseSeparableConvBlock,
    3: CoordAttentionConvBlock,
}


def _residual(in_ch, out_ch):
    """Residual path of a block: a 1x1 projection if the channel count changes."""
    if in_ch == out_ch:
        return nn.Identity()
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 1, bias=False),
        nn.GroupNorm(min(32, out_ch), out_ch)
    )


# ==============================
#   Attention modules
# ==============================

class SEBlock(nn.Module):
    """Squeeze-and-excitation: reweights channels from their global average."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, reduced, bias=False),
            nn.GELU(),
            nn.Linear(reduced, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c = x.shape[:2]
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1, 1)


class CoordinateAttention(nn.Module):
    """Attention weights along rows and along columns, from the map averaged
    over the other axis."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        reduced = max(channels // reduction, 8)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.conv1 = nn.Conv2d(channels, reduced, 1, bias=False)
        self.bn1 = nn.GroupNorm(min(8, reduced), reduced)
        self.act = nn.GELU()
        self.conv_h = nn.Conv2d(reduced, channels, 1, bias=False)
        self.conv_w = nn.Conv2d(reduced, channels, 1, bias=False)

    def forward(self, x):
        _, _, H, W = x.shape
        x_h = self.pool_h(x)                            # B, C, H, 1
        x_w = self.pool_w(x).permute(0, 1, 3, 2)        # B, C, W, 1
        y = self.act(self.bn1(self.conv1(torch.cat([x_h, x_w], dim=2))))
        y_h, y_w = torch.split(y, [H, W], dim=2)
        a_h = self.conv_h(y_h).sigmoid()
        a_w = self.conv_w(y_w.permute(0, 1, 3, 2)).sigmoid()
        return x * a_h * a_w


class SelfAttention(nn.Module):
    """Self-attention over all positions of the map (used at the bottleneck,
    where the map is smallest)."""
    def __init__(self, channels):
        super().__init__()
        qk_channels = max(1, channels // 8)
        self.query = nn.Conv2d(channels, qk_channels, 1)
        self.key = nn.Conv2d(channels, qk_channels, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(B, -1, H * W)
        v = self.value(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1)).view(B, C, H, W)
        return self.gamma * out + x


# ==============================
#   Regression heads (selected by head_type)
# ==============================

class AdaptiveFeatureExtractor(nn.Module):
    """'pooled' head, per level: the map averaged over 1x1, 2x2 and 4x4 grids
    plus its global maximum -> 22 values per channel, whatever the map size."""
    FEATURES_PER_CHANNEL = 1 + 4 + 16 + 1

    def __init__(self, channels):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(s) for s in [1, 2, 4]])
        self.max_pool = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        B = x.shape[0]
        features = [pool(x).view(B, -1) for pool in self.pools]
        features.append(self.max_pool(x).view(B, -1))
        return torch.cat(features, dim=1)


class RegressionHead(nn.Module):
    """'pooled' head: an MLP narrowing to 32 features, with a skip connection
    from the input to the 128-wide layer."""
    def __init__(self, input_dim, output_size, dropout=0.1):
        super().__init__()
        dims = [input_dim, 256, 128, 64, 32]
        layers = []
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(dims[-1], output_size))
        self.fc = nn.Sequential(*layers)
        self.skip = nn.Linear(input_dim, 128)

    def forward(self, x):
        out = x
        for i, layer in enumerate(self.fc):
            out = layer(out)
            if i == 4:      # after the first Linear/LayerNorm/GELU/Dropout (128 wide)
                out = out + self.skip(x)
        return out


class SpatialRegressionHead(nn.Module):
    """'spatial' head: keeps where each feature is on the map.

    A 1x1 convolution projects the feature map down to a few channels, the
    result is resampled to a fixed grid x grid map -- so the head's size
    doesn't depend on the input size -- and flattened. The outputs are a
    hidden layer ``width`` wide plus a direct linear map from the flattened
    features, so that a linear relation between the map and the outputs
    doesn't have to go through the nonlinearity.
    """
    def __init__(self, in_channels, output_size, grid=32, channels=4, width=512, dropout=0.1):
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, channels, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d(grid)
        n_features = channels * grid * grid
        self.hidden = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(n_features, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(width, output_size)
        self.direct = nn.Linear(n_features, output_size)

    def forward(self, x):
        features = self.pool(self.reduce(x)).flatten(1)
        return self.out(self.hidden(features)) + self.direct(features)


# ==============================
#   Network
# ==============================

class UNetRegressor(nn.Module):
    """
    Parameters
    ----------
    input_channels : int
        Channels of the input maps.
    output_size : int
        Number of outputs (modes).
    base_channels : int
        Channels at the first encoder level; doubled at every level down.
    dropout_level : float
        Dropout in the convolution blocks and in the head.
    depth : int
        Encoder/decoder levels. Each halves the map (average pooling), so
        e.g. 60x60 maps become 7x7 at the bottleneck with depth 3. The maps
        must be at least 2 ** depth pixels wide (see check_input_size()).
    conv_block_type : int
        Convolution block: 0 ConvBlock, 1 DepthwiseSeparableConvBlock,
        3 CoordAttentionConvBlock.
    head_type : str
        'pooled': every decoder level (and the bottleneck) is pooled to
        grids of at most 4x4 and goes through an MLP narrowing to 32
        features -- at most 32 independent outputs, and no fine spatial
        detail.
        'spatial': the last, full-resolution decoder output goes through
        SpatialRegressionHead -- for outputs, such as higher-order modes,
        that depend on fine spatial detail.
    head_grid : int
        For the 'spatial' head only: the grid its feature map is resampled
        to (see SpatialRegressionHead). It should be fine enough to keep
        the detail the outputs need -- e.g. for a pyramid's four pupil
        images in one map, a grid coarser than the pupil sampling averages
        neighbouring subapertures together, and across pupils.
    """
    def __init__(self, input_channels=1, output_size=5, base_channels=32,
                 dropout_level=0.1, depth=5, conv_block_type=0, head_type='pooled',
                 head_grid=32):
        super().__init__()
        if conv_block_type not in CONV_BLOCKS:
            raise ValueError(f'conv_block_type must be one of {sorted(CONV_BLOCKS)}, '
                             f'got {conv_block_type!r}')
        if head_type not in ('pooled', 'spatial'):
            raise ValueError(f"head_type must be 'pooled' or 'spatial', got {head_type!r}")
        self.head_type = head_type
        block = CONV_BLOCKS[conv_block_type]

        def level_channels(i):
            return base_channels * (2 ** i)

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = input_channels
        for i in range(depth):
            self.encoders.append(block(in_ch, level_channels(i), dropout=dropout_level))
            self.pools.append(nn.AvgPool2d(2))
            in_ch = level_channels(i)

        bottleneck_ch = level_channels(depth)
        self.bottleneck = nn.Sequential(
            block(in_ch, bottleneck_ch, dropout=dropout_level),
            SelfAttention(bottleneck_ch),
            block(bottleneck_ch, bottleneck_ch, dropout=dropout_level)
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(depth)):
            out_ch = level_channels(i)
            self.ups.append(nn.ConvTranspose2d(level_channels(i + 1), out_ch, kernel_size=2, stride=2))
            self.decoders.append(block(2 * out_ch, out_ch, dropout=dropout_level))   # + skip connection

        if head_type == 'spatial':
            self.regressor = SpatialRegressionHead(base_channels, output_size, grid=head_grid,
                                                   dropout=dropout_level)
        else:
            level_chs = [bottleneck_ch] + [level_channels(i) for i in reversed(range(depth))]
            self.feature_extractors = nn.ModuleList([AdaptiveFeatureExtractor(c) for c in level_chs])
            feature_dim = AdaptiveFeatureExtractor.FEATURES_PER_CHANNEL * sum(level_chs)
            self.regressor = RegressionHead(feature_dim, output_size, dropout_level)

    def _trunk(self, x):
        """Encoder, bottleneck and decoder. Returns the bottleneck output and
        the list of decoder outputs (coarsest first)."""
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        bottleneck = x = self.bottleneck(x)

        decoder_outputs = []
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = up(x)
            # Odd sizes along the way (e.g. 15 -> 7 -> 14) leave the upsampled
            # map one pixel off its skip connection. The spatial head resizes
            # the map up, so the last one keeps the input's resolution; the
            # pooled head shrinks the skip instead.
            if x.shape[2:] != skip.shape[2:]:
                if self.head_type == 'spatial':
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
                else:
                    skip = F.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=False)
            x = decoder(torch.cat([x, skip], dim=1))
            decoder_outputs.append(x)

        return bottleneck, decoder_outputs

    @staticmethod
    def check_input_size(shape, depth):
        """Raise a clear error if maps of the given (height, width) are too
        small for ``depth`` levels: each level halves them, and the bottleneck
        needs at least one pixel."""
        if min(shape) < 2 ** depth:
            raise ValueError(f'input maps of {shape[0]}x{shape[1]} pixels are too small for depth={depth}: '
                             f'each level halves them, so they must be at least {2 ** depth} pixels wide. '
                             f'Use depth <= {max(int(min(shape)).bit_length() - 1, 0)}')

    def output_layers(self):
        """The final linear layers whose outputs are summed into the result.
        Rescaling their weights and biases together rescales the output
        exactly, which is how Conv2dNetTrainer keeps the predictions unchanged
        when it updates the output normalization."""
        if self.head_type == 'spatial':
            return [self.regressor.out, self.regressor.direct]
        return [self.regressor.fc[-1]]

    def forward(self, x):
        bottleneck, decoder_outputs = self._trunk(x)
        if self.head_type == 'spatial':
            return self.regressor(decoder_outputs[-1])
        features = [extractor(level) for extractor, level
                    in zip(self.feature_extractors, [bottleneck] + decoder_outputs)]
        return self.regressor(torch.cat(features, dim=1))
