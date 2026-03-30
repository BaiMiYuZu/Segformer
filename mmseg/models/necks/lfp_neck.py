import torch.nn as nn

from ..builder import NECKS


@NECKS.register_module()
class LFPNeck(nn.Module):
    """Apply LFP purification to each backbone stage before decoding."""

    def __init__(self,
                 in_channels,
                 active_stages=(0, 1, 2, 3),
                 wave='haar',
                 mode='zero',
                 with_gauss=True,
                 gauss_gate=0.5):
        super(LFPNeck, self).__init__()
        assert isinstance(in_channels, (list, tuple)), \
            '`in_channels` should describe the four backbone stages.'

        try:
            from LFP import LFP
        except ImportError as exc:
            raise ImportError(
                'LFPNeck requires `LFP.py` and its wavelet dependencies. '
                'Please ensure packages such as `pytorch_wavelets` and '
                '`PyWavelets` are installed in the active environment.'
            ) from exc

        self.in_channels = list(in_channels)
        self.active_stages = set(active_stages)
        assert self.active_stages.issubset(set(range(len(self.in_channels)))), \
            '`active_stages` must refer to valid backbone stage indices.'

        self.lfp_modules = nn.ModuleList([
            LFP(
                in_channels=channels,
                wave=wave,
                mode=mode,
                with_gauss=with_gauss,
                gauss_gate=gauss_gate)
            for channels in self.in_channels
        ])

    def init_weights(self):
        """Keep the default initialization from the wrapped LFP modules."""

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels), \
            'The number of backbone features must match `in_channels`.'

        outs = []
        for stage_idx, (feat, expected_channels, lfp) in enumerate(
                zip(inputs, self.in_channels, self.lfp_modules)):
            assert feat.size(1) == expected_channels, \
                'Backbone feature channels do not match the LFP stage setup.'
            if stage_idx in self.active_stages:
                feat = lfp(feat)
            outs.append(feat)

        return tuple(outs)
