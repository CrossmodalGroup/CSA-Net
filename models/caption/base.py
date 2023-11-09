# ------------------------------------------------------------------------
# GRIT: Faster and Better Image captioning Transformer
# Licensed under the Creative Commons Attribution.
# ------------------------------------------------------------------------
# Modified from Meshed Memory Transformer
# https://github.com/aimagelab/meshed-memory-transformer
# ------------------------------------------------------------------------

import torch
from models.caption.containers import Module


class BaseCaptioner(Module):

    def __init__(self):
        super(BaseCaptioner, self).__init__()

    def init_weights(self):
        raise NotImplementedError

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        raise NotImplementedError

    def forward(self, images, seq, *args):
        device = images.device
        b_s = images.size(0)
        seq_len = seq.size(1)
        state = self.init_state(b_s, device)
        out = None

        outputs = []
        for t in range(seq_len):
            out, state = self.step(t, state, out, images, seq, *args, mode='teacher_forcing')
            outputs.append(out)

        outputs = torch.cat([o.unsqueeze(1) for o in outputs], 1)
        return outputs

    # def beam_search(self, visual, max_len: int, eos_idx: int, beam_size: int, out_size=1, return_probs=False, **kwargs):
    #     bs = BeamSearch(self, max_len, eos_idx, beam_size)
    #     return bs.apply(visual, out_size, return_probs, **kwargs)
