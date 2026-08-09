import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class CCModel(nn.Module):
    def __init__(
        self,
        config,
        transforms=None,
        batch_size=16,
        to_device=False,
        device="cuda",
    ):
        super(CCModel, self).__init__()
        self.dt = config.dt
        self.nma = config.nma
        self.channel_shift = config.channel_shift
        self.reduce_t = config.reduce_t  # time reduction
        self.reduce_x = config.reduce_x  # station reduction
        self.reduce_c = config.reduce_c  # channel reduction
        self.domain = config.domain
        self.mccc = config.mccc
        self.use_pair_index = config.use_pair_index
        self.pre_fft = config.pre_fft

        self.transforms = transforms
        self.batch_size = batch_size
        self.to_device = to_device
        self.device = device

        # TM
        self.shift_t = config.shift_t
        self.normalize = config.normalize

        # AN
        self.nlag = config.nlag
        self.nfft = self.nlag * 2
        self.window = torch.hann_window(self.nfft, periodic=False).to(self.device)
        self.spectral_whitening = config.spectral_whitening

    def forward(self, x):
        """Perform cross-correlation on input data
        Args:
            x (tuple):
                - x[0] (dict):
                    - data (torch.Tensor): data1 with shape (batch, nsta/nch, nt)
                    - info (dict): attributes information of data1
                - x[1] (dict):
                    - data (torch.Tensor): data2 with shape (batch, nsta/nch, nt)
                    - info (dict): information information of data2
        """

        x1, x2 = x
        if self.to_device:
            data1 = x1["data"].to(self.device)
            data2 = x2["data"].to(self.device)
        else:
            data1 = x1["data"]
            data2 = x2["data"]

        if self.domain == "frequency":
            # xcorr with fft in frequency domain
            nfast = (data1.shape[-1] - 1) * 2
            nfast = 2 ** int(math.ceil(math.log2(nfast)))
            if not self.pre_fft:
                data1 = torch.fft.rfft(data1, n=nfast, dim=-1)
                data2 = torch.fft.rfft(data2, n=nfast, dim=-1)
            if self.channel_shift != 0:
                xcor_freq = data1 * torch.roll(torch.conj(data2), self.channel_shift, dims=-2)
            else:
                xcor_freq = data1 * torch.conj(data2)
            xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
            xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - self.nlag : nfast // 2 + self.nlag + 1]

        elif self.domain == "time":
            ## using conv1d in time domain
            if self.normalize:
                data1 = data1.to(torch.float64)
                data2 = data2.to(torch.float64)

            nb1, nc1, nx1, nt1 = data1.shape
            nb2, nc2, nx2, nt2 = data2.shape

            eps = torch.finfo(data1.dtype).eps * 10.0
            nlag = nt2 // 2

            if self.shift_t:
                nt_index = torch.arange(nt1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                shift_index = x2["info"]["traveltime_index"]
                shift_index = shift_index.repeat_interleave(
                    3, dim=1
                )  # repeat to match three channels for P and S wave templates
                shift_index = (nt_index + shift_index.unsqueeze(-1)) % nt1
                data1 = data1.gather(-1, shift_index)

            ## A template slot is nt2 samples wide, but a P window truncated at the S
            ## arrival fills only part of it and the rest is zero padding. The mask marks
            ## the real samples, so the padding is left out of both the template's own
            ## normalization and the running mean/std of the continuous data it is compared
            ## against. Without it a template that fills 250 of 400 samples correlates with
            ## a copy of itself at ~0.6 instead of 1.0.
            if "template_mask" in x2["info"]:
                mask = x2["info"]["template_mask"]
                if not torch.is_tensor(mask):
                    mask = torch.as_tensor(np.asarray(mask))
                mask = mask.to(device=data2.device, dtype=data2.dtype)
            else:
                mask = torch.ones_like(data2)
            n_valid = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            ## channels that carry no data at all -- a single-component station leaves two
            ## of the three empty -- must not be averaged in below, or the best cc such a
            ## station can reach is 1/3 and it never clears min_cc
            n_chan = (mask.sum(dim=-1) > 0).sum(dim=-2, keepdim=True).clamp(min=1)

            if self.normalize:
                ## population std over the valid samples, not torch.std's sample std: the
                ## denominator below counts the same n, so the pair is the plain Pearson
                ## coefficient. (torch.std divides by n-1 and left cc high by sqrt(n/(n-1)).)
                data2 = data2 * mask
                data2 = (data2 - data2.sum(dim=-1, keepdim=True) / n_valid) * mask
                data2 = data2 / (torch.sqrt((data2**2).sum(dim=-1, keepdim=True) / n_valid) + eps)
                data1 = data1 - torch.mean(data1, dim=-1, keepdim=True)
                data1 = data1 / (torch.std(data1, dim=-1, keepdim=True) + eps)

            data1 = data1.view(1, nb1 * nc1 * nx1, nt1)  # long
            data2 = data2.reshape(nb2 * nc2 * nx2, 1, nt2)  # short
            mask = mask.reshape(nb2 * nc2 * nx2, 1, nt2)

            # xcorr
            data1 = F.pad(data1, (nlag, nt2 - 1 - nlag), mode="constant", value=0)
            xcor = F.conv1d(data1, data2, stride=1, groups=nb1 * nc1 * nx1)

            if self.normalize:
                ## the same grouped convolution, with the mask as the kernel, gives the
                ## running sums of the continuous data over each template's own window
                n = n_valid.reshape(1, nb2 * nc2 * nx2, 1)
                sum1 = F.conv1d(data1, mask, stride=1, groups=nb1 * nc1 * nx1)
                sum2 = F.conv1d(data1**2, mask, stride=1, groups=nb1 * nc1 * nx1)
                std1 = torch.sqrt(torch.clamp(sum2 / n - (sum1 / n) ** 2, 0))
                xcor = xcor / n / (std1 + eps)

            xcor = xcor.view(nb2, nc2, nx2, -1)

            if self.reduce_x:
                xcor = torch.sum(xcor, dim=(-2), keepdim=True)
            if self.reduce_c:
                xcor = torch.sum(xcor, dim=(-3), keepdim=True) / n_chan.unsqueeze(-1)

        elif self.domain == "stft":
            nlag = self.nlag
            nb1, nc1, nx1, nt1 = data1.shape
            # nb2, nc2, nx2, nt2 = data2.shape
            data1 = data1.view(nb1 * nc1 * nx1, nt1)
            # data2 = data2.view(nb2 * nc2 * nx2, nt2)
            data2 = data2.view(nb1 * nc1 * nx1, nt1)
            if not self.pre_fft:
                data1 = torch.stft(
                    data1,
                    n_fft=self.nlag * 2,
                    hop_length=self.nlag,
                    window=self.window,
                    center=True,
                    return_complex=True,
                )
                data2 = torch.stft(
                    data2,
                    n_fft=self.nlag * 2,
                    hop_length=self.nlag,
                    window=self.window,
                    center=True,
                    return_complex=True,
                )
            if self.spectral_whitening:
                # freqs = np.fft.fftfreq(self.nlag*2, d=self.dt)
                # data1 = data1 / torch.clip(torch.abs(data1), min=1e-7) #float32 eps
                # data2 = data2 / torch.clip(torch.abs(data2), min=1e-7)
                data1 = torch.exp(1j * data1.angle())
                data2 = torch.exp(1j * data2.angle())

            xcor = torch.fft.irfft(torch.sum(data1 * torch.conj(data2), dim=-1), dim=-1)
            xcor = torch.roll(xcor, self.nlag, dims=-1)
            xcor = xcor.view(nb1, nc1, nx1, -1)

        else:
            raise ValueError("domain should be frequency or time or stft")

        # pair_index = [(i.item(), j.item()) for i, j in zip(x1["info"]["index"], x2["info"]["index"])]
        pair_index = [(i, j) for i, j in zip(x1["index"], x2["index"])]

        meta = {
            "xcorr": xcor,
            "pair_index": pair_index,
            "nlag": nlag,
            "data1": x1["data"],
            "data2": x2["data"],
            "info1": x1["info"],
            "info2": x2["info"],
        }

        if self.transforms is not None:
            meta = self.transforms(meta)

        output = {}
        for key in meta:
            # if key not in ["data1", "data2", "info1", "info2"]:
            if key not in ["data1", "data2"]:
                if isinstance(meta[key], torch.Tensor):
                    output[key] = meta[key].cpu()
                else:
                    output[key] = meta[key]

        return output

    def forward_map(self, x):
        """Perform cross-correlation on input data (dataset_type == map)
        Args:
            x (dict):
                - data (torch.Tensor): input data with shape (batch, nsta/nch, nt)
                - pair_index (torch.Tensor): pair index
                - info (dict): attributes information
        """
        if self.to_device:
            data = x["data"].to(self.device)
            pair_index = x["pair_index"].to(self.device)
        else:
            data = x["data"]
            pair_index = x["pair_index"]
        num_pairs = pair_index.shape[0]

        for i in tqdm(range(0, num_pairs, self.batch_size)):
            c1 = pair_index[i : i + self.batch_size, 0]
            c2 = pair_index[i : i + self.batch_size, 1]
            if len(c1) == 1:  ## returns a view of the original tensor
                data1 = torch.select(data, 0, c1[0]).unsqueeze(0)
            else:
                data1 = torch.index_select(data, 0, c1)
            if len(c2) == 1:
                data2 = torch.select(data, 0, c1[0]).unsqueeze(0)
            else:
                data2 = torch.index_select(data, 0, c2)

            if self.domain == "time":
                ## using conv1d in time domain
                nb1, nc1, nt1 = data1.shape
                data1 = data1.view(1, nb1 * nc1, nt1)
                nb2, nc2, nt2 = data2.shape
                data2 = data2.view(nb2 * nc2, 1, nt2)
                if self.channel_shift != 0:
                    xcor = F.conv1d(
                        data1, torch.roll(data2, self.channel_shift, dims=-2), padding=self.nlag + 1, groups=nb1 * nc1
                    )
                else:
                    xcor = F.conv1d(data1, data2, padding=self.nlag + 1, groups=nb1 * nc1)
                xcor = xcor.view(nb1, nc1, -1)

            elif self.domain == "frequency":
                # xcorr with fft in frequency domain
                nfast = (data1.shape[-1] - 1) * 2
                if self.channel_shift != 0:
                    xcor_freq = data1 * torch.roll(torch.conj(data2), self.channel_shift, dims=-2)
                else:
                    xcor_freq = data1 * torch.conj(data2)
                xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
                xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[
                    ..., nfast // 2 - self.nlag + 1 : nfast // 2 + self.nlag
                ]

            meta = {"xcorr": xcor, "pair_index": pair_index}
            if self.transforms is not None:
                meta = self.transforms(meta)

        return meta
