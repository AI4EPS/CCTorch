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
        self.nlag = config.nlag
        self.nma = config.nma
        self.channel_shift = config.channel_shift
        self.reduce_t = config.reduce_t
        self.reduce_x = config.reduce_x
        self.domain = config.domain
        self.mccc = config.mccc
        self.use_pair_index = config.use_pair_index
        self.transforms = transforms
        self.batch_size = batch_size
        self.to_device = to_device
        self.device = device

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

        if self.domain == "time":
            ## using conv1d in time domain
            nb1, nc1, nx1, nt1 = data1.shape
            data1 = data1.view(1, nb1 * nc1 * nx1, nt1)
            nb2, nc2, nx2, nt2 = data2.shape
            data2 = data2.view(nb2 * nc2 * nx2, 1, nt2)
            if self.channel_shift != 0:
                xcor = F.conv1d(
                    data1, torch.roll(data2, self.channel_shift, dims=-2), padding=self.nlag, groups=nb1 * nc1 * nx1
                )
            else:
                xcor = F.conv1d(data1, data2, padding=self.nlag, groups=nb1 * nc1 * nx1)
            xcor = xcor.view(nb1, nc1, nx1, -1)

        elif self.domain == "frequency":
            # xcorr with fft in frequency domain
            nfast = (data1.shape[-1] - 1) * 2
            if self.channel_shift != 0:
                xcor_freq = data1 * torch.roll(torch.conj(data2), self.channel_shift, dims=-2)
            else:
                xcor_freq = data1 * torch.conj(data2)
            xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
            xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - self.nlag : nfast // 2 + self.nlag + 1]

        pair_index = [(i.item(), j.item()) for i, j in zip(x1["info"]["index"], x2["info"]["index"])]
        meta = {"xcorr": xcor, "pair_index": pair_index, 
                "nlag": self.nlag,
                "data1": x1["data"], "data2": x2["data"]}
        if self.transforms is not None:
            meta = self.transforms(meta)

        return meta


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
