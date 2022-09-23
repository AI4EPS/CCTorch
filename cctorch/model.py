import torch
import torch.nn as nn


class CCModel(nn.Module):
    def __init__(self, dt, maxlag, nma=20, device="cuda", channel_shift=0, reduce_t=True, reduce_x=False, to_device=True, batching=True):
        super(CCModel, self).__init__()
        self.device = device
        self.to_device = to_device
        self.batching = batching
        self.dt = dt
        self.nlag = int(maxlag / dt)
        self.nma = nma
        self.channel_shift = channel_shift
        self.reduce_t = reduce_t
        self.reduce_x = reduce_x

    def forward(self, x):
        x1, x2 = x
        if self.to_device:
            data1 = x1["data"].to(self.device)
            data2 = x2["data"].to(self.device)
        else:
            data1 = x1["data"]
            data2 = x2["data"]
        if not self.batching:
            data1 = data1.unsqueeze(0)
            data2 = data2.unsqueeze(0)
            event1 = x1["event"].unsqueeze(0)
            event2 = x2["event"].unsqueeze(0)
        else:
            ## temporary solution for compelx number
            data1 = torch.view_as_complex(data1)
            data2 = torch.view_as_complex(data2)
            event1 = x1["event"]
            event2 = x2["event"]

        # xcorr
        nfast = (data1.shape[-1] - 1) * 2
        if self.channel_shift > 0:
            xcor_freq = torch.conj(data1) *  torch.roll(data2, self.channel_shift, dims=1)
        else:
            xcor_freq = torch.conj(data1) * data2
        xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
        xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - self.nlag + 1 : nfast // 2 + self.nlag]
        # moving average
        xcor[:] = self.moving_average(xcor, nma=self.nma)
        # pick
        if self.reduce_t:
            if self.reduce_x:
                vmax = torch.mean(torch.max(torch.abs(xcor), dim=-1).values, dim=-1, keepdim=True)
                imax = 0
            else:
                vmax, imax = torch.max(xcor, dim=-1)
                vmin, imin = torch.min(xcor, dim=-1)
                ineg = torch.abs(vmin) > vmax
                vmax[ineg] = vmin[ineg]
                imax[ineg] = imin[ineg]
            return {"id1": event1, "id2": event2, "cc": vmax, "dt": (imax - self.nlag + 1) * self.dt, "channel_shift": self.channel_shift}
        else:
            return {"id1": event1, "id2": event2, "cc": xcor, "dt": self.dt, "channel_shift": self.channel_shift}

    def moving_average(self, x, nma=20):
        m = torch.nn.AvgPool1d(nma, stride=1, padding=nma // 2)
        return m(x.permute(0, 2, 1))[:, :, : x.shape[1]].permute(0, 2, 1)
