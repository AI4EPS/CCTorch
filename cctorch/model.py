import torch
import torch.nn as nn

from .transforms import FFT, FFT_NORMALIZE, MA, nextpow2, xcorr_lag


class CCModel(nn.Module):
    def __init__(self, device, dt, maxlag, nma=20):
        super(CCModel, self).__init__()
        self.device = device
        self.nlag = int(maxlag / dt)
        self.xcor_time_axis = xcorr_lag(self.nlag) * dt
        self.nma = nma

    def forward(self, x):
        x1, x2 = x
        data1 = x1["data"].to(self.device)
        data2 = x2["data"].to(self.device)

        # xcorr
        nfast = data1.shape[-1] - 1

        ## temporary solution for compelx number
        data1 = torch.view_as_complex(data1)
        data2 = torch.view_as_complex(data2)
        xcor_freq = torch.conj(data1) * data2
        xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
        xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[:, nfast // 2 - self.nlag + 1 : nfast // 2 + self.nlag]
        # moving average
        xcor = MA(xcor, nma=self.nma)
        # pick
        vmax, imax = torch.max(xcor, dim=1)
        vmin, imin = torch.min(xcor, dim=1)
        ineg = torch.abs(vmin) > vmax
        vmax[ineg] = vmin[ineg]
        imax[ineg] = imin[ineg]
        return {"cc": vmax, "dt": self.xcor_time_axis[imax]}
