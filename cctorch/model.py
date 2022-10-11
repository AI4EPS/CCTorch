import torch
import torch.nn as nn
import torch.nn.functional as F

from .transforms import (
    interp_time_cubic_spline,
    pick_mccc_refine,
    pick_Rkt_maxabs,
    pick_Rkt_mccc,
)


class CCModel(nn.Module):
    def __init__(
        self,
        dt,
        maxlag,
        nma=20,
        device="cuda",
        channel_shift=0,
        mccc=False,
        reduce_t=True,
        reduce_x=False,
        domain="time",
        to_device=True,
        batching=True,
    ):
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
        self.domain = domain
        self.mccc = mccc

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

        if self.domain == "time":
            ## using conv1d
            nb1, nc1, nt1 = data1.shape
            data1 = data1.view(1, nb1 * nc1, nt1)
            nb2, nc2, nt2 = data2.shape
            data2 = data2.view(nb2 * nc2, 1, nt2)
            assert nt2 <= nt1
            if self.channel_shift > 0:
                xcor = F.conv1d(
                    data1, torch.roll(data2, self.channel_shift, dims=-1), padding=self.nlag + 1, groups=nb1 * nc1
                )
            else:
                xcor = F.conv1d(data1, data2, padding=self.nlag + 1, groups=nb1 * nc1)
            xcor = xcor.view(nb1, nc1, -1)
        elif self.domain == "frequency":
            # xcorr with fft in frequency domain
            nfast = (data1.shape[-1] - 1) * 2
            if self.channel_shift > 0:
                xcor_freq = torch.conj(data1) * torch.roll(data2, self.channel_shift, dims=-1)
            else:
                xcor_freq = torch.conj(data1) * data2
            xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
            xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - self.nlag + 1 : nfast // 2 + self.nlag]

        # cross-correlation matrix for one event pair
        result = {"id1": event1, "id2": event2, "xcor": xcor, "dt": self.dt, "channel_shift": self.channel_shift}
        # picking traveltime difference
        if self.reduce_t:
            # MCCC pick
            if self.mccc:
                scale_factor = 1
                # xcor_interp = interp_time_cubic_spline(xcor[0, :, :], scale_factor=scale_factor)
                pick_dt, G0, d0 = pick_Rkt_mccc(
                    xcor[0, :, :], self.dt / scale_factor, scale_factor=1, verbose=False, cuda=True
                )
                vmax, vmin, cc_dt = pick_mccc_refine(
                    xcor[0, :, :], self.dt / scale_factor, pick_dt, G0=G0, d0=d0, verbose=False
                )
                result["cc_mean"] = torch.mean(torch.abs(vmax))
                if not self.reduce_x:
                    result["cc_main"] = vmax
                    result["cc_side"] = vmin
                    result["cc_dt"] = cc_dt
                    result["cc_mean"] = torch.mean(torch.abs(vmax))
            # Simple pick
            else:
                xcor[:] = self.moving_average(xcor, nma=self.nma)
                if self.reduce_x:
                    vmax = torch.mean(torch.max(torch.abs(xcor), dim=-1).values, dim=-1, keepdim=True)
                    result["cc_mean"] = vmax
                else:
                    vmax, vmin, tmax = pick_Rkt_maxabs(xcor, self.dt)
                    result["cc_main"] = vmax
                    result["cc_side"] = vmin
                    result["cc_dt"] = tmax
        return result

    def moving_average(self, x, nma=20):
        m = torch.nn.AvgPool1d(nma, stride=1, padding=nma // 2)
        return m(x.permute(0, 2, 1))[:, :, : x.shape[1]].permute(0, 2, 1)
