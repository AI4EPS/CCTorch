import torch


def xcorr_lag(nt):
    nxcor = 2 * nt - 1
    return torch.arange(-(nxcor // 2), -(nxcor // 2) + nxcor)


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def fft_real(x):
    """
    assume fft axis in dim=-1
    """
    ntime = x.shape[-1]
    nfast = nextpow2(2*ntime-1)
    return torch.fft.rfft(x, n=nfast, dim=-1)


def fft_normalize(x):
    """"""
    x -= torch.mean(x, dim=-1, keepdims=True)
    x /= x.square().sum(dim=-1, keepdims=True).sqrt()
    return fft_real(x)
