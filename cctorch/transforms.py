import torch


def xcorr_lag(nt):
    nxcor = 2 * nt - 1
    return torch.arange(-(nxcor // 2), -(nxcor // 2) + nxcor)


def nextpow2(i):
    n = 1
    while n < i:
        n *= 2
    return n


def FFT(x):
    """
    assume fft axis in dim=-1
    """
    ntime = x.shape[-1]
    cclag = xcorr_lag(ntime)
    nxcor = len(cclag)
    nfast = nextpow2(nxcor)
    return torch.fft.rfft(x, n=nfast, dim=-1)


def FFT_NORMALIZE(x):
    """"""
    x[:] = FFT(x)
    mean = torch.mean(x, dim=-1, keepdims=True)
    std = torch.std(x, dim=-1, keepdims=True, unbiased=False)
    x[:] = ((x-mean) / std) / torch.sqrt(torch.Tensor([x.shape[-1]]))
    return x


def MA(x, nma=20):
    """
    Moving average in dim=-1
    """
    m = torch.nn.AvgPool1d(nma, stride=1, padding=nma // 2)
    return m(x.transpose(1, 0))[:, : x.shape[0]].transpose(1, 0)
