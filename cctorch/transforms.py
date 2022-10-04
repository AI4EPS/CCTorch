import torch
import numpy as np
from scipy import sparse
from scipy.signal import tukey
from scipy.sparse.linalg import lsmr
from tqdm import tqdm


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
    nfast = nextpow2(2 * ntime - 1)
    return torch.fft.rfft(x, n=nfast, dim=-1)


def fft_real_normalize(x):
    """"""
    x -= torch.mean(x, dim=-1, keepdims=True)
    x /= x.square().sum(dim=-1, keepdims=True).sqrt()
    return fft_real(x)


# torch helper functions
def count_tensor_byte(*args):
    total_byte_size = 0
    for arg in args:
        total_byte_size += arg.element_size()*arg.nelement()
    return total_byte_size


def print_total_size(total_byte_size):
    size_unit = ['Byte', 'KB', 'MB', 'GB', 'TB']
    i = 0
    while total_byte_size > 1024 and i < len(size_unit):
        total_byte_size /= 1024
        i += 1
    print(f'Total memory is {total_byte_size} {size_unit[i]}')


def moving_average(data, ma):
    """
    moving average with AvgPool1d along axis=0
    """
    if isinstance(data, np.ndarray):
        data = torch.tensor(data)
    m = torch.nn.AvgPool1d(ma, stride=1, padding=ma//2)
    data_ma = m(data.transpose(1, 0))[:, :data.shape[0]].transpose(1, 0)
    return data_ma


def gather_roll(data, shift_index):
    """
    roll data[irow, :] along axis 1 by the amount of shift[irow]
    """
    nrow, ncol = data.shape
    index = torch.arange(ncol, device=data.device).view([1, ncol]).repeat((nrow, 1))
    index = (index-shift_index.view([nrow, 1]))%ncol
    return torch.gather(data, 1, index)


def taper_time(data, alpha=0.8):
    taper = tukey(data.shape[-1], alpha)
    return data * torch.tensor(taper, device=data.device)


def h_poly(t):
    tt = t[None, :]**torch.arange(4, device=t.device)[:, None]
    A = torch.tensor([
        [1, 0, -3, 2],
        [0, 1, -2, 1],
        [0, 0, 3, -2],
        [0, 0, -1, 1]
    ], dtype=t.dtype, device=t.device)
    return A @ tt


def interp1d_cubic_spline(x, y, xs):
    m = (y[1:] - y[:-1]) / (x[1:] - x[:-1])
    m = torch.cat([m[[0]], (m[1:] + m[:-1]) / 2, m[[-1]]])
    idxs = torch.searchsorted(x[1:], xs)
    dx = (x[idxs + 1] - x[idxs])
    hh = h_poly((xs - x[idxs]) / dx)
    return hh[0] * y[idxs] + hh[1] * m[idxs] * dx + hh[2] * y[idxs + 1] + hh[3] * m[idxs + 1] * dx


def interp_time_cubic_spline(data, scale_factor=10):
    """"""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    nrow, ncol = data.shape
    xb = torch.arange(ncol, device=data.device)
    xq = torch.linspace(0, ncol-1, scale_factor*(ncol-1)+1, device=data.device)
    dataq = torch.zeros((nrow, len(xq)), device=data.device)
    for i in range(nrow):
        dataq[i, :] = interp1d_cubic_spline(xb, data[i, :], xq)
    return dataq


def xcorr_phase(data1, data2, dt, maxlag=0.5, channel_shift=0):
    """
    cross-correlatin between event phase data
    """
    # xcorr
    data_freq1 = fft_real_normalize(data1)
    data_freq2 = fft_real_normalize(data2)
    nlag = int(maxlag / dt)
    nfast = (data_freq1.shape[-1] - 1) * 2
    if channel_shift > 0:
        xcor_freq = torch.conj(data_freq1) * torch.roll(data_freq2, channel_shift, dims=1)
    else:
        xcor_freq = torch.conj(data_freq1) * data_freq2
    xcor_time = torch.fft.irfft(xcor_freq, n=nfast, dim=-1)
    xcor = torch.roll(xcor_time, nfast // 2, dims=-1)[..., nfast // 2 - nlag + 1 : nfast // 2 + nlag]
    xcor_time_axis = (xcorr_lag(nlag)*dt).numpy()
    xcor_info = {'nx': data1.shape[0], 'nt': len(xcor_time_axis), 'dt': dt, 'time_axis': xcor_time_axis}
    return xcor, xcor_info


def pick_Rkt_mccc(Rkt_ij, dt, 
        maxlag=0.3, taper=0.8, scale_factor=10, ma=40,
        damp=1, cc_threshold=0.7, mccc_maxlag=0.04, win_threshold=10,
        chunk_size=50000, cuda=False, device_id=0, verbose=False):
    """
    pick cc matrix's peaks using multi-channel cross-correlation
    Args:
        Rkt_ij: xcor matrix between event pair: (i, j), shape=[nchan, ntime]
        dt: sampling interval of Rkt_ij, [sec]
        maxlag: maximum lag to examine for the xcor matrix
        taper: taper ratio
        scale_factor: interpolation scaling factor
        ma: moving average window length
    Returns:
        pick_tt_mccc: purely relative traveltime pick
        G, d: matrices for least-square inversion
    """
    if isinstance(Rkt_ij, np.ndarray):
        Rkt_ij = torch.tensor(Rkt_ij)
    # if cuda:
    #     device = torch.device(device_id)
    #     Rkt_ij = Rkt_ij.cuda(device)
    nt = Rkt_ij.shape[-1]
    nlag = int(maxlag/dt)
    nt = Rkt_ij.shape[-1]
    ic = nt//2
    ib = max([0, ic-nlag+1])
    ie = min([nt, ic+nlag])
    # taper selected window and do spline interpolation along time-axis
    if scale_factor > 1:
        Rkt_win_interp = interp_time_cubic_spline(taper_time(Rkt_ij[:, ib:ie], alpha=taper), scale_factor=scale_factor)
    else:
        Rkt_win_interp = taper_time(Rkt_ij[:, ib:ie], alpha=taper)
    # moving average along channel-axis
    if ma > 0:
        Rkt_win_interp[:] = moving_average(Rkt_win_interp, ma) 
    # multi-channel cross correlation
    solution = mccc(Rkt_win_interp, dt/scale_factor, cc_threshold, damp=damp, mccc_maxlag=mccc_maxlag, win_threshold=win_threshold, 
            chunk_size=chunk_size, cuda=cuda, device_id=device_id, verbose=verbose)
    pick_tt_mccc = torch.tensor(solution[0], device=Rkt_ij.device)
    # G and d
    G = solution[-2]
    d = solution[-1]
    return (pick_tt_mccc, G, d)


def pick_Rkt_maxabs(Rkt_ij, dt, maxlag=0.3, ma=0):
    """
    reduce time-axis for cc matrix
    Args:
        Rkt_ij: xcor matrix between event pair: (i, j), shape=[nchan, ntime]
    Returns:
        Ck: vector cc, shape = [nchan]
    """
    if ma > 0:
        Rkt_ij = moving_average(Rkt_ij, ma)
    nlag = int(maxlag/dt)
    nt = Rkt_ij.shape[-1]
    ic = nt//2
    ib = max([0, ic-nlag+1])
    ie = min([nt, ic+nlag])
    vmax, imax = torch.max(Rkt_ij[:, ib:ie], dim=-1)
    vmin, imin = torch.min(Rkt_ij[:, ib:ie], dim=-1)
    ineg = torch.abs(vmin) > vmax
    vmax[ineg], vmin[ineg] = vmin[ineg], vmax[ineg]
    imax[ineg] = imin[ineg]
    tmax = (imax - nlag + 1) * dt
    return (vmax, vmax, tmax)


def pick_mccc_refine(Rkt_ij, dt, pick_tt, ma=60, win_main=0.3, win_side=0.1, w0=10, G0=None, d0=None, max_niter=5, verbose=True):
    """
    iteratively pick peak cc for main lobe and side lobe along pick_tt
    Args:
        Rkt_ij: xcor matrix between event pair: (i, j), shape=[nchan, ntime]
        win_main: half width of window for picking peak cc of main lobe: 0.5/freq_dominant
        win_side: half width of window for picking peak cc of side lobe: 1/freq_dominant
    """
    nt = Rkt_ij.shape[-1]
    ic = nt//2
    # window including main and side lobes
    nwin2 = int(win_side/dt)
    indx_lb = max([0, ic-nwin2+1])
    indx_le = max([0, ic-nwin2//2+1])
    indx_rb = min([nt, ic+nwin2//2])
    indx_re = min([nt, ic+nwin2]) 
    # allocate
    Rkt_ij_shift = torch.zeros_like(Rkt_ij)
    pick_tt_refine = pick_tt.clone()
    tmax = torch.zeros_like(pick_tt)
    #tmax_prev = pick_tt.clone()
    # main lobe max abs cc
    kiter = 0
    if verbose:
        pbar = tqdm(total=max_niter)
    while kiter <= max_niter:
        # shift the xcor data to flatten the cc peaks
        shift_index = -torch.round(pick_tt_refine/dt).int()
        Rkt_ij_shift[:] = gather_roll(Rkt_ij, shift_index)
        if ma > 0:
            Rkt_ij_shift[:] = moving_average(Rkt_ij_shift, ma=ma)
        # pick main lobe cc peak
        nwin1 = int(win_main/dt)
        ib1 = max([0, ic-nwin1+1])
        ie1 = min([nt, ic+nwin1])
        vmax, imax = torch.max(Rkt_ij_shift[:, ib1:ie1], dim=-1)
        vmin, imin = torch.min(Rkt_ij_shift[:, ib1:ie1], dim=-1)
        ineg = torch.abs(vmin) > vmax
        vmax[ineg] = vmin[ineg]
        imax[ineg] = imin[ineg]
        tmax[:] = imax * dt - (ic-ib1) * dt
        tmax += pick_tt_refine
        # tmax change:
        #dtmax0 = torch.mean(torch.abs(tmax-tmax_prev))
        #print(f'{dtmax0=:.3f}')
        #tmax_prev[:] = tmax
        # side lobe max abs cc
        vmin = torch.maximum(torch.max(torch.abs(Rkt_ij_shift[:, indx_lb:indx_le]), dim=-1).values, 
                            torch.max(torch.abs(Rkt_ij_shift[:, indx_rb:indx_re]), dim=-1).values)
        # vmax: max abs(cc) for main lobe
        # vmin: max abs(cc) for side lobe
        # tmax: refined pick_tt for main lobe
        if G0 is None or d0 is None: # or dtmax0 < dt/2:
            return (vmax, vmin, tmax)
        elif kiter < max_niter:
            isel = torch.where(torch.abs(vmax) > torch.quantile(torch.abs(vmax), 0.15))[0]
            if not isel.device.type == 'cpu':
                tsel = tmax[isel].cpu()
                isel = isel.cpu()
            else:
                tsel = tmax[isel]
            nsel = len(isel)
            G1 = sparse.coo_matrix((np.ones(nsel), (np.arange(nsel), isel.numpy())), shape=(nsel, len(vmax)))
            d1 = tsel.numpy()
            G = sparse.vstack([w0*G0, G1])
            d = np.concatenate([w0*d0, d1])
            sol = lsmr(G, d)
            pick_tt_refine[:] = torch.tensor(sol[0])
            w0 /= 1.1
            win_main /= 2
            if kiter == max_niter-1:
                win_main = min([0.01, win_main])
            #print(f'{win_main=}')
        kiter += 1
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    return (vmax, vmin, tmax)


def mccc(data, dt, cc_threshold, damp=1, mccc_maxlag=0.04, win_threshold=None, chunk_size=50000, cuda=False, verbose=True, device_id=0):
    """
    multi channel cross correlation
    Ref: VanDecar-1990-Determination of teleseismic relative phase arrival times using
        multil-channel cross-correlation and least squares
    Args:
        data: multi-channel data, shape=[nchan, ntime]
        dt: sampling interval
        cc_threshold: minimum cc threshold
        damp: damping factor for t[i+1] - t[i] = 0
        mccc_maxlag: maximum time lag for mccc
        win_threshold: maximum channel spacing for mccc
    Returns:
        solution: lsmr solution list
    """
    #
    if cuda:
        if data.device.type == 'cuda':
            device = data.device
        else:
            device = torch.device(device_id)
    else:
        device = data.device
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, device=device)
    # fft
    nchan, ntime = data.shape
    cclag = xcorr_lag(ntime)*dt
    nxcor = len(cclag)
    nfast = nextpow2(nxcor)
    nfast_r = nfast//2+1
    # window for selecting mccc cc values
    it_xcor_win = np.where(np.logical_and(cclag>=-mccc_maxlag, cclag<=mccc_maxlag))[0]
    tt_xcor_win = cclag[it_xcor_win]
    nt_xcor_win = len(it_xcor_win)
    # allocation
    # cross-correlation channel pair indices (i, j)
    if win_threshold is None:
        index_i = torch.tensor([i for i in range(nchan) for _ in range(i+1, nchan)], device=device)
        index_j = torch.tensor([j for i in range(nchan) for j in range(i+1, nchan)], device=device)
    else:
        index_i = torch.tensor([i for i in range(nchan) for _ in range(i+1, min(i+win_threshold, nchan))], device=device)
        index_j = torch.tensor([j for i in range(nchan) for j in range(i+1, min(i+win_threshold, nchan))], device=device)
    npair = len(index_i)
    # ffts
    # data_freq = torch.fft.rfft(data, n=nfast, dim=-1)
    data_freq = fft_real_normalize(data)
    xcor_freq = torch.complex(torch.zeros(chunk_size, nfast_r), torch.zeros(chunk_size, nfast_r))
    xcor_time = torch.zeros(chunk_size, nfast, device=device)
    xcor_time_win = torch.zeros(chunk_size, nt_xcor_win, device=device)
    # cc value and time shift
    value_cc = torch.zeros(npair)
    value_dt = torch.zeros(npair)
    # memory allocation size
    if verbose:
        total_byte_size = count_tensor_byte(index_i, index_j, xcor_freq, xcor_time, 
                xcor_time_win, value_cc, value_cc)
        print_total_size(total_byte_size)
    # to gpu
    if cuda:
        xcor_freq = xcor_freq.cuda(device)
    nchunk = int(np.ceil(npair/chunk_size))
    ib = 0
    if verbose:
        pbar = tqdm(total=nchunk)
    for _ in range(nchunk):
        ie = min(ib+chunk_size, npair)
        ii = index_i[ib:ie]
        jj = index_j[ib:ie]
        nn = ie-ib
        xcor_freq[:nn, :] = torch.conj(data_freq[ii, :]) * data_freq[jj, :]
        xcor_time[:nn, :] = torch.fft.irfft(xcor_freq[:nn, :], n=nfast, dim=-1)
        xcor_time_win[:nn, :] = torch.roll(xcor_time[:nn, :], nfast//2, dims=-1)[:, nfast//2-nt_xcor_win//2:nfast//2+(nt_xcor_win+1)//2]
        vmax, imax = torch.max(xcor_time_win[:nn, :], dim=-1)
        vmin, imin = torch.min(xcor_time_win[:nn, :], dim=-1)
        ineg = torch.abs(vmin) > vmax
        vmax[ineg] = vmin[ineg]
        imax[ineg] = imin[ineg]
        value_cc[ib:ie] = vmax
        value_dt[ib:ie] = tt_xcor_win[imax]
        ib = ie
        if verbose:
            pbar.update(1)
    if verbose:
        pbar.close()
    # back to cpu
    if cuda:
        index_i = index_i.cpu()
        index_j = index_j.cpu()
        value_cc = value_cc.cpu()
        value_dt = value_dt.cpu()
        del xcor_freq
        del xcor_time
        del xcor_time_win
        torch.cuda.empty_cache()
    # G matrix, dt_ij = ti-tj = 0
    igood = torch.where(torch.abs(value_cc) > cc_threshold)[0]
    ngood = len(igood)
    value_cc = value_cc[igood]
    value_dt = -value_dt[igood] # cc[d[ti], d[tj=ti+dt_ij]] vs dt_ij=ti-tj, needs a "-1"
    # weight = torch.abs(value_cc).numpy()
    weight = np.ones(len(value_cc))
    index_ii = np.tile(np.arange(ngood), 2)
    index_jj = torch.concat([index_i[igood], index_j[igood]]).numpy()
    value_ij = np.concatenate([np.ones(ngood)*weight, -np.ones(ngood)*weight])
    G = sparse.coo_matrix((value_ij, (index_ii, index_jj)), shape=(ngood, nchan))
    d = np.concatenate([value_dt.numpy()*weight, []])
    # regularization: t[i+1]-t[i] = 0, i=0, ..., n-2
    D = (np.diag(np.ones(nchan)) - np.diag(np.ones(nchan - 1), k=-1))[1:, :]
    D = sparse.csr_matrix(D) * damp
    G = sparse.vstack((G, D))
    d = np.concatenate([d, np.zeros(D.shape[0])])
    # least-square
    solution = lsmr(G, d)
    solution = list(solution)
    solution.append(G)
    solution.append(d)
    return solution

