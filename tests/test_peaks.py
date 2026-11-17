"""DetectPeaksCC keeps the three largest peaks of |cc|, each refined: the right alignment must be among them.

Synthetic pairs through the path of a template run (normalize -> CCModel in the time domain -> DetectPeaksCC): window 2
is window 1 delayed by a known number of samples (fractions too), with a known sign, on three channels. The pulse has
several cycles, so the correlation peaks of opposite sign half a period away are nearly as high as the right one, and
band-limited noise makes one of them the largest in part of the pairs.

    pytest tests/test_peaks.py
"""
import sys
from types import SimpleNamespace

import numpy as np
import torch
from scipy.signal import butter, filtfilt

sys.path.insert(0, __file__.rsplit("/", 2)[0])
from cctorch import CCModel
from cctorch.transforms import DetectPeaksCC, normalize

NT, FS, F0, WIDTH = 400, 100.0, 6.0, 0.15  # samples, Hz, Hz, s: the half period is 8.3 samples
CONFIG = SimpleNamespace(dt=1 / FS, nma=(20, 0), channel_shift=0, reduce_t=False, reduce_x=False, reduce_c=False, domain="time", mccc=False,
                         use_pair_index=False, pre_fft=False, shift_t=False, normalize=False, dtype=torch.float32, nlag=int(0.5 * FS), spectral_whitening=True)


def pairs(n, noise, seed=0):
    """n pairs of windows (n, 3, 1, NT) float32, the delay of window 2 (n,) in samples and its sign (n,)."""
    rng = np.random.default_rng(seed)
    delay, sign = rng.uniform(-20, 20, n), rng.choice([-1.0, 1.0], n)
    t = (np.arange(NT) - NT / 2) / FS
    pulse = lambda t: np.exp(-(t**2) / (2 * WIDTH**2)) * np.cos(2 * np.pi * F0 * t)
    gain = np.array([1.0, 0.6, 0.3])[:, None, None]  # (3, 1, 1): the channels differ in amplitude only
    b, a = butter(2, [2, 12], "bandpass", fs=FS)
    band_noise = lambda: noise * filtfilt(b, a, rng.standard_normal((n, 3, 1, NT)))
    x1 = gain * pulse(t)[None, None, None, :] + band_noise()
    x2 = gain * (sign[:, None] * pulse(t[None, :] - delay[:, None] / FS))[:, None, None, :] + band_noise()
    return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32), delay, sign


def detect(x1, x2, detector):
    """The detector's output for the pairs (x1[i], x2[i])."""
    model = CCModel(config=CONFIG, transforms=detector, device="cpu")
    info = {"idx": np.arange(len(x1))}
    return model(({"data": normalize(x1.clone()), "index": info["idx"], "info": info}, {"data": normalize(x2.clone()), "index": info["idx"], "info": info}))


def stored(out):
    """The three peaks of each pair: shift (n, 3) samples and signed cc (n, 3), the top peak first."""
    shift = np.concatenate([out["cc_shift"][..., None], out["cc_peaks_shift"]], -1).reshape(-1, 3)
    return shift, np.concatenate([out["cc_max"][..., None], out["cc_peaks_cc"]], -1).reshape(-1, 3)


def test_clean_pairs():
    x1, x2, delay, sign = pairs(500, noise=0.0)
    shift, cc = stored(detect(x1, x2, DetectPeaksCC(topk=3)))
    assert np.abs(shift[:, 0] + delay).max() < 0.02, "the top peak is the true alignment: shift = -delay of window 2"
    assert (np.sign(cc[:, 0]) == sign).all() and (np.abs(cc[:, 0]) > 0.99).all() and (np.abs(cc[:, 0]) < 1.001).all()  # a cc, not a sum of three
    half = np.abs(shift[:, 1:] - shift[:, :1])  # the two lesser peaks sit half a period from the top one, with the other sign
    assert (np.abs(half - FS / F0 / 2) < 0.5).all() and (np.sign(cc[:, 1:]) == -sign[:, None]).all()
    assert (np.abs(cc[:, 0]) > np.abs(cc[:, 1:]).max(1)).all()  # the lesser two are ordered by their sample value, not by the refined one


def test_noisy_pairs_keep_the_true_alignment():
    x1, x2, delay, sign = pairs(4000, noise=0.25)
    shift, cc = stored(detect(x1, x2, DetectPeaksCC(topk=3)))
    near = np.abs(shift + delay[:, None]) <= 1.0  # (n, 3): this stored peak is the true alignment
    skipped = ~near[:, 0]
    assert 0.1 < skipped.mean() < 0.5, f"the noise must make another peak the largest in part of the pairs: {skipped.mean():.3f}"
    assert near.any(1).mean() > 0.97, f"the true alignment is one of the three stored peaks: {near.any(1).mean():.4f}"
    k = near.argmax(1)[near.any(1)]
    assert (np.sign(cc[near.any(1), k]) == sign[near.any(1)]).all(), "the stored peak at the true alignment has the true sign"
    assert near[:, :2].any(1).mean() < near.any(1).mean(), "the third peak is needed in part of the pairs"
    print(f"top peak wrong {skipped.mean():.3f}; true alignment among two peaks {near[:, :2].any(1).mean():.4f}, among three {near.any(1).mean():.4f}")


def test_top_peak_does_not_depend_on_the_number_kept():
    x1, x2, _, _ = pairs(2000, noise=0.25, seed=1)
    two, three = detect(x1, x2, DetectPeaksCC(topk=2)), detect(x1, x2, DetectPeaksCC(topk=3))
    for key in ("cc_max", "cc_shift", "cc_dt", "cc_weight", "cc_contrast", "cc_peak_lag"):
        assert np.array_equal(two[key], three[key]), key
    assert np.array_equal(two["cc_peaks_shift"][..., 0], three["cc_peaks_shift"][..., 0]) and three["cc_peaks_cc"].shape == (2000, 1, 1, 2)


def test_no_peak_is_nan():
    x1, x2, _, _ = pairs(4, noise=0.0)
    x2[:2] = 0  # a window without data: the correlation is 0 everywhere
    out = detect(x1, x2, DetectPeaksCC(topk=3))
    assert (out["cc_max"][:2] == 0).all() and np.isnan(out["cc_peaks_shift"][:2]).all() and np.isnan(out["cc_peaks_cc"][:2]).all()
    assert np.isfinite(out["cc_peaks_shift"][2:]).all() and np.isfinite(out["cc_peaks_cc"][2:]).all()


def test_empty_channels_do_not_change_a_pair():
    """A vertical-only station in a three-channel memmap: two slots of zeros. The pair must give what one channel gives."""
    x1, x2, _, _ = pairs(100, noise=0.3)
    one = stored(detect(x1[:, :1], x2[:, :1], DetectPeaksCC(topk=3)))
    x1[:, 1:], x2[:, 1:] = 0, 0
    three = stored(detect(x1, x2, DetectPeaksCC(topk=3)))
    assert all(np.array_equal(a, b, equal_nan=True) for a, b in zip(one, three))

