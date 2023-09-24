# %%
import numpy as np
import json
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime, timezone
import fsspec

# %%
root_path = "local"
region = "demo"
protocal = "gs"
bucket = "quakeflow_share"
fs = fsspec.filesystem(protocol=protocal)
fs.get(
    f"{bucket}/{region}/waveforms/",
    f"{root_path}/{region}/waveforms/",
    recursive=True,
)
fs.get(
    f"{bucket}/{region}/cctorch/",
    f"{root_path}/{region}/cctorch/",
    recursive=True,
)

# %%
with open(f"{root_path}/{region}/cctorch/config.json", "r") as template:
    config = json.load(template)

print(json.dumps(config, indent=4, sort_keys=True))

# %%
template = np.memmap(
    f"{root_path}/{region}/cctorch/template.dat", dtype=np.float32, mode="r", shape=tuple(config["template_shape"])
)
traveltime_type = np.memmap(
    f"{root_path}/{region}/cctorch/traveltime_type.dat",
    dtype=np.int32,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)
traveltime_index = np.memmap(
    f"{root_path}/{region}/cctorch/traveltime_index.dat",
    dtype=np.int32,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)
arrivaltime_index = np.memmap(
    f"{root_path}/{region}/cctorch/arrivaltime_index.dat",
    dtype=np.int64,
    mode="r",
    shape=tuple(config["traveltime_shape"]),
)

# %%
station_index = pd.read_csv(
    f"{root_path}/{region}/cctorch/station_index.txt", header=None, names=["index", "station_id", "component"]
)
event_index = pd.read_csv(f"{root_path}/{region}/cctorch/event_index.txt", header=None, names=["index", "event_index"])

# %%
year = "2019"
jday = "185"
hour = "17"

meta = obspy.read(f"{root_path}/{region}/waveforms/{year}-{jday}/{hour}/*.mseed")
meta.merge(fill_value="latest")
for tr in meta:
    if tr.stats.sampling_rate != config["sampling_rate"]:
        if tr.stats.sampling_rate % config["sampling_rate"] == 0:
            tr.decimate(int(tr.stats.sampling_rate / config["sampling_rate"]))
        else:
            tr.resample(config["sampling_rate"])
begin_time = min([tr.stats.starttime for tr in meta])
end_time = max([tr.stats.endtime for tr in meta])
meta.detrend("constant")
meta.trim(begin_time, end_time, pad=True, fill_value=0)

# %%
nt = meta[0].stats.npts
data = np.zeros([3, len(station_index), nt])
for i, sta in station_index.iterrows():
    if len(sta["component"]) == 3:
        for j, c in enumerate(sta["component"]):
            tr = meta.select(id=f"{sta['station_id']}{c}")
            data[j, i, :] = tr[0].data
    else:
        j = config["component_mapping"][sta["component"]]
        tr = meta.select(id=f"{sta['station_id']}{c}")
        data[j, i, :] = tr[0].data

# %%
ieve = 2
ista = 0
icomp = config["component_mapping"][c]
print(traveltime_type[ieve, ista, icomp])
print(arrivaltime_index[ieve, ista, icomp])
arrivaltime_ = datetime.fromtimestamp(
    float(arrivaltime_index[ieve, ista, icomp]) / config["sampling_rate"], tz=timezone.utc
)
# arrivaltime_ = datetime.fromtimestamp(arrivaltime_index[ieve, ista, icomp], tz=timezone.utc)
print(arrivaltime_)


# %%
plt.figure(figsize=(20, 5))
plt.plot(template[ieve, ista, icomp, :])
plt.show()

# %%


def roll_by_index(tensor, indices):
    nb, nc, nx, nt = tensor.shape
    # assert nx == indices.numel(), "Mismatch in tensor and indices shape."

    # Compute the new indices after rolling
    roll_idx = (torch.arange(nt).unsqueeze(0) - indices.unsqueeze(1)) % nt

    # Use advanced indexing to get the rolled tensor
    result = tensor[:, torch.arange(nx).unsqueeze(1), roll_idx]

    return result


def sliding_window_mad(sqeuence, window_size):
    # Get tensor shape
    nb, nc, nx, nt = sqeuence.shape

    # Convert tensor to 2D with each row being a moving window
    unfolded = sqeuence.unfold(-1, window_size, 1)

    # Move the 'unfold' dimension to the end and reshape
    unfolded_reshaped = unfolded.permute(0, 1, 2, 4, 3).reshape(-1, window_size)

    # Compute the median for each row
    medians = unfolded_reshaped.median(dim=-1).values

    # Compute absolute deviations from the median for each element
    absolute_deviations = torch.abs(unfolded_reshaped - medians.unsqueeze(-1))

    # Compute MAD (median of the absolute deviations) for each row
    mad = absolute_deviations.median(dim=-1).values

    # Reshape MAD back to original tensor shape (but with nt1 reduced due to windowing)
    mad_reshaped = mad.reshape(nb, nc, nx, nt - window_size + 1)

    return mad_reshaped


def sliding_window_pearson_correlation(long_seq, short_seq):
    assert long_seq.shape[:-1] == short_seq.shape[:-1], "The first three dimensions should match"
    nt1 = long_seq.size(-1)
    nt2 = short_seq.size(-1)

    # Get sliding windows
    windows = long_seq.unfold(dimension=-1, size=nt2, step=1)

    # Expand dimensions of short_seq for broadcasting
    short_seq_expanded = short_seq.unsqueeze(-2)

    # Compute means for each window and the short sequence
    window_means = windows.mean(dim=-1, keepdim=True)
    short_seq_mean = short_seq_expanded.mean(dim=-1, keepdim=True)

    # Compute terms for Pearson correlation formula
    numerator = ((windows - window_means) * (short_seq_expanded - short_seq_mean)).sum(dim=-1)
    denominator = torch.sqrt(
        ((windows - window_means) ** 2).sum(dim=-1) * ((short_seq_expanded - short_seq_mean) ** 2).sum(dim=-1)
    )

    # Add a small value to the denominator to prevent division by zero
    denominator += 1e-9

    # Compute correlation coefficients
    correlations = numerator / denominator

    return correlations


# def detect_peaks(scores, vmin=0.3, kernel=101, stride=1, K=0):
#     nb, nc, nt, nx = scores.shape
#     pad = kernel // 2
#     smax = F.max_pool2d(scores, (kernel, 1), stride=(stride, 1), padding=(pad, 0))[:, :, :nt, :]
#     keep = (smax == scores).float()
#     scores = scores * keep

#     batch, chn, nt, ns = scores.size()
#     scores = torch.transpose(scores, 2, 3)
#     if K == 0:
#         K = max(round(nt * 10.0 / 3000.0), 3)
#     if chn == 1:
#         topk_scores, topk_inds = torch.topk(scores, K)
#     else:
#         topk_scores, topk_inds = torch.topk(scores[:, 1:, :, :].view(batch, chn - 1, ns, -1), K)
#     topk_inds = topk_inds % nt

#     return topk_scores.detach().cpu(), topk_inds.detach().cpu()


def detect_peaks(scores, ratio=10, kernel=101, stride=1, K=100):
    nb, nc, nx, nt = scores.shape
    padding = kernel // 2
    smax = F.max_pool2d(scores, (1, kernel), stride=(1, stride), padding=(0, padding))[:, :, :, :nt]
    scores_ = F.pad(scores, (0, kernel - nt % kernel), mode="reflect")
    scores_ = scores_.view(nb, nc, nx, -1, kernel)
    ## MAD = median(|x_i - median(x)|)
    mad = (scores_ - scores_.median(dim=-1, keepdim=True)).abs().median(dim=-1, keepdim=True)
    mad = mad.repeat(1, 1, 1, 1, kernel).view(nb, nc, nx, -1)[:, :, :, :nt]
    keep = (smax == scores).float()
    scores = scores * keep * (scores > ratio * mad).float()

    if K == 0:
        K = max(round(nt * 10.0 / 3000.0), 3)
    topk_scores, topk_inds = torch.topk(scores, K)

    return topk_scores.detach().cpu(), topk_inds.detach().cpu()


# %%
# input = torch.tensor(tr.data, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
input = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
input = input.repeat(1, 2, 1, 1)
weight = torch.tensor(template[ieve], dtype=torch.float32).unsqueeze(0)
shift_index = torch.tensor(traveltime_index[ieve], dtype=torch.int64).unsqueeze(0)
shift_index = shift_index.repeat_interleave(3, dim=1)

data1 = input[:, :, :, :]  # .to("mps")
data2 = weight[:, :, :, :]  # .to("mps")
shift_index = shift_index[:, :, :]
channel_shift = 0
nlag = 0

nb1, nc1, nx1, nt1 = data1.shape
nb2, nc2, nx2, nt2 = data2.shape

nt_index = torch.arange(nt1).unsqueeze(0).unsqueeze(0).unsqueeze(0)
adjusted_index = (nt_index + shift_index.unsqueeze(-1)) % nt
data1 = data1.gather(-1, adjusted_index)


# xcor = sliding_window_pearson_correlation(data1, data2)
# raise
data1.to("mps")
data2.to("mps")

data1 = data1.view(1, nb1 * nc1 * nx1, nt1)
data2 = data2.view(nb2 * nc2 * nx2, 1, nt2)

eps = torch.finfo(data1.dtype).eps
data2 = (data2 - torch.mean(data2, dim=-1, keepdim=True)) / (torch.std(data2, dim=-1, keepdim=True) + eps)
data1_ = F.pad(data1, (nt2 // 2, nt2 - 1 - nt2 // 2), mode="reflect")
local_mean = F.avg_pool1d(data1_, nt2, stride=1)
local_std = F.lp_pool1d(data1 - local_mean, norm_type=2, kernel_size=nt2, stride=1) * np.sqrt(nt2)
# local_std = torch.where(data1_ == 0, torch.ones_like(local_std), local_std)

# if channel_shift != 0:
#     xcor = F.conv1d(data1, torch.roll(data2, channel_shift, dims=-2), padding=nlag, groups=nb1 * nc1 * nx1)
# else:
#     xcor = F.conv1d(data1, data2, padding=nlag, groups=nb1 * nc1 * nx1)

# data1 = torch.roll(data1, shift, dims=-2)
xcor = F.conv1d(data1, data2, stride=1, groups=nb1 * nc1 * nx1)
# xcor = F.conv1d(data1, data2, stride=1, groups=nb1 * nc1 * nx1)

xcor = xcor / (local_std + eps)
xcor = xcor.view(nb1, nc1, nx1, -1)

cc = xcor.cpu().numpy()


# MAD = sliding_window_mad(torch.mean(xcor, dim=(-3, -2), keepdim=True), 1000)
xcor = torch.sum(xcor, dim=(-3, -2), keepdim=True)

# topk_scores, topk_index = detect_peaks(xcor, ratio=10, kernel=101, stride=1, K=100)

kernel = 101
median_window = 1000
stride = 1
scores = xcor
ratio = 10
K = 100
nb, nc, nx, nt = scores.shape
padding = kernel // 2
smax = F.max_pool2d(scores, (1, kernel), stride=(1, stride), padding=(0, padding))[:, :, :, :nt]
scores_ = F.pad(scores, (0, median_window - nt % median_window), mode="constant", value=0)
scores_ = scores_.view(nb, nc, nx, -1, median_window)
## MAD = median(|x_i - median(x)|)
mad = (scores_ - scores_.median(dim=-1, keepdim=True)[0]).abs().median(dim=-1)[0]
mad = F.interpolate(mad, scale_factor=(1, median_window), mode="bilinear", align_corners=False)[:, :, :, :nt]
keep = (smax == scores).float() * (scores > ratio * mad).float()
scores = scores * keep

if K == 0:
    K = max(round(nt * 10.0 / 3000.0), 3)
topk_scores, topk_inds = torch.topk(scores, K, dim=-1, sorted=True)

topk_scores = topk_scores.flatten()
topk_inds = topk_inds.flatten()
topk_inds = topk_inds[topk_scores > 0]
topk_scores = topk_scores[topk_scores > 0]
# nt = weight.shape[-1]

# weight = (weight - torch.mean(weight)) / torch.std(weight)
# cc = F.conv1d(input, weight, stride=1)  # / nt

# input_ = F.pad(input, (nt // 2, nt - 1 - nt // 2), mode="reflect")
# local_mu = F.avg_pool1d(input_, nt, stride=1)
# local_std = F.lp_pool1d(
#     input - local_mu,
#     norm_type=2,
#     kernel_size=nt,
#     stride=1,
# )  # / np.sqrt(nt)

# cc = cc / local_std / np.sqrt(nt)

event_time = [begin_time + x / config["sampling_rate"] for x in topk_inds.numpy()]

# %%
meta_ = meta.select(channel="*Z").copy()
# meta_.filter("highpass", freq=1.0)
for e in event_time:
    tmp = meta_.slice(obspy.UTCDateTime(e) - 1, obspy.UTCDateTime(e) + 20.0)
    tmp.sort()
    tmp.plot(equal_scale=False, size=(800, 600))

# %%
plt.figure(figsize=(20, 5))
plt.plot(xcor[0, 0, 0, :].numpy(), label="xcorr")
plt.plot(10 * mad[0, 0, 0, :].numpy(), label="10*MAD")
# plt.xlim([280_000, 290_000])
# plt.xlim([150_000-5000, 150_000+5000])
plt.legend()
plt.show()
raise

# %%
plt.figure(figsize=(20, 20))
for i in range(cc.shape[2]):
    plt.plot(cc[0, 0, i, :] * 0.8 + i, "k")
plt.grid()
plt.xlim([280_000, 290_000])
plt.savefig("debug_cc.png")

# %%
plt.figure(figsize=(20, 1))
# for i in range(MAD.shape[2]):
#     plt.plot(MAD[0, 0, i, :] * 2 + i, "k")
# plt.grid()
# plt.plot(MAD[0, 0, 0, :], "k")

plt.plot(torch.sum(xcor, dim=(-3, -2), keepdim=True)[0, 0, 0, :].numpy())
plt.xlim([280_000, 290_000])
# plt.savefig("debug_mad.png")

raise

# %%
plt.figure(figsize=(20, 5))
# plt.plot(st[0].data)
plt.plot(input.squeeze().numpy())
plt.grid()
# plt.xlim([6000, 8000])

# %%
plt.figure(figsize=(20, 5))
plt.plot(weight[0, 0, :].numpy())

# %%


# # %%
# for i in range(template.shape[0]):
#     for j in range(template.shape[1]):
#         for k in range(template.shape[2]):
#             print(i, j, k, l, np.min(template[i, j, k]), np.max(template[i, j, k]))
#             plt.figure()
#             plt.plot(template[i, j, k])
#             plt.show()
#             raise

# # %%
# template = np.memmap(
#     f"{root_path}/{region}/cctorch/snr.dat", dtype=np.float32, mode="r", shape=tuple(config["snr_shape"])
# )

# # %%
# for i in range(template.shape[0]):
#     for j in range(template.shape[1]):
#         for k in range(template.shape[2]):
#             print(i, j, k, l, np.min(template[i, j, k]), np.max(template[i, j, k]))
#             raise

# # %%

# %%
