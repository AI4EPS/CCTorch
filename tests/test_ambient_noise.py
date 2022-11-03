if __name__ == "__main__":

    tmp = []
    with h5py.File(f"{args.result_path}/{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5", "a") as f:
        for i in range(1250):
            for j in [500]:
                pair_index = f"{i}_{j}"
                if pair_index in f:
                    tmp.append(f[pair_index][:])
    xcorr = np.array(tmp)
    xcorr = xcorr.transpose(1, 0, 2)
    print(xcorr.shape)
    # raise

    plt.figure()
    # vmax = np.max(np.abs(xcorr[0, :, :]))
    _, nch, nt = xcorr.shape
    # xcorr[0, :, nt // 2 - 10 : nt // 2 + 11] *= 0.0
    # vmax = np.std(xcorr[0, :, :])
    mask = np.ones(nt)
    mask[nt // 2 - 10 : nt // 2 + 11] = 0.0
    # xcorr[0, :, nt // 2 - 10 : nt // 2 + 11] *= 0.0
    vmax = np.std(xcorr[0, :, mask == 1.0]) * 3
    plt.imshow(xcorr[0, :, :], cmap="seismic", vmax=vmax, vmin=-vmax)
    plt.colorbar()
    plt.savefig("xcorr.png", dpi=300)
    ## TODO: cleanup writting

    plt.figure()
    ccall = xcorr[0, :, :]
    max_lag = 30
    vmax = np.percentile(np.abs(ccall), 99)
    plt.imshow(
        # filter(ccall, 25, 1, 10),
        ccall,
        aspect="auto",
        vmax=vmax,
        vmin=-vmax,
        # extent=(-max_lag, max_lag, ccall.shape[0], 0),
        cmap="RdBu",
    )
    plt.colorbar()
    plt.savefig("test_no_whitening_no_filtering.png", dpi=300)
    plt.show()
