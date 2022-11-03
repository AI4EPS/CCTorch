from pathlib import Path
import h5py


def write_results(results, result_path, ccconfig, dim=1, rank=0, world_size=1):
    if ccconfig.mode == "CC":
        ## TODO: add writting for CC
        pass
    elif ccconfig.mode == "TM":
        ## TODO: add writting for CC
        pass
    elif ccconfig.mode == "AM":
        write_ambient_noise(results, result_path, ccconfig, dim=dim, rank=rank, world_size=world_size)
    else:
        raise ValueError(f"{ccconfig.mode} not supported")


def write_ambient_noise(results, result_path, ccconfig, dim=1, rank=0, world_size=1):
    if not isinstance(result_path, Path):
        result_path = Path(result_path)
    with h5py.File(result_path / f"{ccconfig.mode}_{rank:03d}_{world_size:03d}.h5", "a") as fp:
        for result in results:
            for i, pair_index in enumerate(result["info"]["pair_index"]):
                if dim == 0:
                    xcorr = result["xcorr"][i, :, :]
                elif dim == 1:
                    xcorr = result["xcorr"][:, i, :]
                else:
                    raise ValueError(f"{dim} not supported")
                if pair_index in fp:
                    data = fp[pair_index][:]
                    count = fp[pair_index].attrs["count"]
                    data = (data * count + xcorr) / (count + 1)
                    fp[pair_index][:] = data
                    fp[pair_index].attrs["count"] = count + 1
                else:
                    fp.create_dataset(
                        f"{pair_index}",
                        data=xcorr,
                    )
                    fp[pair_index].attrs["count"] = 1
