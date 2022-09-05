import pandas as pd
import numpy as np


def write_xcor_to_csv(result, path_result):
    nbatch = result["cc"].shape[0]
    if np.isscalar(result["id1"]):
        id1 = [result["id1"]]
        id2 = [result["id2"]]
    else:
        id1 = result["id1"].cpu().numpy()
        id2 = result["id2"].cpu().numpy()
    cc = result["cc"].cpu().numpy()
    dt = result["dt"].cpu().numpy()
    for ibatch in range(nbatch):
        fn = f"{path_result}/{id1[ibatch]}_{id2[ibatch]}.csv"
        pd.DataFrame({"cc": cc[ibatch, :], "dt": dt[ibatch, :]}).to_csv(fn, index=False)
