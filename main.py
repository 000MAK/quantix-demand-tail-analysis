"""Fit a Peaks-over-Threshold Generalized Pareto model to bike demand extremes."""
from __future__ import annotations
import argparse, io, json, urllib.request, zipfile
from pathlib import Path

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"

def exceedances(values, threshold: float) -> list[float]:
    """Return positive excesses over a threshold, discarding non-exceedances."""
    return [float(x) - float(threshold) for x in values if float(x) > float(threshold)]

def empirical_quantile(values, probability: float) -> float:
    """Linear-interpolated quantile without relying on a third-party library."""
    if not 0 <= probability <= 1: raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(x) for x in values)
    if not ordered: raise ValueError("values cannot be empty")
    position = (len(ordered) - 1) * probability
    lo, hi = int(position), min(int(position) + 1, len(ordered) - 1)
    fraction = position - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * fraction

def pot_quantile(threshold: float, shape: float, scale: float, exceedance_rate: float, probability: float) -> float:
    """Estimate a quantile above a POT threshold using a fitted GPD tail."""
    import math
    if scale <= 0 or not 0 < exceedance_rate < 1 or not 0 < probability < 1:
        raise ValueError("scale must be positive; rates/probability must be in (0, 1)")
    if probability <= 1 - exceedance_rate:
        raise ValueError("probability must exceed the empirical threshold quantile")
    survival_ratio = exceedance_rate / (1 - probability)
    if abs(shape) < 1e-8:
        return float(threshold) + float(scale) * math.log(survival_ratio)
    return float(threshold) + float(scale) / float(shape) * (survival_ratio ** float(shape) - 1)

def load_counts(cache_dir="data"):
    import pandas as pd
    cache=Path(cache_dir); cache.mkdir(parents=True,exist_ok=True); archive=cache/"Bike-Sharing-Dataset.zip"
    if not archive.exists(): urllib.request.urlretrieve(DATA_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        member=next((n for n in zf.namelist() if n.endswith("hour.csv")),None)
        if member is None: raise RuntimeError("hour.csv not found in UCI archive")
        with zf.open(member) as f: return pd.read_csv(f)["cnt"].dropna().astype(float).tolist()

def main(output_dir="outputs", threshold_quantile=.9, return_probability=.99):
    from scipy.stats import genpareto
    import numpy as np
    values=load_counts()
    threshold=empirical_quantile(values,threshold_quantile)
    exc=exceedances(values,threshold)
    if len(exc)<30: raise RuntimeError("too few exceedances for a stable classroom fit; lower the threshold quantile")
    shape, loc, scale = genpareto.fit(exc, floc=0)
    rate=len(exc)/len(values)
    estimate=pot_quantile(threshold,float(shape),float(scale),rate,return_probability)
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    report={"n_observations":len(values),"threshold_quantile":threshold_quantile,"threshold":threshold,
      "n_exceedances":len(exc),"exceedance_rate":rate,"gpd_shape":float(shape),"gpd_scale":float(scale),
      "return_probability":return_probability,"estimated_return_level":estimate,
      "warning":"Exploratory demonstration; serial dependence, stationarity and threshold stability are not validated."}
    (out/"tail_risk.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default="outputs"); p.add_argument("--threshold-quantile",type=float,default=.9); p.add_argument("--return-probability",type=float,default=.99)
    a=p.parse_args(); print(json.dumps(main(a.output_dir,a.threshold_quantile,a.return_probability),indent=2))
