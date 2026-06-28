from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path


ERA5_VARIABLES = {
    "10m_u_component_of_wind":   "u10",
    "10m_v_component_of_wind":   "v10",
    "2m_temperature":            "t2m",
    "surface_pressure":          "sp",
    "2m_dewpoint_temperature":   "d2m",
    "total_precipitation":       "tp",
    "100m_u_component_of_wind":  "u100",
    "100m_v_component_of_wind":  "v100",
}


def download_era5(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    variables: list[str] | None = None,
    output_path: str = "data/external/era5_raw.nc",
) -> Path:
    """
    Belirtilen koordinat ve tarih aralığı için ERA5 saatlik verisi indirir.
    start_date / end_date: "YYYY-MM-DD" formatında.
    Döner: indirilen NetCDF dosyasının yolu.
    """
    import cdsapi

    if variables is None:
        variables = list(ERA5_VARIABLES.keys())

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Tarih aralığından yıl/ay listesi
    dates = pd.date_range(start_date, end_date, freq="MS")
    years  = sorted(set(str(d.year) for d in dates))
    months = sorted(set(f"{d.month:02d}" for d in dates))

    c = cdsapi.Client(quiet=True)
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": variables,
            "year":  years,
            "month": months,
            "day":   [f"{d:02d}" for d in range(1, 32)],
            "time":  [f"{h:02d}:00" for h in range(24)],
            "area":  [lat + 0.25, lon - 0.25, lat - 0.25, lon + 0.25],
            "format": "netcdf",
        },
        str(out),
    )
    return out


def netcdf_to_dataframe(nc_path: str | Path) -> pd.DataFrame:
    """
    ERA5 NetCDF dosyasını standart DataFrame'e dönüştürür.
    Rüzgar hızı ve yönü U/V bileşenlerinden hesaplanır.
    Sıcaklık Kelvin'den Celsius'a çevrilir.
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError("xarray paketi gerekli: pip install xarray netcdf4")

    ds = xr.open_dataset(nc_path)
    df = ds.mean(dim=["latitude", "longitude"]).to_dataframe().reset_index()
    df = df.rename(columns={"valid_time": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)

    # Rüzgar hızı ve yönü (10m)
    if "u10" in df.columns and "v10" in df.columns:
        df["wind_speed_10m"] = np.sqrt(df["u10"] ** 2 + df["v10"] ** 2)
        df["wind_dir_10m"]   = (np.degrees(np.arctan2(-df["u10"], -df["v10"])) + 360) % 360

    # Rüzgar hızı (100m)
    if "u100" in df.columns and "v100" in df.columns:
        df["wind_speed_100m"] = np.sqrt(df["u100"] ** 2 + df["v100"] ** 2)

    # Sıcaklık K → °C
    for col in ["t2m", "d2m"]:
        if col in df.columns:
            df[col] = df[col] - 273.15

    # Basınç Pa → hPa
    if "sp" in df.columns:
        df["sp"] = df["sp"] / 100

    # Zaman özellikleri ekle
    df["hour"]       = df["datetime"].dt.hour
    df["day_of_year"] = df["datetime"].dt.dayofyear
    df["month"]      = df["datetime"].dt.month

    # Döngüsel zaman kodlaması (modelin anlayacağı form)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365)

    return df


def load_era5_csv(path: str | Path) -> pd.DataFrame:
    """Daha önce kaydedilmiş ERA5 CSV'yi yükler."""
    return pd.read_csv(path, parse_dates=["datetime"])
