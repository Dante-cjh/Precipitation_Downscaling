import os
import xarray as xr
import numpy as np


def process_datasets(base_dir='../../processed_datasets'):
    """
    使用双三次插值将全球地形数据调整到 0.125° 分辨率，
    并裁剪局部区域，将地形数据 (z, lsm) 合并到降水数据中。
    """
    # 加载全球地形数据
    geopotential_path = os.path.join(base_dir, 'Geopotential.nc')
    geopotential_ds = xr.open_dataset(geopotential_path)

    # 如果地形数据的纬度是从北到南，可以先 sortby('latitude')，保证它是升序：
    # geopotential_ds = geopotential_ds.sortby('latitude')

    # -------------------------------------------------------------------------
    # (可选) 第一步：全局上采样到 0.125° 分辨率
    # -------------------------------------------------------------------------
    print("开始插值地形数据到 0.125° 分辨率 (bicubic interpolation)...")

    # 注意：如果你真的想要“全球 0.125°”的网格，可以显式设置范围为 [-90, 90], [0, 360] 等
    # 也可以用下列自动获取 min/max 的方式
    lat_min_global = geopotential_ds['latitude'].values.min()
    lat_max_global = geopotential_ds['latitude'].values.max()
    lon_min_global = geopotential_ds['longitude'].values.min()
    lon_max_global = geopotential_ds['longitude'].values.max()

    # 生成 0.125° 的经纬度坐标
    new_lat = np.arange(lat_min_global, lat_max_global + 0.125, 0.125)
    new_lon = np.arange(lon_min_global, lon_max_global + 0.125, 0.125)

    # 在此做全局插值
    interpolated_geopotential = geopotential_ds.interp(
        latitude=new_lat,
        longitude=new_lon,
        method='cubic'  # bicubic
    )
    print("地形数据插值完成。")
    print("插值后纬度范围:",
          interpolated_geopotential['latitude'].values.min(),
          interpolated_geopotential['latitude'].values.max())
    print("插值后经度范围:",
          interpolated_geopotential['longitude'].values.min(),
          interpolated_geopotential['longitude'].values.max())

    # -------------------------------------------------------------------------
    # 第二步：逐文件裁剪 + interp_like 到降水网格
    # -------------------------------------------------------------------------
    for dataset_type in ['train', 'val']:
        dataset_dir = os.path.join(base_dir, dataset_type)
        new_dataset_dir = os.path.join(base_dir, f'new_{dataset_type}')
        os.makedirs(new_dataset_dir, exist_ok=True)

        for file_name in os.listdir(dataset_dir):
            if file_name.endswith('.nc'):
                file_path = os.path.join(dataset_dir, file_name)
                ds = xr.open_dataset(file_path)

                # 提取降水图的经纬度范围
                lat_min = ds['latitude'].values.min()
                lat_max = ds['latitude'].values.max()
                lon_min = ds['longitude'].values.min()
                lon_max = ds['longitude'].values.max()

                print(f"Processing {file_name}: "
                      f"lat=({lat_min}, {lat_max}), lon=({lon_min}, {lon_max})")

                # ----------------------------
                # 1) 先裁剪大范围，减少内存
                # ----------------------------
                # 这里 slice 的顺序要跟实际坐标顺序一致；如果 latitude 升序，
                # 则 slice(lat_min, lat_max)，若是降序则反过来。
                # 如担心浮点误差，可加一点小偏移：
                EPS = 1e-6
                cropped_geo = interpolated_geopotential.sel(
                    latitude=slice(lat_min - EPS, lat_max + EPS),
                    longitude=slice(lon_min - EPS, lon_max + EPS)
                )

                # ----------------------------
                # 2) 再用 interp_like, bicubic 到 ds 的精确网格
                # ----------------------------
                cropped_geopotential = cropped_geo.interp_like(
                    ds,
                    method='cubic',
                    kwargs={
                        "bounds_error": False,
                        "fill_value": None
                    }
                )

                # 一定要先确保 monotonic
                cropped_geopotential = cropped_geopotential.sortby('latitude')
                cropped_geopotential = cropped_geopotential.sortby('longitude')

                # 再用 xarray 的 interpolate_na 来用“最近邻”填充 NaN
                cropped_geopotential = (
                    cropped_geopotential
                    .interpolate_na(dim="longitude", method="nearest")
                    .interpolate_na(dim="latitude", method="nearest")
                )

                # 合并地形数据到降水数据
                merged_ds = xr.merge([ds, cropped_geopotential[['z', 'lsm']]])

                # 保存合并后的数据到新的目录
                new_file_path = os.path.join(new_dataset_dir, file_name)
                merged_ds.to_netcdf(new_file_path)
                print(f"Processed and saved: {new_file_path}")


if __name__ == '__main__':

    # 调用函数处理数据集
    process_datasets()