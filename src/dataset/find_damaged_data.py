import os
import xarray as xr

def find_and_remove_damaged_files(data_dir):
    """
    检查数据集中无法打开的文件，并将其删除。

    参数:
    - data_dir: 数据文件所在目录路径

    返回:
    - removed_files: 一个列表，包含所有已删除文件的路径和原因
    """
    removed_files = []

    # 获取所有 .nc 文件
    nc_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.nc')]

    for file_path in nc_files:
        try:
            # 尝试打开数据集
            ds = xr.open_dataset(file_path)
            ds.close()  # 成功打开后关闭文件

        except Exception as e:
            # 捕获异常并删除文件
            removed_files.append((file_path, str(e)))
            try:
                os.remove(file_path)
                print(f"已删除损坏文件: {file_path}")
            except Exception as delete_error:
                print(f"删除文件时发生错误: {file_path} - {delete_error}")

    return removed_files


# 使用示例
data_dir = '../processed_datasets/dataset'
removed_files = find_and_remove_damaged_files(data_dir)

# 打印结果
if removed_files:
    print("以下文件已被删除：")
    for file, error in removed_files:
        print(f"{file} - 错误: {error}")
else:
    print("未发现损坏文件，未删除任何文件")