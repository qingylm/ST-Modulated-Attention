import os
from datasets import load_from_disk, concatenate_datasets
from tqdm import tqdm
import shutil
import tempfile



def merge_datasets_from_dirs(temp_paths, target_path, delete_temp=False, verbose=True):
    """
    合并多个数据集分片目录到目标目录。

    参数:
        temp_paths (list of str): 临时分片目录路径列表。
        target_path (str): 合并后数据集的保存路径。
        delete_temp (bool): 是否在合并成功后删除临时分片目录。
        verbose (bool): 是否打印进度信息。
    """
    # 过滤掉不存在的路径
    valid_paths = [p for p in temp_paths if os.path.exists(p)]
    if not valid_paths:
        print("❌ 没有有效的分片目录，请检查路径。")
        return

    if verbose:
        print(f"正在合并 {len(valid_paths)} 个分片...")

    # 加载所有分片数据集
    datasets = []
    for p in tqdm(valid_paths, desc="加载分片", disable=not verbose):
        try:
            ds = load_from_disk(p)
            datasets.append(ds)
        except Exception as e:
            print(f"⚠️ 加载 {p} 失败: {e}，跳过该分片。")
            continue

    if not datasets:
        print("❌ 没有成功加载任何分片，合并失败。")
        return

    # 合并所有数据集
    if verbose:
        print("正在拼接数据集...")
    combined = concatenate_datasets(datasets)

    # 保存最终数据集
    combined.save_to_disk(target_path)
    if verbose:
        print(f"✅ 合并完成，共 {len(combined)} 条样本，保存至 {target_path}")

    # 可选删除临时分片
    if delete_temp:
        for p in valid_paths:
            try:
                shutil.rmtree(p)
                if verbose:
                    print(f"🗑️ 已删除临时分片：{p}")
            except Exception as e:
                print(f"⚠️ 删除 {p} 失败: {e}")


def merge_in_batches(temp_paths, target_path, batch_size=10, temp_base=None):
    if temp_base is None:
        temp_base = r"I:\temp\temp_merge"  # 可自定义
    os.makedirs(temp_base, exist_ok=True)

    valid = [p for p in temp_paths if os.path.exists(p)]
    if not valid:
        print("❌ 没有有效分片路径")
        return
    print(f"✅ 找到 {len(valid)} 个有效分片")
    temp_merged = []
    for i in range(0, len(valid), batch_size):
        batch = valid[i:i + batch_size]
        temp_dir = tempfile.mkdtemp(prefix="merge_batch_", dir=temp_base)  # 指定 dir
        print(f"处理批次 {i // batch_size + 1}, 临时目录: {temp_dir}")
        merge_datasets_from_dirs(batch, temp_dir, delete_temp=False, verbose=True)
        temp_merged.append(temp_dir)
    merge_datasets_from_dirs(temp_merged, target_path, delete_temp=True, verbose=True)




# 执行合并（不删除临时分片）
if __name__ == "__main__":
    TEMP_DIR = [
        r"I:\data\database\cci_hq_dp\database_part1",
        r"I:\data\database\cci_hq_dp\database_part2",
        r"I:\data\database\cci_hq_dp\database_part3",
        r"I:\data\database\cci_hq_dp\database_part4",
        r"I:\data\database\cci_hq_dp\database_part5",
        r"I:\data\database\cci_hq_dp\database_part6",
        r"I:\data\database\cci_hq_dp\database_part7",
        r"I:\data\database\cci_hq_dp\database_part8",
        r"I:\data\database\cci_hq_dp\database_part9",
        r"I:\data\database\cci_hq_dp\database_part10",
    ]

    TARGET_DIR = r"I:\data\database\CCI_HQ_Database"

    os.environ['TMP'] = r"I:\temp"
    os.environ['TEMP'] = r"I:\temp"
    os.environ['HF_DATASETS_CACHE'] = r"I:\temp\huggingface_cache"
    try:
        merge_in_batches(TEMP_DIR, TARGET_DIR)
    except Exception as e:
        print(f"❌ 合并过程发生错误: {e}")
        import traceback
        traceback.print_exc()