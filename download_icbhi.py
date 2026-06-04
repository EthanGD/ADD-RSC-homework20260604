"""
下载 ICBHI 2017 呼吸音数据集并解压到指定目录
数据来源: https://bhichallenge.med.auth.gr/ICBHI_2017_Challenge
"""
import os
import urllib.request
import zipfile

# ========== 配置 ==========
SAVE_DIR = "./data/ICBHI"
ZIP_PATH = os.path.join(SAVE_DIR, "ICBHI_final_database.zip")
DATA_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_2017_training.zip"
# 备用链接（Zenodo 镜像）
FALLBACK_URL = "https://zenodo.org/record/4009889/files/ICBHI_final_database.zip"


def download_file(url, save_path):
    print(f"正在从 {url} 下载...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def report_hook(count, block_size, total_size):
        if total_size > 0:
            percent = count * block_size / total_size * 100
            print(f"\r下载进度: {percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, save_path, reporthook=report_hook)
    print("\n下载完成!")


def extract_zip(zip_path, extract_dir):
    print(f"正在解压 {zip_path} 到 {extract_dir} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("解压完成!")


def main():
    # 如果目标目录已存在则跳过
    target_dir = os.path.join(SAVE_DIR, "ICBHI_final_database")
    if os.path.isdir(target_dir):
        print(f"数据集已存在: {target_dir}")
        print("如需重新下载，请先删除该目录。")
        return

    # 尝试主链接，失败则用备用链接
    for url in [DATA_URL, FALLBACK_URL]:
        try:
            download_file(url, ZIP_PATH)
            break
        except Exception as e:
            print(f"下载失败 ({url}): {e}")
            if os.path.exists(ZIP_PATH):
                os.remove(ZIP_PATH)
            continue
    else:
        print("所有下载链接均失败，请手动下载数据集。")
        print(f"下载地址1: {DATA_URL}")
        print(f"下载地址2: {FALLBACK_URL}")
        print(f"解压目标: {os.path.abspath(SAVE_DIR)}")
        return

    # 解压
    extract_zip(ZIP_PATH, SAVE_DIR)

    # 清理 zip
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("已清理压缩包。")

    print(f"\n数据集准备完毕，路径: {os.path.abspath(target_dir)}")


if __name__ == "__main__":
    main()
