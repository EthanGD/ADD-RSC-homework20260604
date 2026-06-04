"""
下载 ICBHI 额外文件：人口统计信息和事件数据
"""
import os
import sys
import urllib.request
import zipfile
import ssl

sys.stdout.reconfigure(encoding='utf-8')
ssl._create_default_https_context = ssl._create_unverified_context

SAVE_DIR = "./data/ICBHI"
DEMOGRAPHIC_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/ICBHI_Challenge_demographic_information.txt"
EVENTS_URL = "https://bhichallenge.med.auth.gr/sites/default/files/ICBHI_final_database/events.zip"


def download_file(url, save_path):
    print(f"正在从 {url} 下载...")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def report_hook(count, block_size, total_size):
        if total_size > 0:
            percent = count * block_size / total_size * 100
            print(f"\r下载进度：{percent:.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, save_path, reporthook=report_hook)
    print("\n下载完成!")


def extract_zip(zip_path, extract_dir):
    print(f"正在解压 {zip_path} 到 {extract_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    print("解压完成!")


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 下载人口统计信息
    demographic_path = os.path.join(SAVE_DIR, "ICBHI_Challenge_demographic_information.txt")
    if os.path.exists(demographic_path):
        print(f"人口统计信息已存在：{demographic_path}")
    else:
        try:
            download_file(DEMOGRAPHIC_URL, demographic_path)
        except Exception as e:
            print(f"下载人口统计信息失败：{e}")

    # 下载事件文件
    events_path = os.path.join(SAVE_DIR, "events.zip")
    events_dir = os.path.join(SAVE_DIR, "events")
    if os.path.exists(events_dir):
        print(f"事件数据已存在：{events_dir}")
    else:
        try:
            download_file(EVENTS_URL, events_path)
            extract_zip(events_path, SAVE_DIR)
            if os.path.exists(events_path):
                os.remove(events_path)
                print("已清理 events.zip。")
        except Exception as e:
            print(f"下载事件文件失败：{e}")

    print(f"\n额外文件下载完成，路径：{os.path.abspath(SAVE_DIR)}")


if __name__ == "__main__":
    main()
