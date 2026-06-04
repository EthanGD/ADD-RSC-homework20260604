"""
下载 AST 预训练权重 (ast-finetuned-audioset-10-10-0.4593.pth)
来源：https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593
"""
import os
import sys
import urllib.request
import ssl

sys.stdout.reconfigure(encoding='utf-8')
ssl._create_default_https_context = ssl._create_unverified_context

# ========== 配置 ==========
SAVE_PATH = "./ast-finetuned-audioset-10-10-0.4593.pth"
MODEL_URL = "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593/resolve/main/pytorch_model.bin"


def download_file(url, save_path):
    print(f"正在从 Hugging Face 下载 AST 预训练权重...")
    print(f"URL: {url}")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    def report_hook(count, block_size, total_size):
        if total_size > 0:
            downloaded = count * block_size
            percent = downloaded / total_size * 100
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"\r下载进度: {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(url, save_path, reporthook=report_hook)
    print("\n下载完成!")


def main():
    if os.path.exists(SAVE_PATH):
        print(f"权重文件已存在: {SAVE_PATH}")
        print("如需重新下载，请先删除该文件。")
        return

    try:
        download_file(MODEL_URL, SAVE_PATH)
    except Exception as e:
        print(f"\n自动下载失败: {e}")
        print("\n请手动下载:")
        print(f"  方式1: 访问 https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593")
        print(f"  方式2: git lfs clone 仓库后复制 pytorch_model.bin")
        print(f"  方式3: pip install huggingface_hub; python -c \"from huggingface_hub import hf_hub_download; hf_hub_download('MIT/ast-finetuned-audioset-10-10-0.4593', 'pytorch_model.bin', local_dir='.')\"")
        print(f"\n下载后将文件重命名为: {os.path.abspath(SAVE_PATH)}")
        return

    # 验证文件大小（AST 权重约 340MB）
    file_size = os.path.getsize(SAVE_PATH) / (1024 * 1024)
    if file_size < 100:
        print(f"警告: 文件仅 {file_size:.1f} MB，可能下载不完整，请检查。")
    else:
        print(f"文件大小: {file_size:.1f} MB")

    print(f"\n权重准备完毕，路径: {os.path.abspath(SAVE_PATH)}")


if __name__ == "__main__":
    main()
