from huggingface_hub import snapshot_download

download_path = snapshot_download(repo_id="wikimedia/wit_base", cache_dir="./cache", local_dir="./wit_base")
