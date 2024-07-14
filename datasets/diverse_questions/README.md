#概要
conda仮想環境でvllmの環境を設定する手順です。
python1、CUDAのバージョン、requierments.txtは20240707の公式を参照しています。
https://github.com/vllm-project/vllm

##インストール手順
conda create --name vllm python=3.11
conda install CUDA=12.1 -c nvidia
pip install vllm -r requierments.txt

###CUDA 11.8
# Install vLLM with CUDA 11.8.
export VLLM_VERSION=0.4.0
export PYTHON_VERSION=310
pip install https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu118-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu118

# flash attention
pip install flash-attn --no-build-isolation
