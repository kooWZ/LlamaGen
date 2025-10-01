cd external_tokenizers
git clone https://github.com/bytedance/1d-tokenizer.git TiTok
cd TiTok
pip install -r requirements.txt
pip install nvidia-cudnn-cu12==9.3.0.75
cd ../..

cd outputs
cd ckpts
mkdir titok_bl128
cd titok_bl128
huggingface-cli download yucornetto/tokenizer_titok_bl128_vq8k_imagenet --local-dir .
cd ../..

