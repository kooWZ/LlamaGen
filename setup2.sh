cd external_tokenizers
git clone git@github.com:bytedance/1d-tokenizer.git TiTok
cd TiTok
pip install -r requirements.txt
cd ../..

cd outputs
cd ckpts
mkdir titok_bl128
cd titok_bl128
huggingface-cli download yucornetto/tokenizer_titok_bl128_vq8k_imagenet --local-dir .