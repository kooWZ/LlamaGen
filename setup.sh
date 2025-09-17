pip install tensorflow
pip install nvidia-cudnn-cu12==9.3.0.75
pip install scipy

cd external_tokenizers
git clone https://github.com/apple/ml-flextok.git flextok
cd ..

cd dataset
mkdir ImageNet-1k
cd ImageNet-1k
mkdir flextok_codes
cd flextok_codes
huggingface-cli download --repo-type dataset lykong/ImageNet1k-Flextok-Codes --local-dir .
cd ..

mkdir reference
cd reference
wget https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz

cd ../../../

mkdir outputs
cd outputs
mkdir ckpts
cd ckpts
huggingface-cli download --token ${HF_TOKEN} lykong/ar_1d_tok --local-dir .
mkdir flextok_d12_d12_in1k
cd flextok_d12_d12_in1k
huggingface-cli download EPFL-VILAB/flextok_d12_d12_in1k --local-dir .
