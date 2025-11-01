cd external_tokenizers
git clone https://github.com/dzj441/postTok
cd ..

pip install -r extra_req.txt

cd outputs
cd ckpts
mkdir dinov2
cd dinov2
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth

cd ..
mkdir postTok_sim128
cd postTok_sim128
huggingface-cli download DZJ181u2u/checkpoints --local-dir .

