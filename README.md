conda create -n patho python=3.12
conda activate patho
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# internvl2_5_78b_mpo.py 이미지 파일이름 (지금은 /examples 폴더 안에 있음), 프롬프트 설정 필요
python internvl2_5_78b_mpo.py 