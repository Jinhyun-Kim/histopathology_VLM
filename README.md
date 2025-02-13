## InternVL2_5
conda create -n patho python=3.12
conda activate patho
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
python main_InternVL.py

## QWEN
pip install git+https://github.com/huggingface/transformers accelerate
pip install qwen-vl-utils[decord]==0.0.8

## OVIS
conda create -n patho11 python=3.11
pip install torch==2.4.0 transformers==4.46.2 numpy==1.25.0 pillow==10.3.0 pandas openpyxl

python main_ovis.py 
