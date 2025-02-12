
import pandas as pd
from PIL import Image
# path_img = '/nfs_share/students/jinhyun/TCGA/patches/'


img_dir_list = pd.read_excel('os_img_list.xlsx')
print(img_dir_list)

TCGA_Reports = pd.read_csv('TCGA_Reports.csv')

# '.'을 기준으로 나누고 첫 번째 요소만 추출
TCGA_Reports['img_dir'] = TCGA_Reports['patient_filename'].str.split('.').str[0]

# 새로운 DataFrame 생성
TCGA_Reports_Dir = TCGA_Reports.copy()

TCGA_Reports_Dir.to_excel('TCGA_Reports_Dir.xlsx')
TCGA_Reports_Dir.to_csv('TCGA_Reports_Dir.csv')

# def load_image(image_file, input_size=448, max_num=12):
#     image = Image.open(image_file).convert('RGB')
#     transform = build_transform(input_size=input_size)
#     images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
#     pixel_values = [transform(image) for image in images]
#     pixel_values = torch.stack(pixel_values)
#     return pixel_values

# pixel_values = load_image('./examples/image1.png', max_num=12).to(torch.bfloat16).cuda()