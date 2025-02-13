import math
import os
import numpy as np
import pandas as pd

import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    num_layers = {
        'InternVL2_5-1B': 24, 'InternVL2_5-2B': 24, 'InternVL2_5-4B': 36, 'InternVL2_5-8B': 32,
        'InternVL2_5-26B': 48, 'InternVL2_5-38B': 64, 'InternVL2_5-78B': 80}[model_name]
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


# path = 'OpenGVLab/InternVL2_5-78B-MPO'
# path = 'OpenGVLab/InternVL2_5-78B'
# device_map = split_model('InternVL2_5-78B')

path = 'OpenGVLab/InternVL2_5-38B-MPO'
# path = 'OpenGVLab/InternVL2_5-38B'
device_map = split_model('InternVL2_5-38B')

# path = 'OpenGVLab/InternVL2_5-8B-MPO'
# path = 'OpenGVLab/InternVL2_5-8B'
# device_map = split_model('InternVL2_5-8B')

# path = 'OpenGVLab/InternVL2_5-4B-MPO'
# path = 'OpenGVLab/InternVL2_5-4B'
# device_map = split_model('InternVL2_5-4B')

# If you set `load_in_8bit=True`, you will need two 80GB GPUs.
# If you set `load_in_8bit=False`, you will need at least three 80GB GPUs.
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=True)


def predict_prognosis(base_directory, target_options = None):
    """Predicts prognosis for a given WSI's image patches."""

    image_files = sorted([os.path.join(base_directory, f) for f in os.listdir(base_directory) if f.endswith(".png")])
    image_files_selected = image_files[:20]
    print(f"Found {len(image_files_selected)}/{len(image_files)} images from {base_directory}")

    pixel_values_list, num_patches_list = [], []
    for img_file in image_files_selected:
        # set the max number of tiles in `max_num`
        pvs = load_image(img_file, max_num=12).to(torch.bfloat16).cuda()
        pixel_values_list.append(pvs)
        num_patches_list.append(pvs.shape[0])
    pixel_values = torch.cat(pixel_values_list, dim=0)

    
    ## multi-image multi-round conversation
    question = f"<image>\nYou are a professional pathologist who predict patient's prognosis based on the whole-slide image (WSI). Based on these image patches randomly selected from the WSI, what is the predicted survival range of this patient?\nPossible choices:\n{target_options}.\nPlease select and reply only among one of the possible choices, even if unsure." # only answer

    # question = ("<image>\n"
    #         "You are an expert pathologist specializing in analyzing whole-slide images (WSIs) to predict the patient's prognosis. "
    #         "Based on the provided image patches extracted from the WSI, assess the survival range of this patient. "
    #         f"Select the most appropriate survival range from the following options: {target_options}. "
    #         "Ensure your response follows a structured format, providing both a prediction and a concise rationale. "
    #         "Respond strictly in the following format:\n"
    #         "{your prediction}\n"
    #         "Explanation: {a well-reasoned explanation within one paragraph based on your expert assessment of the image features.}") # with explanation

    # question = f"<image>\nYou are a professional pathologist who predicts patient's prognosis based on the whole-slide image (WSI). Based on these image patches extracted from the WSI with different magnifications, what is the predicted survival range of this patient?\nPossible choices:\n{target_options}.\nPlease select and reply only among one of the possible choices, even if unsure." # multi-scale


    response, history = model.chat(tokenizer, pixel_values, question, generation_config, 
                                   history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')
    return response

    # question = 'Please write a official report according to the image.'
    # response, history = model.chat(tokenizer, pixel_values, question, generation_config, 
    #                                history=history, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

    ## multi-image multi-round conversation, separate images 
    # question = 'Image-1: <image>\nImage-2: <image>\nDescribe the two images in detail.'
    # response, history = model.chat(tokenizer, pixel_values, question, generation_config,
    #                                num_patches_list=num_patches_list,
    #                                history=None, return_history=True)
    # print(f'User: {question}\nAssistant: {response}')

def prepare_target(label_file_path):
    num_bins = 10
    df = pd.read_csv(label_file_path, sep="\t")

    # Quantizes the days_to_death values into discrete bins.
    df = df[["case_submitter_id", "days_to_death"]].drop_duplicates()
    df = df[df["days_to_death"] != "'--"]  # Remove missing values
    df["days_to_death"] = df["days_to_death"].astype(float)
    bins = np.linspace(df["days_to_death"].min(), df["days_to_death"].max(), num_bins)
    labels = [f"{int(bins[i])}-{int(bins[i+1])} days" for i in range(len(bins)-1)]
    df["days_to_death_category"] = pd.cut(df["days_to_death"], bins=bins, labels=labels, include_lowest=True)

    return df


def main():
    base_path = "/nfs_share/students/jinhyun/TCGA/patches"
    label_file_path = "/home/jinhyun/data/TCGA/LUAD/labels/clinical/clinical.tsv"
    result_file_path = "results/InternVL2_5-38B-MPO_randompatch448_20.xlsx"
    os.makedirs("results", exist_ok=True)

    df_clinical = prepare_target(label_file_path)
    print(f"Target distribution: \n", df_clinical[["days_to_death_category"]].value_counts())
    target_options = "\n".join([f"- {label}" for label in df_clinical["days_to_death_category"].unique()])

    sample_paths = sorted([os.path.join(base_path, d) for d in df_clinical["case_submitter_id"] if os.path.isdir(os.path.join(base_path, d))])

    print(f"Found {len(sample_paths)}/{df_clinical.shape[0]} number of cases for analysis")


    results_list = []
    for sample_path in sample_paths:
        patient_id = os.path.basename(sample_path)
        predicted_prognosis = predict_prognosis(sample_path, target_options)
        actual_prognosis = df_clinical[df_clinical["case_submitter_id"] == patient_id]["days_to_death_category"].values[0]
        results_list.append([patient_id, predicted_prognosis, actual_prognosis])
        print("Predicted: ", predicted_prognosis, " Actual:", actual_prognosis)
    
    results_df = pd.DataFrame(results_list, columns=["Patient ID", "Predicted Prognosis", "Actual Prognosis"])
    results_df.to_excel(result_file_path, index=False)


if __name__ == "__main__":
    main()
