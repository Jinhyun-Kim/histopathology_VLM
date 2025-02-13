from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

import os
import pandas as pd
import numpy as np

# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-72B-Instruct", 
#     torch_dtype="auto", 
#     device_map="auto"
# )

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-72B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-72B-Instruct")


def predict_prognosis(base_directory, target_options = None):
    """Predicts prognosis for a given WSI's image patches."""

    image_files = sorted([os.path.join(base_directory, f) for f in os.listdir(base_directory) if f.endswith(".png")])
    image_files_selected = image_files[:20]
    print(f"Found {len(image_files_selected)}/{len(image_files)} images from {base_directory}")

    img_contents = [{"type": "image", "image": img_file} for img_file in image_files_selected]
    # print(img_contents)
    messages = [
        {
            "role": "user",
            "content": img_contents + 
            [
                {"type": "text", "text": f"You are a professional pathologist who predict patient's prognosis based on the whole-slide image (WSI). Based on these image patches randomly selected from the WSI, what is the predicted survival range of this patient?\nPossible choices:\n{target_options}.\nPlease select and reply only among one of the possible choices, even if unsure."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])
    return output_text[0]


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
    result_file_path = "results/Qwen2.5-VL-72B-Instruct_randompatch448_20.xlsx"
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
