import torch
from PIL import Image
from transformers import AutoModelForCausalLM

import os
import pandas as pd
import numpy as np

model = AutoModelForCausalLM.from_pretrained("AIDC-AI/Ovis1.6-Gemma2-27B",
                                             torch_dtype=torch.bfloat16,
                                             multimodal_max_length=8192,
                                             trust_remote_code=True).cuda()
text_tokenizer = model.get_text_tokenizer()
visual_tokenizer = model.get_visual_tokenizer()


def predict_prognosis(base_directory, target_options = None):
    """Predicts prognosis for a given WSI's image patches."""

    image_files = sorted([os.path.join(base_directory, f) for f in os.listdir(base_directory) if f.endswith(".png")])
    image_files_selected = image_files[:20]
    print(f"Found {len(image_files_selected)}/{len(image_files)} images from {base_directory}")

    images = [Image.open(path) for path in image_files_selected]

    query = f"<image>\nYou are a professional pathologist who predict patient's prognosis based on the whole-slide image (WSI). Based on these image patches randomly selected from the WSI, what is the predicted survival range of this patient?\nPossible choices:\n{target_options}.\nPlease select and reply only among one of the possible choices, even if unsure."

    # format conversation
    prompt, input_ids, pixel_values = model.preprocess_inputs(query, images)
    attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
    input_ids = input_ids.unsqueeze(0).to(device=model.device)
    attention_mask = attention_mask.unsqueeze(0).to(device=model.device)
    pixel_values = [pixel_values.to(dtype=visual_tokenizer.dtype, device=visual_tokenizer.device)]

    # generate output
    with torch.inference_mode():
        gen_kwargs = dict(
            max_new_tokens=1024,
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
            repetition_penalty=None,
            eos_token_id=model.generation_config.eos_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            use_cache=True
        )
        output_ids = model.generate(input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **gen_kwargs)[0]
        output = text_tokenizer.decode(output_ids, skip_special_tokens=True)

    # print(f'Output:\n{output}')
    return output


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
    result_file_path = "results/Ovis1.6-Gemma2-27B_randompatch448_20.xlsx"
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
