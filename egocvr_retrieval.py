import argparse
import random
import sys
from ast import literal_eval
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent
sys.path.insert(0, ROOT_DIR.as_posix())

from model.models import (
    forward_blip,
    forward_blip_text,
    forward_clip,
    forward_clip_text,
    forward_egovlpv2,
    forward_egovlpv2_text,
    forward_languagebind,
    forward_languagebind_text,
    init_BLIP,
    init_CLIP,
    init_EgoVLPv2,
    init_languagebind,
)

CUDA_DEVICE = "cuda:0"
EMBEDDING_DIR = "./embeddings"
VIDEO_DIR = "./data"

parser = argparse.ArgumentParser(
    "Script to perform Composed Video Retrieval on EgoCVR dataset"
)

parser.add_argument(
    "--models",
    nargs="*",
    default=["languagebind", "egovlpv2"],
    type=str,
    help="Which models to use for retrieval.",
)
parser.add_argument(
    "--modalities",
    default=["visual", "text"],
    nargs="*",
    type=str,
    help="Query modalities to use for retrieval.",
)
parser.add_argument(
    "--evaluation",
    default="global",
    choices=[
        "local",
        "global",
    ],
    type=str,
    help="Type of evaluation. Local: within the same video, Global: across all videos",
)
parser.add_argument(
    "--finetuned",
    action="store_true",
    help="Use finetuned CVR model if available (only BLIP).",
)
parser.add_argument(
    "--query_frames", default=15, type=int, help="Number of video query frames."
)
parser.add_argument(
    "--target_frames", default=15, type=int, help="Number of video target frames."
)
parser.add_argument(
    "--text",
    default="tfcvr",
    choices=["instruction", "tfcvr", "gt"],
    type=str,
    help="Type of query text to use for retrieval. instruction: instruction text, tfcvr: modified captions, gt: target clip narration",
)
parser.add_argument(
    "--fusion",
    default="avg",
    choices=["crossattn", "avg"],
    type=str,
    help="Query fusion strategy when using visual-text modality.",
)
parser.add_argument(
    "--min_gallery_size", default=2, type=int, help="Minimum gallery size. default=2"
)
parser.add_argument(
    "--no_precomputed", action="store_true", help="Do not use precomputed embeddings."
)
parser.add_argument(
    "--neighbors",
    default=15,
    type=int,
    help="Number of neighbors to use for the first stage of 2-stage retrieval.",
)

args = parser.parse_args()

#####################
###### CONFIG #######
#####################
config = {
    "blip": {
        "annotations": f"{ROOT_DIR}/annotation/egocvr/egocvr_annotations_gallery.csv",
        "embedding_path": f"{EMBEDDING_DIR}/EgoCVR_blip-large.csv",
        "ckpt_path_finetuned": "./checkpoints/webvid-covr.ckpt",
        "ckpt_path_notfinetuned": "https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_large_retrieval_coco.pth",
        "video_folder": f"{VIDEO_DIR}/egocvr_clips",
    },
    "egovlpv2": {
        "annotations": f"{ROOT_DIR}/annotation/egocvr/egocvr_annotations_gallery.csv",
        "embedding_path": f"{EMBEDDING_DIR}/EgoCVR_EgoVLPv2.csv",
        "ckpt_path": "./checkpoints/EgoVLPv2.pth",
        "video_folder": f"{VIDEO_DIR}/egocvr_clips_256",
    },
    "clip": {
        "annotations": f"{ROOT_DIR}/annotation/egocvr/egocvr_annotations_gallery.csv",
        "embedding_path": f"{EMBEDDING_DIR}/EgoCVR_ViT-L-14_datacomp_xl_s13b_b90k.csv",
        "video_folder": f"{VIDEO_DIR}/egocvr_clips",
    },
    "languagebind": {
        "annotations": f"{ROOT_DIR}/annotation/egocvr/egocvr_annotations_gallery.csv",
        "embedding_path": f"{EMBEDDING_DIR}/EgoCVR_LanguageBind.csv",
        "video_folder": f"{VIDEO_DIR}/egocvr_clips",
    },
}

modalities = args.modalities
assert len(modalities) <= 2, "We implemented only 2 stages"
evaluation = args.evaluation
finetuned = args.finetuned
num_query_frames = args.query_frames
num_target_frames = args.target_frames
fusion = args.fusion
text_variant = args.text
min_gallery_size = args.min_gallery_size
no_precomputed = args.no_precomputed
num_neighbors = args.neighbors

# Recalls
recalls = [1, 5, 10] if not evaluation == "local" else [1, 2, 3]

if "blip" in args.models:
    config["blip"]["ckpt_path"] = (
        config["blip"]["ckpt_path_finetuned"]
        if finetuned
        else config["blip"]["ckpt_path_notfinetuned"]
    )

query_frame_method = "middle" if num_query_frames == 1 else "sample"
if text_variant == "tfcvr":
    text_variant = "modified_captions"
elif text_variant == "gt":
    text_variant = "target_clip_narration"
else:
    text_variant = "instruction"

for _, config_ in config.items():
    config_["embedding_path_raw"] = (
        config_["embedding_path"].replace(".csv", ".pt")
        if Path(config_["embedding_path"].replace(".csv", ".pt")).exists()
        else None
    )

assert len(args.models) == len(args.modalities)


def seed_everything(seed=42):
    # Set Python seed
    random.seed(seed)

    # Set NumPy seed
    np.random.seed(seed)

    # Set PyTorch seed for CPU
    torch.manual_seed(seed)

    # Set PyTorch seed for GPU, if available
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_embeddings(path, emb_path=None):
    df = pd.read_csv(path)
    if emb_path:
        embeddings = torch.load(emb_path)
        embeddings = embeddings.to(CUDA_DEVICE)
    else:
        embeddings = df["clip_embeddings"].apply(
            lambda emb: np.array(literal_eval(emb))
        )
        embeddings = np.stack(embeddings)
        embeddings = torch.tensor(embeddings, device=CUDA_DEVICE, dtype=torch.float32)
    return df, embeddings


def dump_embeddings(path, emb_path=None):
    # dump embeddings to .pt file if not already present
    if not emb_path:
        dump_path = path.replace(".csv", ".pt")
        print(
            f"Dumping embeddings to {dump_path}. Improves loading time for future runs."
        )
        df = pd.read_csv(path)
        embeddings = df["clip_embeddings"].apply(
            lambda emb: np.array(literal_eval(emb))
        )
        embeddings = np.stack(embeddings)
        embeddings = torch.tensor(embeddings, device=CUDA_DEVICE, dtype=torch.float32)
        torch.save(embeddings, dump_path)
        return dump_path
    return emb_path


def nearest_neighbors(
    candidate_embeddings,
    query,
    k,
    normalize=True,
    return_distances=True,
):
    if query.ndim == 1:
        query = query.unsqueeze(0)
    if normalize:
        candidate_embeddings = F.normalize(candidate_embeddings, dim=-1)
        query = F.normalize(query, dim=-1)

    similarities = torch.matmul(query, candidate_embeddings.T)
    topk_values, topk_indices = torch.topk(similarities, k, largest=True)

    if return_distances:
        return topk_values, topk_indices
    else:
        return topk_indices


def compute_recall_at_k(
    query_embeddings,
    candidate_embeddings,
    ground_truth,
    k,
    gallery,
    min_gallery_size,
    modalities,
    num_neighbors=None,
):

    _, indices = nearest_neighbors(
        candidate_embeddings[args.models[0]],
        torch.stack(query_embeddings[args.models[0]][modalities[0]]),
        (len(candidate_embeddings[args.models[0]])),
    )

    total_relevant = 0
    total_retrieved_relevant = 0

    num_queries = len(ground_truth)
    for i in range(num_queries):

        relevant_items = set(ground_truth[i])
        filtered_indices = torch.tensor(list(set(gallery[i])), device=CUDA_DEVICE)

        filter_mask = torch.isin(indices[i], filtered_indices)
        filtered_indices = indices[i][filter_mask]

        if len(filtered_indices) < min_gallery_size:
            # skip this query
            continue

        if len(modalities) > 1:
            new_query = query_embeddings[args.models[1]][modalities[1]][i]
            if new_query.ndim < 2:
                new_query = new_query.unsqueeze(0)

            new_candidates_indices = filtered_indices[:num_neighbors]
            new_candidates = candidate_embeddings[args.models[1]][
                new_candidates_indices
            ]
            _, new_indices = nearest_neighbors(
                new_candidates,
                new_query,
                k=(len(new_candidates)),
            )

            filtered_indices = new_candidates_indices.cpu().numpy()[
                new_indices.cpu().numpy()[0]
            ]

        retrieved_items = set(filtered_indices[:k].tolist())
        relevant_retrieved = relevant_items.intersection(retrieved_items)

        total_relevant += 1
        total_retrieved_relevant += min(len(relevant_retrieved), 1)

    recall_at_k = total_retrieved_relevant / total_relevant if total_relevant > 0 else 0
    return recall_at_k


def main():
    print(
        f"Running {args.models} retrieval with {modalities} using {evaluation} evaluation."
    )
    seed_everything(123)

    tqdm.pandas()
    models = {}
    frame_loaders = {}
    tokenizers = {}
    model_forwards = {}
    text_forwards = {}
    if "blip" in args.models:
        model_blip, frame_loader_blip, tokenizer_blip = init_BLIP(
            checkpoint_path=config["blip"]["ckpt_path"],
            query_frame_method=query_frame_method,
            num_query_frames=num_query_frames,
            device=CUDA_DEVICE,
        )
        models["blip"] = model_blip
        frame_loaders["blip"] = frame_loader_blip
        tokenizers["blip"] = tokenizer_blip
        model_forwards["blip"] = forward_blip
        text_forwards["blip"] = forward_blip_text

    if "egovlpv2" in args.models:
        model_egovlpv2, frame_loader_egovlpv2, tokenizer_egovlpv2 = init_EgoVLPv2(
            checkpoint_path=config["egovlpv2"]["ckpt_path"], device=CUDA_DEVICE
        )
        models["egovlpv2"] = model_egovlpv2
        frame_loaders["egovlpv2"] = frame_loader_egovlpv2
        tokenizers["egovlpv2"] = tokenizer_egovlpv2
        model_forwards["egovlpv2"] = forward_egovlpv2
        text_forwards["egovlpv2"] = forward_egovlpv2_text

    if "clip" in args.models:
        model_clip, frame_loader_clip, tokenizer_clip = init_CLIP(
            query_frame_method=query_frame_method,
            num_query_frames=num_query_frames,
            device=CUDA_DEVICE,
        )
        models["clip"] = model_clip
        frame_loaders["clip"] = frame_loader_clip
        tokenizers["clip"] = tokenizer_clip
        model_forwards["clip"] = forward_clip
        text_forwards["clip"] = partial(forward_clip_text, tokenizer=tokenizer_clip)

    if "languagebind" in args.models:
        model_languagebind, frame_loader_languagebind, tokenizer_languagebind = (
            init_languagebind(device=CUDA_DEVICE)
        )
        models["languagebind"] = model_languagebind
        frame_loaders["languagebind"] = frame_loader_languagebind
        tokenizers["languagebind"] = tokenizer_languagebind
        model_forwards["languagebind"] = forward_languagebind
        text_forwards["languagebind"] = forward_languagebind_text

    df_dict = {}
    model_embeddings_dict = {}
    for model in set(args.models):
        dump_embeddings(
            config[model]["embedding_path"], config[model]["embedding_path_raw"]
        )
        df_dict[model], model_embeddings_dict[model] = load_embeddings(
            config[model]["embedding_path"], config[model]["embedding_path_raw"]
        )

    # Fix for LanguageBind embeddings due to possible unnecessary extra dimension
    if "languagebind" in args.models:
        model_embeddings_dict["languagebind"] = model_embeddings_dict[
            "languagebind"
        ].squeeze(1)

    annotation_df = pd.read_csv(config[args.models[0]]["annotations"])

    all_targets = annotation_df["target_clip_ids"].apply(literal_eval)

    gallery = (
        annotation_df[f"{args.evaluation}_idx"].progress_apply(literal_eval).tolist()
    )

    query_embeddings = {}
    for model in set(args.models):
        query_embeddings[model] = defaultdict(list)
    candidate_embeddings = []
    ground_truth = []

    index_mapping = {}
    for model in set(args.models):
        for i in range(len(model_embeddings_dict[model])):
            clip_id = df_dict[model].iloc[i]["clip_name"]
            index_mapping[clip_id] = i

    print(f"Generating {args.models} {modalities} embeddings")
    for i in tqdm(range(len(annotation_df))):

        modifier_text = annotation_df.iloc[i][text_variant]

        video_uid = annotation_df.iloc[i]["video_clip_id"].split("_")[0]
        clip_name = annotation_df.iloc[i]["video_clip_id"]

        with torch.no_grad():
            for modality, model in zip(modalities, args.models):
                video_path = (
                    Path(config[model]["video_folder"]) / video_uid / f"{clip_name}.mp4"
                )
                query_video = model_embeddings_dict[model][index_mapping[clip_name]]
                query_caption = modifier_text

                query_embedding = model_forwards[model](
                    modality,
                    models[model],
                    tokenizers[model],
                    query_video,
                    query_caption,
                    video_path,
                    frame_loaders[model],
                    fusion,
                    num_query_frames,
                    query_frame_method,
                    use_precomputed=(not no_precomputed),
                )

                query_embeddings[model][modality].append(query_embedding)

        all_gts = []
        for entry in all_targets[i]:
            all_gts.append(index_mapping[entry])
        ground_truth.append(all_gts)

    candidate_embeddings = model_embeddings_dict
    if num_target_frames == 1:
        # use only the middle frame for target clips
        for model in set(args.models):
            if "languagebind" not in model and "egovlpv2" not in model:
                temporal_mid = candidate_embeddings[model].shape[1] // 2
                candidate_embeddings[model] = candidate_embeddings[model][
                    :, temporal_mid, :
                ]

    for model in set(args.models):
        if candidate_embeddings[model].ndim > 2:
            candidate_embeddings[model] = candidate_embeddings[model].mean(1)

    recall_results = []
    for k in recalls:
        recall = compute_recall_at_k(
            query_embeddings,
            candidate_embeddings,
            ground_truth,
            k,
            gallery,
            min_gallery_size,
            modalities,
            num_neighbors,
        )
        recall_results.append(recall)
    print(
        f"Recall@{','.join([str(r) for r in recalls])}: {' & '.join([str('{0:.3f}'.format(res)) for res in recall_results])} \\\\"
    )


if __name__ == "__main__":
    main()
