from pathlib import Path
import importlib

import torch
import torch.nn.functional as F

from utils import FrameLoader, pre_caption

TEXT_MAX_WORDS = 30


def temporal_sample(embeddings, num_frames):
    if embeddings.ndim == 1:
        return embeddings
    len_emb = len(embeddings)
    if num_frames == 1:
        # take the middle frame
        return embeddings[len_emb // 2]
    else:
        # for now just averaging, could be improved to sample num_frames frames
        return embeddings.mean(0)


def forward_blip(
    modality,
    model,
    tokenizer,
    ref_img,
    caption,
    video_path,
    frame_loader,
    modality_fusion,
    num_query_frames,
    query_frame_method,
    use_precomputed=True,
    device="cuda",
):
    if not use_precomputed or (
        modality == "visual-text" and modality_fusion == "crossattn"
    ):
        ref_img = frame_loader(video_path.as_posix()).to(device)
        if query_frame_method == "sample":
            cross_embed = []
            for i in range(len(ref_img)):
                embed = forward_blip_crossattn(model, ref_img[i], caption)
                cross_embed.append(embed)
            cross_embed = torch.stack(cross_embed, dim=0)
            cross_embed = cross_embed.mean(0)
            return cross_embed

    if modality == "visual":
        return (
            temporal_sample(ref_img, num_query_frames)
            if use_precomputed
            else forward_blip_visual(model, ref_img)
        )
    elif modality == "visual-text":
        if modality_fusion == "crossattn":
            # can't use pre-extracted embeddings for crossattn fusion
            return forward_blip_crossattn(model, ref_img, caption)
        elif modality_fusion == "avg":
            if use_precomputed:
                if isinstance(caption, torch.Tensor):
                    text_embed = caption
                else:
                    text_embed = forward_blip_text(model, caption)
                ref_img = temporal_sample(ref_img, num_query_frames)
                return (ref_img + text_embed) / 2
            else:
                return forward_blip_visual_text_avg(model, ref_img, caption)
        else:
            raise NotImplementedError(f"Fusion {modality_fusion} not implemented")
    elif modality == "text":
        if use_precomputed and isinstance(caption, torch.Tensor):
            return caption
        return forward_blip_text(model, caption)
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")


def forward_blip_visual(model, ref_img):
    ref_img = ref_img.unsqueeze(0)

    model.eval()
    ref_img_embs = model.visual_encoder(ref_img)
    query_feat = F.normalize(model.vision_proj(ref_img_embs[:, 0, :]), dim=-1)

    query_feat = query_feat.squeeze(0)
    return query_feat


def forward_blip_text(model, caption, device="cuda"):
    caption = pre_caption(caption, TEXT_MAX_WORDS)
    text = model.tokenizer(
        caption,
        padding="longest",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    # Shift encoder
    query_embs = model.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    query_feat = query_embs.last_hidden_state[:, 0, :]
    query_feat = F.normalize(model.text_proj(query_feat), dim=-1)

    query_feat = query_feat.squeeze(0)
    return query_feat


def forward_blip_crossattn(model, ref_img, caption):
    ref_img = ref_img.unsqueeze(0)

    model.eval()
    device = ref_img.device

    ref_img_embs = model.visual_encoder(ref_img)
    ref_img_atts = torch.ones(ref_img_embs.size()[:-1], dtype=torch.long).to(device)

    caption = pre_caption(caption, TEXT_MAX_WORDS)
    text = model.tokenizer(
        caption,
        padding="longest",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    # Shift encoder
    encoder_input_ids = text.input_ids.clone()
    encoder_input_ids[:, 0] = model.tokenizer.enc_token_id
    query_embs = model.text_encoder(
        encoder_input_ids,
        attention_mask=text.attention_mask,
        encoder_hidden_states=ref_img_embs,
        encoder_attention_mask=ref_img_atts,
        return_dict=True,
    )
    query_feat = query_embs.last_hidden_state[:, 0, :]
    query_feat = F.normalize(model.text_proj(query_feat), dim=-1)

    query_feat = query_feat.squeeze(0)
    return query_feat


def forward_blip_visual_text_avg(model, ref_img, caption):
    ref_img = ref_img.unsqueeze(0)

    model.eval()
    device = ref_img.device

    # visual forward
    ref_img_embs = model.visual_encoder(ref_img)
    query_feat_vis = F.normalize(model.vision_proj(ref_img_embs[:, 0, :]), dim=-1)

    caption = pre_caption(caption, TEXT_MAX_WORDS)
    text = model.tokenizer(
        caption,
        padding="longest",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    ).to(device)

    # Shift encoder
    query_text_embs = model.text_encoder(
        text.input_ids,
        attention_mask=text.attention_mask,
        return_dict=True,
        mode="text",
    )
    query_feat_txt = query_text_embs.last_hidden_state[:, 0, :]
    query_feat_txt = F.normalize(model.text_proj(query_feat_txt), dim=-1)

    query_feat = (query_feat_vis + query_feat_txt) / 2
    query_feat = query_feat.squeeze(0)
    return query_feat


def forward_egovlpv2(
    modality,
    model,
    tokenizer,
    ref_img,
    caption,
    video_path,
    frame_loader,
    modality_fusion,
    num_query_frames,
    query_frame_method,
    use_precomputed=True,
    device="cuda",
):
    if not use_precomputed or (
        modality == "visual-text" and modality_fusion == "crossattn"
    ):
        ref_img = frame_loader(video_path.as_posix()).to(device)

    if modality == "visual":
        return ref_img if use_precomputed else forward_egovlpv2_visual(model, ref_img)

    elif modality == "visual-text":
        if modality_fusion == "crossattn":
            # can't use pre-extracted embeddings for crossattn fusion
            raise NotImplementedError("Crossattn fusion not implemented for EgoVLPv2")
        elif modality_fusion == "avg":
            if use_precomputed:
                if isinstance(caption, torch.Tensor):
                    text_embed = caption
                else:
                    text_embed = forward_egovlpv2_text(model, tokenizer, caption)
                return (ref_img + text_embed) / 2
            else:
                text_embed = forward_egovlpv2_text(model, tokenizer, caption)
                video_embed = forward_egovlpv2_visual(model, ref_img)
                return (video_embed + text_embed) / 2
        else:
            raise NotImplementedError(f"Fusion {modality_fusion} not implemented")
    elif modality == "text":
        if use_precomputed and isinstance(caption, torch.Tensor):
            return caption
        return forward_egovlpv2_text(model, tokenizer, caption)
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")


def forward_egovlpv2_text(model, tokenizer, caption, device="cuda"):
    text = tokenizer(caption, return_tensors="pt", padding=True, truncation=True)
    text = {key: val.cuda(device) for key, val in text.items()}
    text_embed = model.compute_text(text)
    text_embed /= text_embed.norm(dim=-1, keepdim=True)
    return text_embed


def forward_egovlpv2_visual(model, ref_img):
    ref_img = ref_img.unsqueeze(0)
    video_embed = model.compute_video(ref_img)
    video_embed /= video_embed.norm(dim=-1, keepdim=True)
    video_embed = video_embed.squeeze(0)
    return video_embed


def forward_clip(
    modality,
    model,
    tokenizer,
    ref_img,
    caption,
    video_path,
    frame_loader,
    modality_fusion,
    num_query_frames,
    query_frame_method,
    use_precomputed=True,
    device="cuda",
):
    if not use_precomputed:
        ref_img = frame_loader(video_path.as_posix()).to(device)
        if query_frame_method == "sample":
            raise NotImplementedError

    if modality == "visual":
        return (
            temporal_sample(ref_img, num_query_frames)
            if use_precomputed
            else forward_clip_visual(model, ref_img)
        )
    elif modality == "visual-text":
        if modality_fusion == "crossattn":
            # can't use pre-extracted embeddings for crossattn fusion
            raise NotImplementedError("Crossattn fusion not implemented for CLIP")
        elif modality_fusion == "avg":
            if use_precomputed:
                if isinstance(caption, torch.Tensor):
                    text_embed = caption
                else:
                    text_embed = forward_clip_text(model, tokenizer, caption, device)
                ref_img = temporal_sample(ref_img, num_query_frames)
                return (ref_img + text_embed) / 2
            else:
                text_embed = forward_clip_text(model, tokenizer, caption, device)
                video_embed = forward_clip_visual(model, ref_img)
                return (video_embed + text_embed) / 2
        else:
            raise NotImplementedError(f"Fusion {modality_fusion} not implemented")

    elif modality == "text":
        if use_precomputed and isinstance(caption, torch.Tensor):
            return caption
        return forward_clip_text(model, tokenizer, caption, device)
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")


def forward_clip_text(model, tokenizer, caption, device="cuda"):
    text = tokenizer(caption).to(device)
    with torch.cuda.amp.autocast():
        text_embed = model.encode_text(text)
        text_embed /= text_embed.norm(dim=-1, keepdim=True)

        text_embed = text_embed.squeeze(0)
    text_embed = text_embed.float()
    return text_embed


def forward_clip_visual(model, ref_img):
    with torch.cuda.amp.autocast():
        video_embed = model.encode_image(ref_img)
        video_embed /= video_embed.norm(dim=-1, keepdim=True)
    video_embed = video_embed.float()

    return video_embed


def init_EgoVLPv2(checkpoint_path, device="cuda", no_temporal=False, small_proj=False):
    from src.model.egovlpv2.model import FrozenInTime
    from src.model.egovlpv2.video_utils import FrameLoader as EgoFrameLoader
    import transformers

    video_params = {
        "model": "SpaceTimeTransformer",
        "arch_config": "base_patch16_224",
        "num_frames": 16,
        "pretrained": True,
        "time_init": "zeros",
    }
    text_params = {"model": "roberta-base", "pretrained": True, "input": "text"}

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        "roberta-base", TOKENIZERS_PARALLELISM=False
    )
    projection = "small" if small_proj else "default"

    model = FrozenInTime(
        video_params,
        text_params,
        projection_dim=4096,
        load_checkpoint=checkpoint_path,
        projection=projection,
        load_temporal_fix="bilinear",
        task_names="EgoNCE_ITM_MLM",
        norm_layer=None,
        embed_dim=768,
    )
    model = model.to(device)

    if no_temporal:
        frame_method = "middle_repeat"
    else:
        frame_method = "uniform"
    frame_loader = EgoFrameLoader(16, method=frame_method)
    model.eval()
    return model, frame_loader, tokenizer


class SimpleEgoVLPDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, frame_loader, transform=None):
        self.video_paths = video_paths
        self.frame_loader = frame_loader
        self.transform = transform
        # remove extension and only filename
        self.video_ids = [Path(p).stem for p in video_paths]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_id = self.video_ids[idx]
        frames = self.frame_loader(video_path)
        if self.transform:
            frames = self.transform(frames)
        return video_id, idx, frames


def init_BLIP(checkpoint_path, query_frame_method, num_query_frames, device="cuda"):
    from src.model.blip_cir import blip_cir, BLIPCir
    from src.data.transforms import transform_test

    model = BLIPCir(
        med_config="./configs/med_config.json",
        image_size=384,
        vit="large",
        vit_grad_ckpt=True,
        vit_ckpt_layer=12,
        embed_dim=256,
        train_vit=False,
        loss=None,
    )
    model = blip_cir(model, checkpoint_path)
    model = model.to(device)

    transform = transform_test(384)
    # frame loader for query videos. "middle" or "sample"
    frame_loader = FrameLoader(
        transform=transform, method=query_frame_method, frames_video=num_query_frames
    )
    model.eval()
    return model, frame_loader, None


def init_CLIP(query_frame_method, num_query_frames, device="cuda"):
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="datacomp_xl_s13b_b90k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-L-14")

    frame_loader = FrameLoader(
        transform=preprocess, method=query_frame_method, frames_video=num_query_frames
    )

    model = model.to(device)
    model.eval()
    return model, frame_loader, tokenizer


class LanguageBindFrameLoader:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, video_path):
        if isinstance(video_path, Path):
            video_path = video_path.as_posix()
        return self.transform([video_path])


class LanguageBindTensorDivider(torch.nn.Module):
    def __init__(self, value=255.0):
        super().__init__()
        self.value = value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


def languagebind_middle_frame_processor(
    video_path,
    transform,
    video_decode_backend="opencv",
    clip_start_sec=0.0,
    clip_end_sec=None,
    num_frames=8,
):
    # Use repeated middle frame
    cv2 = importlib.import_module("cv2")
    cv2_vr = cv2.VideoCapture(video_path)
    duration = int(cv2_vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = duration // 2

    video_data = []
    cv2_vr.set(1, frame_idx)
    _, frame = cv2_vr.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2_vr.release()

    frame = torch.from_numpy(frame).permute(2, 0, 1)
    video_data = [frame] * num_frames
    video_data = torch.stack(video_data, dim=1)
    video_outputs = transform(video_data)

    return video_outputs


def init_languagebind(
    device="cuda", variant="LanguageBind_Video_FT", no_temporal=False
):
    from languagebind import (
        LanguageBind,
        transform_dict,
        LanguageBindVideoTokenizer,
    )

    tokenizer = LanguageBindVideoTokenizer.from_pretrained(f"LanguageBind/{variant}")
    model = LanguageBind(
        clip_type={"video": variant},
    ).to(device)
    model.eval()
    video_transform = transform_dict["video"](model.modality_config["video"])
    # patch the video transform to not use random horizontal flipping
    video_transform.transform.transforms = video_transform.transform.transforms[:4]
    video_transform.transform.transforms[0] = LanguageBindTensorDivider(255.0)

    if no_temporal:
        video_transform.image_processor = languagebind_middle_frame_processor
    frame_loader = LanguageBindFrameLoader(video_transform)

    return model, frame_loader, tokenizer


def forward_languagebind_text(model, tokenizer, caption, device="cuda"):
    to_device = getattr(importlib.import_module("languagebind"), "to_device")

    data = {
        "language": to_device(
            tokenizer(
                [caption],
                max_length=77,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ),
            device,
        )
    }
    embeddings = model(data)
    return embeddings["language"][0]


def forward_languagebind_visual(model, ref_img, device="cuda"):
    to_device = getattr(importlib.import_module("languagebind"), "to_device")

    if ref_img["pixel_values"].ndim == 6:
        ref_img["pixel_values"] = ref_img["pixel_values"].squeeze(0)

    data = {"video": to_device(ref_img, device)}
    embeddings = model(data)
    return embeddings["video"][0]


def forward_languagebind(
    modality,
    model,
    tokenizer,
    ref_img,
    caption,
    video_path,
    frame_loader,
    modality_fusion,
    num_query_frames,
    query_frame_method,
    use_precomputed=True,
    device="cuda",
):

    if not use_precomputed:
        ref_img = frame_loader(video_path.as_posix())

    if modality == "visual":
        return (
            ref_img
            if use_precomputed
            else forward_languagebind_visual(model, ref_img, device=device)
        )

    elif modality == "visual-text":
        if modality_fusion == "crossattn":
            raise NotImplementedError(
                "Crossattn fusion not implemented for LanguageBind"
            )
        elif modality_fusion == "avg":
            if use_precomputed:
                if isinstance(caption, torch.Tensor):
                    text_embed = caption
                else:
                    text_embed = forward_languagebind_text(
                        model, tokenizer, caption, device=device
                    )
                if ref_img.ndim == 2:
                    ref_img = ref_img.squeeze(0)
                return (ref_img + text_embed) / 2
            else:
                text_embed = forward_languagebind_text(
                    model, tokenizer, caption, device=device
                )
                video_embed = forward_languagebind_visual(model, ref_img, device=device)
                return (video_embed + text_embed) / 2
        else:
            raise NotImplementedError(f"Fusion {modality_fusion} not implemented")
    elif modality == "text":
        if use_precomputed and isinstance(caption, torch.Tensor):
            return caption
        return forward_languagebind_text(model, tokenizer, caption, device=device)
    else:
        raise NotImplementedError(f"Modality {modality} not implemented")
