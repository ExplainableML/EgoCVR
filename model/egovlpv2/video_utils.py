import random
import numpy as np
import cv2
import torch

from torchvision import transforms
from torchvision.transforms._transforms_video import NormalizeVideo


def sample_frames(num_frames, vlen, sample="rand", fix_start=None):
    acc_samples = min(num_frames, vlen)
    intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
    ranges = []
    for idx, interv in enumerate(intervals[:-1]):
        ranges.append((interv, intervals[idx + 1] - 1))
    if sample == "rand":
        frame_idxs = [random.choice(range(x[0], x[1])) for x in ranges]
    elif fix_start is not None:
        frame_idxs = [x[0] + fix_start for x in ranges]
    elif sample == "uniform":
        frame_idxs = [(x[0] + x[1]) // 2 for x in ranges]
    elif sample == "middle_repeat":
        frame_idxs = [vlen // 2] * num_frames
    else:
        raise NotImplementedError

    return frame_idxs


def read_frames_cv2(video_path, num_frames, sample="rand", fix_start=None):
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()
    vlen = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # get indexes of sampled frames
    frame_idxs = sample_frames(num_frames, vlen, sample=sample, fix_start=fix_start)
    frames = []
    success_idxs = []
    for index in frame_idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame)
            # (H x W x C) to (C x H x W)
            frame = frame.permute(2, 0, 1)
            frames.append(frame)
            success_idxs.append(index)
        else:
            pass
            print(frame_idxs, " fail ", index, f"  (vlen {vlen})")

    frames = torch.stack(frames).float() / 255
    cap.release()
    return frames, success_idxs


class FrameLoader:
    def __init__(self, num_frames, method="rand", fix_start=None):
        self.num_frames = num_frames
        self.method = method
        self.fix_start = fix_start

        normalize = NormalizeVideo(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        self.transforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.Resize(224),
                normalize,
            ]
        )

    def __call__(self, video_path):
        frames, success_idxs = read_frames_cv2(
            video_path, self.num_frames, sample=self.method, fix_start=self.fix_start
        )

        if self.num_frames > 1:
            frames = frames.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
            frames = self.transforms(frames)
            frames = frames.transpose(0, 1)  # recover
        else:
            frames = self.transforms(frames)

        return frames
