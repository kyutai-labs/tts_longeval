# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# uv run --with=audioseal,julius test_watermark.py FOLDER
"""Script used to test the resilience of a watermark to encoding with Mimi."""

import argparse
from pathlib import Path
import sphn

from audioseal import AudioSeal
from huggingface_hub import hf_hub_download
import julius.resample
from moshi.models import loaders
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-duration", type=float, default=10.0)
    parser.add_argument("--nq", type=int, default=32)
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(8)

    model = AudioSeal.load_generator("audioseal_wm_16bits")
    detector = AudioSeal.load_detector("audioseal_detector_16bits")
    mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device="cpu")
    mimi.set_num_codebooks(args.nq)

    total_found = 0.0
    total_mimi_found = 0.0
    total = 0.0
    files = args.folder.glob("**/*.wav")
    for file in files:
        print(file)
        wav_np, sr = sphn.read(file)
        wav = torch.from_numpy(wav_np)
        le = wav.shape[-1]
        segment = int(sr * args.segment_duration)
        for offset in range(0, le - segment // 2, segment):
            chunk = wav[..., offset : offset + segment].mean(dim=0).view(1, 1, -1)

            watermark = model.get_watermark(chunk, sr)
            chunk = chunk + watermark

            chunk_mimi = julius.resample.resample_frac(chunk, sr, mimi.sample_rate)
            with torch.no_grad():
                codes = mimi.encode(chunk_mimi)
                chunk_mimi = mimi.decode(codes)
                chunk_mimi = julius.resample.resample_frac(chunk_mimi, mimi.sample_rate, sr)

                result, _ = detector.detect_watermark(chunk, sr)
                result_mimi, _ = detector.detect_watermark(chunk_mimi, sr)

            total += 1
            total_found += bool(result)
            total_mimi_found += bool(result_mimi)

            perc_found = total_found / total
            perc_mimi_found = total_mimi_found / total

            print(f"Results (no mimi): {100 * perc_found:.1f}%")
            print(f"Results (mimi): {100 * perc_mimi_found:.1f}%")


if __name__ == "__main__":
    main()
