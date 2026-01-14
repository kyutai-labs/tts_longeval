# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import json
from pathlib import Path
import sys

from chatterbox.tts import ChatterboxTTS
from chatterbox.models.t3.modules import learned_pos_emb
import sphn
import torch

from external_tools.speaker import Smoother, hf_get


def new_forward(self, x):
    """
    Returns positional embeddings for index 0 up to the length of x
    """
    max_len = self.emb.weight.shape[0]
    sl = x.shape[1]
    if sl > max_len:
        raise ValueError("Too long generation.")
    return self.emb(torch.arange(0, sl, device=x.device))


@torch.no_grad()
def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, help="Maximum number of turns to merge.")
    args = parser.parse_args()

    learned_pos_emb.LearnedPositionEmbeddings.forward = new_forward

    model = ChatterboxTTS.from_pretrained(device="cuda")

    while True:
        print("Ready to process batch.")
        line = sys.stdin.readline()
        batch = json.loads(line)
        assert len(batch) == 1
        item = batch[0]

        num_speakers = len(item["speaker_audios"])

        all_audios = []
        segments = []
        start = 0.0
        sample_rate = 24000

        turns = [(idx % num_speakers, turn) for idx, turn in enumerate(item["turns"])]
        while turns:
            if num_speakers == 1:
                if args.max_turns:
                    turns_to_generate = min(len(turns), args.max_turns)
                else:
                    turns_to_generate = len(turns)
            else:
                turns_to_generate = 1
            speaker_index = turns[0][0]
            wav = None
            while turns_to_generate >= 1:
                to_generate = " ".join(turn.strip() for _, turn in turns[:turns_to_generate])
                turns = turns[turns_to_generate:]
                print("Generating", to_generate)
                audio_prompt_path = hf_get(item["speaker_audios"][speaker_index])
                try:
                    wav = model.generate(to_generate, audio_prompt_path=str(audio_prompt_path))
                except ValueError:
                    print("Input was too long, removing one turn")
                    turns_to_generate -= 1
                break
            assert wav is not None
            wav = Smoother()(wav.cpu())
            duration = wav.shape[-1] / sample_rate
            segments.append((speaker_index, (start, start + duration)))
            start += duration
            all_audios.append(wav.squeeze())

        wav = torch.cat(all_audios, dim=-1)
        wav.clamp_(-0.99, 0.99)
        output_file = Path(item["output_file"])
        sphn.write_wav(output_file, wav.numpy(), sample_rate)
        with open(output_file.with_suffix(".segments.json"), "w") as f:
            json.dump({"segments": segments}, f)
        print("saved", item["output_file"])
        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
