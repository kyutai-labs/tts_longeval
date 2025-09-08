# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapted from
# https://colab.research.google.com/drive/10v9MIEbZOr_3V8ZcPAIh8MN7q2LjcstS?usp=sharing
# originally released under the Apache 2 license (see LICENSE-apache for a copy.).
# See https://github.com/canopyai/Orpheus-TTS
"""Wrapper around Orpheus with voice cloning. Not 100% sure it is doing the right thing."""

import argparse
import json
from pathlib import Path
import sys

from snac import SNAC  # type: ignore
import sphn
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
import torch

from external_tools.speaker import get_speaker_audio, Smoother


def load_models():
    model_name = "canopylabs/orpheus-tts-0.1-pretrained"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map=0)
    return tokenizer, snac_model, model


def prepare_prompt(tokenizer, snac_model, audio_array, audio_transcript, text_to_generate):
    myts = tokenise_audio(snac_model, audio_array)
    start_tokens = torch.tensor([[128259]], dtype=torch.int64)
    end_tokens = torch.tensor([[128009, 128260, 128261, 128257]], dtype=torch.int64)
    final_tokens = torch.tensor([[128258, 128262]], dtype=torch.int64)
    voice_prompt = audio_transcript
    prompt_tokked = tokenizer(voice_prompt, return_tensors="pt")

    input_ids = prompt_tokked["input_ids"]

    zeroprompt_input_ids = torch.cat([
        start_tokens, input_ids, end_tokens, torch.tensor([myts]), final_tokens], dim=1)  # SOH SOT Text EOT EOH

    prompts = [text_to_generate]

    all_modified_input_ids = []
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        second_input_ids = torch.cat([zeroprompt_input_ids, start_tokens, input_ids, end_tokens], dim=1)
        all_modified_input_ids.append(second_input_ids)

    all_padded_tensors = []
    all_attention_masks = []

    max_length = max([modified_input_ids.shape[1] for modified_input_ids in all_modified_input_ids])

    for modified_input_ids in all_modified_input_ids:
        padding = max_length - modified_input_ids.shape[1]
        padded_tensor = torch.cat([torch.full((1, padding), 128263, dtype=torch.int64), modified_input_ids], dim=1)
        attention_mask = torch.cat([
            torch.zeros((1, padding), dtype=torch.int64),
            torch.ones((1, modified_input_ids.shape[1]), dtype=torch.int64)], dim=1)
        all_padded_tensors.append(padded_tensor)
        all_attention_masks.append(attention_mask)

    all_padded_tensors = torch.cat(all_padded_tensors, dim=0)
    all_attention_masks = torch.cat(all_attention_masks, dim=0)

    input_ids = all_padded_tensors.to("cuda")
    attention_mask = all_attention_masks.to("cuda")
    return input_ids, attention_mask


def tokenise_audio(snac_model, waveform):
    waveform = torch.from_numpy(waveform).unsqueeze(0)
    waveform = waveform.to(dtype=torch.float32)
    waveform = waveform.unsqueeze(0)

    with torch.inference_mode():
        codes = snac_model.encode(waveform)

        all_codes = []
        for i in range(codes[0].shape[1]):
            all_codes.append(codes[0][0][i].item()+128266)  # noqa
            all_codes.append(codes[1][0][2*i].item()+128266+4096)  # noqa
            all_codes.append(codes[2][0][4*i].item()+128266+(2*4096))  # noqa
            all_codes.append(codes[2][0][(4*i)+1].item()+128266+(3*4096))  # noqa
            all_codes.append(codes[1][0][(2*i)+1].item()+128266+(4*4096))  # noqa
            all_codes.append(codes[2][0][(4*i)+2].item()+128266+(5*4096))  # noqa
            all_codes.append(codes[2][0][(4*i)+3].item()+128266+(6*4096))  # noqa
        return all_codes


def convert_to_audio(snac_model, generated_ids):
    token_to_find = 128257
    token_to_remove = 128258

    # Check if the token exists in the tensor
    token_indices = (generated_ids == token_to_find).nonzero(as_tuple=True)

    if len(token_indices[1]) > 0:
        last_occurrence_idx = token_indices[1][-1].item()
        cropped_tensor = generated_ids[:, last_occurrence_idx + 1:]
    else:
        cropped_tensor = generated_ids

    processed_rows = []
    for row in cropped_tensor:
        # Apply the mask to each row
        masked_row = row[row != token_to_remove]
        processed_rows.append(masked_row)

    code_lists = []
    for row in processed_rows:
        # row is a 1D tensor with its own length
        row_length = row.size(0)
        new_length = (row_length // 7) * 7  # largest multiple of 7 that fits in this row
        trimmed_row = row[:new_length]
        trimmed_row = [t - 128266 for t in trimmed_row]
        code_lists.append(trimmed_row)

    def redistribute_codes(code_list):
        layer_1 = []
        layer_2 = []
        layer_3 = []
        for i in range((len(code_list)+1)//7):  # noqa
            layer_1.append(code_list[7*i])  # noqa
            layer_2.append(code_list[7*i+1]-4096)  # noqa
            layer_3.append(code_list[7*i+2]-(2*4096))  # noqa
            layer_3.append(code_list[7*i+3]-(3*4096))  # noqa
            layer_2.append(code_list[7*i+4]-(4*4096))  # noqa
            layer_3.append(code_list[7*i+5]-(5*4096))  # noqa
            layer_3.append(code_list[7*i+6]-(6*4096))  # noqa
        codes = [torch.tensor(layer_1).unsqueeze(0),
                 torch.tensor(layer_2).unsqueeze(0),
                 torch.tensor(layer_3).unsqueeze(0)]
        audio_hat = snac_model.decode(codes)
        return audio_hat

    my_samples = []
    for code_list in code_lists:
        samples = redistribute_codes(code_list)
        print("ADDING SAMPLES", samples.shape)
        my_samples.append(samples)
    return torch.cat(my_samples, dim=-1)[0]


@torch.no_grad()
def main():
    stdout = sys.stdout
    sys.stdout = sys.stderr

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-turns", type=int, help="Maximum number of turns to merge.")
    parser.add_argument("--context", type=int, default=1,
                        help="Context to provide when doing needing than one generation.")
    args = parser.parse_args()

    tokenizer, snac_model, model = load_models()

    while True:
        print("Ready to process batch.")
        line = sys.stdin.readline()
        batch = json.loads(line)
        assert len(batch) == 1
        item = batch[0]

        speakers = []
        for speaker_audio in item['speaker_audios']:
            audio, text = get_speaker_audio(speaker_audio, 24000)
            speakers.append((audio.numpy()[0], text))
        single_speaker = len(speakers) == 1

        all_audios = []
        all_text = []
        segments = []
        start = 0.
        sample_rate = 24000

        turns = [(idx % len(speakers), turn) for idx, turn in enumerate(item['turns'])]

        while turns:
            if single_speaker:
                if args.max_turns:
                    turns_to_generate = min(len(turns), args.max_turns)
                else:
                    turns_to_generate = len(turns)
            else:
                turns_to_generate = 1

            speaker_index = turns[0][0]
            speaker_audio, speaker_text = speakers[speaker_index]

            if single_speaker and args.context:
                audio_context = [torch.from_numpy(speaker_audio)] + all_audios[-args.context:]
                text_context = [speaker_text] + all_text[-args.context:]
                speaker_text = " ".join(text_context)
                speaker_audio = torch.cat(audio_context, dim=-1).numpy()

            success = False
            generated_ids = None
            to_generate = None
            while turns_to_generate >= 1:
                to_generate = " ".join(turn.strip() for _, turn in turns[:turns_to_generate])
                print("Generating", to_generate)
                input_ids, attention_mask = prepare_prompt(
                    tokenizer, snac_model, speaker_audio, speaker_text, to_generate)
                print(input_ids.shape, attention_mask.shape)
                max_length = 8192
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_length=max_length,
                    attention_mask=attention_mask,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    num_return_sequences=1,
                    eos_token_id=128258,
                    # end_token_id=128009,
                )
                num_generated = generated_ids.shape[-1]
                print("Generated samples", num_generated)
                if num_generated >= max_length:
                    print("We reached the maximum size, removing one turn and retrying.")
                    turns_to_generate -= 1
                else:
                    success = True
                    turns = turns[turns_to_generate:]
                    break
            if not success:
                raise RuntimeError("Even one turn was too long to generate!")
            assert generated_ids is not None
            assert to_generate is not None
            audio = convert_to_audio(snac_model, generated_ids.cpu())
            audio = Smoother()(audio)
            duration = audio.shape[-1] / sample_rate
            segments.append((speaker_index, (start, start + duration)))
            start += duration
            all_audios.append(audio.squeeze())
            all_text.append(to_generate)

        wav = torch.cat(all_audios, dim=-1)
        wav.clamp_(-0.99, 0.99)

        output_file = Path(item['output_file'])
        sphn.write_wav(output_file, wav.numpy(), sample_rate)
        with open(output_file.with_suffix('.segments.json'), 'w') as f:
            json.dump({'segments': segments}, f)
        print("saved", item['output_file'])
        stdout.write("external_tts:" + json.dumps({"status": "ok"}) + "\n")
        stdout.flush()


if __name__ == "__main__":
    main()
