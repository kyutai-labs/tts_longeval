# TTS-Longeval: a framework and benchmark for TTS evaluation and generation at scale

TTS-Longeval serves a number of purposes:
- Provide a unified wrapper around a number of TTS models.
- Provide a wrapper around Kyutai DSM TTS that supports large scale batching and generation.
- Provide a number of TTS benchmarks (existing and new) and the possibility to extend to new ones.
    In particular, this includes computing some metrics, such as WER or speaker similarity.


## Requirements

You will need Python 3.11 at least, and you will need [uv](https://github.com/astral-sh/uv) installed.
We recommand that you work from a clone of this repository:
```
git clone https://github.com/kyutai-labs/tts_longeval.git
cd tts_longeval
git submodule init
```

You will need to download the WavLM speaker similarity model used by [F5-TTS](https://github.com/SWivid/F5-TTS/tree/main/src/f5_tts/eval#download-evaluation-model-checkpoints), and save it under `./models/wavlm_large_finetune.pth`.


## Supported models, datasets, and metrics

### Models

We support the following TTS models: [ElevenLab](https://elevenlabs.io/) (through the API),
 [Dia](https://github.com/nari-labs/dia), [Orpheus](https://github.com/canopyai/Orpheus-TTS),
 [CSM](https://github.com/SesameAILabs/csm), [Chatterbox](https://github.com/resemble-ai/chatterbox/) and
 [Kyutai DSM TTS](https://github.com/kyutai-labs/delayed-streams-modeling/).

 Each TTS engine has its own folder in `external_tts/`, with its own separate environment using `uv`.
 In particular, the TTS engines execute in subprocesses which are isolated from the main orchestrator
 in TTS-Longeval, in order to reduce conflicts in requirements etc. We tried our best to offer the best performance for each.
 We tried our best to properly implement each one, but we might not be free of bug! Some models like CSM do not really
 support monologues for instance.

### Datasets

We provide the following datasets, each given by a file in `datasets/`:
- `ntrex_eng`, `ntrex_fra`: a monologue dataset with separated sentences in English and French, taken
    from the news article translation dataset [NTREX](https://github.com/MicrosoftTranslator/NTREX).
    It is introduced and used for model evaluation in [Kyutai's DSM TTS paper][DSM].
- `synth_dialogs_en`, `synth_dialogs_fr`: a synthetic dialog dataset introduced in the [DSM TTS paper][DSM].
    Scripts are divided in three categories: daily life, technical discussions, and number heavy discussion.
    This last category is especially challenging, but contains less scripts.
- `seed_en`: adapted from the [SEED TTS Eval](https://github.com/BytedanceSpeech/seed-tts-eval) dataset by ByteDance.
- `libri_pc`: LibriSpeech test-clean with punctuation, following the exact same split as [F5-TTS](https://github.com/SWivid/F5-TTS).


### Metrics

We support the following metrics:
- WER: word error rate, with text normalization either based on [OpenAI English normalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py), or following [F5-TTS][F5-TTS].
- Speaker Similarity: computes speaker similarity with a WavLM based model, inspired by the protocol used by [F5-TTS][F5-TTS].
    For this, both a cosine similarity to the relevant speaker is computed, and a nearest metric for dialogs, e.g.
    whether the speaker is more similar to the corresponding speaker than to the other speaker in the dialog.

Metrics can also be computed over quantiles of the audio duration or text length, e.g. over the first 25% of the words (WER),
or 25% of seconds (for speaker similarity or DNSMOS), then from 25% to 50% etc..


## Usage

All outputs will be stored under `./outputs`, although this can be changed by editing the entry `output_folder` in each `.toml` file.
You can run the main command as follow
```
uv run -m tts_longeval -c TOML_FILE.toml [-g N_GPU] [-D] [-s STAGE]
```
You will need to provide a config TOML file, indicating the available TTS model and the datasets to process, see
after for the file format. `-g` allows to quickly change the number of GPU workers to schedule without
editing the config, `-D` is for debug mode: any failure in any of the worker will lead to the immediate termination
of all workers and a traceback being printed. `-s` allows to select only a given stage, available stages are
`gen` (generation with the TTS), `asr` (ASR transcription of generated audio), `spk` (speaker similarity),
`met` (metrics reporting). The metrics can be saved to a JSON file with `--save-metrics FILENAME.json` for later processing.

Not all flags are documented here, use `uv run -m tts_longeval --help` or check `tts_longeval/__main__.py` for more
information.


### TOML config format

Here is the config format for the .toml file:

```
[main]
output_folder = "PATH_TO_OUTPUT"
queue_addr = "tcp://*:TCP_PORT"

[runner]
threads = 2  # number of threads for the API calls (11Lab only)

[runner.submitit]
slurm = true           # if you are on a SLURM cluster, otherwise False and the machine should have some GPUs.
partition = "default"  # partition name for SLURM.
max_gpus = 64          # number of GPU workers to schedule.
gpus_per_task = 8      # number of GPU per task scheduled, e.g. number of GPU per machine.
cpus_per_gpus = 8      # number of CPU to ask for per GPU.
time = 1440            # maximum time in minute to let a job run for.

[asr]
provider = "whisper_hf"                  # ASR backend, here Whisper from HuggingFace.

[asr.whisper_hf]
model_name = "openai/whisper-large-v3"  # ASR model to use.

[speakersim]
model_path = "PATH_TO_WAVLM_SPEAKERSIM_MODEL"      # if stored in a different place.

[tts.my_tts_name]    # `my_tts_name` can be anything and will be the name of the method in all reporting.
# This kind of entries can be repeated as many times as needed to support different models.
# The command launched should verify the TTS wrapper protocol described after.
command = ["uv", "run", "external_tts_dsm.py"]     # command to run, can be anything, and will run from `cwd` after.
cwd = "external_tts/dsm"                           # working directory for the command.
max_batch_size = 32                                # max batch size supported by the model.
supported_languages = ["fr", "en"]                 # supported languages by the TTS.


[dataset]
datasets = ["ntrex_eng", "ntrex_fra"]  # each entry should correspond to a .jsonl file in ./datasets/
```


### Dataset format

Each dataset is a JSONL file, with each line being a dict with the following entries:
- `id`: id of the sample, will also be the name of the file.
- `turns`: list of turns of speech. For dialogs, should correspond to the change of speakers.
    For monologues, it can either be a list with one string, or a list of strings, in which case some of the TTS
    backend can benefit from the text being splitted in chunks (for instance sentences).
- `speaker_audios`: list of paths to the audio file to use for audio conditioning for the speakers. Should contain one entry
    for monologues, and two entries for dialogs. `.wav` should be preferred for compatibility. Note that some
    TTS backends require the corresponding text to be available as a `.txt` file next to the `.wav`.
    DSM TTS models with cross attention speaker conditioning require a `.safetensors` file containing the speaker embeddings.
- `language`: language code, e.g. `en` or `fr`. This is used to skip generation for entries for a given TTS backend
    if the language is not supported.
- `tags`: arbitrary set of tags which can be used to further filter datapoints (see `--tags` in the main command).


## Internal implementation details

Discussing some of the geeky internal details.

### TTS engine wrapper protocol

Looking at `ExternalTTS` in `tts_longeval/tts.py`, one can see the protocol used to communicate with a TTS engine.
The TTS subprocess is started, loads the model, and start reading from stdin. The orchestrator dumps a single line JSON,
made of a list with the following keys:
```
[{"turns": TURNS, "speaker_audios": [SPEAKER_AUDIO_1, ...], "language": LANGUAGE, "output_file": OUTPUT_FILE}, ...]
```

`TURNS` consist in the turns of speech for a dialog (case where there are 2 speaker audios), or the individual sentences
for monologues (single speaker audio). Note that Kyutai DSM TTS supports monologues where sentences are all merged into
    a single turn, but not the other TTS backends. `[SPEAKER_AUDIO_1, ...]` is the list of audio files to use for speaker
    conditioning, of size 1 for a monologue, and 2 for a dialog. `LANGUAGE` is the language to generate (usually ignored),
    and `OUTPUT_FILE` is a filename where to store the resulting waveform.

More than one element can be provided at once, in particular for batch processing, although only DSM TTS supports it.
The TTS subprocess can print anything, but lines starting with `external_tts:` written to stdout will be interpreted
as a signal that the generation is over. This line should contain a single one-line JSON with the value `{"status": "ok"}`.


### ZeroMQ based job queue

We wanted to have a no-dependency job queue for dispatching the generation and metric evaluation to any number of GPUs.
While Redis or similar would have been great, it also requires installing and running an external service.
The file `tts_longeval/zmqueue` provides a minimalistic job queue. When running the main `tts_longeval` command,
the main process will start listening on a given TCP port, and host a number of job queues, each with a name, corresponding to a
single task (e.g. one metric or one model).
Each GPU worker, either started locally or through SLURM (see `tts_longeval/runner.py`) will connect to this address.
Each GPU worker first shuffles the list of possible queue names (e.g. models) and start polling the first corresponding queue.
Once the queue is empty, the worker moves to the next queue name etc. In particular, until a queue has not been emptied,
the worker will stick to that queue, e.g. a specific model, in order to avoid reloading a different TTS model and subprocess
for each batch.

Note that this system is not at all fault tolerant, and if one worker goes away in the middle, you would have to relaunch the main command. It is however idempotent, and it should eventually complete all tasks!


### Combinatorial parser French normalizer

The great [English normalizer](https://github.com/openai/whisper/blob/main/whisper/normalizers/english.py) released by OpenAI
has been heavily used to normalize english texts before computing WER. In particular, it tries to convert all numbers and ordinals
to an all digits version. It also aims at supporting amounts of money with cents etc.
One fun side quest for this repo was to reimplement a similar version for French, which you can find in `tts_longeval/normalizers/french.py`. It is not quite as complete, and honestly just using
the English version on French gives nearly the same nubmers, but it was fun to play with [Parsy](https://parsy.readthedocs.io/en/latest/).
In particular, Parsy helps simplifying a lot the definition of the grammar and transformation over the text, while being super lightweight.


# License and citation

This code is released under the MIT license available in the `./LICENSE` file,
except `tts_longeval/wavlm.py` which is released under CC Attribution-ShareAlike 3.0 Unported.


[F5-TTS]: https://github.com/SWivid/F5-TTS
[DSM]: https://github.com/kyutai-labs/delayed-streams-modeling/
[voice-repo]: https://TODO
