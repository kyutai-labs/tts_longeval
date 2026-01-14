# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import argparse
from collections import defaultdict
import json
import logging
import os
from pathlib import Path
import tomllib

from tts_longeval.asr import ASRTask
from tts_longeval.config import Config
from tts_longeval.metrics import print_results_for_dataset
from tts_longeval.task import Tasker, MultiTasker
from tts_longeval.zmqueue import ZMQueue
from tts_longeval.speakersim import SpeakerSimilarityTask
from tts_longeval.tts import TTSTask, LoadableTTS
from tts_longeval.utils import init_logging


logger = logging.getLogger(__name__)


def any_is_prefix(name: str, prefixes: list[str]) -> bool:
    for prefix in prefixes:
        if name.startswith(prefix):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="TTS Long Eval.")
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        default="tts_longeval.toml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "-D",
        "--debug",
        action="store_true",
        help="Debug mode, errors will be fatal, logs will be tailed.",
    )
    parser.add_argument(
        "-g",
        "--gpus",
        type=int,
        help="Override the number of gpus.",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Override the number of threads.",
    )
    parser.add_argument(
        "-T",
        "--tags",
        action="append",
        default=[],
        help="Required tags.",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        type=str,
        action="append",
        help="Select a subset of datasets.",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="models",
        type=str,
        action="append",
        help="Select a subset of models.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="More verbose logging.",
    )

    STAGES = ["gen", "asr", "spk", "met"]
    parser.add_argument(
        "-s",
        "--stages",
        dest="stages",
        type=str,
        action="append",
        choices=STAGES,
        help="Select which steps to run. Repeat the flag to select multiples.",
    )
    parser.add_argument(
        "--save-metrics",
        type=Path,
        help="Path to a JSON file where to save the metrics.",
    )

    args = parser.parse_args()
    init_logging(args.verbose)
    if args.verbose:
        os.environ["_TTS_LONGEVAL_VERBOSE"] = "1"
    raw = tomllib.load(args.config.open("rb"))
    config = Config.model_validate(raw)
    if args.gpus is not None:
        config.runner.submitit.max_gpus = args.gpus
    if args.threads is not None:
        config.runner.threads = args.threads
    if args.debug is not None:
        config.main.debug = args.debug
    if not args.stages:
        args.stages = STAGES

    queue_root = config.main.output_folder / "queues"
    queue_root.mkdir(exist_ok=True, parents=True)
    output_folder = config.main.output_folder / "outputs"
    output_folder.mkdir(exist_ok=True, parents=True)

    tts_models: dict[str, LoadableTTS] = {}
    to_load = list(config.tts.items())
    to_load += list(config.tts11.items())
    for name, tts_config in to_load:
        tts_model = tts_config.get(name)
        if tts_model is None:
            continue
        if args.models and not any_is_prefix(name, args.models):
            logger.debug(f"Skipping model {name}")
            continue

        tts_models[name] = tts_model

    all_pairs = []
    pairs_per_dataset_per_method = defaultdict(lambda: defaultdict(list))
    pairs_per_method = defaultdict(list)

    for dataset in config.dataset.datasets:
        if args.datasets and not any_is_prefix(dataset, args.datasets):
            logger.debug(f"Skipping dataset {dataset}")
            continue
        samples = config.dataset.get(dataset)
        samples.sort(key=lambda x: sum(len(turn.split(" ")) for turn in x.turns))
        logger.info(f"Found {len(samples)} samples for dataset {dataset}")
        for tts_name, tts_model in tts_models.items():
            this_output_folder = output_folder / dataset / tts_name
            this_output_folder.mkdir(exist_ok=True, parents=True)
            for sample in samples:
                if sample.language not in tts_model.supported_languages:
                    continue
                missing_tag = False
                for needed_tag in tts_model.need_tags + args.tags:
                    if needed_tag not in sample.tags:
                        missing_tag = True
                        break
                if missing_tag:
                    continue
                output_file = this_output_folder / (sample.id + f".{config.main.output_format}")
                all_pairs.append((sample, output_file))
                pairs_per_dataset_per_method[dataset][tts_name].append((sample, output_file))
                pairs_per_method[tts_name].append((sample, output_file))

    with ZMQueue(pull_address=config.main.queue_addr) as zmqueue:
        if "gen" in args.stages:
            gpu_taskers = []
            cpu_taskers = []
            for name, tts_model in tts_models.items():
                this_pairs = pairs_per_method[name]
                kept_pairs = []
                for sample, file in this_pairs:
                    if file.with_suffix(".done").exists():
                        continue
                    kept_pairs.append((sample, file))
                if not kept_pairs:
                    continue
                logger.info(f"Will generate {len(kept_pairs)} audio files for {name}.")
                queue = zmqueue.new_queue(name)
                with queue.pusher() as pusher:
                    for pair in kept_pairs:
                        pusher.push(pair)
                tasker = Tasker(tts_model.max_batch_size, TTSTask(debug=config.main.debug), tts_model, queue)
                if tts_model.is_api:
                    cpu_taskers.append(tasker)
                else:
                    gpu_taskers.append(tasker)
            if gpu_taskers or cpu_taskers:
                runner = config.runner.get(config.main.output_folder / "submitit", config.main.debug)
                runner.run(MultiTasker(gpu_taskers), MultiTasker(cpu_taskers, should_init_logging=False))
        if "asr" in args.stages:
            all_pairs_for_asr = [pair for pair in all_pairs if not pair[1].with_suffix(".asr.json").exists()]
            if all_pairs_for_asr:
                logger.info(f"All samples are generated, will now transcribe {len(all_pairs_for_asr)} files.")
                asr_model = config.asr.get()
                queue = zmqueue.new_queue("__asr")
                tasker = Tasker(1, ASRTask(debug=config.main.debug), asr_model, queue)
                with queue.pusher() as pusher:
                    for pair in all_pairs_for_asr:
                        if pair[1].exists():
                            pusher.push(pair)
                runner = config.runner.get(config.main.output_folder / "submitit", config.main.debug)
                runner.run(MultiTasker([tasker]), MultiTasker([]))
                logger.info("All transcript are in.")
        if "spk" in args.stages:
            all_pairs_for_spk = [pair for pair in all_pairs if not pair[1].with_suffix(".speaker.json").exists()]
            if all_pairs_for_spk:
                logger.info(f"Will compute speaker similarity over {len(all_pairs_for_spk)} files.")
                queue = zmqueue.new_queue("__spk")
                tasker = Tasker(1, SpeakerSimilarityTask(debug=config.main.debug), config.speakersim, queue)
                with queue.pusher() as pusher:
                    for pair in all_pairs_for_spk:
                        if pair[1].exists():
                            pusher.push(pair)
                runner = config.runner.get(config.main.output_folder / "submitit", config.main.debug)
                runner.run(MultiTasker([tasker]), MultiTasker([]))
                logger.info("All speaker sims are in.")

    if "met" in args.stages:
        with config.metrics.get_pool() as pool:
            all_metrics = {}
            for dataset, pairs_per_method in pairs_per_dataset_per_method.items():
                print("=" * 64)
                print(f"Results for dataset: {dataset}")
                lines = print_results_for_dataset(pool, config.metrics, pairs_per_method)
                all_metrics[dataset] = lines
                print("\n" + "-" * 64 + "\n")
            if args.save_metrics is not None:
                args.save_metrics.write_text(json.dumps(all_metrics))


if __name__ == "__main__":
    main()
