import csv
import hashlib
import shutil
import time
import torch
import yaml

from pathlib import Path
from src import *


def stringify(value):
	if isinstance(value, float):
		return f"{value:.6f}"
	if isinstance(value, list):
		return "[" + ", ".join(stringify(item) for item in value) + "]"
	return str(value)


def compact_mapping(mapping):
	return "(" + ", ".join(f"{key}={stringify(value)}" for key, value in mapping.items()) + ")"


def run_id():
	return hashlib.sha1(str(time.time()).encode()).hexdigest()[:6]


def ensure_run_dir(root, identifier):
	run_dir = Path(root) / "runs" / identifier
	(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
	return run_dir


def snapshot_sources(run_dir, config_path, data_path):
	shutil.copy2(config_path, run_dir / "config.yaml")
	shutil.copy2(data_path, run_dir / "data.md")
	shutil.copy2(Path(__file__).resolve().parent / "src.py", run_dir / "src.py")


def write_records(path, rows, fieldnames):
	with path.open("w", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def append_scoreboard(path, identifier, config, scores, duration):
	scoreboard = Path(path)
	scoreboard.parent.mkdir(parents=True, exist_ok=True)
	header = ["id", "config", "scores", "duration"]
	write_header = not scoreboard.exists() or not scoreboard.read_text().strip()
	with scoreboard.open("a", newline="") as handle:
		writer = csv.writer(handle)
		if write_header:
			writer.writerow(header)
		writer.writerow([identifier, compact_mapping(config), compact_mapping(scores), int(round(duration))])


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
	start = time.time()
	root = Path(__file__).resolve().parent
	config_path = root / "config.yaml"
	data_path = root / "data.md"
	scoreboard_path = root / "scoreboard.csv"
	config = yaml.safe_load(config_path.read_text())
	device = device_for()
	set_seed(int(config["seed"]))
	datasets = materialize_data_file(data_path)
	rows = datasets["pretrain"] + datasets["finetune"] + datasets["val"] + datasets["test"]
	tokenizer = build_tokenizer(rows)

	if config["architecture"] != "GPT2":
		raise ValueError(f'Unsupported architecture {config["architecture"]}')

	model = build_model(config, tokenizer, rows).to(device)
	identifier = run_id()
	run_dir = ensure_run_dir(root, identifier)
	snapshot_sources(run_dir, config_path, data_path)
	pretrain_name = config["dataset"]

	if pretrain_name not in datasets:
		raise ValueError(f"Unknown dataset {pretrain_name}")

	pretrain_train, pretrain_val = split_rows(datasets[pretrain_name], int(config["seed"]))
	epoch = 0
	epoch, pretrain_history = train_stage(model, "pretrain", pretrain_train, pretrain_val, tokenizer, device, run_dir, config, epoch)
	epoch, finetune_history = train_stage(model, "finetune", datasets["finetune"], datasets["val"], tokenizer, device, run_dir, config, epoch)
	history = pretrain_history + finetune_history
	metrics = instantiate_metrics(config["metrics"])
	max_tokens = int(config["max_tokens"])
	val_scores, val_details = evaluate_rows(model, datasets["val"], tokenizer, metrics, device, max_tokens)
	test_scores, test_details = evaluate_rows(model, datasets["test"], tokenizer, metrics, device, max_tokens)
	scores = {f"val_{key}": value for key, value in val_scores.items()} | {f"test_{key}": value for key, value in test_scores.items()}
	write_records(run_dir / "training.log", history, ["epoch", "stage", "train_loss", "val_loss", "grad_norm"])
	write_records(run_dir / "evaluation.csv", val_details + test_details, ["input", "gold", "output", "ttlt"])
	append_scoreboard(scoreboard_path, identifier, config, scores, time.time() - start)


if __name__ == "__main__":
	main()
