import hashlib
import time
from dotenv import load_dotenv
import torch
import wandb
import yaml

from pathlib import Path
from src import *

DEFAULT_DATASET = "pretrain"
DEFAULT_METRICS = ["NLS", "TTLT"]


def run_id():
	return hashlib.sha1(str(time.time()).encode()).hexdigest()[:6]


def ensure_run_dir(root, identifier):
	run_dir = Path(root) / "runs" / identifier
	(run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
	return run_dir


def device_for():
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
	start = time.time()
	load_dotenv()
	root = Path(__file__).resolve().parent
	config_path = root / "config.yaml"
	data_path = root / "data.md"
	config = yaml.safe_load(config_path.read_text())
	config["dataset"] = DEFAULT_DATASET
	config["metrics"] = DEFAULT_METRICS
	device = device_for()
	set_seed(int(config["seed"]))
	datasets = materialize_data_file(data_path)
	rows = datasets["pretrain"] + datasets["finetune"] + datasets["val"] + datasets["test"]
	tokenizer = build_tokenizer(rows)

	if config["architecture"] != "GPT2":
		raise ValueError(f"Unsupported architecture {config['architecture']}")

	model = build_model(tokenizer, rows).to(device)
	batch_size = infer_batch_size(model, rows, tokenizer, device)
	max_tokens = infer_max_tokens(rows, tokenizer)
	config = config | {"batch_size": batch_size, "dataset": DEFAULT_DATASET, "max_tokens": max_tokens, "metrics": DEFAULT_METRICS}
	identifier = run_id()
	run_dir = ensure_run_dir(root, identifier)

	if DEFAULT_DATASET not in datasets:
		raise ValueError(f"Unknown dataset {DEFAULT_DATASET}")

	with wandb.init(config=config, dir=str(run_dir), name=identifier, project="mlops") as run:
		pretrain_train, pretrain_val = split_rows(datasets[DEFAULT_DATASET], int(config["seed"]))
		epoch = 0
		epoch, _ = train_stage(model, "pretrain", pretrain_train, pretrain_val, tokenizer, device, run_dir, config, batch_size, epoch, run)
		epoch, _ = train_stage(model, "finetune", datasets["finetune"], datasets["val"], tokenizer, device, run_dir, config, batch_size, epoch, run)
		metrics = instantiate_metrics(DEFAULT_METRICS)
		val_scores, val_details = evaluate_rows(model, datasets["val"], tokenizer, metrics, device, max_tokens)
		test_scores, test_details = evaluate_rows(model, datasets["test"], tokenizer, metrics, device, max_tokens)
		scores = {f"val_{key}": value for key, value in val_scores.items()} | {f"test_{key}": value for key, value in test_scores.items()}
		run.log({
			"evaluation": wandb.Table(
				columns=["input", "gold", "output", "ttlt"],
				data=[[row["input"], row["gold"], row["output"], row["ttlt"]] for row in val_details + test_details],
			)
		})
		run.summary["duration"] = time.time() - start
		for key, value in scores.items():
			run.summary[key] = value


if __name__ == "__main__":
	main()
