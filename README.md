# mlops

## Setup

Follow these steps if you want to run our code locally.

1. Clone repository. For example:
	```bash
	git clone https://github.com/gregorylimeurhen/sutd404
	```

### Web Application

Follow these steps if you want to run our web application locally.

1. Optionally, download and extract latest weights from [http://huggingface.co/gregorylimeurhen/sutd404](http://huggingface.co/gregorylimeurhen/sutd404) into `./experiments/runs/`.
2. Optionally, run `./app/build.py`.
3. Open `./app/index.html`. For example:
	```bash
	open ./app/index.html # MacOS
	xdg-open ./app/index.html # Linux
	```

### Experiments

Follow these steps if you want to run our experiments locally.

1. Rename `./experiments/.env.example` to `./experiments/.env`. For example:
	```bash
	mv ./experiments/.env.example ./experiments/.env
	```
2. Set `WANDB_API_KEY` in `./experiments/.env` to working [Weights & Biases (W&B)](http://wandb.ai) API key.
3. Install packages in `./experiments/requirements.txt`. For example:
	```bash
	python3 -m pip install -r ./experiments/requirements.txt
	```

## Structure

```
.
├── .gitignore
├── app
│   ├── assets.json
│   ├── build.py
│   ├── deploy.py
│   ├── favicon.jpeg
│   ├── index.css
│   ├── index.html
│   ├── index.js
│   ├── search.svg
│   ├── weights.bin
│   └── worker.js
├── experiments
│   ├── .env
│   ├── .env.example
│   ├── config.toml
│   ├── data
│   │   ├── aliases.tsv
│   │   ├── boundaries.txt
│   │   ├── edges.tsv
│   │   ├── layout.txt
│   │   ├── n2a.tsv
│   │   ├── neighbors.json
│   │   ├── test.tsv
│   │   ├── train.tsv
│   │   └── val.tsv
│   ├── preprocess.py
│   ├── requirements.txt
│   ├── runs
│   │   └── <MMSS>
│   │       ├── test
│   │       │   ├── results
│   │       │   └── snapshot.zip
│   │       └── train
│   │           ├── latest.pt
│   │           ├── model.pt
│   │           ├── snapshot.zip
│   │           └── wandb
│   ├── test.py
│   ├── train.py
│   └── utils.py
├── paper
│   ├── bibliography.bib
│   ├── extra.tex
│   ├── main.pdf
│   └── main.tex
└── README.md
```
