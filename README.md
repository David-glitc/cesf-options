# CESF Options Predictor

Paper put-write dashboard: CESF vs a raw Monte Carlo control, with [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c) for C-only GPT training.

**Live:** [https://cesfoptions.kierkegaard.space](https://cesfoptions.kierkegaard.space)

**Code:** [github.com/David-glitc/cesf-options](https://github.com/David-glitc/cesf-options)

## What trains in C

The GPT is [vixhal-baraiya/microgpt-c](https://github.com/vixhal-baraiya/microgpt-c): one C file (`microgpt-c/src/microgpt.c`), libc only, Adam + sampling. This tree compiles it with `-DBLOCK_SIZE=48` and small CLI flags (`--prefix`, `--steps`, `--no-bench`). Python never trains the transformer. Python only builds CESF teacher labels and serves the paper dashboard.

```bash
python3 run_train.py --synthetic --paths 150 --stride 15 --steps 1500 --samples 5
# compiles microgpt.c, then runs ./microgpt-c/microgpt on the corpus
```

Direct C path (after the corpus exists):

```bash
cc -O3 -march=native -ffast-math -Wall -DBLOCK_SIZE=48 -o microgpt-c/microgpt microgpt-c/src/microgpt.c -lm
./microgpt-c/microgpt data/options_corpus.txt --steps=1500 --samples=5 --no-bench --temp=0.2 --greedy
```

Upstream credit: Vixhal Baraiya, [microgpt-c](https://github.com/vixhal-baraiya/microgpt-c).

## Models

| Book | Rule |
|------|------|
| CESF | Compress GBM futures to E_H(Q), score puts on all-path EV, skip short puts if crash-mass ≥ 0.20 |
| RAW | Same paths and grid, no ε-graph, no crash veto |
| microgpt-c | ~4k-parameter character GPT in C, trained on CESF teacher lines |

## Dashboard

```bash
python3 -m pip install -r requirements.txt
python3 live_server.py --ticker SPY --port 8766 --paths 48 --days 520
python3 -m unittest tests.test_options_predictor
```

Paper only. Marks are Black–Scholes on Yahoo last. No broker.

## Query (put-write)

| Field | Value |
|-------|-------|
| Underlying | Close price (GBM calibrated on 120-day lookback) |
| ε | 0.088 |
| H | 42 trading days |
| Grid | ATM / 5% OTM puts, 21–30 DTE |
| Goal | EV 1% of spot, delta 0.45, gamma −0.02 |

## Deploy

Docker Compose on the Coolify network, Traefik host `cesfoptions.kierkegaard.space`:

```bash
docker compose -f deploy/docker-compose.yml up -d --build
```
