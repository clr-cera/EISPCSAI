# EISPCSAI

## Entering running environment

To enter the environment, run the following command:

```bash
docker compose run --rm --service-ports interactive fish
```

## Running

The project was built to be run using uv:

```bash
uv run src/main.py
```

It is built as a pipeline with independent steps, so that if the program is interrupted it can be continued from the last checkpoint

It will firstly run with all Proxy Tasks, then backup the features, visualizations and results, and run the same pipeline with dino
