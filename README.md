# EISPCSAI

## Entering running environment

To enter the environment, run the following command:

```bash
docker compose run --rm --service-ports interactive fish
```

The project was built to be run using uv:

```bash
uv run src/main.py <arguments>
```

Each parameter can be explained running help:

```bash
uv run src/main.py -h
```

To download used datasets and models, run the following commands:

```bash
uv run src/main.py --download_dataset sentiment
uv run src/main.py --download_models
```

To extract features:

```bash
uv run src/main.py --sentiment-dataset-feature
uv run src/main.py --process_pca_sentiment
```

To generate visualizations:

```bash
uv run src/main.py --generate_tsne
uv run src/main.py --generate_tsne_per_feature
uv run src/main.py --generate_umap
```

For ensemble training:

```bash
uv run src/main.py --train_ensemble_sentiment
uv run src/main.py --train_ensemble_sentiment_pca
uv run src/main.py --train_ensemble_sentiment_combinatorics
uv run src/main.py --train_ensemble_sentiment_combinatorics_pca
```
