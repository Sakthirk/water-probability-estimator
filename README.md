# water-probability-estimator

This repository contains a project that uses logistic regression to predict the probability of water presence in a given area. The model is built and trained on relevant datasets to provide accurate estimations. The project includes data preprocessing, model training, evaluation, and deployment steps.

# Deployment environment

1. Default branch -> development.
2. Deployment branches -> development, stage and main.
3. PR changes merged to deployment branches are automatically deployed to respective environments.

```
Branch environment mapping

development -> development environment
stage -> Stage environment
main -> Production environment
```

# Deployment lifecycle

```
development -> Stage -> main
```

1. Deployment feature changes should be first merged to development branch.
2. Once the changes are validated, raise a PR to stage from development branch.
3. For release changes, raise a PR to main from stage branch.

# Local Development

### Prerequisites

- Python 3.10
- Docker

### Steps

```shell
sh serve_local/serve_inference_app.sh
```

# Local Testing

```shell
pip3 install -r requirements.txt
python tests/test_model.py
```

# Local Training

```shell
pip3 install -r requirements.txt
python src/train.py
```
