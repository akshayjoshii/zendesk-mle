# Zendesk Senior MLE Coding Challenge

Follow the instructions in the [README](README.md) to setup the environments and run the code.

## 1. Setup Environments

Instead of using pip to install the dependencies, we will use [Poetry](https://python-poetry.org/) to manage the dependencies and virtual environments. Poetry is a dependency management tool for Python that allows you to declare, manage, and install dependencies in a consistent and reproducible manner.

### 1.1 Install Poetry

To install Poetry, you can use the following command:

```pipx install poetry```
or follow the instructions in the [Poetry documentation](https://python-poetry.org/docs/#installation).

### 1.2 Environments

We have 4 environments to prevent conflicts between the dependencies of different components. Environments:

- `Training`': for training local LLM models.
  
- `Inference`: for running inference on local LLM models & 3rd-party LLM services.
  
- `Evaluation`: for evaluating LLM models' performance on sequence/intent understanding benchmarks.
  
- `App Server`: for hosting the AI REST apis & serving the LLM models.

### 1.3 Create Environments

