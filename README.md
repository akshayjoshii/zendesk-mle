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

- `Training`: for training local LLM models.
  
- `Inference`: for running inference on local LLM models & 3rd-party LLM services.
  
- `Evaluation`: for evaluating LLM models' performance on sequence/intent understanding benchmarks.
  
- `App Server`: for hosting the REST apis & serving the LLM services.


### 1.3 Create Environments
I know you might not be a fan of a yet another new package management tool like Poetry, so I have also included a `requirements.txt` file for the entire project in the root directory. You can use it to install all the dependencies using pip if you prefer.

```bash
pip install -r requirements.txt
```

or if you wish to use Poetry, you can create the environments using the following command (make sure you're in the subdirectory of the component you want to create the environment for):

```bash
poetry install
```

This will create a virtual environment for each component and install the dependencies in the respective environments.

**NOTE:** Once the environments are created, you can activate them & either start
model training or run inference using the pretrained model which I have put in the `results` directory.

## 2. Direct Inference [IMPORTANT]

Suppose you don't care about the training part and just want to run model inference using the pretrained model, you can do so by running the following command (**make sure you're in the root directory of the project**):

```bash
./RUN_DOCKER_INFERENCE.sh --port 9000 --container-name intent-classifier-inference --image-name intent-classifier-inference --tag v1.1 --device cpu
```

This script will build a docker image/container & then run the inference server on port 9000 and use the pretrained model to run inference. You can change the port number and the container name as per your requirement. The inference RESP API  follow the same schema provided in [HERE](/coding_task/README.md)


## 3. Research on Models, Task & ATIS Dataset

Before we actually start with the modeling, we need to understand the task & the dataset. Therefore, we perform exploratory data analysis (EDA) on the ATIS dataset & also outline the research & decisions we made while formulating the task, models, and evaluation metrics. You can read the detailed report here: [EDA & Research Decisions](/coding_task/evaluation/README.md).


## 4. Model Training [IMPORTANT]

Before you actually go ahead and start the training process, I would recommend you to read the [Intent Classification PEFT Training](/coding_task/train/README.md) document in the `train` dir to understand the model architecture, training process & the different training parameters used in the pipeline.

To train the model, I have provided 2 scripts that will run the training process. You can start the model training via these 2 methods:

### 4.1 Bash Scripts [PREFERRED]

You can run the training process using the bash scripts provided in the `train` directory. The scripts does not use docker but instead directly start the training process in the local environment.

Before you could actually run the training process, you need to make sure that you have installed all the dependencies in the `requirements.txt` file & be in the **ROOT directory** of the project.

#### 4.1.1 MultiClass Classification

To run the training process for the multiclass classification task, you can use the following command (**You can change training parameters in the script**):

```bash
./coding_task/train/train_multiclass.sh
```

#### 4.1.2 MultiLabel Classification [PREFERRED]

To run the training process for the multilabel classification task, you can use the following command (**You can change training parameters in the script**):

```bash
./coding_task/train/train_multilabel.sh
```

### 4.2 Docker [NOT PREFERRED]

Although this would be the preferred way to laucnh the training process, due to time constraints & my potato laptop, I was unable to test & run the training process in docker completely. However, I have provided a script to build a docker image & run the container that starts the training process in an interactive mode.

To run the training process using docker, you can use the following command (**Make sure you're in the ROOT dir of the project**):

```bash
./RUN_DOCKER_TRAINING.sh multilabel --image-name intent-classifier-training --container-name intent-classifier-training
```

## 5. Training & Evaluation Summary

I have summarized some key insights and performance metrics from the best model that we trained and evaluated on Validation (extracted from Train set) & the provided Test sets.
The detailed report can be read here: [Training & Evaluation Summary](/results/README.md).


## 6. What I'd Do With More Time

1. **Model Training**: I would have trained the model for more epochs to see if the performance improves. I would also have tried different descriminative models such as `t5-base` or `mbert` or some of the newer autoregressive generative architectures like `llama` or `Phi` or `Qwen` to see if the performance on the eval metrics improves or also just test their general multilingual capabilities. I would also have tried different hyperparameters like learning rate, batch size, etc. to see if the performance improves.

2. **Model Evaluation**: I would have evaluated the model on more datasets like `SNIPS` or `MultiATIS` to see if the performance improves on similar Intent Classification datasets. 

3. **Orchestration**: I would've used AirFlow or MLFlow to manage the training process & also use it to track the model performance on different datasets. Instead of Docker, I would have used Kubernetes to manage the training process & also use it to deploy the model in production (with Autoscale). I would have also used a cloud provider like AWS or GCP to run both training/inference processes on a GPU instance to accelerate everything.

4. **Test Coverage**: I have used Pytest to add a few fixtures & test cases for /data/utils module. But due to time constraints, I was unable to add more tests for the other modules. I would have added more tests for the other modules as well (mainly for training & inference pipelines). I would have also used `tox` to run the tests in different environments & also use `black` or `isort` to format the code.

5. **Documentation**: I would have added more documentation for the code & also added a few more comments to explain the code better.

6. **CI/CD**: I would have used GitHub Actions & Azure DevOps to set up a CI/CD pipeline to run the tests & also deploy the model in dev/production. I would have also used Snyk or Dependabot to check for vulnerabilities in the dependencies & also update them automatically.

7. **Model Serving**: Currently I'm using basic constructs from `FastAPI` to serve the model without any security or user management implementations. I would've further extended the code to use use `Redis` or `RabbitMQ` to queue the requests & also use `Celery` to run the inference process in the background. I would have also used `Prometheus` or `Grafana` to monitor the model performance in production & also use `Sentry` to track the errors in production. 

8. **Model Explainability**: I would have used `IntegratedGRADS` or other `POST-HOC` xAI methods to explain the model predictions & also use `Streamlit` or `Gradio` to create a web app to visualize the model predictions & also use it to explain the model predictions.