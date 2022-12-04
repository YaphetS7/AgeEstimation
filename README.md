# AgeEstimation
Age estimation project with telegram bot

### Installation
0. clone this repo
1. insert ur path to .pth model in utils.py, insert ur tg-bot token in app.py
2. pip install -r requirements.txt
3. python app.py
4. fun :)

### About work
In this work:
* Experiments were conducted to implement the algorithm proposed in the [paper](https://arxiv.org/pdf/2203.13122.pdf). The experiments were unsuccessful for a number of reasons. The code for the experiments can be seen in [MWR_experements.ipynb](https://github.com/YaphetS7/AgeEstimation/blob/main/MWR_experements.ipynb)
* Inference pipeline for the model proposed in the [article](https://arxiv.org/pdf/2103.02140.pdf) has been implemented. The (partial) implementation and the pre-trained model were taken from the [official github](https://github.com/SanyeungWang/PML) of the authors of the article
* The model is wrapped in a telegram bot  

### Demo
Use colab notebook [AgeEstimation_tg_bot.ipynb](https://github.com/YaphetS7/AgeEstimation/blob/main/AgeEstimation_tg_bot.ipynb) for demo of the project

