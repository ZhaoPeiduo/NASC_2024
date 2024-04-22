## LLM Sensei: Leveraging Large Language Model(LLM) to explain multiple-choice questions for JLPT preparation

By Team LLMers: Cao Han, Zhao Peiduo

![Demo](./static/demo.jpg)

This repository contains the model and user interface for LLM Sensei as the entry for National AI Student Challenge 2024, which has been selected as one of the finalist projects.

### Objective

LLM Sensei aims to generate answers and explanations for the input Japanese grammar questions, by using pretrained Japanese LLM and prompt engineering to obtain the desired output from the model.

### Requirement and Quick Start

This solution was written in Python 3.8.10, tested on a laptop with RTX4080 GPU, and will work optimally for other devices with compatible GPU resources.

To install the dependencies, you may want to create a virtual environement (using [virtualenvwrapper](https://virtualenvwrapper.readthedocs.io/en/latest/) for instance) and simply run: 

```
pip install -r requirements.txt
```

To run the application, use the following command:
```
uvicorn backend:app --reload
```

and open the corresponding localhost as indicated by the logging INFO (the default should be http://127.0.0.1:8000).

### Model Specification

After comparing three candidate models, japanese-stablelm-instruct-gamma-7b by stabilityai with half precision is chosen as the LLM backend for this project.

The half-precision model is able to fit within a laptop RTX4080 GPU, occupying approximately 11GB of GPU memory. 

Dataset used for evaluation and other candidate models can be found under the Evaluation directory. 

### Prompt Engineering Methodlogy

The strategy is to use multi-step prompting 
1. Appplying context specification by stating that the LLM should act as a Japanese teacher. 

2. Provide clear instruction for the LLM to indicate the correct answer clearly.

3. Applying the chain-of-thought:
- Firstly, ask the LLM to make a choice among available option;
- Secondly, based on the previous answer, ask the LLM to explain the choice.

### Poster
![Poster](./static/poster.jpg)
