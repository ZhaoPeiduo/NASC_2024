## Japanese Sensei: Leveraging Large Language Model(LLM) to explain multiple-choice questions for JLPT preparation

Team LLMers: Cao Han, Zhao Peiduo

This repository contains the model and user interface for Japanese Sensei as the entry for National AI Studen Challenge 2024. 

### Objective

Japanese Sensei aims to generate answers and explanations for the input Japanese grammar questions, by using pretrained Japanese LLM and prompt engineering to obtain the desired output from the model.

### Setup

This solution was tested on a laptop with RTX4080 GPU, and will work optimally for other devices with compatible GPU resources.

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

Todo (stabilityai model currently with half precision)

### Prompt Engineering Methodlogy

Todo
