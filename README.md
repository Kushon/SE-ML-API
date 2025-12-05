# Toxicity API

A REST API for evaluating text toxicity. Allows you to send comments and receive an assessment of their toxicity (1 toxic, 0 non-toxic).
## Local launch
### Requirements
- [uv](https://docs.astral.sh/uv/)

### Installation
```
$ git clone https://github.com/Kushon/SE-ML-API.git
```

### Launch
In the project folder:
``
$ uv run uvicorn main:app
``

## Documentation and simple interface
When the project is running, go to `localhost:8000/docs`