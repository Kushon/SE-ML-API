from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
import torch
from pydantic import BaseModel
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, PreTrainedModel, PretrainedConfig
import torch.nn as nn

class Input(BaseModel):
    comments: list[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer, app.state.device = load_model_and_tokenizer()
    yield
    


app = FastAPI(lifespan=lifespan)


@app.post("/process")
async def process_data(input: Input, request: Request):
    tokenizer, model, device = request.app.state.tokenizer, request.app.state.model, request.app.state.device
    texts = input.comments
    predictions = predict(tokenizer, model, device, texts)
    return {"predictions": predictions}


