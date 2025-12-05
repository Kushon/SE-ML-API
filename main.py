from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel

from ml import load_model_and_tokenizer


class Input(BaseModel):
    comments: list[str]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model, app.state.tokenizer, app.state.device = load_model_and_tokenizer()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/process")
async def process_data(input: Input, request: Request):
    tokenizer, model, device = (
        request.app.state.tokenizer,
        request.app.state.model,
        request.app.state.device,
    )
    texts = input.comments
    predictions = predict(tokenizer, model, device, texts)
    return {"predictions": predictions}
