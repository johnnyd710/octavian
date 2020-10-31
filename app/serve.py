from datetime import datetime
import logging
import os
import io
import re
import base64
from fastapi import FastAPI
import torch
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms
from PIL import Image

from model import AmdResnet

class RetinalImage(BaseModel):
    b64image: str
    description: Optional[str] = None
    label: Optional[str] = None

logger = logging.getLogger("octavian")

v1_path = "/v1"
model_path = "/octavian"

logger.info("Loading API")
app = FastAPI(
    title="OCTAVIAN - classify retinal images with Age Macular Degeneration (AMD)",
    description="This API will classify an image with AMD or not.",
    version="0.0.1",
    openapi_url=f"{v1_path}{model_path}/openapi.json",
    docs_url=f"{v1_path}{model_path}/docs",
    redoc_url=f"{v1_path}{model_path}/redoc",
)

checkpoint_path = os.getenv("CHECKPOINT_PATH", "checkpoint.pt")
logger.info(f"Loading checkpoint file from {checkpoint_path}")
model = AmdResnet()
model.load_state_dict(torch.load(f"{checkpoint_path}"))
model.eval()

transformer = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

@app.post(f"{v1_path}{model_path}/predict")
def read_root(
    retinal_image: RetinalImage
) -> torch.Tensor:
    image = base64.b64decode(re.sub('^data:image/.+;base64,', '', retinal_image.b64image))
    image = io.BytesIO(image)
    image = Image.open(image)
    processed_image = transformer(image)
    processed_image = processed_image.unsqueeze(1)  # resize
    pred: torch.Tensor = model(processed_image)
    print(pred)
    index = torch.argmax(pred, 1)
    return index.item()

@app.get(f"{v1_path}{model_path}/health")
def redirect():
    return {"detail": "ok"}


if __name__ == "__main__":
    uvicorn.run("serve:app", host="0.0.0.0", port=8080, reload=True)
