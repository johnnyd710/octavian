from datetime import datetime
import logging
import os
import io
import re
import base64
from fastapi import FastAPI, File, UploadFile
import torch
import uvicorn
from pydantic import BaseModel
from typing import List, Optional
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet


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
model = EfficientNet.from_name('efficientnet-b1', num_classes=2)
model.load_state_dict(torch.load(f"{checkpoint_path}"))
model.eval()

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: t[:3, ...]),  # remove alpha channel
    transforms.Normalize((0.5,), (0.5,))
])

@app.post(f"{v1_path}{model_path}/predict")
def read_root(
    file: UploadFile = File(...)
) -> str:
    image = Image.open(file.file)
    processed_image: torch.Tensor = transformer(image)
    processed_image = processed_image.unsqueeze(0)  # resize
    pred: torch.Tensor = model(processed_image)
    index = torch.argmax(pred, 1)
    # pred is a logit function, to convert to probability, use softmax:
    confidence = torch.softmax(pred, dim=1).detach().numpy()[0]
    print(confidence)
    if index.item() == 1:
        return 'no amd with probability {:.2f} %'.format(confidence[1] * 100)
    else:
        return 'no amd with probability {:.2f} %'.format(confidence[0] * 100)

@app.get(f"{v1_path}{model_path}/health")
def redirect():
    return {"detail": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
