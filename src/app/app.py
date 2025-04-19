from fastapi import FastAPI, HTTPException
from src.pred.classify import *
from fastapi.middleware.cors import CORSMiddleware
from src.schemas.input_schema import Input_Schema

app = FastAPI(title="Image Classifier API using MobileNet V2")

origins = [
    "http://localhost:3000",
    "localhost:3000",
    "*",
    "http://127.0.0.1:8089/",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/")
async def main():
    return {"msg": "FastAPI Server is up for Image Classifier API using MobileNet V2."}


@app.post(
    "/predict",
    status_code=200,
    description="Classify an image using MobileNet V2",
    response_description="Returns the predicted class and confidence score",
    responses={
        200: {
            "description": "Successful prediction",
            "content": {
                "application/json": {
                    "example": {
                        "predicted_class": "golden retriever",
                        "probability": 0.95
                    }
                }
            }
        },
        404: {
            "description": "Image URL not accessible",
            "content": {
                "application/json": {
                    "example": {"detail": "Can't fetch the image from the provided URL."}
                }
            }
        }
    }
)
async def predict(request: Input_Schema):
    """
    Classify an image from a given URL using MobileNet V2 model
    
    Parameters:
    - img_url: URL of the image to classify (must be publicly accessible)
    
    Returns:
    - Dictionary containing predicted class and confidence score
    - 404 error if image URL is not accessible
    """
    prediction = run_classifier(request.img_url)
    if not prediction:
        raise HTTPException(
            status_code=404,
            detail="Can't fetch the image from the provided URL."
        )
    return prediction