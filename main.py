
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
from mediapipe.tasks.python.vision import GestureRecognizer
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark

# Load labels
with open("label_letters.json", "r") as f:
    label_map = json.load(f)

# Load the MediaPipe model
gesture_recognizer = GestureRecognizer.create_from_model_path("gesture_recognizer.task")

# Initialize FastAPI
app = FastAPI()

# Enable CORS for Flutter frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(request: Request):
    try:
        body = await request.json()
        frames = body.get("landmarks")

        if not frames or len(frames) != 30:
            return JSONResponse(content={"error": "Expected 30 frames of landmarks"}, status_code=400)

        predictions = []

        for frame in frames:
            if len(frame) != 126:
                return JSONResponse(content={"error": "Each frame must have 126 values"}, status_code=400)

            landmarks = [
                NormalizedLandmark(x=frame[i], y=frame[i+1], z=frame[i+2])
                for i in range(0, 126, 3)
            ]

            result = gesture_recognizer.recognize_async([landmarks], timestamp_ms=0)

            if result.gestures:
                predictions.append(result.gestures[0][0].category_name)
            else:
                predictions.append("unknown")

        most_common = max(set(predictions), key=predictions.count)
        return {"prediction": most_common}

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
