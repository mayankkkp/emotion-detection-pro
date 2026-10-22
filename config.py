# config.py

# The order MUST match the training output perfectly
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Model paths
MODEL_JSON_PATH = "emotiondetector.json"
MODEL_WEIGHTS_PATH = "emotiondetector.h5"

# Face detection settings
# Higher scaleFactor = faster but less accurate. Lower minNeighbors = more false positives.
FACE_SCALE_FACTOR = 1.2
FACE_MIN_NEIGHBORS = 7
FACE_MIN_SIZE = (80, 80)