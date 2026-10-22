# face_detection.py
import cv2
import config

# Load the pre-trained Haar Cascade model
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces(image):
    """
    Takes an RGB image (from Gradio), converts it to grayscale, 
    and returns a list of bounding boxes (x, y, w, h).
    """
    if image is None:
        return []
        
    # Convert RGB to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=config.FACE_SCALE_FACTOR,
        minNeighbors=config.FACE_MIN_NEIGHBORS,
        minSize=config.FACE_MIN_SIZE
    )
    return faces

def extract_face_roi(gray_image, x, y, w, h):
    """
    Extracts the face region of interest (ROI) and preprocesses it 
    to match the model's expected input (48x48, normalized).
    """
    roi = gray_image[y:y+h, x:x+w]
    roi = cv2.resize(roi, (48, 48))
    roi = roi.astype("float32") / 255.0
    return roi