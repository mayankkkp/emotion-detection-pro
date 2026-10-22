# app.py
import os
import cv2
import gradio as gr
from face_detection import detect_faces, extract_face_roi
from inference import predict_emotion

def process_image(image):
    """Handles a single image (upload or single webcam snapshot)"""
    if image is None:
        return None, "No image provided."
        
    faces = detect_faces(image)
    if len(faces) == 0:
        return image, "No face detected."
        
    results = []
    # Convert to grayscale for ROI extraction
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    for (x, y, w, h) in faces:
        roi = extract_face_roi(gray, x, y, w, h)
        emotion, confidence = predict_emotion(roi)
        results.append(f"{emotion} ({confidence:.1%})")
        
        # Draw on image
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return image, "Detected: " + ", ".join(results)

def process_webcam(frame):
    """Handles live webcam frames (streaming)"""
    # Gradio webcam passes frames in a specific format, process it same as image
    return process_image(frame)

# Professional UI using Gradio Blocks
with gr.Blocks(title="Facial Emotion Recognition System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Live Facial Emotion Recognition")
    gr.Markdown("Detect faces and classify emotions in real-time using a CNN model.")
    
    with gr.Tabs():
        with gr.TabItem("📸 Image Upload"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="Upload Image", type="numpy")
                    image_button = gr.Button("Analyze Image", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Result")
                    image_text = gr.Textbox(label="Prediction")
            image_button.click(process_image, inputs=image_input, outputs=[image_output, image_text])
            
        with gr.TabItem("🎥 Live Webcam"):
            with gr.Row():
                with gr.Column():
                    # streaming=True makes it process frames continuously
                    webcam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam", type="numpy")
                with gr.Column():
                    webcam_text = gr.Textbox(label="Live Prediction")
            
            # When a frame is captured, process it
            webcam_input.stream(process_webcam, inputs=webcam_input, outputs=[webcam_input, webcam_text])

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)