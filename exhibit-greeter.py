#!/usr/bin/env python
"""
Continuous People Detection, Tracking, Badge OCR, Message Generation, Audio Playback,
and Recording of Video Frames & Audio

Dependencies:
  pip install ultralytics opencv-python playsound openai requests torch sounddevice

  - "ultralytics" provides YOLOv8.
  - The OCR now uses NVIDIA's ocdrnet via NVAI.
  - "sounddevice" is used for audio recording.
  
This script uses Ultralytics YOLOv8 for person detection, a simple centroid tracker
to assign IDs and track individuals across frames, OCR (via NVIDIA's ocdrnet) to extract badge information,
and then generates a welcome/goodbye message. The message is converted to audio using the Elevenlabs API
and played back. Only one message is generated per person.

Additionally, it records the processed video frames to an output file and captures audio 
from the default input device. (To capture system audio on Ubuntu, you might need to configure 
a loopback device or use another method.)
"""

import os
import cv2
import numpy as np
import time
import openai
import requests
import uuid
import zipfile
import io
import json
import tempfile
from playsound import playsound
import math
import sounddevice as sd
import wave

# Disable GPU devices to ensure the code runs only on CPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# From Ultralytics for YOLOv8:
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
# Replace these with your actual credentials:
ELEVENLABS_API_KEY = <add key>
VOICE_ID = "ErXwobaYiN019PkySvjV"
OPENAI_API_KEY = "your_openai_api_key_here"  # Only needed if you set USE_LLM=True

# Use LLM for message generation? (If False, a simple template is used.)
USE_LLM = False

# Set OpenAI API key (if using LLM)
openai.api_key = OPENAI_API_KEY

# Define NVIDIA Green color (BGR format since OpenCV uses BGR)
NVIDIA_GREEN = (0, 185, 118)  # BGR format (76B900 in hex is RGB 118, 185, 0)

# -----------------------------------------------------------------------------
# NVIDIA OCR Configuration and Helper Functions
# -----------------------------------------------------------------------------
nvai_url = "https://ai.api.nvidia.com/v1/cv/nvidia/ocdrnet"

# Get the API key from the environment (or use the default)
api_key = os.environ.get('NGC_PERSONAL_API_KEY', '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')
if not api_key.startswith('Bearer '):
    header_auth = f'Bearer {api_key}'
else:
    header_auth = api_key

def _upload_asset(image_bytes, description):
    """
    Uploads an asset (image) to the NVCF API.
    
    :param image_bytes: Binary image data.
    :param description: Description of the asset.
    :return: The UUID of the uploaded asset.
    """
    assets_url = "https://api.nvcf.nvidia.com/v2/nvcf/assets"
    headers = {
        "Authorization": header_auth,
        "Content-Type": "application/json",
        "accept": "application/json",
    }
    s3_headers = {
        "x-amz-meta-nvcf-asset-description": description,
        "content-type": "image/jpeg",
    }
    payload = {"contentType": "image/jpeg", "description": description}

    response = requests.post(assets_url, headers=headers, json=payload, timeout=30)
    response.raise_for_status()

    asset_url = response.json()["uploadUrl"]
    asset_id = response.json()["assetId"]

    response = requests.put(
        asset_url,
        data=image_bytes,
        headers=s3_headers,
        timeout=300,
    )
    response.raise_for_status()
    return uuid.UUID(asset_id)

def process_image_and_get_text(image_path):
    """
    Processes the image using NVIDIA's OCDRNet model, extracts the OCR results
    from the returned ZIP entirely in memory, and returns a single string containing
    only the detected text (the 'label' fields).
    
    :param image_path: Path to the input image.
    :return: A string containing the detected text.
    """
    with open(image_path, "rb") as image_file:
        asset_id = _upload_asset(image_file.read(), "Input Image")
    
    inputs = {"image": f"{asset_id}", "render_label": False}
    asset_list = f"{asset_id}"
    headers = {
        "Content-Type": "application/json",
        "NVCF-INPUT-ASSET-REFERENCES": asset_list,
        "NVCF-FUNCTION-ASSET-IDS": asset_list,
        "Authorization": header_auth,
    }
    response = requests.post(nvai_url, headers=headers, json=inputs)
    response.raise_for_status()

    zip_bytes = io.BytesIO(response.content)
    detected_text = ""
    with zipfile.ZipFile(zip_bytes, "r") as z:
        for filename in z.namelist():
            try:
                with z.open(filename) as f:
                    file_content = f.read().decode("utf-8", errors="replace")
                    data = json.loads(file_content)
                    if "metadata" in data:
                        labels = [entry.get("label", "") for entry in data.get("metadata", [])]
                        detected_text = " ".join(labels)
                        break
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    return detected_text

# -----------------------------------------------------------------------------
# Centroid Tracker (for tracking individual detections)
# -----------------------------------------------------------------------------
class CentroidTracker:
    def __init__(self, max_disappeared=10, max_distance=50):
        self.nextObjectID = 0
        self.objects = {}
        self.bboxes = {}
        self.disappeared = {}
        self.message_sent = {}
        self.initial_centroid = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def register(self, centroid, bbox):
        self.objects[self.nextObjectID] = centroid
        self.bboxes[self.nextObjectID] = bbox
        self.disappeared[self.nextObjectID] = 0
        self.message_sent[self.nextObjectID] = False
        self.initial_centroid[self.nextObjectID] = centroid
        self.nextObjectID += 1
        
    def deregister(self, objectID):
        del self.objects[objectID]
        del self.bboxes[objectID]
        del self.disappeared[objectID]
        del self.message_sent[objectID]
        del self.initial_centroid[objectID]
        
    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.max_disappeared:
                    self.deregister(objectID)
            return self.bboxes
        
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            input_centroids[i] = (cx, cy)
            
        if len(self.objects) == 0:
            for i in range(0, len(input_centroids)):
                self.register(tuple(input_centroids[i]), rects[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = np.linalg.norm(np.array(objectCentroids)[:, np.newaxis] - input_centroids, axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            usedRows = set()
            usedCols = set()
            
            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = tuple(input_centroids[col])
                self.bboxes[objectID] = rects[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)
            
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)
            
            if D.shape[0] >= D.shape[1]:
                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1
                    if self.disappeared[objectID] > self.max_disappeared:
                        self.deregister(objectID)
            else:
                for col in unusedCols:
                    self.register(tuple(input_centroids[col]), rects[col])
                    
        return self.bboxes

# -----------------------------------------------------------------------------
# Detection Functions using Ultralytics YOLO (YOLOv8)
# -----------------------------------------------------------------------------
def load_detection_model():
    model = YOLO('yolov8s.pt')
    model.to('cpu')
    return model

def detect_people_ultralytics(frame, model):
    results = model(frame)
    boxes_out = []
    if len(results) > 0:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            if cls_id == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                boxes_out.append((x1, y1, w, h))
    return boxes_out

# -----------------------------------------------------------------------------
# Badge Cropping and OCR (Using NVIDIA ocdrnet)
# -----------------------------------------------------------------------------
def crop_badge(frame, bbox):
    x, y, w, h = bbox
    return frame[y: y + h, x: x + w]

def read_badge_info(cropped_image):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp_file:
        temp_filename = tmp_file.name
        cv2.imwrite(temp_filename, cropped_image)
    
    try:
        detected_text = process_image_and_get_text(temp_filename)
    finally:
        os.remove(temp_filename)
    
    print("OCR Detected Text:", detected_text)
    lines = [line.strip() for line in detected_text.splitlines() if line.strip()]
    name = lines[0] if len(lines) > 0 else "Guest"
    job = lines[1] if len(lines) > 1 else ""
    return {"name": name, "job": job, "raw_text": detected_text}

# -----------------------------------------------------------------------------
# Message Generation (Template or via LLM)
# -----------------------------------------------------------------------------
def generate_message(badge_info, direction, use_llm=USE_LLM):
    name = badge_info.get("name", "Guest")
    job = badge_info.get("job", "")
    if not use_llm:
        if direction == "in":
            message = f"Welcome {name}, {job}! We're glad to have you here."
        else:
            message = f"Goodbye {name}, have a great day ahead!"
        return message
    else:
        if direction == "in":
            prompt = f"Generate a warm and professional welcome message for a conference attendee named {name} who works as {job}."
        else:
            prompt = f"Generate a polite and sincere goodbye message for a conference attendee named {name} who works as {job}."
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        message = response.choices[0].message.content.strip()
        return message

# -----------------------------------------------------------------------------
# Text-to-Speech Conversion & Audio Playback using Elevenlabs API
# -----------------------------------------------------------------------------
def text_to_speech(message, voice_id, api_key):
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    headers = {"xi-api-key": api_key, "Content-Type": "application/json"}
    data = {"text": message}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        output_file = "output.mp3"
        with open(output_file, "wb") as f:
            f.write(response.content)
        return output_file
    else:
        raise Exception(f"Text-to-speech API call failed: {response.text}")

def play_audio(file_path):
    playsound(file_path)

# -----------------------------------------------------------------------------
# Audio Recording Setup using sounddevice
# -----------------------------------------------------------------------------
# Global list to hold recorded audio chunks
audio_frames = []
audio_samplerate = 44100
audio_channels = 2

def audio_callback(indata, frames, time_info, status):
    audio_frames.append(indata.copy())

# -----------------------------------------------------------------------------
# Main Continuous Processing Loop (with audio recording context)
# -----------------------------------------------------------------------------
def main():
    print("Loading YOLOv8 detection model...")
    model = load_detection_model()
    tracker = CentroidTracker(max_disappeared=10, max_distance=50)
    
    entry_count = 0
    exit_count = 0
    display_texts = {}  # key: objectID, value: (message, timestamp, bbox)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video device.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Instead of writing frames immediately, store them in a list.
    recorded_frames = []
    
    frame_count = 0
    start_time = time.time()

    # Open the audio input stream for the duration of the loop.
    with sd.InputStream(samplerate=audio_samplerate, channels=audio_channels, callback=audio_callback):
        while True:
            iteration_start = time.time()
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
            
            boxes = detect_people_ultralytics(frame, model)
            objects = tracker.update(boxes)
            
            for objectID, bbox in objects.items():
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), NVIDIA_GREEN, 2)
                cv2.putText(frame, f"ID {objectID}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, NVIDIA_GREEN, 2)
                
                if not tracker.message_sent.get(objectID, False):
                    init_x = tracker.initial_centroid[objectID][0]
                    current_x = tracker.objects[objectID][0]
                    threshold = 30  # Minimum pixel movement to trigger a message
                    direction = None
                    if current_x < init_x - threshold:
                        direction = "in"
                    elif current_x > init_x + threshold:
                        direction = "out"
                    
                    if direction is not None:
                        badge_img = crop_badge(frame, bbox)
                        badge_info = read_badge_info(badge_img)
                        message = generate_message(badge_info, direction, use_llm=USE_LLM)
                        print(f"Generated message for ID {objectID}: {message}")
                        
                        try:
                            audio_file = text_to_speech(message, VOICE_ID, ELEVENLABS_API_KEY)
                            play_audio(audio_file)
                        except Exception as e:
                            print("Error during text-to-speech conversion:", e)
                        
                        if direction == "in":
                            entry_count += 1
                        elif direction == "out":
                            exit_count += 1
                        
                        tracker.message_sent[objectID] = True
                        display_texts[objectID] = (message, time.time(), bbox)
            
            current_time = time.time()
            for obj_id in list(display_texts.keys()):
                msg, timestamp, disp_bbox = display_texts[obj_id]
                if current_time - timestamp < 3:
                    dx, dy, dw, dh = disp_bbox
                    cv2.putText(frame, msg, (dx, dy + dh + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, NVIDIA_GREEN, 2)
                else:
                    del display_texts[obj_id]
            
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NVIDIA_GREEN, 2)
            cv2.putText(frame, f"Entries: {entry_count}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NVIDIA_GREEN, 2)
            cv2.putText(frame, f"Exits: {exit_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, NVIDIA_GREEN, 2)
            
            cv2.imshow("Conference Welcome System", frame)
            
            # Save a copy of the processed frame.
            recorded_frames.append(frame.copy())
            
            # Optionally enforce a fixed loop duration if processing is faster than desired.
            desired_fps = 20.0
            desired_frame_time = 1.0 / desired_fps
            iteration_duration = time.time() - iteration_start
            if iteration_duration < desired_frame_time:
                time.sleep(desired_frame_time - iteration_duration)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    
    # Compute the actual (effective) FPS based on the real-time recording duration.
    total_duration = time.time() - start_time
    effective_fps = len(recorded_frames) / total_duration if total_duration > 0 else 20.0
    print(f"Recording duration: {total_duration:.2f} seconds, Effective FPS: {effective_fps:.2f}")
    
    # Write the recorded frames to a video file using the effective FPS.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("output_video.avi", fourcc, effective_fps, (frame_width, frame_height))
    for frame in recorded_frames:
        video_writer.write(frame)
    video_writer.release()
    print("Video saved as output_video.avi")

# -----------------------------------------------------------------------------
# Entry Point
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    try:
        main()
    finally:
        if audio_frames:
            # Concatenate all captured audio frames.
            audio_data = np.concatenate(audio_frames, axis=0)
            # Convert from float32 to int16.
            audio_data_int16 = np.int16(audio_data * 32767)
            with wave.open("output_audio.wav", "wb") as wav_file:
                wav_file.setnchannels(audio_channels)
                wav_file.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
                wav_file.setframerate(audio_samplerate)
                wav_file.writeframes(audio_data_int16.tobytes())
            print("Audio saved as output_audio.wav")
        else:
            print("No audio data captured.")
