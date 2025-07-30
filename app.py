import os
import cv2
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file, url_for
from werkzeug.utils import secure_filename
import tempfile
import threading
import time
from pathlib import Path
from PIL import Image
import torch
import random

# Disable torch jit for compatibility
torch.jit.script = lambda f: f

try:
    from transparent_background import Remover
    TRANSPARENT_BG_AVAILABLE = True
except ImportError:
    print("Warning: transparent_background not installed. Only basic methods will be available.")
    TRANSPARENT_BG_AVAILABLE = False

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024   # 100MB max file size

# Create necessary directories
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Store processing status
processing_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class BackgroundRemover:
    def __init__(self):
        global TRANSPARENT_BG_AVAILABLE  # Add this line
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500
        )
        self.ai_remover_normal = None
        self.ai_remover_fast = None

        if TRANSPARENT_BG_AVAILABLE:
            try:
                print("Initializing AI background removers...")
                self.ai_remover_fast = Remover(mode='fast')
                self.ai_remover_normal = Remover()
                print("AI background removers initialized successfully")
            except Exception as e:
                print(f"Error initializing AI removers: {e}")
                TRANSPARENT_BG_AVAILABLE = False  # Modifying global var

    
    def remove_background(self, input_path, output_path, method='mog2', progress_callback=None):
        """Remove background from video using specified method"""
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        writer = None
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame for faster processing (e.g., half size)
                frame = cv2.resize(frame, (width // 2, height // 2))
                
                # Check for timeout (19 minutes for safety)
                if time.time() - start_time >= 19 * 60:
                    print("Processing timeout reached")
                    break
                
                if method in ['ai_normal', 'ai_fast']:
                    processed_frame = self._process_with_ai(frame, method)
                    
                    # Initialize writer with PIL image dimensions
                    if writer is None:
                        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, pil_img.size)
                else:
                    # Traditional methods
                    if writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                    
                    if method == 'mog2':
                        processed_frame = self._process_with_mog2(frame)
                    elif method == 'grabcut':
                        processed_frame = self._process_with_grabcut(frame)
                    else:
                        processed_frame = self._process_with_contours(frame)
                
                writer.write(processed_frame)
                frame_count += 1
                
                # Update progress
                if progress_callback and total_frames > 0:
                    progress = (frame_count / total_frames) * 100
                    progress_callback(progress)
                    
                print(f"Processing frame {frame_count}/{total_frames}")
        
        finally:
            cap.release()
            if writer:
                writer.release()
    
    def _process_with_ai(self, frame, method):
        """Process frame using AI-based transparent background removal"""
        if not TRANSPARENT_BG_AVAILABLE:
            raise ValueError("Transparent background library not available")
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb).convert('RGB')
        
        # Choose remover based on method
        if method == 'ai_fast' and self.ai_remover_fast:
            remover = self.ai_remover_fast
        elif method == 'ai_normal' and self.ai_remover_normal:
            remover = self.ai_remover_normal
        else:
            raise ValueError(f"AI remover not available for method: {method}")
        
        # Process with transparent background removal
        # Using 'green' type for green screen effect (transparent background)
        processed_pil = remover.process(pil_img, type='green')
        
        # Convert back to BGR for OpenCV
        processed_array = np.array(processed_pil)
        processed_frame = cv2.cvtColor(processed_array, cv2.COLOR_RGB2BGR)
        
        return processed_frame
        """Process frame using MOG2 background subtraction"""
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply Gaussian blur to smooth edges
        fg_mask = cv2.GaussianBlur(fg_mask, (5, 5), 0)
        
        # Create 3-channel mask
        mask_3channel = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Apply mask to original frame
        result = frame * mask_3channel
        
        return result.astype(np.uint8)
    
    def _process_with_grabcut(self, frame):
        """Process frame using GrabCut algorithm"""
        height, width = frame.shape[:2]
        
        # Create initial mask
        mask = np.zeros((height, width), np.uint8)
        
        # Define rectangle for probable foreground (center 80% of image)
        rect = (
            int(width * 0.1), int(height * 0.1),
            int(width * 0.8), int(height * 0.8)
        )
        
        # Initialize for GrabCut
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Create final mask
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Apply mask
        result = frame * mask2[:, :, np.newaxis]
        
        return result
    
    def _process_with_contours(self, frame):
        """Process frame using contour detection"""
        # Convert to grayscale and apply threshold
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create mask from largest contours
        mask = np.zeros(gray.shape, np.uint8)
        
        # Sort contours by area and keep the largest ones
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
        
        # Fill the largest contours
        for contour in contours:
            if cv2.contourArea(contour) > 1000:  # Filter small contours
                cv2.fillPoly(mask, [contour], 255)
        
        # Apply mask
        mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        result = frame * mask_3channel
        
        return result.astype(np.uint8)

bg_remover = BackgroundRemover()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    method = request.form.get('method', 'mog2')
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload a video file.'}), 400
    
    filename = secure_filename(file.filename)
    timestamp = str(int(time.time()))
    filename = f"{timestamp}_{filename}"
    
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)
    
    # Generate output filename
    output_filename = f"processed_{filename}"
    output_path = os.path.join(PROCESSED_FOLDER, output_filename)
    
    # Initialize processing status
    task_id = timestamp
    processing_status[task_id] = {'progress': 0, 'status': 'processing', 'output_file': output_filename}
    
    # Start processing in background
    def process_video():
        try:
            def progress_callback(progress):
                processing_status[task_id]['progress'] = progress
            
            bg_remover.remove_background(input_path, output_path, method, progress_callback)
            processing_status[task_id]['status'] = 'completed'
            processing_status[task_id]['progress'] = 100
            
            # Clean up input file
            os.remove(input_path)
            
        except Exception as e:
            processing_status[task_id]['status'] = 'error'
            processing_status[task_id]['error'] = str(e)
    
    thread = threading.Thread(target=process_video)
    thread.start()
    
    return jsonify({'task_id': task_id})

@app.route('/status/<task_id>')
def get_status(task_id):
    if task_id not in processing_status:
        return jsonify({'error': 'Invalid task ID'}), 404
    
    return jsonify(processing_status[task_id])

@app.route('/download/<filename>')
def download_video(filename):
    try:
        file_path = os.path.join(PROCESSED_FOLDER, filename)
        if os.path.exists(file_path):
            return send_file(file_path, as_attachment=True)
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/cleanup/<task_id>', methods=['POST'])
def cleanup_task(task_id):
    if task_id in processing_status:
        # Remove processed file if it exists
        if 'output_file' in processing_status[task_id]:
            file_path = os.path.join(PROCESSED_FOLDER, processing_status[task_id]['output_file'])
            if os.path.exists(file_path):
                os.remove(file_path)
        
        # Remove from processing status
        del processing_status[task_id]
    
    return jsonify({'message': 'Cleanup completed'})

if __name__ == '__main__':
    app.run(debug=True, threaded=True)