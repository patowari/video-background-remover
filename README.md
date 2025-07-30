# Video Background Removal Application Setup Guide
<img width="1018" height="921" alt="image" src="https://github.com/user-attachments/assets/56bebd7a-8931-4e00-a2e0-06c66aba209d" />


## Project Structure
```
video-bg-removal/
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Frontend template
├── uploads/              # Temporary upload folder (created automatically)
├── processed/            # Processed videos folder (created automatically)
└── README.md            # This file
```

## Features

### AI-Powered Background Removal
- **AI Normal Mode**: High-accuracy deep learning model for best results
- **AI Fast Mode**: Faster processing with good quality
- **Traditional Methods**: MOG2, GrabCut, and Contours for fallback

### Key Capabilities
- Supports multiple video formats (MP4, AVI, MOV, MKV, WEBM)
- Real-time progress tracking
- Drag & drop file upload
- Responsive web interface
- Automatic cleanup of temporary files
- GPU acceleration support (if available)

## Installation

### 1. Clone or Download the Project
Create a new directory and save all the files:
- `app.py` - Main Flask application
- `requirements.txt` - Dependencies
- Create a `templates` folder and save `index.html` inside it

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv video-bg-env
source video-bg-env/bin/activate  # On Windows: video-bg-env\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: The `transparent-background` library will download pre-trained models on first use (~100MB).

### 4. Install Additional Requirements (if needed)
For better performance, you might need:
```bash
# For GPU support (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (if you don't have GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Running the Application

### 1. Start the Flask Server
```bash
python app.py
```

### 2. Access the Application
Open your browser and go to: `http://localhost:5000`

## Usage Guide

### 1. Upload Video
- Click the upload area or drag & drop your video file
- Supported formats: MP4, AVI, MOV, MKV, WEBM
- Maximum file size: 100MB

### 2. Choose Method
- **AI Normal**: Best quality, slower processing (recommended)
- **AI Fast**: Good quality, faster processing
- **MOG2**: Traditional method, good for moving subjects
- **GrabCut**: Good for static backgrounds
- **Contours**: Fastest, basic edge detection

### 3. Process Video
- Click "Process Video" button
- Monitor progress in real-time
- Processing time varies based on video length and method

### 4. Download Result
- Preview the processed video
- Download the result
- Process another video or cleanup

## Technical Details

### AI Models
The application uses the `transparent-background` library which includes:
- InSPyReNet (Inference for Segmentation Pyramid Network)
- U²-Net based architectures
- Pre-trained on large datasets for accurate segmentation

### Processing Methods

1. **AI Normal/Fast**: 
   - Uses deep learning models
   - Processes frame-by-frame
   - Converts to PIL → AI processing → OpenCV

2. **Traditional Methods**:
   - MOG2: Background subtraction using Mixture of Gaussians
   - GrabCut: Interactive foreground extraction
   - Contours: Edge-based object detection

### Performance Considerations
- **GPU**: Significantly faster for AI methods
- **CPU**: Works but slower, especially for AI methods
- **Memory**: ~2-4GB RAM recommended for smooth operation
- **Timeout**: 19-minute processing limit for safety

## Troubleshooting

### Common Issues

1. **Import Error for transparent_background**
   ```
   pip install transparent-background
   ```

2. **CUDA/GPU Issues**
   - Install appropriate PyTorch version for your CUDA version
   - Or use CPU-only version: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

3. **Memory Issues**
   - Use AI Fast mode instead of Normal
   - Process shorter videos
   - Ensure sufficient RAM

4. **Slow Processing**
   - Use AI Fast mode
   - Try traditional methods (MOG2, Contours)
   - Consider GPU acceleration

### Model Download
On first run, the AI models will be downloaded automatically:
- Models are stored in `~/.cache/transparent-background/`
- Total size: ~100-200MB
- Internet connection required for first run

## API Endpoints

- `GET /` - Main interface
- `POST /upload` - Upload and start processing
- `GET /status/<task_id>` - Check processing status
- `GET /download/<filename>` - Download processed video
- `POST /cleanup/<task_id>` - Cleanup temporary files

## Configuration

### File Size Limits
Modify in `app.py`:
```python
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
```

### Processing Timeout
Modify in `app.py`:
```python
if time.time() - start_time >= 19 * 60:  # 19 minutes
```

### GPU/CPU Selection
The application automatically detects and uses available GPU. To force CPU:
```python
import torch
torch.cuda.is_available = lambda: False
```

## Advanced Usage

### Custom Background
Modify the AI processing method to use different background types:
```python
# In _process_with_ai method
processed_pil = remover.process(pil_img, type='map')  # or 'white', 'blur', etc.
```

### Batch Processing
The current implementation processes one video at a time. For batch processing, you'd need to implement a queue system.

## Security Notes

- The app includes basic file validation
- Temporary files are cleaned up automatically
- Consider adding authentication for production use
- File size limits prevent abuse

## License & Credits

This application integrates several open-source libraries:
- Flask - Web framework
- OpenCV - Computer vision
- transparent-background - AI background removal
- PyTorch - Deep learning framework

## Support

For issues related to:
- **Flask/Web Interface**: Check Flask documentation
- **AI Models**: Check transparent-background GitHub repository
- **OpenCV**: Check OpenCV documentation
- **General Python**: Check respective library documentation
