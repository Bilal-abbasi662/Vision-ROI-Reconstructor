# Task #01: Vision ROI Reconstructor

This project processes videos to detect characters using YOLO11 and applies creative visual transformations, creating a demonic glow effect on detected regions.

## Features

- **Character Detection**: Uses YOLO11 for robust character detection
- **ROI Transformation Pipeline**:
  1. Darkens characters by 70% to create silhouette effect
  2. Extracts edges using Canny Edge Detection
  3. Dilates edges for emphasis
  4. Applies bright red glow (BGR: 0, 0, 255) to edges
  5. Blends glow onto darkened character
- **Batch Processing**: Automatically processes all videos in `/data` folder
- **Flexible Resolution & Frame Rate**: Handles videos of any resolution and frame rate
- **Modular Design**: Well-organized code with clear separation of concerns

## Project Structure

```
Vision-ROI-Reconstructor/
├── task_01_roi_reconstructor.py    # Main script
├── requirements.txt                 # Dependencies
├── data/                            # Input videos (place your .mp4, .avi, .mov files here)
├── output/                          # Processed videos (auto-generated)
└── README.md                        # This file
```

## Installation

### 1. Clone/Setup the project
```bash
cd Vision-ROI-Reconstructor
```

### 2. Create a Python virtual environment (recommended)
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

**Note**: First run will automatically download YOLO11 model (~50MB)

## Usage

### Basic Usage
```bash
python task_01_roi_reconstructor.py
```

### How it works:

1. **Input**: Place video files in the `/data` folder
   - Supported formats: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`

2. **Processing**: Script automatically:
   - Detects characters in each frame using YOLO11
   - Extracts character regions (ROIs)
   - Applies transformation pipeline (darken → edge detect → dilate → red glow)
   - Re-inserts transformed ROIs into original frame positions
   - Preserves original video properties (resolution, FPS)

3. **Output**: Processed videos saved to `/output` folder
   - Filename format: `{original_name}_processed.mp4`
   - Maintains original resolution and frame rate

## Code Structure

### Classes

#### `ROITransformer`
Handles all ROI transformation operations:
- `darken_roi()`: Reduces brightness by 70%
- `extract_edges()`: Canny edge detection
- `dilate_edges()`: Thickens edges for glow effect
- `apply_red_glow()`: Applies bright red color to edges
- `transform_roi()`: Complete transformation pipeline

#### `CharacterDetector`
Manages YOLO11 detection:
- `detect_characters()`: Returns bounding boxes for detected characters

#### `VideoProcessor`
Handles frame and video manipulation:
- `get_video_properties()`: Extracts FPS, resolution, frame count
- `process_frame()`: Applies transformations to all ROIs in a frame

#### `VideoPipeline`
Main orchestrator:
- `get_input_videos()`: Finds all videos in data folder
- `process_video()`: Processes a single video file
- `run()`: Processes all videos in the data folder

## Key Implementation Details

### Pixel Replacement Logic
The core transformation happens in `apply_red_glow()`:
```python
# Find all pixels where edges exist
edge_mask = edges > 0

# Replace edge pixels with bright red (BGR: 0, 0, 255)
result[edge_mask] = [0, 0, 255]
```

This approach:
- Preserves the darkened background (silhouette)
- Overlays red glow only where edges were detected
- Creates efficient in-place replacement without blending artifacts

### ROI Re-insertion
After transformation, the modified ROI is placed back at original coordinates:
```python
output_frame[y1:y2, x1:x2] = transformed_roi
```

This ensures the glow effect appears in the exact spatial location where characters were detected.

## Configuration

You can customize the behavior by editing the script:

```python
# Adjust darkness factor (0.7 = 70% darker)
darkened = ROITransformer.darken_roi(roi, darkness_factor=0.7)

# Adjust edge dilation kernel size and iterations
dilated_edges = ROITransformer.dilate_edges(edges, kernel_size=3, iterations=2)

# Adjust detection confidence threshold
pipeline.run(conf_threshold=0.5)

# Change YOLO model (speed vs accuracy trade-off)
detector = CharacterDetector("yolo11m.pt")  # m=medium, l=large, x=xlarge
```

## Performance Tips

- Use `yolo11n.pt` (nano) for speed on CPU
- Use `yolo11m.pt` (medium) or `yolo11l.pt` (large) for better accuracy on GPU
- Reduce video resolution before processing for faster execution
- Process shorter clips for testing

## Troubleshooting

### YOLO model not downloading
```bash
# Manually download YOLO11 model
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt')"
```

### Out of memory errors
- Reduce video resolution
- Use smaller YOLO model (nano)
- Process videos in smaller batches

### Video codec issues
- Ensure FFmpeg is installed: `pip install opencv-python-headless[ffmpeg]`
- Or use system FFmpeg: `choco install ffmpeg` (Windows) or `brew install ffmpeg` (macOS)

## Example Output

Input video properties are preserved in output:
- Original: `sample.mp4` (1920x1080, 30 FPS, 300 frames)
- Output: `sample_processed.mp4` (1920x1080, 30 FPS, 300 frames)

Each detected character appears with:
- Darkened background (silhouette effect)
- Bright red edge glow around the character outline
- Perfect spatial alignment with original frame

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.8.0 | Image/video processing |
| ultralytics | ≥8.0.0 | YOLO11 model |
| numpy | ≥1.24.0 | Numerical operations |
| torch | ≥2.0.0 | Deep learning backend |

## License

See LICENSE file for details.

---

**Created**: May 2026  
**Task**: Vision ROI Reconstructor - Task #01
