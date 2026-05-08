"""
Task: Vision ROI Reconstructor with Temporal Tracking
Processes videos using YOLO11 tracking to maintain consistent object identity
and applies visual transformations to create a demonic glow effect on a single target character.

Workflow:
1. Detection & Tracking: YOLO11.track() assigns persistent track IDs
2. Initialization: In frame 1, identify the largest character and assign it as target
3. Persistence: Apply glow effect only to the target across subsequent frames
4. Reset: If target disappears for >150 frames (5 seconds), find new largest character
5. ROI Extraction: Isolate target's bounding box
6. Transformation: Darken → edge detect → dilate → blend red glow
7. Re-insertion: Place modified ROI back into frame
8. Output: Save processed video to /output folder

This ensures the demonic glow effect stays on the same character throughout the video,
preventing the effect from "jumping" between different characters.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import logging
from typing import Tuple, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ROITransformer:
    """Handles ROI extraction and transformation operations."""
    
    @staticmethod
    def darken_roi(roi: np.ndarray, darkness_factor: float = 0.7) -> np.ndarray:
        """
        Darken the ROI to create a silhouette effect.
        
        Args:
            roi: Input ROI (BGR image)
            darkness_factor: Darkening intensity (0.7 = 70% darker)
            
        Returns:
            Darkened ROI
        """
        darkened = roi.astype(float) * (1 - darkness_factor)
        return darkened.astype(np.uint8)
    
    @staticmethod
    def extract_edges(roi: np.ndarray) -> np.ndarray:
        """
        Apply Canny Edge Detection to extract character outlines.
        
        Args:
            roi: Input ROI
            
        Returns:
            Binary edge map
        """
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return edges
    
    @staticmethod
    def dilate_edges(edges: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
        """
        Dilate edges to make them thicker for the demonic glow effect.
        
        Args:
            edges: Binary edge map
            kernel_size: Size of dilation kernel
            iterations: Number of dilation iterations
            
        Returns:
            Dilated edge map
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        dilated = cv2.dilate(edges, kernel, iterations=iterations)
        return dilated
    
    @staticmethod
    def apply_red_glow(roi: np.ndarray, edges: np.ndarray) -> np.ndarray:
        """
        Apply bright red color to edges and blend with darkened ROI.
        This creates the demonic glow effect.
        
        Args:
            roi: Darkened ROI (background)
            edges: Dilated edge map (binary)
            
        Returns:
            ROI with red glow effect blended
        
        Note: Pixel replacement logic:
        - Where edges exist (value > 0), we set pixel to bright red (0, 0, 255) in BGR
        - This overlays the red edges on top of the darkened character
        - The result creates a neon/demonic glow appearance
        """
        # Create output array as copy of darkened ROI
        result = roi.copy()
        
        # Find all pixels where edges exist (edge value > 0)
        edge_mask = edges > 0
        
        # Replace edge pixels with bright red (BGR: 0, 0, 255)
        # By replacing only where edge_mask is True, we preserve the darkened background
        # and overlay the red glow only on the detected edges
        result[edge_mask] = [0, 0, 255]
        
        return result
    
    @staticmethod
    def transform_roi(roi: np.ndarray) -> np.ndarray:
        """
        Apply complete transformation pipeline to ROI:
        1. Darken to silhouette
        2. Extract edges with Canny
        3. Dilate edges
        4. Blend red glow
        
        Args:
            roi: Original ROI
            
        Returns:
            Transformed ROI with demonic glow effect
        """
        # Step 1: Darken the ROI
        darkened = ROITransformer.darken_roi(roi, darkness_factor=0.7)
        
        # Step 2: Extract edges from original ROI
        edges = ROITransformer.extract_edges(roi)
        
        # Step 3: Dilate edges to make glow effect more pronounced
        dilated_edges = ROITransformer.dilate_edges(edges, kernel_size=3, iterations=2)
        
        # Step 4: Apply red glow effect
        glowing_roi = ROITransformer.apply_red_glow(darkened, dilated_edges)
        
        return glowing_roi


class CharacterDetector:
    """Handles YOLO11-based character detection with temporal tracking."""
    
    def __init__(self, model_name: str = "yolo11n.pt"):
        """
        Initialize YOLO11 model for tracking.
        
        Args:
            model_name: YOLO model to use (default: nano for speed)
        """
        try:
            self.model = YOLO(model_name)
            logger.info(f"YOLO11 model '{model_name}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
    
    def track_characters(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[dict]:
        """
        Track characters in a frame using YOLO11.track() with persistent track IDs.
        
        This uses YOLO's built-in tracking algorithm (ByteTrack) to maintain consistent IDs
        across frames for the same character, with robustness against occlusions and fast motion.
        
        Args:
            frame: Input frame (BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of dicts with keys: 'track_id', 'x1', 'y1', 'x2', 'y2', 'area'
        """
        # Use YOLO's track() with ByteTrack for robust tracking during occlusions and fast motion
        # ByteTrack is more resilient to temporary disappearances (like character jumps)
        results = self.model.track(
            frame, 
            conf=conf_threshold, 
            verbose=False, 
            persist=True,
            tracker='bytetrack.yaml'  # Use ByteTrack for better robustness
        )
        
        tracked_detections = []
        for result in results:
            # Defensive check: ensure boxes and id exist before accessing
            if result.boxes is None or result.boxes.id is None:
                continue
            
            try:
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.cpu().numpy().astype(int)
                
                # Safety check: ensure we have matching boxes and IDs
                if len(boxes) != len(track_ids):
                    logger.warning(f"Mismatch: {len(boxes)} boxes but {len(track_ids)} track IDs")
                    continue
                
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = map(int, box)
                    tracked_detections.append({
                        'track_id': track_id,
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2,
                        'area': (x2 - x1) * (y2 - y1)  # Store area for finding largest bbox
                    })
            except Exception as e:
                logger.warning(f"Error processing tracked detections: {e}")
                continue
        
        return tracked_detections


class VideoProcessor:
    """Handles video processing and frame manipulation."""
    
    @staticmethod
    def get_video_properties(video_path: str) -> dict:
        """
        Extract video properties for accurate playback and writing.
        
        Args:
            video_path: Path to input video
            
        Returns:
            Dictionary with fps, width, height, frame_count
        """
        cap = cv2.VideoCapture(video_path)
        
        properties = {
            'fps': cap.get(cv2.CAP_PROP_FPS),
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        }
        
        cap.release()
        return properties
    
    @staticmethod
    def process_frame(frame: np.ndarray, tracked_detections: List[dict], target_track_id: Optional[int] = None) -> np.ndarray:
        """
        Process a single frame by applying transformations to the target tracked object.
        
        Args:
            frame: Input frame (BGR)
            tracked_detections: List of dicts with track_id and bounding box info
            target_track_id: The track_id to apply effects to. If None, apply to largest object.
            
        Returns:
            Processed frame with transformed ROI for target object re-inserted
        """
        output_frame = frame.copy()
        
        # Filter to only the target object if specified
        detections_to_process = []
        
        if target_track_id is not None:
            # Only process the specific tracked object
            for detection in tracked_detections:
                if detection['track_id'] == target_track_id:
                    detections_to_process.append(detection)
        else:
            # If no target specified, just take all detections
            detections_to_process = tracked_detections
        
        # Apply transformation to target detection(s)
        for detection in detections_to_process:
            x1 = max(0, detection['x1'])
            y1 = max(0, detection['y1'])
            x2 = min(frame.shape[1], detection['x2'])
            y2 = min(frame.shape[0], detection['y2'])
            
            # Skip invalid bounding boxes
            if x1 >= x2 or y1 >= y2:
                continue
            
            # Step 1: Extract ROI from original frame
            roi = frame[y1:y2, x1:x2].copy()
            
            # Step 2: Apply transformation pipeline
            transformed_roi = ROITransformer.transform_roi(roi)
            
            # Step 3: Re-insert transformed ROI back into exact coordinates
            # This is the critical pixel-replacement logic:
            # We directly replace pixels in the output frame at the original bounding box coordinates
            # with the processed pixels from the transformed ROI. This ensures the glow effect
            # appears in the exact location where the character was detected.
            output_frame[y1:y2, x1:x2] = transformed_roi
        
        return output_frame


class VideoPipeline:
    """Main pipeline orchestrating the complete workflow with temporal smoothing."""
    
    def __init__(self, data_folder: str = "./data", output_folder: str = "./output"):
        """
        Initialize the pipeline.
        
        Args:
            data_folder: Path to folder containing input videos
            output_folder: Path to save processed videos
        """
        self.data_folder = Path(data_folder)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        self.detector = CharacterDetector()
        logger.info("VideoPipeline initialized")
    
    @staticmethod
    def smooth_bbox(bbox_history: List[dict], current_bbox: dict) -> dict:
        """
        Apply temporal smoothing to bounding box by averaging over 3 frames.
        This prevents flickering during fast motion or occlusions.
        
        Args:
            bbox_history: List of previous bounding boxes (max 3 frames)
            current_bbox: Current frame's bounding box
            
        Returns:
            Smoothed bounding box
        """
        # Keep history to max 3 frames
        bbox_history.append(current_bbox)
        if len(bbox_history) > 3:
            bbox_history.pop(0)
        
        # Average bounding box coordinates over available history
        avg_x1 = int(np.mean([b['x1'] for b in bbox_history]))
        avg_y1 = int(np.mean([b['y1'] for b in bbox_history]))
        avg_x2 = int(np.mean([b['x2'] for b in bbox_history]))
        avg_y2 = int(np.mean([b['y2'] for b in bbox_history]))
        
        return {
            'x1': avg_x1,
            'y1': avg_y1,
            'x2': avg_x2,
            'y2': avg_y2,
            'area': (avg_x2 - avg_x1) * (avg_y2 - avg_y1)
        }
    
    def get_input_videos(self) -> List[Path]:
        """
        Recursively find all video files in data folder.
        
        Returns:
            List of video file paths
        """
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}
        videos = [
            f for f in self.data_folder.rglob('*')
            if f.is_file() and f.suffix.lower() in video_extensions
        ]
        return sorted(videos)
    
    def process_video(self, video_path: Path, conf_threshold: float = 0.5) -> bool:
        """
        Process a single video file with temporal tracking.
        
        Implements:
        1. Initialization: Find largest bbox in first frame and track that object
        2. Persistence: Only apply glow effect to the tracked object across frames
        3. Reset: If tracked object disappears for >150 frames, find new largest object
        
        Args:
            video_path: Path to input video
            conf_threshold: Confidence threshold for detections
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Processing video: {video_path.name}")
            
            # Get video properties
            properties = VideoProcessor.get_video_properties(str(video_path))
            fps = properties['fps']
            width = properties['width']
            height = properties['height']
            frame_count = properties['frame_count']
            
            logger.info(f"  Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {frame_count}")
            
            # Set up video writer for output
            output_name = video_path.stem + "_processed.mp4"
            output_path = self.output_folder / output_name
            
            # Use H.264 codec for compatibility
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            
            if not writer.isOpened():
                logger.error(f"Failed to initialize video writer for {output_path}")
                return False
            
            # TEMPORAL TRACKING STATE VARIABLES
            # Initialize target: None until we detect the first (largest) object
            target_track_id = None
            # Reset threshold: 150 frames (5 seconds at 30fps)
            frames_since_detection = 0
            reset_threshold = 150
            # Temporal smoothing: Keep history of bounding boxes for the target object
            bbox_history = []  # Stores last 3 bounding boxes for smoothing
            
            # Open input video
            cap = cv2.VideoCapture(str(video_path))
            frame_num = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Step 1: Track characters in frame using YOLO.track() with ByteTrack
                tracked_detections = self.detector.track_characters(frame, conf_threshold)
                
                # Step 2: INITIALIZATION - First time we see an object, track the largest one
                if target_track_id is None and tracked_detections:
                    # Find the largest bounding box (by area)
                    largest = max(tracked_detections, key=lambda d: d['area'])
                    target_track_id = largest['track_id']
                    frames_since_detection = 0
                    bbox_history = []  # Reset smoothing history
                    logger.info(f"  [Frame {frame_num}] Initialized tracking: target_id={target_track_id}, area={largest['area']}")
                
                # Step 3: PERSISTENCE - Check if target object still exists in current frame
                target_found = False
                current_target_bbox = None
                if target_track_id is not None:
                    for detection in tracked_detections:
                        if detection['track_id'] == target_track_id:
                            target_found = True
                            frames_since_detection = 0
                            # Store current bbox for smoothing
                            current_target_bbox = {
                                'x1': detection['x1'],
                                'y1': detection['y1'],
                                'x2': detection['x2'],
                                'y2': detection['y2'],
                                'area': detection['area']
                            }
                            break
                
                # Step 4: TEMPORAL SMOOTHING - Apply 3-frame averaging to prevent flickering
                if current_target_bbox is not None:
                    current_target_bbox = self.smooth_bbox(bbox_history, current_target_bbox)
                
                # Step 5: RESET - If target disappears for too long, find new target
                if target_track_id is not None and not target_found:
                    frames_since_detection += 1
                    if frames_since_detection > reset_threshold:
                        logger.info(f"  [Frame {frame_num}] Target disappeared for {frames_since_detection} frames. Resetting...")
                        target_track_id = None
                        frames_since_detection = 0
                        bbox_history = []  # Reset smoothing history
                        
                        # If new detections exist, reinitialize with largest
                        if tracked_detections:
                            largest = max(tracked_detections, key=lambda d: d['area'])
                            target_track_id = largest['track_id']
                            frames_since_detection = 0
                            bbox_history = []  # Reset smoothing history
                            logger.info(f"  [Frame {frame_num}] New target acquired: target_id={target_track_id}")
                
                # Step 6: Process frame - apply glow effect only to target object (with smoothed bbox)
                if current_target_bbox is not None:
                    # Create a temporary detection list with only the smoothed target
                    smoothed_detection = [{
                        'track_id': target_track_id,
                        **current_target_bbox
                    }]
                    processed_frame = VideoProcessor.process_frame(
                        frame, 
                        smoothed_detection, 
                        target_track_id=target_track_id
                    )
                else:
                    # No target found, return unchanged frame
                    processed_frame = frame.copy()
                
                # Write processed frame to output video
                writer.write(processed_frame)
                
                frame_num += 1
                if frame_num % 30 == 0:
                    status = f"target_id={target_track_id}" if target_track_id is not None else "searching"
                    logger.info(f"  Processed {frame_num}/{frame_count} frames [{status}]")
            
            cap.release()
            writer.release()
            
            logger.info(f"✓ Successfully saved: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing {video_path}: {e}", exc_info=True)
            return False
    
    def run(self, conf_threshold: float = 0.5) -> None:
        """
        Run the complete pipeline on all videos in data folder.
        
        Args:
            conf_threshold: Confidence threshold for detections
        """
        logger.info("Starting Vision ROI Reconstructor pipeline...")
        
        videos = self.get_input_videos()
        
        if not videos:
            logger.warning(f"No video files found in {self.data_folder}")
            return
        
        logger.info(f"Found {len(videos)} video(s) to process")
        
        success_count = 0
        for i, video_path in enumerate(videos, 1):
            logger.info(f"\n[{i}/{len(videos)}] Processing: {video_path.name}")
            if self.process_video(video_path, conf_threshold):
                success_count += 1
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Pipeline complete! Successfully processed {success_count}/{len(videos)} videos")
        logger.info(f"Output folder: {self.output_folder}")


def main():
    """Entry point for the script."""
    try:
        # Initialize and run pipeline
        pipeline = VideoPipeline(data_folder="./data", output_folder="./output")
        pipeline.run(conf_threshold=0.5)
        
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)


if __name__ == "__main__":
    main()
