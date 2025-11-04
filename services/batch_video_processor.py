"""
Batch image processing and video frame detection module
Handles multiple images and video files for fire/smoke detection
"""

import cv2
import numpy as np
from PIL import Image
import io
import base64
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

def extract_video_frames(video_path, fps=1, max_frames=30):
    """
    Extract frames from video at specified FPS
    
    Args:
        video_path: Path to video file or video bytes
        fps: Frames per second to extract (default: 1 frame per second)
        max_frames: Maximum number of frames to extract
    
    Returns:
        List of PIL Images
    """
    frames = []
    
    try:
        # Open video
        if isinstance(video_path, bytes):
            # Save bytes to temporary file
            temp_path = 'temp_video.mp4'
            with open(temp_path, 'wb') as f:
                f.write(video_path)
            video_path = temp_path
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # print(f"📹 Video: {duration:.1f}s, {video_fps:.1f} FPS → Extracting {min(max_frames, int(duration * fps))} frames...")
        
        # Calculate frame interval
        frame_interval = int(video_fps / fps) if fps > 0 else 1
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                pil_image = Image.fromarray(frame_rgb)
                frames.append(pil_image)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Clean up temp file
        if isinstance(video_path, str) and video_path == 'temp_video.mp4':
            os.remove(video_path)
        
        # print(f"✅ Extracted {len(frames)} frames")
        return frames
    
    except Exception as e:
        print(f"❌ Error extracting video frames: {e}")
        return []

def process_batch_images(images, model_id, api_key, confidence=40, max_workers=5):
    """
    Process multiple images in parallel
    
    Args:
        images: List of PIL Images or file paths
        model_id: Roboflow model ID
        api_key: Roboflow API key
        confidence: Confidence threshold
        max_workers: Number of parallel workers
    
    Returns:
        List of detection results
    """
    results = []
    
    def process_single_image(image, index):
        """Process a single image"""
        try:
            # Decode base64 to PIL Image for proper optimization
            if isinstance(image, str) and not image.endswith(('.jpg', '.jpeg', '.png', '.gif')):
                # Decode base64 string to PIL Image
                image_data = base64.b64decode(image)
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image, str):
                # File path
                image = Image.open(image)
            
            # OPTIMIZE IMAGE (same as app.py optimize_image function)
            # Handle transparency and convert to RGB
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Create white background for transparent images
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[3])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Resize if too large (max 1920x1080) with high-quality LANCZOS resampling
            max_size = (1920, 1080)
            if image.width > max_size[0] or image.height > max_size[1]:
                image.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            # Convert optimized image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=85, optimize=True)
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Make API request
            url = f"https://detect.roboflow.com/{model_id}?api_key={api_key}&confidence={confidence}"
            
            response = requests.post(
                url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                predictions = result.get('predictions', [])
                
                # Case-insensitive class detection
                has_fire = any(p.get('class', '').lower() == 'fire' for p in predictions)
                has_smoke = any(p.get('class', '').lower() == 'smoke' for p in predictions)
                
                return {
                    "index": index,
                    "success": True,
                    "predictions": predictions,
                    "count": len(predictions),
                    "has_fire": has_fire,
                    "has_smoke": has_smoke,
                    "max_confidence": max([p.get('confidence', 0) for p in predictions]) if predictions else 0
                }
            else:
                return {
                    "index": index,
                    "success": False,
                    "error": f"API error: {response.status_code}"
                }
        
        except Exception as e:
            return {
                "index": index,
                "success": False,
                "error": str(e)
            }
    
    # Process images in parallel
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_image, img, i): i 
            for i, img in enumerate(images)
        }
        
        # Collect results as they complete
        fire_count = 0
        smoke_count = 0
        error_count = 0
        
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            
            if result['success']:
                if result['has_fire']:
                    fire_count += 1
                elif result['has_smoke']:
                    smoke_count += 1
            else:
                error_count += 1
                # Only log errors
                print(f"   ⚠️ Error on image {result['index'] + 1}: {result['error']}")
    
    # Sort results by index
    results.sort(key=lambda x: x['index'])
    
    elapsed_time = time.time() - start_time
    
    # Summary log (commented out for production)
    # total = len(results)
    # detections = fire_count + smoke_count
    # print(f"🔥 Fire: {fire_count}/{total} | 💨 Smoke: {smoke_count}/{total} | ⚠️ Errors: {error_count}")
    # print(f"✅ Complete in {elapsed_time:.2f}s")
    
    return results

def analyze_batch_results(results):
    """
    Analyze batch processing results and generate summary
    
    Args:
        results: List of detection results
    
    Returns:
        Summary statistics
    """
    total = len(results)
    successful = sum(1 for r in results if r['success'])
    failed = total - successful
    
    fire_detected = sum(1 for r in results if r.get('has_fire', False))
    smoke_detected = sum(1 for r in results if r.get('has_smoke', False))
    
    total_detections = sum(r.get('count', 0) for r in results if r['success'])
    
    summary = {
        "total_images": total,
        "successful": successful,
        "failed": failed,
        "fire_detected_count": fire_detected,
        "smoke_detected_count": smoke_detected,
        "total_detections": total_detections,
        "success_rate": (successful / total * 100) if total > 0 else 0,
        "fire_rate": (fire_detected / successful * 100) if successful > 0 else 0,
        "smoke_rate": (smoke_detected / successful * 100) if successful > 0 else 0
    }
    
    return summary

def process_video_detection(video_path, model_id, api_key, fps=1, max_frames=30, confidence=40):
    """
    Process video for fire/smoke detection
    
    Args:
        video_path: Path to video file
        model_id: Roboflow model ID
        api_key: Roboflow API key
        fps: Frames per second to analyze
        max_frames: Maximum frames to process
        confidence: Confidence threshold
    
    Returns:
        Detection results with timeline
    """
    print("🎬 Starting video detection...")
    
    # Extract frames
    frames = extract_video_frames(video_path, fps=fps, max_frames=max_frames)
    
    if not frames:
        return {
            "success": False,
            "error": "Could not extract frames from video"
        }
    
    # Process frames
    results = process_batch_images(frames, model_id, api_key, confidence)
    
    # Analyze results
    summary = analyze_batch_results(results)
    
    # Create timeline with frame images and predictions
    timeline = []
    for i, result in enumerate(results):
        if result['success']:
            frame_data = {
                "frame": i + 1,
                "timestamp": f"{i * (1/fps):.1f}s",
                "has_fire": result.get('has_fire', False),
                "has_smoke": result.get('has_smoke', False),
                "detection_count": result.get('count', 0),
                "predictions": result.get('predictions', [])
            }
            
            # Add frame image as base64 if there are detections
            if result.get('count', 0) > 0 and i < len(frames):
                frame_img = frames[i]
                if isinstance(frame_img, Image.Image):
                    buffered = io.BytesIO()
                    frame_img.save(buffered, format="JPEG", quality=70)
                    frame_data["image_base64"] = base64.b64encode(buffered.getvalue()).decode()
            
            timeline.append(frame_data)
    
    return {
        "success": True,
        "frames_analyzed": len(frames),
        "summary": summary,
        "timeline": timeline,
        "detailed_results": results
    }

# Example usage
if __name__ == '__main__':
    # Test with sample images
    print("Batch Video Processor - Test Mode")
    print("=" * 60)
    
    # This would be used in the API
    # Example: process_video_detection('fire_video.mp4', 'fire-and-smoke-0izsi/2', 'YOUR_API_KEY')
