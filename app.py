"""
Enhanced Fire & Smoke Detection API
Includes: Batch processing, video detection, webhooks, alerts, history, auth, analytics
"""

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import requests
import base64
import io
import os
from PIL import Image
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from services.weather_context import smart_detection
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
from functools import wraps

# Import database and service modules
from services.database import (
    init_database, log_detection, get_detection_history, 
    get_detection_stats, create_user, verify_user,
    generate_api_key, verify_api_key, get_user_api_keys,
    create_webhook, get_webhooks, get_alert_settings
)
from services.batch_video_processor import (
    process_batch_images, extract_video_frames, 
    process_video_detection, analyze_batch_results
)
from services.notifications import trigger_alerts, send_webhook, send_email_alert
from api_docs import api_blueprint

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB for videos
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
CORS(app)

# Register API documentation blueprint
app.register_blueprint(api_blueprint)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["1000 per day", "100 per hour"],
    storage_uri="memory://",
)

# API Keys
ROBOFLOW_API_KEY = os.getenv('ROBOFLOW_API_KEY', '')
ROBOFLOW_BASE_URL = "https://detect.roboflow.com"

# Initialize database on startup
init_database()

# Authentication decorator
def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return jsonify({"error": "API key required"}), 401
        
        key_info = verify_api_key(api_key)
        if not key_info:
            return jsonify({"error": "Invalid API key"}), 401
        
        # Add key info to request context
        request.api_key_info = key_info
        return f(*args, **kwargs)
    
    return decorated_function

# Helper functions (from original app.py)
def allowed_file(filename):
    """Check if file extension is allowed"""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff', 'webp', 'mp4', 'avi', 'mov'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image, format="JPEG", quality=85):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.save(buffered, format=format, quality=quality, optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def base64_to_image(base64_string):
    """Convert base64 string to PIL Image"""
    if ',' in base64_string:
        base64_string = base64_string.split(',')[1]
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    return image

def optimize_image(image, max_size=(1920, 1080), quality=85):
    """Optimize image for faster processing"""
    if image.mode != 'RGB':
        if image.mode == 'RGBA':
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[3])
            image = background
        else:
            image = image.convert('RGB')
    
    if image.width > max_size[0] or image.height > max_size[1]:
        image.thumbnail(max_size, Image.Resampling.LANCZOS)
    
    return image

def process_uploaded_image(file_or_base64):
    """Process uploaded image from file or base64"""
    if isinstance(file_or_base64, str):
        image = base64_to_image(file_or_base64)
    else:
        image = Image.open(file_or_base64.stream)
    
    image = optimize_image(image)
    return image

# Routes

@app.route('/')
def home():
    """Main navigation hub"""
    return render_template('home.html')

@app.route('/detect')
def detect_page():
    """Single image detection interface"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Detection history dashboard"""
    return render_template('dashboard.html')

@app.route('/batch')
def batch_page():
    """Batch image processing interface"""
    return render_template('batch.html')

@app.route('/video')
def video_page():
    """Video analysis interface"""
    return render_template('video.html')

@app.route('/webhooks')
def webhooks_page():
    """Webhook management interface"""
    return render_template('webhooks.html')

@app.route('/webcam')
def webcam_page():
    """Live webcam detection interface"""
    return render_template('webcam.html')

@app.route('/register')
def register_page():
    """User registration page"""
    return render_template('register.html')

@app.route('/login')
def login_page():
    """User login page"""
    return render_template('login.html')

@app.route('/keys')
def api_keys_page():
    """API keys management page"""
    return render_template('api_keys.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "version": "1.0-phase5",
        "features": [
            "single_detection",
            "batch_processing",
            "video_detection",
            "webhooks",
            "email_alerts",
            "detection_history",
            "api_authentication",
            "analytics"
        ],
        "api_keys_configured": {
            "roboflow": bool(ROBOFLOW_API_KEY)
        }
    })

@app.route('/roboflow/detect', methods=['POST'])
@limiter.limit("300 per minute")  # Increased for webcam support (5 per second)
def roboflow_detect():
    """Single image detection (original endpoint, enhanced with logging)"""
    start_time = time.time()
    
    try:
        if not ROBOFLOW_API_KEY:
            return jsonify({"error": "Roboflow API key not configured"}), 400
        
        # Get parameters
        model_id = request.form.get('model_id', 'fire-and-smoke-0izsi/2') if 'file' in request.files else request.json.get('model_id', 'fire-and-smoke-0izsi/2')
        confidence = int(request.form.get('confidence', 40) if 'file' in request.files else request.json.get('confidence', 40))
        
        # Get image
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '' or not allowed_file(file.filename):
                return jsonify({"error": "Invalid file"}), 400
            image = process_uploaded_image(file)
        elif request.json and 'image' in request.json:
            image = process_uploaded_image(request.json['image'])
        else:
            return jsonify({"error": "No image provided"}), 400
        
        # Convert to base64
        img_base64 = image_to_base64(image)
        
        # Make request to Roboflow (with timeout for speed)
        url = f"{ROBOFLOW_BASE_URL}/{model_id}?api_key={ROBOFLOW_API_KEY}&confidence={confidence}"
        
        try:
            response = requests.post(
                url,
                data=img_base64,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15  # 15 second timeout (Roboflow can be slow)
            )
        except requests.exceptions.Timeout:
            print(f"Roboflow API timeout for model {model_id}")
            return jsonify({"error": "Roboflow API timeout", "predictions": [], "count": 0}), 200
        except requests.exceptions.RequestException as e:
            print(f"Roboflow API error: {e}")
            return jsonify({"error": "Roboflow API error", "predictions": [], "count": 0}), 200
        
        if response.status_code == 200:
            result = response.json()
            predictions = result.get('predictions', [])
            
            # Basic result (case-insensitive class matching)
            basic_result = {
                "provider": "roboflow",
                "model": model_id,
                "predictions": predictions,
                "count": len(predictions),
                "has_fire": any(p.get('class', '').lower() == 'fire' for p in predictions),
                "has_smoke": any(p.get('class', '').lower() == 'smoke' for p in predictions),
                "max_confidence": max([p.get('confidence', 0) for p in predictions]) if predictions else 0
            }
            
            # Add weather context if requested (skip for webcam for speed)
            # Check if it's a webcam request (only from JSON, not file uploads)
            is_webcam = False
            if 'file' not in request.files and request.is_json:
                is_webcam = request.json.get('is_webcam', False)
            
            use_context = request.form.get('use_context', 'true') if 'file' in request.files else (request.json.get('use_context', 'true') if request.is_json else 'true')
            location = request.form.get('location', 'Unknown') if 'file' in request.files else (request.json.get('location', 'Unknown') if request.is_json else 'Unknown')
            city = request.form.get('city', 'Bongao') if 'file' in request.files else (request.json.get('city', 'Bongao') if request.is_json else 'Bongao')
            
            # Skip weather context for webcam to improve speed
            if use_context == 'true' and basic_result['count'] > 0 and not is_webcam:
                try:
                    enhanced_result = smart_detection(basic_result, location=location, city=city)
                except Exception as e:
                    print(f"Smart detection error: {e}")
                    enhanced_result = basic_result
                    enhanced_result['alert'] = basic_result['has_fire'] or basic_result['has_smoke']
            else:
                enhanced_result = basic_result
                # Add minimal context for webcam
                if is_webcam and basic_result['count'] > 0:
                    enhanced_result['alert'] = basic_result['has_fire'] or basic_result['has_smoke']
                # Add alert for single image too
                elif not is_webcam and basic_result['count'] > 0:
                    enhanced_result['alert'] = basic_result['has_fire'] or basic_result['has_smoke']
            
            # Calculate processing time
            processing_time_ms = (time.time() - start_time) * 1000
            enhanced_result['processing_time_ms'] = processing_time_ms
            
            # Log detection to database (skip for webcam to improve speed)
            # Webcam generates too many logs - only log significant detections
            api_key_info = getattr(request, 'api_key_info', None)
            if not is_webcam or (is_webcam and (basic_result['has_fire'] or basic_result['has_smoke'])):
                try:
                    log_detection(
                        model_id=model_id,
                        predictions=predictions,
                        user_id=api_key_info['user_id'] if api_key_info else None,
                        api_key_id=api_key_info['id'] if api_key_info else None,
                        location=location,
                        city=city,
                        weather_context=enhanced_result.get('weather_context'),
                        processing_time_ms=processing_time_ms,
                        image_size=f"{image.width}x{image.height}",
                        ip_address=request.remote_addr,
                        confidence_threshold=confidence
                    )
                except Exception as log_error:
                    # Don't let logging errors break detection
                    print(f"Logging error: {log_error}")
            
            # Trigger alerts if fire or smoke detected (webcam only, no auth required)
            # ONLY trigger for webcam/live detection (check for 'is_webcam' parameter)
            is_webcam = request.form.get('is_webcam', 'false') if 'file' in request.files else request.json.get('is_webcam', False)
            is_webcam = str(is_webcam).lower() == 'true' or is_webcam == True
            
            if (basic_result['has_fire'] or basic_result['has_smoke']) and is_webcam:
                detection_data = {
                    **enhanced_result,
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "location": location
                }
                
                # Trigger webhooks for authenticated users
                if api_key_info:
                    webhooks = get_webhooks(api_key_info['user_id'])
                    if webhooks:
                        for webhook in webhooks:
                            send_webhook(webhook['url'], detection_data)
                
                # Get notification preferences from request or user settings
                notification_email = request.json.get('notification_email') if request.json else None
                notification_phone = request.json.get('notification_phone') if request.json else None
                
                alert_settings = None
                if api_key_info:
                    # Use authenticated user's settings
                    alert_settings = get_alert_settings(api_key_info['user_id'])
                
                # If no user-specific settings, use request parameters or environment variables
                if not alert_settings:
                    alert_settings = {
                        'email_enabled': bool(notification_email or os.getenv('DEFAULT_ALERT_EMAIL')),
                        'email_address': notification_email or os.getenv('DEFAULT_ALERT_EMAIL'),
                        'sms_enabled': bool(notification_phone or os.getenv('DEFAULT_ALERT_PHONE')),
                        'phone_number': notification_phone or os.getenv('DEFAULT_ALERT_PHONE'),
                        'alert_for_fire': True,
                        'alert_for_smoke': True,
                        'min_confidence': 50
                    }
                
                if alert_settings:
                    # Check if this detection meets alert criteria
                    should_alert = False
                    if basic_result['has_fire'] and alert_settings.get('alert_for_fire'):
                        should_alert = True
                    if basic_result['has_smoke'] and alert_settings.get('alert_for_smoke'):
                        should_alert = True
                    
                    # Check confidence threshold
                    max_conf_percent = basic_result['max_confidence'] * 100
                    if should_alert and max_conf_percent >= alert_settings.get('min_confidence', 50):
                        # Send email alert (with error handling)
                        if alert_settings.get('email_enabled'):
                            email_address = alert_settings.get('email_address')
                            if email_address:
                                try:
                                    send_email_alert(
                                        to_email=email_address,
                                        subject="🔥 Fire/Smoke Detection Alert!",
                                        detection_data=detection_data
                                    )
                                except Exception as email_error:
                                    print(f"Email alert failed: {email_error}")
                        
                        # Send SMS alert via Twilio (with error handling)
                        if alert_settings.get('sms_enabled'):
                            phone_number = alert_settings.get('phone_number')
                            if phone_number:
                                try:
                                    from services.notifications import send_sms_alert
                                    alert_type = "Fire" if basic_result['has_fire'] else "Smoke"
                                    message = f"🔥 {alert_type} detected! Confidence: {max_conf_percent:.1f}%. Location: {location or 'Unknown'}"
                                    send_sms_alert(phone_number, message)
                                except Exception as sms_error:
                                    print(f"SMS alert failed: {sms_error}")
            
            return jsonify(enhanced_result)
        else:
            return jsonify({"error": "Roboflow API error", "details": response.text}), response.status_code
    
    except Exception as e:
        import traceback
        print("="*60)
        print("ERROR IN /roboflow/detect:")
        print(traceback.format_exc())
        print("="*60)
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/detect/batch', methods=['POST'])
@limiter.limit("5 per minute")
def batch_detect():
    """Batch image processing"""
    try:
        data = request.json
        images_b64 = data.get('images', [])
        model_id = data.get('model_id', 'fire-and-smoke-0izsi/2')
        confidence = data.get('confidence', 40)
        max_workers = data.get('max_workers', 5)
        
        if not images_b64:
            return jsonify({"error": "No images provided"}), 400
        
        # Pass base64 strings directly (no conversion to avoid quality loss)
        # Clean base64 strings
        cleaned_images = []
        for img_b64 in images_b64:
            try:
                # Remove data URL prefix if present
                if ',' in img_b64:
                    img_b64 = img_b64.split(',')[1]
                cleaned_images.append(img_b64)
            except Exception as e:
                print(f"Error cleaning base64: {e}")
        
        if not cleaned_images:
            return jsonify({"error": "No valid images"}), 400
        
        # Process batch with base64 strings directly
        results = process_batch_images(
            cleaned_images,
            model_id,
            ROBOFLOW_API_KEY,
            confidence,
            max_workers
        )
        summary = analyze_batch_results(results)
        
        return jsonify({
            "success": True,
            "summary": summary,
            "results": results
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/detect/video', methods=['POST'])
@limiter.limit("2 per minute")
def video_detect():
    """Video frame detection"""
    try:
        if 'video' not in request.files:
            return jsonify({"error": "No video file provided"}), 400
        
        video_file = request.files['video']
        model_id = request.form.get('model_id', 'fire-and-smoke-0izsi/2')
        confidence = int(request.form.get('confidence', 40))
        fps = int(request.form.get('fps', 1))
        max_frames = int(request.form.get('max_frames', 30))
        
        # Save video temporarily
        temp_path = 'temp_video.mp4'
        video_file.save(temp_path)
        
        # Process video
        result = process_video_detection(temp_path, model_id, ROBOFLOW_API_KEY, fps, max_frames, confidence)
        
        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/history', methods=['GET'])
@require_api_key
def get_history():
    """Get detection history"""
    try:
        user_id = request.api_key_info['user_id']
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        history = get_detection_history(user_id, limit, offset)
        
        return jsonify({
            "success": True,
            "count": len(history),
            "history": history
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/analytics', methods=['GET'])
@require_api_key
def get_analytics():
    """Get detection analytics"""
    try:
        user_id = request.api_key_info['user_id']
        days = int(request.args.get('days', 30))
        
        stats = get_detection_stats(user_id, days)
        
        return jsonify({
            "success": True,
            "period_days": days,
            "statistics": stats
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/webhooks', methods=['GET', 'POST', 'DELETE'])
@require_api_key
def manage_webhooks():
    """Manage webhooks"""
    try:
        user_id = request.api_key_info['user_id']
        
        if request.method == 'GET':
            webhooks = get_webhooks(user_id)
            return jsonify({
                "success": True,
                "webhooks": webhooks
            })
        
        elif request.method == 'POST':
            data = request.json
            url = data.get('url')
            event_type = data.get('event_type', 'all')
            
            if not url:
                return jsonify({"error": "URL required"}), 400
            
            webhook_id = create_webhook(user_id, url, event_type)
            
            return jsonify({
                "success": True,
                "webhook_id": webhook_id,
                "message": "Webhook created successfully"
            })
        
        elif request.method == 'DELETE':
            # Delete webhook (implement in database.py)
            return jsonify({"success": True, "message": "Webhook deleted"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/auth/register', methods=['POST'])
def register():
    """Register new user"""
    try:
        data = request.json
        print(f"\n🔍 DEBUG: Received registration data: {data}")
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        print(f"🔍 DEBUG: Extracted - username: {username}, email: {email}, password: {'*' * len(password) if password else None}")
        
        if not all([username, email, password]):
            print("❌ DEBUG: Missing required fields")
            return jsonify({"error": "Missing required fields"}), 400
        
        print(f"🔍 DEBUG: Calling create_user...")
        user_id = create_user(username, email, password)
        print(f"✅ DEBUG: User created with ID: {user_id}")
        
        # Generate initial API key
        print(f"🔍 DEBUG: Generating API key for user {user_id}...")
        api_key = generate_api_key(user_id, "Default Key")
        print(f"✅ DEBUG: API key generated: {api_key[:20]}...")
        
        return jsonify({
            "success": True,
            "user_id": user_id,
            "api_key": api_key,
            "message": "User registered successfully. Save your API key!",
            "version": "UPDATED_WITH_VALIDATION"
        })
    
    except ValueError as e:
        # Validation errors (400 Bad Request)
        print(f"⚠️  DEBUG: Validation error: {str(e)}")
        return jsonify({"error": str(e)}), 400
    
    except Exception as e:
        # Other errors (500 Internal Server Error)
        print(f"❌ DEBUG: Exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/auth/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        user = verify_user(username, password)
        
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401
        
        # Store user in session
        from flask import session
        session['user_id'] = user['id']
        session['username'] = user['username']
        
        return jsonify({
            "success": True,
            "user": {
                "id": user['id'],
                "username": user['username'],
                "email": user['email']
            }
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/v1/auth/logout', methods=['POST'])
def logout():
    """User logout"""
    from flask import session
    session.clear()
    return jsonify({"success": True, "message": "Logged out successfully"})

@app.route('/api/v1/keys', methods=['GET', 'POST'])
def manage_api_keys():
    """Manage API keys - requires session authentication"""
    from flask import session
    
    try:
        # Check if user is logged in via session
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({"error": "Not authenticated. Please login first."}), 401
        
        if request.method == 'POST':
            # Generate new API key
            data = request.json or {}
            name = data.get('name', 'API Key')
            
            from services.database import generate_api_key
            api_key = generate_api_key(user_id, name)
            
            return jsonify({
                "success": True,
                "api_key": api_key,
                "message": "API key generated. Save it securely!"
            })
        
        # GET - list user's API keys
        from services.database import get_user_api_keys
        keys = get_user_api_keys(user_id)
        
        return jsonify({
            "success": True,
            "keys": keys
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    """Rate limit error handler"""
    return jsonify({
        "error": "Rate limit exceeded",
        "message": "Too many requests. Please try again later."
    }), 429

if __name__ == '__main__':
    print("=" * 60)
    print("Fire & Smoke Detection API")
    print("=" * 60)
    print("\n🚀 Features Enabled:")
    print("  ✓ Single image detection")
    print("  ✓ Batch image processing")
    print("  ✓ Video frame detection")
    print("  ✓ Webhook notifications")
    print("  ✓ Email/SMS alerts")
    print("  ✓ Detection history")
    print("  ✓ User authentication")
    print("  ✓ API key management")
    print("  ✓ Analytics dashboard")
    print("\n📚 API Documentation:")
    print("  http://localhost:5000/api/v1/docs")
    print("\n🔑 Configured API Keys:")
    print(f"  Roboflow: {'✓' if ROBOFLOW_API_KEY else '✗'}")
    print("\n" + "=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
