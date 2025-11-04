"""
API Documentation using Flask-RESTX (Swagger/OpenAPI)
Provides interactive API documentation
"""

from flask_restx import Api, Resource, fields, Namespace
from flask import Blueprint

# Create API blueprint
api_blueprint = Blueprint('api', __name__, url_prefix='/api/v1')

# Initialize API with Swagger documentation
api = Api(
    api_blueprint,
    version='1.0',
    title='Fire & Smoke Detection API',
    description='AI-powered fire and smoke detection using Roboflow models',
    doc='/docs',
    contact='rahoodsfamily-lang',
    contact_email='support@firedetection.api'
)

# Create namespaces
detection_ns = Namespace('detection', description='Fire and smoke detection operations')
batch_ns = Namespace('batch', description='Batch processing operations')
video_ns = Namespace('video', description='Video detection operations')
history_ns = Namespace('history', description='Detection history operations')
webhook_ns = Namespace('webhooks', description='Webhook management')
auth_ns = Namespace('auth', description='Authentication operations')

api.add_namespace(detection_ns, path='/detect')
api.add_namespace(batch_ns, path='/batch')
api.add_namespace(video_ns, path='/video')
api.add_namespace(history_ns, path='/history')
api.add_namespace(webhook_ns, path='/webhooks')
api.add_namespace(auth_ns, path='/auth')

# Define models for documentation

# Detection request model
detection_request = api.model('DetectionRequest', {
    'image': fields.String(required=True, description='Base64 encoded image or image file'),
    'model_id': fields.String(default='fire-and-smoke-0izsi/2', description='Roboflow model ID'),
    'confidence': fields.Integer(default=40, description='Confidence threshold (0-100)'),
    'use_context': fields.Boolean(default=True, description='Use weather context for better accuracy'),
    'location': fields.String(default='Unknown', description='Detection location'),
    'city': fields.String(default='Bongao', description='City for weather context')
})

# Prediction model
prediction_model = api.model('Prediction', {
    'class': fields.String(description='Detected class (fire or smoke)'),
    'confidence': fields.Float(description='Confidence score (0-100)'),
    'x': fields.Float(description='Bounding box center X'),
    'y': fields.Float(description='Bounding box center Y'),
    'width': fields.Float(description='Bounding box width'),
    'height': fields.Float(description='Bounding box height')
})

# Detection response model
detection_response = api.model('DetectionResponse', {
    'provider': fields.String(description='AI provider (roboflow)'),
    'model': fields.String(description='Model ID used'),
    'predictions': fields.List(fields.Nested(prediction_model)),
    'count': fields.Integer(description='Number of detections'),
    'has_fire': fields.Boolean(description='Fire detected'),
    'has_smoke': fields.Boolean(description='Smoke detected'),
    'max_confidence': fields.Float(description='Highest confidence score'),
    'weather_context': fields.Raw(description='Weather context information'),
    'processing_time_ms': fields.Float(description='Processing time in milliseconds')
})

# Batch request model
batch_request = api.model('BatchRequest', {
    'images': fields.List(fields.String, required=True, description='List of base64 encoded images'),
    'model_id': fields.String(default='fire-and-smoke-0izsi/2', description='Roboflow model ID'),
    'confidence': fields.Integer(default=40, description='Confidence threshold (0-100)'),
    'max_workers': fields.Integer(default=5, description='Number of parallel workers')
})

# Batch response model
batch_response = api.model('BatchResponse', {
    'total_images': fields.Integer(description='Total images processed'),
    'successful': fields.Integer(description='Successfully processed'),
    'failed': fields.Integer(description='Failed to process'),
    'fire_detected_count': fields.Integer(description='Images with fire detected'),
    'smoke_detected_count': fields.Integer(description='Images with smoke detected'),
    'results': fields.List(fields.Raw, description='Individual detection results')
})

# Video request model
video_request = api.model('VideoRequest', {
    'video': fields.String(required=True, description='Base64 encoded video or video file'),
    'model_id': fields.String(default='fire-and-smoke-0izsi/2', description='Roboflow model ID'),
    'confidence': fields.Integer(default=40, description='Confidence threshold (0-100)'),
    'fps': fields.Integer(default=1, description='Frames per second to analyze'),
    'max_frames': fields.Integer(default=30, description='Maximum frames to process')
})

# Video response model
video_response = api.model('VideoResponse', {
    'success': fields.Boolean(description='Processing success'),
    'frames_analyzed': fields.Integer(description='Number of frames analyzed'),
    'summary': fields.Raw(description='Detection summary'),
    'timeline': fields.List(fields.Raw, description='Frame-by-frame timeline')
})

# Webhook model
webhook_model = api.model('Webhook', {
    'id': fields.Integer(description='Webhook ID'),
    'url': fields.String(required=True, description='Webhook URL'),
    'event_type': fields.String(default='all', description='Event type: all, fire_detected, smoke_detected'),
    'is_active': fields.Boolean(default=True, description='Webhook active status'),
    'created_at': fields.DateTime(description='Creation timestamp')
})

# User model
user_model = api.model('User', {
    'username': fields.String(required=True, description='Username'),
    'email': fields.String(required=True, description='Email address'),
    'password': fields.String(required=True, description='Password')
})

# Login model
login_model = api.model('Login', {
    'username': fields.String(required=True, description='Username'),
    'password': fields.String(required=True, description='Password')
})

# API Key model
api_key_model = api.model('APIKey', {
    'id': fields.Integer(description='API Key ID'),
    'key_prefix': fields.String(description='Key prefix (first 12 chars)'),
    'name': fields.String(description='Key name/description'),
    'created_at': fields.DateTime(description='Creation timestamp'),
    'last_used': fields.DateTime(description='Last used timestamp'),
    'is_active': fields.Boolean(description='Key active status')
})

# History model
history_model = api.model('DetectionHistory', {
    'id': fields.Integer(description='Detection ID'),
    'timestamp': fields.DateTime(description='Detection timestamp'),
    'model_id': fields.String(description='Model used'),
    'detection_count': fields.Integer(description='Number of detections'),
    'has_fire': fields.Boolean(description='Fire detected'),
    'has_smoke': fields.Boolean(description='Smoke detected'),
    'max_confidence': fields.Float(description='Maximum confidence'),
    'location': fields.String(description='Detection location')
})

# Statistics model
stats_model = api.model('Statistics', {
    'total_detections': fields.Integer(description='Total detections'),
    'fire_detections': fields.Integer(description='Fire detections'),
    'smoke_detections': fields.Integer(description='Smoke detections'),
    'avg_confidence': fields.Float(description='Average confidence'),
    'avg_processing_time': fields.Float(description='Average processing time (ms)')
})

# Error model
error_model = api.model('Error', {
    'error': fields.String(description='Error message'),
    'details': fields.String(description='Error details')
})

# Example responses for documentation
detection_example = {
    "provider": "roboflow",
    "model": "fire-and-smoke-0izsi/2",
    "predictions": [
        {
            "class": "fire",
            "confidence": 87.4,
            "x": 320.5,
            "y": 240.3,
            "width": 150.2,
            "height": 180.7
        }
    ],
    "count": 1,
    "has_fire": True,
    "has_smoke": False,
    "max_confidence": 87.4,
    "processing_time_ms": 1250.5
}

# API Documentation strings
DETECTION_DOC = """
Detect fire and smoke in a single image.

**Features:**
- Real-time detection using Roboflow AI
- Weather context integration for better accuracy
- Configurable confidence threshold
- Support for multiple image formats

**Example Request:**
```json
{
    "image": "base64_encoded_image_data",
    "model_id": "fire-and-smoke-0izsi/2",
    "confidence": 40,
    "use_context": true,
    "location": "Building A",
    "city": "Bongao"
}
```

**Example Response:**
```json
{
    "provider": "roboflow",
    "model": "fire-and-smoke-0izsi/2",
    "predictions": [
        {
            "class": "fire",
            "confidence": 87.4,
            "x": 320.5,
            "y": 240.3,
            "width": 150.2,
            "height": 180.7
        }
    ],
    "count": 1,
    "has_fire": true,
    "has_smoke": false,
    "max_confidence": 87.4
}
```
"""

BATCH_DOC = """
Process multiple images in parallel for faster batch detection.

**Features:**
- Parallel processing with configurable workers
- Batch summary statistics
- Individual results for each image

**Use Cases:**
- Processing surveillance camera archives
- Analyzing multiple locations simultaneously
- Bulk image analysis
"""

VIDEO_DOC = """
Analyze video files for fire and smoke detection.

**Features:**
- Automatic frame extraction
- Configurable FPS for analysis
- Frame-by-frame timeline
- Detection summary

**Use Cases:**
- Video surveillance analysis
- Historical footage review
- Real-time video monitoring
"""

WEBHOOK_DOC = """
Configure webhooks to receive real-time alerts when fire or smoke is detected.

**Event Types:**
- `all`: Trigger on any detection
- `fire_detected`: Trigger only on fire detection
- `smoke_detected`: Trigger only on smoke detection

**Webhook Payload:**
```json
{
    "timestamp": "2025-11-04T00:18:37",
    "location": "Building A",
    "has_fire": true,
    "has_smoke": false,
    "max_confidence": 87.4,
    "predictions": [...]
}
```
"""

# Example endpoint resources for documentation
# These show up in Swagger UI as examples

@detection_ns.route('/single')
class SingleDetection(Resource):
    @detection_ns.doc('detect_single_image', description=DETECTION_DOC)
    @detection_ns.expect(detection_request)
    @detection_ns.response(200, 'Success', detection_response)
    @detection_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Detect fire and smoke in a single image"""
        return detection_example

@batch_ns.route('/process')
class BatchDetection(Resource):
    @batch_ns.doc('batch_processing', description=BATCH_DOC)
    @batch_ns.expect(batch_request)
    @batch_ns.response(200, 'Success', batch_response)
    @batch_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Process multiple images in parallel"""
        return {
            "success": True,
            "summary": {
                "total_images": 10,
                "successful": 10,
                "fire_detected_count": 3,
                "smoke_detected_count": 2
            }
        }

@video_ns.route('/analyze')
class VideoDetection(Resource):
    @video_ns.doc('video_analysis', description=VIDEO_DOC)
    @video_ns.expect(video_request)
    @video_ns.response(200, 'Success', video_response)
    @video_ns.response(400, 'Bad Request', error_model)
    def post(self):
        """Analyze video for fire and smoke"""
        return {
            "success": True,
            "frames_analyzed": 30,
            "summary": {"fire_detected_count": 5}
        }

@history_ns.route('/list')
class DetectionHistory(Resource):
    @history_ns.doc('get_history', description='Get detection history')
    @history_ns.response(200, 'Success')
    @history_ns.response(401, 'Unauthorized', error_model)
    def get(self):
        """Get detection history (requires API key)"""
        return {"success": True, "history": []}

@webhook_ns.route('/manage')
class WebhookManagement(Resource):
    @webhook_ns.doc('list_webhooks', description=WEBHOOK_DOC)
    @webhook_ns.response(200, 'Success')
    def get(self):
        """List all webhooks"""
        return {"success": True, "webhooks": []}
    
    @webhook_ns.expect(webhook_model)
    @webhook_ns.response(201, 'Created')
    def post(self):
        """Create a new webhook"""
        return {"success": True, "webhook_id": 1}

# NOTE: These mock endpoints are DISABLED to allow real auth endpoints in app_phase5.py to work
# @auth_ns.route('/register')
# class UserRegistration(Resource):
#     @auth_ns.doc('register_user', description='Register a new user account')
#     @auth_ns.expect(user_model)
#     @auth_ns.response(200, 'Success')
#     @auth_ns.response(400, 'Bad Request', error_model)
#     def post(self):
#         """Register a new user and get API key"""
#         return {
#             "success": True,
#             "user_id": 1,
#             "api_key": "fsd_xxxxxxxxxxxxx"
#         }

# @auth_ns.route('/login')
# class UserLogin(Resource):
#     @auth_ns.doc('login_user', description='Login with username and password')
#     @auth_ns.expect(login_model)
#     @auth_ns.response(200, 'Success')
#     @auth_ns.response(401, 'Unauthorized', error_model)
#     def post(self):
#         """Login and get user information"""
#         return {
#             "success": True,
#             "user": {"id": 1, "username": "testuser"}
#         }

if __name__ == '__main__':
    print("API Documentation Module")
    print("Access documentation at: http://localhost:5000/api/v1/docs")
