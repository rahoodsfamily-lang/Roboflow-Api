"""
Notification system for webhooks, email, and SMS alerts
Sends alerts when fire or smoke is detected
"""

import requests
import os
from datetime import datetime
import json

# Email configuration (SendGrid)
SENDGRID_API_KEY = os.getenv('SENDGRID_API_KEY', '')
EMAIL_FROM = os.getenv('EMAIL_FROM', 'Fire Detection System <noreply@firedetection.com>')

# SMS configuration (Twilio)
TWILIO_ACCOUNT_SID = os.getenv('TWILIO_ACCOUNT_SID', '')
TWILIO_AUTH_TOKEN = os.getenv('TWILIO_AUTH_TOKEN', '')
TWILIO_PHONE_NUMBER = os.getenv('TWILIO_PHONE_NUMBER', '')

def send_webhook(url, data, timeout=10):
    """
    Send webhook notification to external URL
    
    Args:
        url: Webhook URL
        data: Data to send (will be JSON encoded)
        timeout: Request timeout in seconds
    
    Returns:
        Success status and response
    """
    try:
        response = requests.post(
            url,
            json=data,
            headers={'Content-Type': 'application/json'},
            timeout=timeout
        )
        
        return {
            "success": response.status_code in [200, 201, 202, 204],
            "status_code": response.status_code,
            "response": response.text[:200]  # First 200 chars
        }
    
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": "Webhook request timed out"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def send_email_alert(to_email, subject, detection_data):
    """
    Send email alert for fire/smoke detection using SendGrid API
    
    Args:
        to_email: Recipient email address
        subject: Email subject
        detection_data: Detection information
    
    Returns:
        Success status
    """
    if not SENDGRID_API_KEY:
        return {
            "success": False,
            "error": "SendGrid API key not configured"
        }
    
    try:
        # Create HTML email body
        html_body = f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .header {{ background-color: #ff4444; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .alert-box {{ background-color: #fff3cd; border-left: 4px solid #ff4444; padding: 15px; margin: 20px 0; }}
                .detection {{ background-color: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                .footer {{ background-color: #f8f9fa; padding: 15px; text-align: center; font-size: 12px; color: #666; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🔥 Fire/Smoke Detection Alert</h1>
            </div>
            <div class="content">
                <div class="alert-box">
                    <h2>⚠️ Detection Alert</h2>
                    <p><strong>Time:</strong> {detection_data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}</p>
                    <p><strong>Location:</strong> {detection_data.get('location', 'Unknown')}</p>
                </div>
                
                <h3>Detection Details:</h3>
                <div class="detection">
                    <p><strong>Fire Detected:</strong> {'Yes 🔥' if detection_data.get('has_fire') else 'No'}</p>
                    <p><strong>Smoke Detected:</strong> {'Yes 💨' if detection_data.get('has_smoke') else 'No'}</p>
                    <p><strong>Confidence:</strong> {detection_data.get('max_confidence', 0) * 100:.1f}%</p>
                    <p><strong>Total Detections:</strong> {detection_data.get('detection_count', 0)}</p>
                </div>
                
                <h3>Predictions:</h3>
        """
        
        # Add predictions
        for pred in detection_data.get('predictions', [])[:5]:  # Show first 5
            conf_percent = pred.get('confidence', 0) * 100
            html_body += f"""
                <div class="detection">
                    <p><strong>Class:</strong> {pred.get('class', 'unknown').upper()}</p>
                    <p><strong>Confidence:</strong> {conf_percent:.1f}%</p>
                </div>
            """
        
        html_body += """
                <p style="margin-top: 20px;">
                    <strong>Action Required:</strong> Please verify the detection and take appropriate action if necessary.
                </p>
            </div>
            <div class="footer">
                <p>This is an automated alert from Fire & Smoke Detection API</p>
                <p>Powered by Roboflow AI</p>
            </div>
        </body>
        </html>
        """
        
        # Send via SendGrid API
        sendgrid_data = {
            "personalizations": [{
                "to": [{"email": to_email}],
                "subject": subject
            }],
            "from": {"email": EMAIL_FROM.split('<')[1].strip('>') if '<' in EMAIL_FROM else EMAIL_FROM,
                     "name": EMAIL_FROM.split('<')[0].strip() if '<' in EMAIL_FROM else "Fire Detection System"},
            "content": [{
                "type": "text/html",
                "value": html_body
            }]
        }
        
        response = requests.post(
            'https://api.sendgrid.com/v3/mail/send',
            headers={
                'Authorization': f'Bearer {SENDGRID_API_KEY}',
                'Content-Type': 'application/json'
            },
            json=sendgrid_data,
            timeout=10
        )
        
        if response.status_code not in [200, 201, 202]:
            raise Exception(f"SendGrid API error: {response.status_code} - {response.text}")
        
        return {
            "success": True,
            "message": f"Email sent to {to_email}"
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def send_sms_alert(to_phone, message):
    """
    Send SMS alert via Twilio
    
    Args:
        to_phone: Recipient phone number (E.164 format)
        message: SMS message text
    
    Returns:
        Success status
    """
    if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
        return {
            "success": False,
            "error": "Twilio credentials not configured"
        }
    
    try:
        from twilio.rest import Client
        
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        
        message = client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=to_phone
        )
        
        return {
            "success": True,
            "message_sid": message.sid
        }
    
    except ImportError:
        return {
            "success": False,
            "error": "Twilio library not installed. Run: pip install twilio"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

def trigger_alerts(detection_data, webhooks=None, email_settings=None, sms_settings=None):
    """
    Trigger all configured alerts for a detection
    
    Args:
        detection_data: Detection information
        webhooks: List of webhook URLs
        email_settings: Email configuration dict
        sms_settings: SMS configuration dict
    
    Returns:
        Summary of alert results
    """
    results = {
        "webhooks": [],
        "emails": [],
        "sms": []
    }
    
    has_fire = detection_data.get('has_fire', False)
    has_smoke = detection_data.get('has_smoke', False)
    
    # Send webhooks
    if webhooks:
        for webhook in webhooks:
            url = webhook.get('url')
            event_type = webhook.get('event_type', 'all')
            
            # Check if webhook should be triggered
            should_trigger = (
                event_type == 'all' or
                (event_type == 'fire_detected' and has_fire) or
                (event_type == 'smoke_detected' and has_smoke)
            )
            
            if should_trigger:
                result = send_webhook(url, detection_data)
                results['webhooks'].append({
                    "url": url,
                    "result": result
                })
    
    # Send email alerts
    if email_settings and email_settings.get('enabled'):
        to_email = email_settings.get('email_address')
        min_confidence = email_settings.get('min_confidence', 50)
        alert_for_fire = email_settings.get('alert_for_fire', True)
        alert_for_smoke = email_settings.get('alert_for_smoke', True)
        
        max_confidence = detection_data.get('max_confidence', 0)
        
        # Check if email should be sent
        should_send = (
            max_confidence >= min_confidence and
            ((has_fire and alert_for_fire) or (has_smoke and alert_for_smoke))
        )
        
        if should_send and to_email:
            subject = "🔥 ALERT: Fire/Smoke Detected!"
            result = send_email_alert(to_email, subject, detection_data)
            results['emails'].append({
                "to": to_email,
                "result": result
            })
    
    # Send SMS alerts
    if sms_settings and sms_settings.get('enabled'):
        to_phone = sms_settings.get('phone_number')
        min_confidence = sms_settings.get('min_confidence', 50)
        
        max_confidence = detection_data.get('max_confidence', 0)
        
        if max_confidence >= min_confidence and to_phone:
            message = f"🔥 ALERT: {'Fire' if has_fire else 'Smoke'} detected at {detection_data.get('location', 'Unknown')} with {max_confidence:.0f}% confidence. Time: {datetime.now().strftime('%H:%M:%S')}"
            
            result = send_sms_alert(to_phone, message)
            results['sms'].append({
                "to": to_phone,
                "result": result
            })
    
    return results

# Example usage
if __name__ == '__main__':
    print("Notification System - Test Mode")
    print("=" * 60)
    
    # Test detection data
    test_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "location": "Building A, Floor 3",
        "has_fire": True,
        "has_smoke": False,
        "max_confidence": 87.4,
        "detection_count": 2,
        "predictions": [
            {"class": "fire", "confidence": 87.4},
            {"class": "fire", "confidence": 72.1}
        ]
    }
    
    print("\n📧 Test Email Configuration:")
    print(f"SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
    print(f"Username: {SMTP_USERNAME if SMTP_USERNAME else '❌ Not configured'}")
    print(f"From Email: {FROM_EMAIL if FROM_EMAIL else '❌ Not configured'}")
    
    print("\n📱 Test SMS Configuration:")
    print(f"Twilio SID: {TWILIO_ACCOUNT_SID[:10] + '...' if TWILIO_ACCOUNT_SID else '❌ Not configured'}")
    print(f"Twilio Phone: {TWILIO_PHONE_NUMBER if TWILIO_PHONE_NUMBER else '❌ Not configured'}")
