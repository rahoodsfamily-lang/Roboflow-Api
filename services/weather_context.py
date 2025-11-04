#!/usr/bin/env python3
"""
Weather Context Detection
Reduces false positives by checking weather conditions
"""

import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# Get API key from environment (you'll need to add this)
OPENWEATHER_API_KEY = os.getenv('OPENWEATHER_API_KEY', '')

class WeatherContext:
    """Get weather context to improve detection accuracy"""
    
    def __init__(self, api_key=None, city="Bongao"):
        self.api_key = api_key or OPENWEATHER_API_KEY
        self.city = city
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def get_weather(self):
        """
        Get current weather conditions
        Returns dict with weather info or None if API fails
        """
        if not self.api_key:
            print("⚠️  No OpenWeather API key configured")
            return None
        
        try:
            params = {
                'q': self.city,
                'appid': self.api_key,
                'units': 'metric'  # Celsius
            }
            
            response = requests.get(self.base_url, params=params, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    'condition': data['weather'][0]['main'],  # 'Fog', 'Clear', 'Rain', etc.
                    'description': data['weather'][0]['description'],
                    'temperature': data['main']['temp'],
                    'humidity': data['main']['humidity'],
                    'visibility': data.get('visibility', 10000),  # meters
                    'city': data['name']
                }
            else:
                print(f"⚠️  Weather API error: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"⚠️  Weather API failed: {e}")
            return None
    
    def adjust_confidence(self, detected_class, confidence, weather=None):
        """
        Adjust detection confidence based on weather context
        
        Args:
            detected_class: 'fire' or 'smoke'
            confidence: Original confidence (0-1)
            weather: Weather data dict (optional, will fetch if None)
        
        Returns:
            Adjusted confidence (0-1) and reason
        """
        if weather is None:
            weather = self.get_weather()
        
        if weather is None:
            # No weather data, return original confidence
            return confidence, "No weather context available"
        
        original_confidence = confidence
        adjustments = []
        
        # Only adjust smoke detections (fire is less affected by weather)
        if detected_class.lower() == 'smoke':
            
            # Check for fog
            if weather['condition'] in ['Fog', 'Mist', 'Haze']:
                confidence *= 0.5  # Reduce by 50%
                adjustments.append(f"Foggy conditions ({weather['condition']})")
            
            # Check visibility
            if weather['visibility'] < 1000:  # Less than 1km
                confidence *= 0.6
                adjustments.append(f"Low visibility ({weather['visibility']}m)")
            
            # Check humidity (high humidity = more likely fog/mist)
            if weather['humidity'] > 85:
                confidence *= 0.8
                adjustments.append(f"High humidity ({weather['humidity']}%)")
            
            # Check temperature (very cold = fog more likely)
            if weather['temperature'] < 10:
                confidence *= 0.9
                adjustments.append(f"Cold temperature ({weather['temperature']}°C)")
        
        # Build reason string
        if adjustments:
            reason = f"Weather context: {', '.join(adjustments)}"
            change = ((confidence - original_confidence) / original_confidence) * 100
            reason += f" | Confidence adjusted by {change:.0f}%"
        else:
            reason = f"Clear weather ({weather['condition']}), no adjustment needed"
        
        return confidence, reason


def smart_detection(image_result, location="Unknown", city="Bongao"):
    """
    Enhanced detection with weather context
    
    Args:
        image_result: Result from Roboflow API
        location: Location name (optional)
        city: City for weather lookup
    
    Returns:
        Enhanced result with weather context
    """
    weather_ctx = WeatherContext(city=city)
    weather = weather_ctx.get_weather()
    
    # Get current time context
    now = datetime.now()
    hour = now.hour
    
    # If no detection, return as-is
    if image_result.get('count', 0) == 0:
        return {
            **image_result,
            'context': {
                'weather': weather,
                'time': hour,
                'location': location,
                'adjustments': 'No detection'
            }
        }
    
    # Get detection details
    prediction = image_result['predictions'][0]
    detected_class = prediction['class']
    original_confidence = prediction['confidence']
    
    # Adjust confidence based on weather
    adjusted_confidence, weather_reason = weather_ctx.adjust_confidence(
        detected_class, 
        original_confidence,
        weather
    )
    
    # Time-based adjustments (morning fog)
    time_reason = ""
    if detected_class.lower() == 'smoke' and 5 <= hour <= 8:
        adjusted_confidence *= 0.8
        time_reason = "Early morning (fog common)"
    
    # Location-based adjustments
    location_reason = ""
    if detected_class.lower() == 'smoke':
        if location.lower() in ['kitchen', 'bathroom']:
            adjusted_confidence *= 0.6
            location_reason = f"Location: {location} (steam expected)"
    
    # Determine if we should alert
    should_alert = adjusted_confidence > 0.3  # 30% threshold after adjustments
    
    # Build enhanced result
    result = {
        **image_result,
        'predictions': [{
            **prediction,
            'original_confidence': original_confidence,
            'adjusted_confidence': adjusted_confidence,
            'confidence_change': adjusted_confidence - original_confidence
        }],
        'alert': should_alert,
        'context': {
            'weather': weather,
            'time': hour,
            'location': location,
            'adjustments': {
                'weather': weather_reason,
                'time': time_reason,
                'location': location_reason
            }
        },
        'recommendation': get_recommendation(
            detected_class, 
            original_confidence, 
            adjusted_confidence,
            should_alert
        )
    }
    
    return result


def get_recommendation(detected_class, original_conf, adjusted_conf, should_alert):
    """Generate human-readable recommendation"""
    
    if not should_alert:
        if detected_class.lower() == 'smoke':
            return (
                "⚠️ Smoke detected but confidence reduced due to weather/context. "
                "Likely fog, steam, or mist. Visual verification recommended."
            )
        else:
            return "Low confidence detection. Likely false positive."
    
    if adjusted_conf < original_conf * 0.7:
        return (
            f"🔥 {detected_class.upper()} DETECTED! "
            f"Confidence reduced by weather context but still concerning. "
            f"Immediate visual verification required."
        )
    
    return (
        f"🚨 {detected_class.upper()} DETECTED! "
        f"High confidence detection. "
        f"Take immediate action!"
    )


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("🌦️  Weather Context Detection Test")
    print("="*60)
    
    # Test weather API
    weather_ctx = WeatherContext(city="Bongao")
    weather = weather_ctx.get_weather()
    
    if weather:
        print(f"\n📍 Location: {weather['city']}")
        print(f"🌤️  Condition: {weather['condition']} ({weather['description']})")
        print(f"🌡️  Temperature: {weather['temperature']}°C")
        print(f"💧 Humidity: {weather['humidity']}%")
        print(f"👁️  Visibility: {weather['visibility']}m")
        
        # Simulate smoke detection
        print("\n" + "="*60)
        print("🧪 Testing Confidence Adjustment")
        print("="*60)
        
        test_cases = [
            ("smoke", 0.75, "Outdoor"),
            ("smoke", 0.60, "Kitchen"),
            ("fire", 0.85, "Bedroom"),
        ]
        
        for detected_class, confidence, location in test_cases:
            print(f"\n📋 Test: {detected_class.upper()} at {confidence:.0%} confidence")
            print(f"   Location: {location}")
            
            adjusted, reason = weather_ctx.adjust_confidence(
                detected_class, 
                confidence, 
                weather
            )
            
            print(f"   Original: {confidence:.1%}")
            print(f"   Adjusted: {adjusted:.1%}")
            print(f"   Reason: {reason}")
            
            if adjusted < 0.3:
                print(f"   ✅ Decision: NO ALERT (likely false positive)")
            else:
                print(f"   🚨 Decision: ALERT!")
    
    else:
        print("\n❌ Weather API not configured")
        print("\n📝 To enable weather context:")
        print("   1. Sign up at https://openweathermap.org/api")
        print("   2. Get free API key")
        print("   3. Add to .env file:")
        print("      OPENWEATHER_API_KEY=your_key_here")
    
    print("\n" + "="*60)
