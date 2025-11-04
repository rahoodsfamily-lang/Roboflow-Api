"""
Database module for storing detection history, users, and API keys
Uses SQLite for simplicity and portability
"""

import os
import sqlite3
import json
from datetime import datetime
import secrets
import hashlib
from contextlib import contextmanager

DATABASE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'database', 'detections.db')

@contextmanager
def get_db():
    """Context manager for database connections"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        conn.close()

def init_database():
    """Initialize database tables"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Detection history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER,
                api_key_id INTEGER,
                model_id TEXT NOT NULL,
                confidence_threshold INTEGER,
                predictions TEXT,  -- JSON array of predictions
                detection_count INTEGER,
                has_fire BOOLEAN,
                has_smoke BOOLEAN,
                max_confidence REAL,
                location TEXT,
                city TEXT,
                weather_context TEXT,  -- JSON object
                processing_time_ms REAL,
                image_size TEXT,
                ip_address TEXT,
                FOREIGN KEY (user_id) REFERENCES users(id),
                FOREIGN KEY (api_key_id) REFERENCES api_keys(id)
            )
        ''')
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_login DATETIME,
                is_active BOOLEAN DEFAULT 1,
                is_admin BOOLEAN DEFAULT 0,
                detection_quota INTEGER DEFAULT 1000,  -- Monthly quota
                detections_used INTEGER DEFAULT 0
            )
        ''')
        
        # API Keys table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS api_keys (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_hash TEXT UNIQUE NOT NULL,
                key_prefix TEXT NOT NULL,  -- First 8 chars for display
                name TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_used DATETIME,
                expires_at DATETIME,
                is_active BOOLEAN DEFAULT 1,
                rate_limit INTEGER DEFAULT 100,  -- Requests per hour
                requests_used INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Webhooks table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS webhooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                url TEXT NOT NULL,
                event_type TEXT NOT NULL,  -- 'fire_detected', 'smoke_detected', 'all'
                is_active BOOLEAN DEFAULT 1,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_triggered DATETIME,
                trigger_count INTEGER DEFAULT 0,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        # Alert settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alert_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                email_enabled BOOLEAN DEFAULT 0,
                sms_enabled BOOLEAN DEFAULT 0,
                email_address TEXT,
                phone_number TEXT,
                min_confidence INTEGER DEFAULT 50,
                alert_for_fire BOOLEAN DEFAULT 1,
                alert_for_smoke BOOLEAN DEFAULT 1,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        ''')
        
        print("✅ Database initialized successfully")

def log_detection(model_id, predictions, user_id=None, api_key_id=None, 
                 location=None, city=None, weather_context=None, 
                 processing_time_ms=None, image_size=None, ip_address=None,
                 confidence_threshold=40):
    """Log a detection to the database"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Analyze predictions
        detection_count = len(predictions)
        has_fire = any(p.get('class') == 'fire' for p in predictions)
        has_smoke = any(p.get('class') == 'smoke' for p in predictions)
        max_confidence = max([p.get('confidence', 0) for p in predictions]) if predictions else 0
        
        cursor.execute('''
            INSERT INTO detections (
                user_id, api_key_id, model_id, confidence_threshold,
                predictions, detection_count, has_fire, has_smoke,
                max_confidence, location, city, weather_context,
                processing_time_ms, image_size, ip_address
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            user_id, api_key_id, model_id, confidence_threshold,
            json.dumps(predictions), detection_count, has_fire, has_smoke,
            max_confidence, location, city, json.dumps(weather_context) if weather_context else None,
            processing_time_ms, image_size, ip_address
        ))
        
        return cursor.lastrowid

def get_detection_history(user_id=None, limit=100, offset=0):
    """Get detection history"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('''
                SELECT * FROM detections 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (user_id, limit, offset))
        else:
            cursor.execute('''
                SELECT * FROM detections 
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            ''', (limit, offset))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def get_detection_stats(user_id=None, days=30):
    """Get detection statistics"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        date_filter = f"datetime('now', '-{days} days')"
        
        if user_id:
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_detections,
                    SUM(has_fire) as fire_detections,
                    SUM(has_smoke) as smoke_detections,
                    AVG(max_confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM detections 
                WHERE user_id = ? AND timestamp > {date_filter}
            ''', (user_id,))
        else:
            cursor.execute(f'''
                SELECT 
                    COUNT(*) as total_detections,
                    SUM(has_fire) as fire_detections,
                    SUM(has_smoke) as smoke_detections,
                    AVG(max_confidence) as avg_confidence,
                    AVG(processing_time_ms) as avg_processing_time
                FROM detections 
                WHERE timestamp > {date_filter}
            ''')
        
        row = cursor.fetchone()
        return dict(row) if row else {}

def create_user(username, email, password):
    """Create a new user with validation"""
    # Validate inputs
    if not username or not email or not password:
        raise ValueError("Username, email, and password are required")
    
    if len(username) < 3:
        raise ValueError("Username must be at least 3 characters")
    
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters")
    
    if '@' not in email:
        raise ValueError("Invalid email format")
    
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Check if username already exists
        cursor.execute('SELECT id FROM users WHERE username = ?', (username,))
        if cursor.fetchone():
            raise ValueError("Username already exists")
        
        # Check if email already exists
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            raise ValueError("Email already exists")
        
        # Hash password and create user
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
            INSERT INTO users (username, email, password_hash)
            VALUES (?, ?, ?)
        ''', (username, email, password_hash))
        return cursor.lastrowid

def verify_user(username, password):
    """Verify user credentials"""
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM users 
            WHERE username = ? AND password_hash = ? AND is_active = 1
        ''', (username, password_hash))
        row = cursor.fetchone()
        return dict(row) if row else None

def generate_api_key(user_id, name=None):
    """Generate a new API key for a user"""
    # Generate random API key
    api_key = f"fsd_{secrets.token_urlsafe(32)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    key_prefix = api_key[:12]  # First 12 chars for display
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO api_keys (user_id, key_hash, key_prefix, name)
            VALUES (?, ?, ?, ?)
        ''', (user_id, key_hash, key_prefix, name))
        
    return api_key  # Return the actual key (only shown once)

def get_user_api_keys(user_id):
    """Get all API keys for a user (without revealing the actual keys)"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT id, key_prefix, name, created_at, last_used
            FROM api_keys
            WHERE user_id = ?
            ORDER BY created_at DESC
        ''', (user_id,))
        
        keys = []
        for row in cursor.fetchall():
            keys.append({
                'id': row[0],
                'key_prefix': row[1],
                'name': row[2],
                'created_at': row[3],
                'last_used': row[4]
            })
        
        return keys

def verify_api_key(api_key):
    """Verify an API key and return user info"""
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT ak.*, u.username, u.email, u.is_active as user_active
            FROM api_keys ak
            JOIN users u ON ak.user_id = u.id
            WHERE ak.key_hash = ? AND ak.is_active = 1 AND u.is_active = 1
        ''', (key_hash,))
        row = cursor.fetchone()
        
        if row:
            # Update last used timestamp
            cursor.execute('''
                UPDATE api_keys 
                SET last_used = CURRENT_TIMESTAMP, requests_used = requests_used + 1
                WHERE id = ?
            ''', (row['id'],))
            
        return dict(row) if row else None

def create_webhook(user_id, url, event_type='all'):
    """Create a webhook for a user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO webhooks (user_id, url, event_type)
            VALUES (?, ?, ?)
        ''', (user_id, url, event_type))
        return cursor.lastrowid

def get_webhooks(user_id, event_type=None):
    """Get webhooks for a user"""
    with get_db() as conn:
        cursor = conn.cursor()
        
        if event_type:
            cursor.execute('''
                SELECT * FROM webhooks 
                WHERE user_id = ? AND (event_type = ? OR event_type = 'all') AND is_active = 1
            ''', (user_id, event_type))
        else:
            cursor.execute('''
                SELECT * FROM webhooks 
                WHERE user_id = ? AND is_active = 1
            ''', (user_id,))
        
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

def update_webhook_trigger(webhook_id):
    """Update webhook trigger count and timestamp"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            UPDATE webhooks 
            SET last_triggered = CURRENT_TIMESTAMP, trigger_count = trigger_count + 1
            WHERE id = ?
        ''', (webhook_id,))

def get_alert_settings(user_id):
    """Get alert settings for a user"""
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM alert_settings 
            WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        return dict(row) if row else None

if __name__ == '__main__':
    # Initialize database
    init_database()
    print("Database setup complete!")
