import json
import os
from threading import Lock
from flask import Blueprint, render_template, jsonify, request
from datetime import datetime, timedelta
from app.config import STATS_FILE

stats_bp = Blueprint('statistics', __name__, url_prefix='/statistics')

lock = Lock()

def get_stats():
    """Reads statistics from the JSON file."""
    with lock:
        if not os.path.exists(STATS_FILE):
            return {'total_recognitions': 0, 'hourly_recognitions': {}}
        try:
            with open(STATS_FILE, 'r', encoding='utf-8') as f:
                stats = json.load(f)
                # Ensure both keys exist for backward compatibility
                if 'total_recognitions' not in stats:
                    stats['total_recognitions'] = 0
                if 'hourly_recognitions' not in stats:
                    stats['hourly_recognitions'] = {}
                return stats
        except (json.JSONDecodeError, IOError):
            return {'total_recognitions': 0, 'hourly_recognitions': {}}

def save_stats(stats):
    """Saves statistics to the JSON file."""
    with lock:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=4, ensure_ascii=False)

def increment_recognition_count():
    """Increments the recognition count for the current hour and the total."""
    stats = get_stats()
    now = datetime.now()
    hour_key = now.strftime('%Y-%m-%d-%H')

    # Increment total and hourly counts
    stats['total_recognitions'] = stats.get('total_recognitions', 0) + 1
    stats.setdefault('hourly_recognitions', {})[hour_key] = stats.get('hourly_recognitions', {}).get(hour_key, 0) + 1
    
    save_stats(stats)
    return stats['total_recognitions']

@stats_bp.route('/')
def index():
    """Renders the statistics page."""
    return render_template('frontend/statistics.html')

@stats_bp.route('/data')
def data():
    """Returns the total statistics data as JSON."""
    stats = get_stats()
    return jsonify({'recognition_count': stats.get('total_recognitions', 0)})

@stats_bp.route('/hourly_data')
def hourly_data():
    """
    Provides data for a specified number of past hours for charting.
    Defaults to 24 hours if not specified.
    """
    try:
        hours = int(request.args.get('hours', 24))
        if not (1 <= hours <= 24):
            hours = 24
    except (ValueError, TypeError):
        hours = 24

    stats = get_stats()
    hourly_recognitions = stats.get('hourly_recognitions', {})
    
    labels = []
    data = []
    
    now = datetime.now()
    for i in range(hours - 1, -1, -1):
        time_point = now - timedelta(hours=i)
        hour_key = time_point.strftime('%Y-%m-%d-%H')
        label = time_point.strftime('%H:00')
        
        labels.append(label)
        data.append(hourly_recognitions.get(hour_key, 0))
        
    return jsonify({'labels': labels, 'data': data, 'requested_hours': hours}) 