#!/usr/bin/env python3
"""
Quick test to see what the statistics API returns
"""

import requests
import json

try:
    response = requests.get('http://localhost:5000/api/statistics', timeout=10)
    
    if response.status_code == 200:
        stats = response.json()
        print("📊 Statistics API Response:")
        print(json.dumps(list(stats.keys()), indent=2))
        
        # Check if we have the expected structure
        if 'dataset_info' in stats:
            print(f"✅ dataset_info found: {stats['dataset_info']}")
        else:
            print(f"❌ dataset_info not found. Available keys: {list(stats.keys())}")
    else:
        print(f"❌ API Error: {response.status_code}")
        print(response.text)
        
except Exception as e:
    print(f"❌ Error: {e}")