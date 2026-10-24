import requests 
import time 
from datetime import datetime 
 
URL = "https://pickleball-ai-engine.onrender.com/" 
print("Script keep-alive is running...") 
 
while True: 
    try: 
        response = requests.get(URL, timeout=10) 
        now = datetime.now().strftime("%%H:%%M:%%S - %%d/%%m/%%Y") 
        if response.status_code == 200: 
            print(f"[{now}] Ping OK - Status 200") 
        else: 
            print(f"[{now}] Status: {response.status_code}") 
    except Exception as e: 
        print(f"Error: {e}") 
    time.sleep(600) 
