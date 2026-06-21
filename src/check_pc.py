import requests
import time

def check_planetary_computer():
    print("Pinging Microsoft Planetary Computer...")
    start_time = time.time()
    
    try:
        # A simple lightweight query to check if the database is responding
        url = "https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items?limit=1"
        response = requests.get(url, timeout=10)
        
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            print(f"\n✅ BACK ONLINE! (Responded in {elapsed:.2f} seconds)")
            print("Microsoft's servers are working again. You can use the fetcher now!")
        else:
            print(f"\n❌ STILL BROKEN! (Status code: {response.status_code})")
            
    except requests.exceptions.Timeout:
        print("\n❌ STILL DEAD! (Request timed out after 10 seconds)")
    except Exception as e:
        print(f"\n❌ STILL BROKEN! (Error: {e})")

if __name__ == "__main__":
    check_planetary_computer()
