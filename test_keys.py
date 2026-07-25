import os
import re
from dotenv import load_dotenv
load_dotenv()

import supabase._sync.client

# Monkey-patch the regex match in the supabase client to bypass JWT validation
# for newer Supabase key formats (e.g. sb_publishable_* or sb_secret_*).
_original_match = re.match
def _custom_match(pattern, string, flags=0):
    if isinstance(pattern, str) and "A-Za-z0-9-_=" in pattern:
        return True
    return _original_match(pattern, string, flags)

supabase._sync.client.re.match = _custom_match

from supabase import create_client

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
service_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

print("Checking Supabase connection...")
print(f"URL: {url}")
print(f"Anon Key prefix: {key[:15] if key else 'Not Set'}...")
print(f"Service Role Key prefix: {service_key[:15] if service_key else 'Not Set'}...")

if not url or "supabase.com/dashboard" in url:
    print("[ERROR] SUPABASE_URL must be the API URL (e.g. https://xxxx.supabase.co), not the dashboard URL.")
    exit(1)

# Test Anon Client
try:
    print("\nInitializing Anon client...")
    client = create_client(url, key)
    # Trigger an auth check
    try:
        client.auth.get_user("dummy_token")
    except Exception as e:
        if "Invalid API key" in str(e):
            print("[FAIL] SUPABASE_KEY (Anon Key) is invalid.")
        else:
            print("[OK] Anon Client: API Key validated successfully (Got expected token check result).")
except Exception as e:
    print(f"[FAIL] Failed to initialize Anon Client: {e}")

# Test Service Role Client
try:
    print("\nInitializing Admin client...")
    admin_client = create_client(url, service_key)
    # Try fetching a row from profiles
    res = admin_client.table("profiles").select("id").limit(1).execute()
    print("[OK] Admin Client: Connected successfully and fetched from profiles table!")
except Exception as e:
    print(f"[FAIL] Admin Client test failed: {e}")
    if "Invalid API key" in str(e):
        print("TIP: Make sure you copied the FULL Secret key using the COPY button in your Supabase settings.")
