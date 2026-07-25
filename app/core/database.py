"""
Supabase client singleton.
Uses the service-role key for admin operations and the anon key for user-scoped queries.
"""

import re
import supabase._sync.client

# Monkey-patch the regex match in the supabase client to bypass JWT validation
# for newer Supabase key formats (e.g. sb_publishable_* or sb_secret_*).
_original_match = re.match
def _custom_match(pattern, string, flags=0):
    if isinstance(pattern, str) and "A-Za-z0-9-_=" in pattern:
        return True
    return _original_match(pattern, string, flags)

supabase._sync.client.re.match = _custom_match

from functools import lru_cache
from supabase import create_client, Client

from app.core.config import get_settings


@lru_cache()
def get_supabase_client() -> Client:
    """Return a cached Supabase client using the service role key (bypasses RLS)."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


@lru_cache()
def get_supabase_admin() -> Client:
    """Return a cached Supabase client using the service-role key (admin ops)."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_SERVICE_ROLE_KEY)


def get_supabase_auth_client() -> Client:
    """Return a non-cached Supabase client specifically for authentication (sign_in / sign_up) so the singleton admin client is never mutated."""
    settings = get_settings()
    return create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
