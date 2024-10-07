"""
Dependencies module for FastAPI app.

This module defines the shared dependencies used across the FastAPI application.
It includes the rate-limiter instance from the `slowapi` library and other utilities
for managing application-wide dependencies.

The `Limiter` instance is configured to apply rate limiting based on the client's
remote IP address (using `get_remote_address`). This limiter can be applied to 
individual routes using decorators to control the rate of incoming requests.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
