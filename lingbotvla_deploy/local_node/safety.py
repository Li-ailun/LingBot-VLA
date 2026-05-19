#!/usr/bin/env python3
"""
Safety utilities:
- joint limit check
- velocity limit check
- workspace boundary check
- emergency stop check
"""

def clip_delta(delta, max_delta):
    return delta.clip(-max_delta, max_delta)
