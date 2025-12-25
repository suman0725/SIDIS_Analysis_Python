# If you just want to check if it's forward:
is_forward = (abs(status) // 1000) & 2 > 0

# Or if you prefer ranges:
is_forward = 2000 <= abs(status) < 4000
