def lead_acceleration_at_time(t):
    if t < 5:
        return 0.0
    elif t < 10:
        return -1.0
    elif t < 15:
        return 1.2
    elif t < 22:
        return 0.2
    else:
        return -2.0
