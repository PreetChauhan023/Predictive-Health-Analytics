import re
def is_valid_email(email):
    return bool(re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', email))

def is_valid_phone(phone):
    return bool(re.match(r'^(?:\+91|0)?[6-9]\d{9}$', phone))

def is_strong_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r'\d', password):
        return "Password must contain at least one number."
    if not re.search(r'[!@#$%^&*]', password):
        return "Password must contain at least one special character (!@#$%^&*)."
    return None