import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or '688hutu66ff6679t4dgjxf550y7thnbg'
    AUTH_TOKEN = os.environ.get('AUTH_TOKEN') or '2ovpxyeiy4YbBTP44SNhLNQxuce_2Dn1doK14eFT3pb6YZEvP'
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app', 'models') 