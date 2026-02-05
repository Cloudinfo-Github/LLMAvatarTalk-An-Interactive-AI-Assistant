from dotenv import load_dotenv
import os

# RIVA server
load_dotenv()
URI = os.getenv('RIVA_URI', '192.168.1.205:50051')
OLLAMA_BASE_URL = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
OLLAMA_MODEL = os.getenv('OLLAMA_MODEL', 'llama3')
voice_config = {
    'en-US': "English-US.Female-1",
    'zh-CN': "Mandarin-CN.Female-1"
}
LANGUAGE = 'en-US'  # or 'en-US'
