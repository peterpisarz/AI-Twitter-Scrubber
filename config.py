from dotenv import load_dotenv
import os

load_dotenv()

# SocialData.tools API Key
SOCIALDATA_API_KEY = os.getenv('SOCIALDATA_API_KEY', 'your_socialdata_api_key_here')

# Stripe API Keys
STRIPE_PUBLIC_KEY = os.getenv('STRIPE_PUBLIC_KEY', 'your_stripe_public_key_here')
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY', 'your_stripe_secret_key_here')

# Flask Secret Key for session management
SECRET_KEY = os.getenv('SECRET_KEY', 'your_flask_secret_key_here') 