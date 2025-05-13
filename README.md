# Twitter Scrubber

A web application that helps users find and manage potentially problematic tweets in their Twitter history.

## Overview

Twitter Scrubber analyzes a user's Twitter timeline to identify potentially offensive or problematic tweets using sentiment analysis and keyword detection. Users can then review and delete flagged tweets to maintain a clean online presence.

## Tech Stack

- **Backend**: Python Flask web framework
- **APIs**: 
  - SocialData.tools API for Twitter data retrieval
  - Stripe for payment processing
- **Analysis**: Custom sentiment analyzer with keyword detection
- **Frontend**: HTML, CSS, JavaScript

## Features

- Fetch tweets from a specified Twitter username
- Analyze tweets for potentially offensive content
- Filter by date range
- Customizable sensitivity and keyword settings
- Payment integration for one-time usage fee
- Direct links to flagged tweets for easy deletion

## Setup

### Requirements

- Python 3.7+
- Flask 2.0+
- API keys for SocialData.tools and Stripe

### Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/twitter-scrubber.git
cd twitter-scrubber
```

2. Create a virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```
pip install -r requirements.txt
```

4. Create a `.env` file with your API keys:
```
SOCIALDATA_API_KEY=your_socialdata_api_key
STRIPE_PUBLIC_KEY=your_stripe_public_key
STRIPE_SECRET_KEY=your_stripe_secret_key
SECRET_KEY=your_flask_secret_key
```

5. Run the application:
```
python app.py
```

The application will be available at http://localhost:5000

## Usage

1. Enter a Twitter username
2. Select a date range for analysis
3. Complete the one-time payment
4. Review flagged tweets and take action

## License

[Your chosen license] 