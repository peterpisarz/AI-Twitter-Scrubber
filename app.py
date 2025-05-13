from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import requests
import re
# Removed transformers import as we'll use our custom analyzer
# from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import stripe
from datetime import datetime
from dateutil.parser import parse
from config import SOCIALDATA_API_KEY, STRIPE_PUBLIC_KEY, STRIPE_SECRET_KEY, SECRET_KEY
import json
import time
import traceback

app = Flask(__name__)
app.secret_key = SECRET_KEY

# Stripe configuration
stripe.api_key = STRIPE_SECRET_KEY

# Create a simple explicit sentiment analyzer that doesn't rely on ML models
class SentimentAnalyzer:
    def __init__(self):
        # Offensive words with their negativity weights
        self.offensive_words = {
            "hate": 0.8, "terrible": 0.7, "awful": 0.7, "horrible": 0.8, "bad": 0.6, "worst": 0.8,
            "stupid": 0.7, "dumb": 0.6, "idiot": 0.7, "ugly": 0.6, "sucks": 0.7, "sad": 0.5,
            "angry": 0.6, "annoying": 0.5, "disappointed": 0.5, "fail": 0.6, "failure": 0.7,
            "poor": 0.5, "disgusting": 0.8, "wrong": 0.5, "useless": 0.6, "waste": 0.6,
            "annoyed": 0.5, "offensive": 0.7, "violent": 0.8, "inappropriate": 0.6,
            "fuck": 0.9, "shit": 0.8, "damn": 0.6, "hell": 0.5, "ass": 0.6, "jerk": 0.7,
            "bitch": 0.8, "asshole": 0.9, "crap": 0.6, "wtf": 0.7, "stfu": 0.8,
            "kill": 0.8, "die": 0.7, "death": 0.7, "murder": 0.9, "weapon": 0.6, "gun": 0.6,
            "racist": 0.9, "sexist": 0.9, "bigot": 0.9, "discrimination": 0.8,
            # Many contexts
            "fake": 0.5, "liar": 0.7, "lying": 0.7, "cheat": 0.7, "stealing": 0.7,
            "corrupt": 0.8, "greedy": 0.7, "selfish": 0.6, "unfair": 0.6, "unjust": 0.7,
            "pathetic": 0.7, "incompetent": 0.7, "ignorant": 0.6, "lazy": 0.6,
            "despise": 0.8, "loser": 0.7, "worthless": 0.8, "garbage": 0.7, "trash": 0.7
        }
        
        # Negative phrase patterns (explicit multi-word phrases)
        self.negative_phrases = [
            "shut up", "go to hell", "hate you", "piss off", "screw you",
            "fuck you", "fuck off", "piece of shit", "full of shit", "what the fuck",
            "don't like", "can't stand", "fed up", "sick of", "tired of", 
            "waste of time", "waste of money", "hate it when", "hate when", 
            "makes me sick", "pisses me off", "rip off"
        ]
        
    def analyze(self, text):
        if not text:
            return {'label': 'NEUTRAL', 'score': 0}
            
        text = text.lower()
        
        # Check for negative phrases (exact matches of multi-word phrases)
        phrase_scores = []
        for phrase in self.negative_phrases:
            if phrase in text:
                phrase_scores.append(0.8)  # Higher weight for phrases
        
        # Check for offensive words (with word boundaries)
        word_scores = []
        for word, weight in self.offensive_words.items():
            # Use regex with word boundaries to match whole words
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, text):
                word_scores.append(weight)
        
        # Calculate the final score
        if phrase_scores or word_scores:
            # Take the maximum score if we have matches
            all_scores = phrase_scores + word_scores
            if all_scores:
                score = min(0.95, max(all_scores))  # Cap at 0.95
            else:
                score = 0
            return {'label': 'NEGATIVE', 'score': score}
        else:
            # No negative content found
            return {'label': 'NEUTRAL', 'score': 0}
    
    def __call__(self, text):
        """Makes the class callable like a function"""
        result = self.analyze(text)
        return [result]  # Return in the same format as Hugging Face pipeline

# Initialize our custom sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()
print("Initialized custom sentiment analyzer")

# Test with known examples
test_texts = [
    "I hate this app, it's terrible",
    "This is just a normal tweet about something",
    "What a massive failing of society that porn is just instantly available",
    "No-code tools absolutely suck when you know how to code"
]

print("Testing sentiment analyzer with sample texts:")
for text in test_texts:
    result = sentiment_analyzer(text)
    print(f"Text: '{text}' → Result: {result}")

# Default offensive keywords - can be customized by users
DEFAULT_OFFENSIVE_KEYWORDS = [
    "hate", "stupid", "idiot", "dumb", "moron", "jerk", "ass", "shit", "damn", "hell",
    "fuck", "fucking", "bitch", "asshole", "crap", "suck", "wtf", "stfu", "lmao", "lmfao",
    "kill", "die", "death", "murder", "violent", "violence", "weapon", "gun", "shoot",
    "racist", "sexist", "bigot", "discrimination", "offensive", "inappropriate",
    "shut up", "screw you", "go to hell", "hate you", "piss off", "pathetic",
    "disgusting", "despise", "loser", "worthless", "garbage", "trash"
]

@app.route('/')
def index():
    # Get custom keywords from session or use defaults
    keywords = session.get('custom_keywords', DEFAULT_OFFENSIVE_KEYWORDS)
    # Get sensitivity setting from session or use default (0.6)
    sensitivity = session.get('sensitivity', 0.6)
    # Get today's date for date picker default value
    today = datetime.now().strftime('%Y-%m-%d')
    return render_template('index.html', keywords=keywords, sensitivity=sensitivity, today=today)

@app.route('/settings', methods=['POST'])
def settings():
    custom_keywords = request.form.get('custom_keywords', '')
    keywords_list = [k.strip() for k in custom_keywords.split(',') if k.strip()]
    
    if keywords_list:
        session['custom_keywords'] = keywords_list
    else:
        session['custom_keywords'] = DEFAULT_OFFENSIVE_KEYWORDS
    
    try:
        sensitivity = float(request.form.get('sensitivity', 0.6))
        if sensitivity < 0:
            sensitivity = 0
        elif sensitivity > 1:
            sensitivity = 1
    except ValueError:
        sensitivity = 0.6
    
    session['sensitivity'] = sensitivity
    
    return redirect(url_for('index'))

@app.route('/analyze', methods=['POST'])
def analyze():
    username = request.form.get('username')
    date_range = request.form.get('date_range')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Debug output for date range values
    print(f"Date range selection: {date_range}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")
    
    # Clear previous session data to avoid size issues
    session.pop('flagged_tweets', None)
    session.pop('tweets_analyzed', None)
    session.pop('tweets_flagged', None)
    session.pop('tweets_pages_fetched', None)
    session.pop('error', None)
    
    # Store date range information for display on results page
    session['date_range_type'] = date_range
    session['start_date'] = start_date if start_date else "earliest"
    session['end_date'] = end_date if end_date else today
    session['twitter_username'] = username
    
    # Save analysis parameters to session for later use after payment
    session['analysis_params'] = {
        'username': username,
        'date_range': date_range,
        'start_date': start_date,
        'end_date': end_date
    }
    
    # Redirect to payment page first
    return redirect(url_for('payment'))

@app.route('/analyze_after_payment')
def analyze_after_payment():
    # Check if payment has been completed
    if not session.get('payment_confirmed', False):
        return redirect(url_for('payment', message="Payment required to analyze tweets"))
    
    # Get analysis parameters from session
    params = session.get('analysis_params', {})
    if not params:
        return redirect(url_for('index'))
        
    username = params.get('username')
    date_range = params.get('date_range')
    start_date = params.get('start_date')
    end_date = params.get('end_date')
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # First, get the user ID from the username
    user_id = get_user_id(username)
    if not user_id:
        session['tweets_analyzed'] = 0
        session['tweets_flagged'] = 0
        session['tweets_pages_fetched'] = 0
        session['error'] = f"Could not find user ID for username: {username}"
        return redirect(url_for('results'))
    
    try:
        if date_range == 'all_time':
            print("Using all_time date range")
            tweets = fetch_tweets(user_id)
        else:
            # Format and validate dates
            if not start_date:
                print("No start date provided, using earliest")
                formatted_start_date = None
            else:
                print(f"Using custom start date: {start_date}")
                formatted_start_date = start_date
                
            if not end_date:
                print("No end date provided, using today")
                formatted_end_date = today
            else:
                formatted_end_date = end_date
                print(f"Using custom end date: {end_date}")
            
            # Ensure start_date is not in the future
            if formatted_start_date and formatted_start_date > today:
                print(f"Start date {formatted_start_date} is in the future, using today instead")
                formatted_start_date = today
            
            # Ensure end_date is not in the future
            if formatted_end_date and formatted_end_date > today:
                print(f"End date {formatted_end_date} is in the future, using today instead")
                formatted_end_date = today
            
            # Ensure start_date is before end_date
            if formatted_start_date and formatted_end_date and formatted_start_date > formatted_end_date:
                print(f"Start date {formatted_start_date} is after end date {formatted_end_date}, swapping")
                formatted_start_date, formatted_end_date = formatted_end_date, formatted_start_date
            
            print(f"Final date range: {formatted_start_date} to {formatted_end_date}")
            tweets = fetch_tweets(user_id, formatted_start_date, formatted_end_date)
        
        # Store the total number of tweets analyzed
        session['tweets_analyzed'] = len(tweets)
        session['tweets_pages_fetched'] = session.get('pages_fetched', 0)
        
        # Get sentiment sensitivity from session
        sensitivity = session.get('sensitivity', 0.6)
        
        # Get custom keywords from session
        custom_keywords = session.get('custom_keywords', DEFAULT_OFFENSIVE_KEYWORDS)
        
        flagged_tweets = analyze_tweets(tweets, sensitivity, custom_keywords)
        
        # Store stats without storing the full tweets (to reduce session size)
        session['tweets_flagged'] = len(flagged_tweets)
        
        # Convert tweet object to a basic dict with just the fields we need for display
        # This helps prevent session size issues
        simplified_flagged_tweets = []
        for tweet in flagged_tweets:
            simplified_tweet = {
                'text': tweet.get('text', ''),
                'link': tweet.get('link', ''),
                'sentiment_score': tweet.get('sentiment_score', 0),
                'keyword_score': tweet.get('keyword_score', 0),
                'combined_score': tweet.get('combined_score', 0),
                'keyword_matches': tweet.get('keyword_matches', []),
                'date': tweet.get('date', '')
            }
            simplified_flagged_tweets.append(simplified_tweet)
        
        # Store the simplified tweets in the session
        if len(simplified_flagged_tweets) > 50:
            # If there are too many tweets, only keep the top 50 (highest combined score)
            session['flagged_tweets'] = simplified_flagged_tweets[:50]
            session['error'] = "Found more than 50 flagged tweets. Only showing the top 50 with highest scores."
        else:
            session['flagged_tweets'] = simplified_flagged_tweets
        
    except Exception as e:
        # Log the error
        print(f"Error during analysis: {str(e)}")
        print(traceback.format_exc())
        
        # Set partial results if available
        if 'tweets_analyzed' not in session:
            session['tweets_analyzed'] = 0
        if 'tweets_flagged' not in session:
            session['tweets_flagged'] = 0
        if 'flagged_tweets' not in session:
            session['flagged_tweets'] = []
        
        # Store the error message
        session['error'] = f"An error occurred during analysis: {str(e)}. Showing partial results if available."
    
    return redirect(url_for('results'))

@app.route('/payment', methods=['GET', 'POST'])
def payment():
    # Get any message passed to the payment page
    message = request.args.get('message', None)
    
    if request.method == 'POST':
        # Save the analysis state so we can retrieve it after payment
        analysis_state = {
            'flagged_tweets': session.get('flagged_tweets', []),
            'tweets_analyzed': session.get('tweets_analyzed', 0),
            'tweets_flagged': session.get('tweets_flagged', 0),
            'tweets_pages_fetched': session.get('tweets_pages_fetched', 0),
            'date_range_type': session.get('date_range_type', 'all_time'),
            'start_date': session.get('start_date', 'earliest'),
            'end_date': session.get('end_date', '')
        }
        session['analysis_state'] = analysis_state
        
        try:
            success_url = url_for('payment_success', _external=True)
            cancel_url = url_for('payment_cancel', _external=True)
            
            checkout_session = stripe.checkout.Session.create(
                payment_method_types=['card'],
                line_items=[{
                    'price_data': {
                        'currency': 'usd',
                        'product_data': {
                            'name': 'Twitter Scrubber One-Time Use',
                            'description': 'Scan and analyze your Twitter history for potentially problematic content',
                            'images': ['https://www.example.com/images/twitter-scrubber.png'],
                        },
                        'unit_amount': 1499,  # $14.99 in cents
                    },
                    'quantity': 1,
                }],
                mode='payment',
                success_url=success_url,
                cancel_url=cancel_url,
            )
            return redirect(checkout_session.url, code=303)
        except Exception as e:
            print(f"Stripe error: {str(e)}")
            print(traceback.format_exc())
            return render_template('payment.html', 
                                  stripe_public_key=STRIPE_PUBLIC_KEY,
                                  error="There was an issue processing your payment. Please try again later.",
                                  username=session.get('twitter_username', ''))
    
    twitter_username = session.get('twitter_username', '')
    return render_template('payment.html', 
                          stripe_public_key=STRIPE_PUBLIC_KEY, 
                          message=message,
                          username=twitter_username)

@app.route('/payment/success')
def payment_success():
    # Restore analysis state from session
    analysis_state = session.get('analysis_state', {})
    for key, value in analysis_state.items():
        session[key] = value
    
    # Add payment confirmation
    session['payment_confirmed'] = True
    session['payment_status'] = 'success'
    
    # Now run the analysis
    return redirect(url_for('analyze_after_payment'))

@app.route('/payment/cancel')
def payment_cancel():
    return render_template('payment.html', 
                         stripe_public_key=STRIPE_PUBLIC_KEY,
                         error="Payment was cancelled. You can try again when you're ready.",
                         username=session.get('twitter_username', ''))

@app.route('/results')
def results():
    # Check if payment is confirmed
    if not session.get('payment_confirmed', False):
        return redirect(url_for('payment', message="Payment is required to view results"))
        
    flagged_tweets = session.get('flagged_tweets', [])
    
    # Ensure each tweet has all required fields with defaults
    for tweet in flagged_tweets:
        if not isinstance(tweet, dict):
            continue
        if 'sentiment_score' not in tweet:
            tweet['sentiment_score'] = 0
        if 'keyword_score' not in tweet:
            tweet['keyword_score'] = 0
        if 'combined_score' not in tweet:
            tweet['combined_score'] = 0
        if 'keyword_matches' not in tweet:
            tweet['keyword_matches'] = []
        if 'date' not in tweet:
            tweet['date'] = ''
            
    error = session.get('error', None)
    tweets_analyzed = session.get('tweets_analyzed', 0)
    tweets_flagged = session.get('tweets_flagged', 0)
    pages_fetched = session.get('tweets_pages_fetched', 0)
    date_range_type = session.get('date_range_type', 'all_time')
    start_date = session.get('start_date', 'earliest')
    end_date = session.get('end_date', datetime.now().strftime('%Y-%m-%d'))
    payment_confirmed = session.get('payment_confirmed', False)
    payment_status = session.get('payment_status', None)
    
    return render_template('results.html', 
                          tweets=flagged_tweets, 
                          error=error,
                          tweets_analyzed=tweets_analyzed,
                          tweets_flagged=tweets_flagged,
                          pages_fetched=pages_fetched,
                          date_range_type=date_range_type,
                          start_date=start_date,
                          end_date=end_date,
                          payment_confirmed=payment_confirmed,
                          payment_status=payment_status)

def fetch_tweets(user_id, start_date=None, end_date=None, max_pages=5):
    all_tweets = []
    url = f"https://api.socialdata.tools/twitter/user/{user_id}/tweets"
    headers = {'Authorization': f'Bearer {SOCIALDATA_API_KEY}', 'Accept': 'application/json'}
    print(f"API Key (first 5 chars for security): {SOCIALDATA_API_KEY[:5]}...")
    params = {}
    
    # Format and add date parameters if provided
    if start_date and end_date:
        # Ensure dates are in YYYY-MM-DD format for the API
        params['start_date'] = start_date
        params['end_date'] = end_date
        print(f"Including date range in API request: {start_date} to {end_date}")
    else:
        print("No date range specified for API request")
    
    # Fetch multiple pages of tweets
    cursor = None
    page_count = 0
    pages_fetched = 0
    
    try:
        while page_count < max_pages:
            if cursor:
                params['cursor'] = cursor
            
            print(f"Fetching page {page_count + 1} of tweets")
            print(f"API Request URL: {url}")
            print(f"API Request Params: {params}")
            
            response = requests.get(url, headers=headers, params=params)
            print(f"API Response Status Code: {response.status_code}")
            
            if response.status_code != 200:
                print(f"API Response Text: {response.text[:500]}...")
                if page_count > 0:  # If we have fetched at least one page, return what we have
                    break
                else:  # If we couldn't fetch even the first page, raise an exception
                    response.raise_for_status()
            
            try:
                data = response.json()
                tweets = data.get('tweets', [])
                
                # Debug for date filtering - check if tweets are within the requested date range
                if start_date and end_date and len(tweets) > 0:
                    first_tweet = tweets[0]
                    last_tweet = tweets[-1]
                    if 'tweet_created_at' in first_tweet:
                        print(f"First tweet date: {first_tweet['tweet_created_at']}")
                    if 'tweet_created_at' in last_tweet:
                        print(f"Last tweet date: {last_tweet['tweet_created_at']}")
                
                all_tweets.extend(tweets)
                print(f"Retrieved {len(tweets)} tweets from page {page_count + 1}")
                
                # Check if there's a next page
                cursor = data.get('next_cursor')
                if not cursor:
                    print("No more pages of tweets available")
                    break
                    
                page_count += 1
                pages_fetched += 1
                
                # Add a small delay to avoid rate limiting
                time.sleep(0.5)
                
            except ValueError as e:
                print(f"JSON Decode Error: {e}")
                break
    except Exception as e:
        print(f"Error fetching tweets: {str(e)}")
        if all_tweets:  # Return partial results if available
            print(f"Returning {len(all_tweets)} tweets despite error")
        else:
            raise  # Re-raise the exception if we have no tweets
    
    session['pages_fetched'] = pages_fetched
    print(f"Total tweets fetched: {len(all_tweets)}")
    return all_tweets

def get_user_id(username):
    url = f"https://api.socialdata.tools/twitter/user/{username}"
    headers = {'Authorization': f'Bearer {SOCIALDATA_API_KEY}', 'Accept': 'application/json'}
    print(f"Fetching User ID for: {username}")
    print(f"User ID Request URL: {url}")
    try:
        response = requests.get(url, headers=headers)
        print(f"User ID Response Status Code: {response.status_code}")
        print(f"User ID Response Text: {response.text[:500]}...")  # Limit to first 500 chars for brevity
        
        if response.status_code != 200:
            return None
            
        data = response.json()
        user_id = data.get('id_str', data.get('id', None))
        print(f"Resolved User ID: {user_id}")
        return user_id
    except (ValueError, KeyError, requests.RequestException) as e:
        print(f"Error resolving User ID: {e}")
        return None

def analyze_tweets(tweets, sensitivity=0.6, offensive_keywords=None):
    if offensive_keywords is None:
        offensive_keywords = DEFAULT_OFFENSIVE_KEYWORDS
    
    flagged = []
    print(f"Analyzing {len(tweets)} tweets with sensitivity threshold: {sensitivity}")
    
    for i, tweet in enumerate(tweets):
        # Skip processing every 20 tweets to display progress
        if i % 20 == 0:
            print(f"Processing tweet {i+1}/{len(tweets)}")
            
        text = tweet.get('full_text', '')
        if not text:  # Handle empty or None full_text
            continue
            
        # Sentiment analysis with our custom analyzer
        sentiment_result = sentiment_analyzer(text)
        
        # Print debug information for the first few tweets
        if i < 5:
            print(f"Tweet {i+1} sentiment result: {sentiment_result}")
            
        # Get the sentiment score
        sentiment_data = sentiment_result[0]
        sentiment_score = sentiment_data.get('score', 0)
        
        # Keyword detection from offensive_keywords list
        lower_text = text.lower()
        keyword_matches = []
        for keyword in offensive_keywords:
            keyword_lower = keyword.lower()
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', lower_text):
                keyword_matches.append(keyword)
        
        keyword_score = len(keyword_matches) * 0.15  # Slightly higher weight for keywords
        
        # Combined score
        combined_score = max(sentiment_score, keyword_score)
        
        # For debug - print scores for some tweets
        if i < 5 or keyword_score > 0 or sentiment_score > 0.2:
            print(f"Tweet: '{text[:50]}...' → Sentiment: {sentiment_score:.2f}, Keywords: {keyword_score:.2f}, Combined: {combined_score:.2f}")
            if keyword_matches:
                print(f"  Keywords found: {keyword_matches}")
        
        # Flag tweet if combined score is significant - using a lower threshold to catch more tweets
        if (sentiment_score > 0.1) or keyword_score > 0 or combined_score > 0.1:
            # Extract date from tweet_created_at
            tweet_date = ""
            if 'tweet_created_at' in tweet and tweet['tweet_created_at']:
                try:
                    # Properly parse the ISO 8601 format date from tweet_created_at
                    # Example format: "2023-12-13T05:39:09.000000Z"
                    created_at = tweet['tweet_created_at']
                    
                    # Print first few date formats for debugging
                    if i < 3:
                        print(f"Raw tweet_created_at: {created_at}")
                    
                    # Parse the date properly
                    if 'T' in created_at:
                        # Split at 'T' to get just the date part (YYYY-MM-DD)
                        date_part = created_at.split('T')[0]
                        # Split date into components
                        year, month, day = date_part.split('-')
                        # Format the date in a readable way
                        tweet_date = f"{month}/{day}/{year}"
                    else:
                        tweet_date = created_at
                except Exception as e:
                    print(f"Error parsing date: {str(e)} for date string: {tweet.get('tweet_created_at', 'N/A')}")
                    tweet_date = "Unknown date"
            else:
                tweet_date = "Unknown date"
            
            # For debug
            if i < 3:
                print(f"Extracted date: {tweet_date}")
            
            flagged.append({
                'text': text,
                'link': f"https://twitter.com/{tweet['user']['screen_name']}/status/{tweet['id_str']}",
                'sentiment_score': round(sentiment_score, 2),
                'keyword_score': round(keyword_score, 2),
                'combined_score': round(combined_score, 2),
                'keyword_matches': keyword_matches,
                'date': tweet_date
            })
    
    print(f"Analysis complete. Flagged {len(flagged)} tweets out of {len(tweets)}")
    
    # Sort by combined score (most offensive first)
    flagged.sort(key=lambda x: x['combined_score'], reverse=True)
    return flagged

if __name__ == '__main__':
    app.run(debug=True) 