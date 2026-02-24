import requests
from bs4 import BeautifulSoup
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
import warnings
import subprocess
import ollama



# Suppress SSL warnings for testing
warnings.filterwarnings("ignore", message="Unverified HTTPS request")


# NLTK setup
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class ContentProcessor:
    def __init__(self):
        """Initialize the ContentProcessor with a set of English stop words."""
        self.stop_words = set(stopwords.words('english'))

    def is_youtube_url(self, url):
        """Check if the given URL is a YouTube link.

        Args:
            url (str): The URL to check.

        Returns:
            bool: True if the URL is a YouTube link, False otherwise.
        """
        parsed = urlparse(url)
        return any(domain in parsed.netloc for domain in ['youtube.com', 'youtu.be'])

    def get_youtube_id(self, url):
        """Extract the YouTube video ID from a given URL.

        Args:
            url (str): The YouTube video URL.

        Returns:
            str or None: The extracted video ID or None if not found.
        """
        parsed = urlparse(url)
        if parsed.netloc == 'youtu.be':
            return parsed.path[1:]
        if 'youtube.com' in parsed.netloc:
            if 'v' in parse_qs(parsed.query):
                return parse_qs(parsed.query)['v'][0]
            if parsed.path.startswith('/embed/'):
                return parsed.path.split('/')[2]
            if parsed.path.startswith('/watch/'):
                return parsed.path.split('/')[2]
        return None

    def extract_youtube_content(self, url):
        """Extracts the transcript and title of a YouTube video.

        Args:
            url (str): The YouTube video URL.

        Returns:
            dict: A dictionary containing the video title, transcript content, and URL.
        """
        try:
            video_id = self.get_youtube_id(url)
            if not video_id:
                return {"error": "Invalid YouTube URL", "url": url}

            api = YouTubeTranscriptApi()
            transcript_obj = api.fetch(video_id)
            transcript = transcript_obj.to_raw_data()
            content = ' '.join([entry['text'] for entry in transcript])

            # Get video title
            embed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
            response = requests.get(embed_url)
            title = response.json().get('title', f"Video {video_id}")

            return {
                "title": title,
                "content": content,
                "url": url
            }
        except Exception as e:
            return {"error": f"YouTube error: {str(e)}", "url": url}

    def extract_web_content(self, url):
        """Extracts text content from a given web page URL.

        Args:
            url (str): The web page URL.

        Returns:
            dict: A dictionary containing the page title, extracted content, and URL.
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept-Language': 'en-US,en;q=0.9'
            }

            response = requests.get(url, headers=headers, timeout=10, verify=False)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')

            # Remove unnecessary elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'form', 'button', 'iframe', 'noscript']):
                element.decompose()

            # Extract content
            article = soup.find('article') or soup.find('main') or soup.find('div', role='main')
            if article:
                content = article.get_text(separator='\n', strip=True)
            else:
                body = soup.find('body')
                paragraphs = body.find_all(['p', 'h1', 'h2', 'h3']) if body else []
                content = '\n'.join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])

            content = re.sub(r'\n{3,}', '\n\n', content.strip())
            return {
                "title": soup.title.string.strip() if soup.title else "Untitled",
                "content": content,
                "url": url
            }
        except Exception as e:
            return {"error": f"Web extraction error: {str(e)}", "url": url}

    def extract_content_from_url(self, url):
        """Determines whether a URL is from YouTube or a webpage and extracts content accordingly.

        Args:
            url (str): The URL to extract content from.

        Returns:
            dict: A dictionary containing extracted content.
        """
        if self.is_youtube_url(url):
            return self.extract_youtube_content(url)
        return self.extract_web_content(url)

    def extract_key_sentences(self, content, max_sentences=20):
        """Extracts the most important sentences from the given content.

        Args:
            content (str): The text content to process.
            max_sentences (int, optional): The maximum number of key sentences to return. Defaults to 20.

        Returns:
            list: A list of extracted key sentences.
        """
        sentences = sent_tokenize(content)
        if not sentences:
            return []

        word_freq = {}
        for sentence in sentences:
            words = sentence.lower().split()
            for word in words:
                if word not in self.stop_words and len(word) > 3:
                    word_freq[word] = word_freq.get(word, 0) + 1

        sentence_scores = {}
        for i, sentence in enumerate(sentences):
            words = sentence.lower().split()
            score = sum(word_freq.get(word, 0) for word in words if word not in self.stop_words)
            if len(words) > 0:
                score /= len(words)
            sentence_scores[i] = score

        top_indices = sorted(sentence_scores.keys(), key=lambda i: sentence_scores[i], reverse=True)[:max_sentences]
        top_indices.sort()
        return [sentences[idx] for idx in top_indices]

class FlashcardGenerator:
    def generate_flashcards_via_ollama(self, content_data, num_cards=5):
        """Generates flashcards using an Ollama LLM model.

        Args:
            content_data (dict): Dictionary containing 'title' and 'content' keys.
            num_cards (int, optional): The number of flashcards to generate. Defaults to 5.

        Returns:
            list: A list of dictionaries with 'question' and 'answer' pairs.
        """
        system_message = """You are an educational content expert. Create flashcards in this EXACT format:
                            [
                                {"question": "...", "answer": "..."},
                                // more items
                            ]
                            ONLY output a valid JSON array with question/answer pairs. Follow these rules:
                            1. Questions should test understanding of key concepts.
                            2. Answers must be concise but complete.
                            3. Never add explanations or markdown.
                            4. Ensure the output is properly enclosed in {} brackets.
                            5. Ensure the JSON array starts with [ and ends with ].
                            6. Generate exactly the requested number of flashcards."""
        
        user_message = f"Content Title: {content_data['title']}\nContent Excerpt:\n{content_data['content']}\n\nGenerate exactly {num_cards} flashcards:"
        
        # Create the prompt for Ollama
        prompt = f"{system_message}\n{user_message}"

        try:
            # Initialize the Ollama client
            client = ollama.Client()

            # Define the model and the input prompt
            model = "mistral"  # Replace with your actual model name

            # Send the query to the model
            response = client.generate(model=model, prompt=prompt)

            # Process JSON response
            result = response.response.strip()
            flashcards = json.loads(result)  # Attempt to parse JSON

            # Validate response type
            if not isinstance(flashcards, list):
                raise ValueError("Response is not a JSON array")

            return flashcards

        except json.JSONDecodeError as e:
            print(f"JSON response parsing error:\n{result}")
            return [{"question": "Format Error", "answer": f"Invalid JSON format: {str(e)}"}]
        except Exception as e:
            return [{"question": "Generation Error", "answer": str(e)}]

def process_url_to_flashcards(url, num_flashcards=5):
    processor = ContentProcessor()
    generator = FlashcardGenerator()
    
    content_data = processor.extract_content_from_url(url)
    if "error" in content_data:
        error = content_data["error"]
        if "YouTube" in error:
            error += ". Ensure captions are enabled."
        return {"error": error, "title": "Error", "url": url, "flashcards": []}
    
    key_sentences = processor.extract_key_sentences(content_data["content"])
    if not key_sentences:
        return {"error": "No meaningful content found", "title": content_data["title"], "url": url, "flashcards": []}
    
    content_data["content"] = " ".join(key_sentences)
    flashcards = generator.generate_flashcards_via_ollama(content_data, num_flashcards)
    return {
        "title": content_data["title"],
        "url": content_data["url"],
        "flashcards": flashcards
    }