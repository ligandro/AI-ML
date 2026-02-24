from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from card import process_url_to_flashcards

app = Flask(__name__)
CORS(app)


# Route to serve the frontend (e.g., React/Vue/HTML)
@app.route('/')
def serve_frontend():
    # Serve 'index.html' from the 'static' directory
    return send_from_directory('static', 'index.html')

# API endpoint to generate flashcards from a given URL
@app.route('/api/generate-flashcards', methods=['POST'])
def generate_flashcards():
    # Get JSON data from the request
    data = request.json
    
    # Validate that the URL is provided
    if not data or 'url' not in data:
        return jsonify({"error": "URL is required"}), 400  # 400 = Bad Request
    
    try:
        # Call the function to process the URL and generate flashcards
        # Default to 5 cards if 'num_cards' is not specified
        result = process_url_to_flashcards(data['url'], data.get('num_cards', 5))
        # Return the result as JSON
        return jsonify(result)
    except Exception as e:
        # Handle any errors and return a 500 Internal Server Error
        return jsonify({"error": str(e)}), 500

# Health check endpoint to verify the server is running
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})  # Simple response to confirm the API is live

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)