# Simple Chatbot

This repository demonstrates a simple AI chatbot with real-time audio/video interaction, implemented using different client and server options. The bot server supports multiple AI backends, and you can connect to it using five different client approaches.

## Two Bot Options

1. **OpenAI Bot** (Default)

   - Uses gpt-4o for conversation
   - Requires OpenAI API key

2. **Gemini Bot**
   - Uses Google's Gemini Multimodal Live model
   - Requires Gemini API key


## Quick Start

### First, start the bot server:

1. Navigate to the server directory:
   ```bash
   cd server
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy env.example to .env and configure:
   - Add your API keys
   - Choose your bot implementation:
     ```ini
     BOT_IMPLEMENTATION=      # Options: 'openai' (default) or 'gemini'
     ```
5. Start the server:
   ```bash
   python server.py
   ```

### Next, connect using the client app:

- [React Guide](client/react/README.md)

## Important Note

The bot server must be running for any of the client implementations to work. Start the server first before trying any of the client apps.

## Requirements

- Python 3.10+
- Node.js 16+ (for JavaScript and React implementations)
- Daily API key
- OpenAI API key (for OpenAI bot)
- Gemini API key (for Gemini bot)
- ElevenLabs API key
- Modern web browser with WebRTC support

## Project Structure

```
simple-chatbot/
├── server/              # Bot server implementation
│   ├── bot-openai.py    # OpenAI bot implementation
│   ├── bot-gemini.py    # Gemini bot implementation
│   ├── runner.py        # Server runner utilities
│   ├── server.py        # FastAPI server
│   └── requirements.txt
└── client/              # Pipecat React client
```
