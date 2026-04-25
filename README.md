# TalkLens: Multimodal Assistive AI System

TalkLens is a production-ready, modular assistive AI application designed to empower individuals with visual and hearing impairments. By leveraging state-of-the-art computer vision and natural language processing, TalkLens provides real-time assistance through three core modules.

## 🌟 Core Features

### 1. Vision Assistance (for the Blind)
- **Object Detection & Scene Description**: Uses YOLOv8 and multimodal LLMs to identify objects and describe the environment.
- **Audio Feedback**: Provides instant spoken feedback about detected objects and their proximity.
- **Hands-Free Interface**: Robust background listener for voice-activated interaction.

### 2. Sign Language Recognition (for the Deaf/Mute)
- **Real-time Interpretation**: Translates American Sign Language (ASL) gestures into text and speech using MediaPipe and LSTM/Transformer models.
- **Dynamic Vocabulary**: Supports a wide range of gestures for seamless communication.

### 3. Voice Control & Interaction
- **Speech-to-Text**: High-accuracy transcription using OpenAI Whisper.
- **Intelligent Conversations**: Integrated AI chat for answering queries and providing guidance.

## 🛠 Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/) for a responsive, interactive UI.
- **Computer Vision**: [YOLOv8](https://ultralytics.com/yolov8), [MediaPipe](https://mediapipe.dev/).
- **AI Models**: [OpenAI GPT-4o](https://openai.com/), [Whisper](https://openai.com/research/whisper), [Google Gemini](https://ai.google.dev/).
- **Backend**: Python 3.9+
- **Sequence Modeling**: LSTM / Transformers.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Webcam and Microphone access

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/TalkLens.git
   cd TalkLens
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies**:
   ```bash
   pip install -r talklens/requirements.txt
   ```

4. **Configure environment variables**:
   Create a `.env` file based on `.env.example`:
   ```bash
   cp talklens/.env.example .env
   # Edit .env with your API keys
   ```

### Running the Application

Start the Streamlit app using the provided script:
```bash
./run_talklens.sh
```

## 📂 Project Structure

```text
TL/
├── talklens/
│   ├── app.py              # Main Streamlit application
│   ├── modules/            # Core logic (vision, sign language, speech)
│   ├── models/             # Pre-trained models and weights
│   ├── config/             # System settings and configuration
│   └── utils/              # Helper functions
├── run_talklens.sh         # Startup script
└── requirements.txt        # Project dependencies
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
Developed with ❤️ by Ganesh Dhere.
