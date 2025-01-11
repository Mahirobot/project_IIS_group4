import os
import pyaudio
import wave
from dotenv import load_dotenv  # type: ignore
import google.generativeai as genai  # type: ignore
from furhat_remote_api import FurhatRemoteAPI  # type: ignore
import ast
import time


# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

class FurhatBartender:
    def __init__(self):
        # Initialize Furhat
        self.furhat = FurhatRemoteAPI("localhost")
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        
        # Gemini configuration
        self.generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        # Initialize Gemini model for conversation
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=(
                "You are a flirty bartender. Consider each prompt is from the customer. "
                "Before the prompt, in brackets, there will be emotion of the customer. "
                "For example, a prompt can be like -\n\n\"(Sad) This day sucks.\"\n\n"
                "Evaluate the sentiment of the customer and classify among happy, sad, angry, or neutral. "
                "Add a confidence score to the sentiment."
                "Your reply should be a dictionary like this: {'emotion': 'happy', 'confidence': 0.8, 'response': 'I'm sorry to hear that. How can I help you?'}"
                "You will give responses as a bartender and will respond according to the mood of the customer. "
                "Your response will be short, maximum 2 lines. "
                "If the customer says goodbye, farewell, nothing more or similar, respond with a goodbye message and add '[END]' at the end of your response."
            ),
        )
        
        # Initialize chat session
        self.chat_session = self.model.start_chat(history=[])
        
        # Initialize audio recording setup
        self.audio_chunk_size = 1024  # Size of each audio chunk (buffer)
        self.audio_format = pyaudio.paInt16  # Audio format
        self.audio_channels = 1  # Mono audio
        self.audio_rate = 16000  # 16 kHz sample rate

        self.audio_filename = "user_audio.wav"  # File to save audio

    def save_audio(self, frames):
        """Save the audio frames to a .wav file"""
        with wave.open(self.audio_filename, 'wb') as wf:
            wf.setnchannels(self.audio_channels)
            wf.setsampwidth(pyaudio.PyAudio().get_sample_size(self.audio_format))
            wf.setframerate(self.audio_rate)
            wf.writeframes(b''.join(frames))
            print(f"Audio saved to {self.audio_filename}")

    def start_audio_recording(self):
        """Start recording audio from the Furhat microphone"""
        p = pyaudio.PyAudio()
        stream = p.open(format=self.audio_format,
                        channels=self.audio_channels,
                        rate=self.audio_rate,
                        input=True,
                        frames_per_buffer=self.audio_chunk_size)

        print("Recording audio...")
        frames = []

        # Record for a fixed duration (e.g., 5 seconds), adjust as needed
        start_time = time.time()
        while time.time() - start_time < 5:
            data = stream.read(self.audio_chunk_size)
            frames.append(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

        self.save_audio(frames)

    def get_response(self, prompt: str) -> str:
        """Get response from Gemini model"""
        response = self.chat_session.send_message(prompt)
        return response.text
    
    def detect_emotion(self, message: str) -> str:
        """Simple emotion detection (replace with sophisticated analysis later)"""
        return {'emotion': "angry", 'confidence': 0.5}
    
    def should_end_conversation(self, response: str) -> bool:
        """Check if the conversation should end"""
        return "[END]" in response
    
    def run_conversation(self):
        """Run the main conversation loop"""
        try:
            # Initial greeting
            self.furhat.say(text="Hello, I'm Alex, I'll be your bartender today. How can I help you?")
            self.furhat.gesture(name="Smile")
            
            vocal_emotion = "neutral"
            vocal_emotion_confidence = 0.5

            while True:
                # Listen for user input and save the audio
                self.start_audio_recording()  # Capture the audio being heard
                
                # Simulate user response (this should be replaced by actual speech recognition)
                user_response = self.furhat.listen()
                if not user_response or not user_response.message:
                    continue
                
                # Detect emotion and format prompt
                facial_emotion = self.detect_emotion(user_response.message)
                if vocal_emotion_confidence > facial_emotion.get('confidence', 0.5):
                    formatted_prompt = f"({vocal_emotion}) {user_response.message}"
                else:
                    formatted_prompt = f"({facial_emotion.get('facial_emotion', 'neutral')}) {user_response.message}"
                
                # Get and speak response
                bartender_response = self.get_response(formatted_prompt)

                # Remove the [END] tag before speaking
                speak_response = bartender_response.replace("[END]", "").strip()
                speak_response = ast.literal_eval(speak_response)
                print(speak_response)
                if type(speak_response) != dict:
                    speak_response = {'emotion': 'neutral', 'confidence': 0.5, 'response': "Sorry, didn't hear you. Can you say that one more time."}
                self.furhat.say(text=speak_response.get('response', "Sorry, didn't hear you. Can you say that one more time."))

                vocal_emotion = speak_response.get('emotion', 'neutral')
                vocal_emotion_confidence = speak_response.get('confidence', 0.5)

                print(f"User: {formatted_prompt}")
                print(f"Bartender: {bartender_response}")
                
                # Check if we should end the conversation
                if self.should_end_conversation(bartender_response):
                    break
                
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            self.chat_session = None
            self.model = None

def main():
    bartender = FurhatBartender()
    bartender.run_conversation()

if __name__ == "__main__":
    main()
