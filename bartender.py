import os
from dotenv import load_dotenv  # type: ignore
import google.generativeai as genai  # type: ignore
from furhat_remote_api import FurhatRemoteAPI  # type: ignore
import ast
import time
import cv2
from facial import open_webcam
from queue import Queue
from feat import Detector
import threading
import pickle
# furhat_say_stop_post_with_http_info

# Load environment variables
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
vocal_weight = float(os.getenv("VOCAL_WEIGHT"))
facial_weight = float(os.getenv("FACIAL_WEIGHT"))

class FurhatBartender:
    def __init__(self):
        # Initialize Models
        with open('model_valence.pkl', 'rb') as f:
            self.model_valence = pickle.load(f)
        with open('model_arousal.pkl', 'rb') as f:
            self.model_arousal = pickle.load(f)
        with open('model_emotion.pkl', 'rb') as f:
            self.model_emotion = pickle.load(f)
         # Initialize Furhat
        self.furhat = FurhatRemoteAPI("localhost")
        self.face_tracker = cv2.CascadeClassifier("frontal_face_features.xml")
        self.path = "temp_face.jpg" 
        self.result_queue = Queue()
        self.frame = None
        self.detector = Detector()
        
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
        
        # Initialize Gemini model
        self.bartender = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=(
                "You are a bartender. Your sole job is to understand, analyze and respond to the customer. In addition, keep track of the conversation."
                "You will be given a dictionary like this: {'response': 'I had a good day', 'emotion': 'happy', 'confidence': 0.8, 'guide': 'excited'}"
                "Here is the drink menu. Margarita, Mojito, Ginger Ale, Chamomile Tea, Gin and Tonic."
                "Margarita - For users who are excited and full of energy (high valence and high arousal)."
                "Mojito - Perfect for those feeling relaxed and content (high valence, low arousal)."
                "Ginger Ale - A calming option for users who might be stressed or frustrated (low valence, high arousal)."
                "Chamomile Tea - Ideal for those feeling down and tired (low valence, low arousal)."
                "Gin and Tonic - A classic choice for users with a neutral mood."
                "The guide is from a sentiment analyzer model, so you can understand the situation better. Use the detected emotion, guide and response to respond to the customer."
                "You will give responses as a bartender and will respond according to the mood of the customer. "
                "Your response will be short, maximum 2 lines."
                "If the customer says goodbye, farewell, or similar, respond with a goodbye message and add '[END]' at the end of your response."
                "If the customer says something that you don't understand, respond with 'Sorry, I didn't understand that. Can you say that again?'"
                "If the customer says something that you can't respond to, respond with 'Sorry, I can't help with that. Can I help you with something else?'"
                "If the customer says Thanks, thank you or something similar, ask a follow up question like 'will that be all?' then understand the situation and respond with a goodbye message and add '[END]'"
                "Don't reply with emojis or any special characters. Only text."
                "At the start and after every 4 interactions, mention the emotion subtly. Example: 'I can understand your sadness.'"
                "If the customers mood changes after your suggestions, make sure to acknowledge that sometimes."
            ),
        )

        self.sentiment_analyzer = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config=self.generation_config,
            system_instruction=(
                "You are a sentiment analysis model. Your job is to unnderstand and track the sentiment of the customer and provide information to another AI model, who will respond to the customer."
                "You will be given a text from a customer. Analyze the text and understand the sentiment."
                "Keep track of the conversation and state of the customer. Understand what the reply should be from a bartender."
                "Add a 'guide' to the response to help the AI bartender understand the situation and give an educated response. It should only be seperated words not sentences. Example: 'excited, party mood'"
                "Evaluate the sentiment of the customer and classify among happy, sad, angry, surprise, fear, disgust, or neutral. "
                "Add a confidence score to the sentiment."
                "Return a gesture from this list: 'Smile', 'Wink', 'ExpressSad', 'CloseEyes', 'Nod', Neutral'"
                "Return a dictionary with 'emotion' and 'confidence' values. Example: {'emotion': 'happy', 'confidence': 0.8, 'guide': '', 'gesture': 'Smile'}"
                "Make sure to always return the dictionary in text format. '{'emotion': 'happy', 'confidence': 0.8, 'guide': '', 'gesture': 'Smile'}'"
                
            ),
        )
        
        # Initialize chat session
        self.chat_session = self.bartender.start_chat(history=[])
        self.sentiment_session = self.sentiment_analyzer.start_chat(history=[])


    def get_response(self, prompt: str, textual_emotion) -> str:
        """Get response from Gemini model"""
        if 'guide' not in textual_emotion:
            textual_emotion['guide'] = textual_emotion["emotion"]
        if 'gesture' in textual_emotion:
            del textual_emotion['gesture']
        textual_emotion['response'] = prompt

        response = None
        while response is None:
            try:
                response = self.chat_session.send_message(str(textual_emotion))
            except Exception as e:
                response = None
                self.chat_session = self.bartender.start_chat(history=[])

        return response.text
    
    def textual_emotion_analysis(self, text: str) -> dict:
        """Analyze the emotion of a text"""
        response = None
        while response is None:
            try:
                response = self.sentiment_session.send_message(text)
            except Exception as e:
                response = None
                self.sentiment_session = self.sentiment_analyzer.start_chat(history=[])
        
        # print('A', response.text)
        return response.text
    
    def detect_emotion(self) -> str:
        """
        Simple emotion detection - can be replaced with more sophisticated analysis
        Returns a default emotion of 'angry' for now
        """
        default_result = {"emotion": 0, "confidence": 0.1}  # Default result

        result = None
        # Check if the result_queue has any results
        if not self.result_queue.empty():
            result = self.result_queue.get()
            # if result is not None:
            #     print(f"Detected Emotion: {result['emotion']}, Confidence: {result['confidence']:.2f}")
            # else:
            #     print("No result received.")
        
        # If result is still the default value, you can log or handle that case as needed
        # Return the result, which will either be the one from the queue or the default
        mapped_emotions = { 0: "neutral", 1: "happy", 2: "sad", 6: "angry", 3: "surprise", 4: "fear", 5: "disgust"}
        
        if result:
            result['emotion'] = mapped_emotions[result.get('emotion', 'neutral')]
        return result if result is not None else default_result
    
    def should_end_conversation(self, response: str) -> bool:
        """Check if the conversation should end"""
        return "[END]" in response

    def emotion_calculation(self, textual_emotion, facial_emotion):
        vocal_emotion_confidence = textual_emotion.get('confidence', 0.5)
        vocal_emotion = textual_emotion.get('emotion', 'neutral')
        sentiment_guide = textual_emotion

        facial_emotion['confidence'] = (facial_weight * facial_emotion.get('confidence', 0.5))
        vocal_emotion_confidence = (vocal_weight * vocal_emotion_confidence)

        if 0.5 < facial_emotion.get('confidence', 0.5):
            sentiment_guide = facial_emotion
        return sentiment_guide

    def run_conversation(self):
        """Run the main conversation loop"""
        stop_event = threading.Event()
        webcam_thread = threading.Thread(
            target=open_webcam,
            args=(self.result_queue, self.path, self.face_tracker, self.detector, self.model_valence, self.model_arousal, self.model_emotion, 15, stop_event)
        )
        webcam_thread.start()
        time.sleep(5)
        try:
            # Initial greeting
            self.furhat.say(text="Hello, I'm Alex. I'll be your bartender today. How can I help you?", blocking=True)
            self.furhat.gesture(name="Smile")
            
            while True:
                t1 = time.time()
                user_response_bool = False
                # Listen for user input
                while user_response_bool == False:
                    user_response = self.furhat.listen()
                    user_response_bool = user_response.message not in ["", " ", None]
                    # print(user_response)
                                
                # Get and speak response
                textual_emotion = self.textual_emotion_analysis(user_response.message)
                textual_emotion = ast.literal_eval(textual_emotion)
                # print("Vocal emotion", textual_emotion)

                gesture = textual_emotion.get('gesture', 'Neutral')

                # Detect emotion and format prompt
                facial_emotion = self.detect_emotion()
                # print('Facial Emotion', facial_emotion)
                sentiment_guide = self.emotion_calculation(textual_emotion, facial_emotion)
                # print('Sentiment guide', sentiment_guide)
                
                if type(sentiment_guide) != dict:
                    sentiment_guide = {'emotion': 'neutral', 'confidence': 0.5, 'guide': "Apologize and ask to repeat what they said."}

                bartender_response = self.get_response(user_response.message, sentiment_guide)
                # print('Bartender Response', bartender_response)
                # Remove the [END] tag before speaking
                speak_response = bartender_response.replace("[END]", "").strip()
                if speak_response == "":
                    self.furhat.say(text=speak_response.get('response', "Sorry, didn't hear you. Can you say that one more time.", blocking=True))
                if gesture in ['Smile', 'Wink', 'ExpressSad', 'CloseEyes', 'Nod']:
                    self.furhat.gesture(name=gesture)

                self.furhat.say(text=speak_response, blocking=True)
                # Save conversation
                print(f"User: {user_response.message}")
                print(f"Bartender: {bartender_response}")
                
                # Check if we should end the conversation
                if self.should_end_conversation(bartender_response):
                    break
                print(time.time() - t1)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
        finally:
            # Cleanup
            self.chat_session = None
            self.bartender = None
            self.sentiment_analyzer = None
            stop_event.set()

def main():
    bartender = FurhatBartender()
    bartender.run_conversation()

if __name__ == "__main__":
    main()