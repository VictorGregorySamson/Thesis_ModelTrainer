import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from tensorflow import keras
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.label import Label
from threading import Thread
from jnius import autoclass  # For Android bindings

# Import the PythonActivity for Android
#PythonActivity = autoclass('org.kivy.android.PythonActivity')

#activity = PythonActivity.mActivity

# Initialize Mediapipe Holistic
mp_holistic = mp.solutions.holistic
# model = keras.models.load_model("251ep-NOZ-INDEX-BEST.keras")

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="fsl_model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Recognized actions
actions = np.array([ 'GOOD MORNING', 'GOOD AFTERNOON', 'GOOD EVENING', 'HELLO', 
    'HOW ARE YOU', 'IM FINE', 'NICE TO MEET YOU', 'THANK YOU', 
    'YOURE WELCOME', 'SEE YOU TOMORROW', 'MONDAY', 'TUESDAY', 
    'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY', 
    'TODAY', 'TOMORROW', 'YESTERDAY', 'BLUE', 'GREEN', 'RED', 
    'BROWN', 'BLACK', 'WHITE', 'YELLOW', 'ORANGE', 'GRAY', 'PINK', 
    'VIOLET', 'LIGHT', 'DARK'])

def extract_keypoints(results):
    pose = np.array([[res.x, res.y] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 2)
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 2)
    rh = np.array([[res.x, res.y] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21 * 2)
    return np.concatenate([pose, lh, rh])

class GestureLayout(BoxLayout):
    """Main layout for Kivy app with Image and Label."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'

        # Kivy Image widget for camera feed
        self.img_widget = Image()
        self.add_widget(self.img_widget)

        # Label for recognized action
        self.action_label = Label(text="Gesture: ...", font_size='20sp', size_hint=(1, 0.1))
        self.add_widget(self.action_label)

        # Start camera feed
        self.cap = cv2.VideoCapture(0)
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.sequence = []
        self.recognized_action = None

        # Schedule the camera update function
        Clock.schedule_interval(self.update_frame, 1.0 / 30.0)

    def update_frame(self, dt):
        """Grabs a frame from the camera, processes it, and updates UI."""
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process the frame with MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)

        # Draw landmarks
        self.draw_landmarks(frame, results)

        # Prediction logic
        keypoints = extract_keypoints(results)
        self.sequence.append(keypoints)
        self.sequence = self.sequence[-120:]

        if len(self.sequence) == 120:
            # Prepare input data for TFLite
            input_data = np.expand_dims(self.sequence, axis=0).astype(np.float32)

            # Set the tensor and invoke the interpreter
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # Get the prediction result
            output_data = interpreter.get_tensor(output_details[0]['index'])[0]
            self.recognized_action = actions[np.argmax(output_data)]
            self.sequence = []

        # Update the action label
        self.action_label.text = f"Gesture: {self.recognized_action if self.recognized_action else '...'}"

        # Convert frame to Kivy texture and display it
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img_widget.texture = texture

    def draw_landmarks(self, frame, results):
        """Draws landmarks on the frame."""
        if results.pose_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        if results.left_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

    def on_stop(self):
        """Releases resources on app stop."""
        self.cap.release()
        self.holistic.close()

class GestureApp(App):
    def build(self):
        return GestureLayout()

# Run the app
if __name__ == '__main__':
    GestureApp().run()
