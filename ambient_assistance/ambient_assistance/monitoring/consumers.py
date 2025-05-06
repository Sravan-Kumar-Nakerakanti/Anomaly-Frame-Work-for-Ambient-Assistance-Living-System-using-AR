# monitoring/consumers.py

import json
from channels.generic.websocket import AsyncWebsocketConsumer
import cv2
import numpy as np
import tensorflow as tf
import base64
from channels.generic.websocket import WebsocketConsumer
from channels.exceptions import StopConsumer

class TaskConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        data = json.loads(text_data)
        task = data['task']
        # Handle task update
        await self.send(text_data=json.dumps({
            'message': f'Task received: {task}'
        }))


model = tf.keras.models.load_model(r'C:/Users/SRAVA/action_recog.h5')
categories = ['wave', 'walk', 'turn', 'throw', 'talk', 'stand', 'smile', 'situp', 'sit', 'shake_hands',
              'run', 'push', 'punch', 'pour', 'pick', 'laugh', 'jump', 'hug', 'hit', 'handstand',
              'fall_floor', 'eat', 'drink', 'dribble', 'climb_stairs', 'climb', 'clap', 'chew',
              'catch', 'brush_hair']

class LiveMonitoringConsumer(WebsocketConsumer):
    def connect(self):
        self.accept()

    def disconnect(self, close_code):
        raise StopConsumer()

    def receive(self, text_data):
        frame_data = text_data.split(",")[1]
        frame_bytes = base64.b64decode(frame_data)
        np_arr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        action = self.predict_action(frame)
        self.send(text_data=json.dumps({
            'action': action
        }))

    def preprocess_frame(self, frame):
        resized_frame = cv2.resize(frame, (224, 224))
        normalized_frame = resized_frame / 255.0
        return normalized_frame

    def predict_action(self, frame):
        preprocessed_frame = self.preprocess_frame(frame)
        predictions = model.predict(np.expand_dims(preprocessed_frame, axis=0))
        predicted_class = np.argmax(predictions, axis=1)
        return categories[predicted_class[0]]

