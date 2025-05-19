import tensorflow as tf
import numpy as np

class HandGestureClassifier:
    def __init__(self, right_model_path, number_model_path,num_threads=1):
        # Initialize the model path and number of threads for inference
        self.right_model_path = right_model_path
        self.number_model_path = number_model_path

        self.num_threads = num_threads
        
        # Load the model during initialization
        self.right_interpreter = self.load_model(self.right_model_path)
        self.number_interpreter = self.load_model(self.number_model_path)

    def load_model(self, model_path):
        """Loads a TensorFlow Lite model."""
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=self.num_threads)
        interpreter.allocate_tensors()
        return interpreter

    def run_inference(self, landmark_list, mode):
        """Runs inference on a given input sample."""

        if mode == "Letter":
            interpreter = self.right_interpreter
        elif mode == "Number":
            interpreter = self.number_interpreter
        
        # Get input and output tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # **Split the input** into two parts:
        keypoints = np.array([landmark_list[:21*2]], dtype=np.float32)  # First 42 values (keypoints)
        global_features = np.array([landmark_list[21*2:]], dtype=np.float32)  # Last 4 values (global features)
        
        # **Set both input tensors**
        interpreter.set_tensor(input_details[0]['index'], keypoints)  
        interpreter.set_tensor(input_details[1]['index'], global_features)  
        
        # Run inference
        interpreter.invoke()
        
        # Get the model output
        result = interpreter.get_tensor(output_details[0]['index'])
        probabilities = np.squeeze(result)
        
        # Get the predicted class and confidence score
        result_index = np.argmax(probabilities)
        confidence = float(probabilities[result_index])

        return result_index, confidence
