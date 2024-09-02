import numpy as np
import serial
import torch
from src.model import *


def real_time_test(model, ser, device, num_points=200):
    """
    Real-time testing function. Reads data from the serial port, preprocesses it, and predicts using the model.
    :param model: The trained model
    :param num_points: Number of data points used for prediction, default is 200
    """
    label_mapping = {
        0: "Rest",
        1: "Thumb",
        2: "Index",
        3: "Little",
    }
    # Set the model to evaluation mode
    model.eval()

    try:
        while True:
            # Read data
            data_buffer = []
            ser.flushInput()  # Clear serial port buffer
            ser.readline()  # Discard the first data point

            while len(data_buffer) < num_points:
                data_point = ser.readline().decode('utf-8').strip()  # Read a data point
                if data_point:
                    data_buffer.append(float(data_point))  # Add data point to buffer

            # Data preprocessing
            data_array = np.array(data_buffer).reshape(1, -1)  # Convert to numpy array and reshape
            data_tensor = torch.tensor(data_array, dtype=torch.float32).to(device).view(1, 1,-1)  # Convert to tensor and reshape

            # Make prediction using the model
            with torch.no_grad():
                outputs = model(data_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()

            # Print prediction result
            #print(f"Predicted class: {label_mapping[predicted_class]}")
            print(predicted_class)

    except KeyboardInterrupt:
        print("Real-time testing stopped.")


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ser = serial.Serial('/dev/ttyUSB0', 921600)
    model = ResNet(num_classes=4, window_size=400).to(device)
    model.load_state_dict(torch.load('trained/model_469_0.9925.pt'))
    model.eval()

    real_time_test(model, ser, device, num_points=400)