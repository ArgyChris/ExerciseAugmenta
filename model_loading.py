import argparse
from tensorflow import keras
import numpy as np
from pickle import load

def validate_target_pixel_value(value):
    value = float(value)
    if value < 3 or value > 255:
        raise argparse.ArgumentTypeError("target_pixel_value must be between 3 and 255")
    return value

def target_sensor_value(value):
    value = float(value)
    if value < 2072 or value > 20115:
        raise argparse.ArgumentTypeError("target_pixel_value must be between 2072 and 20115")
    return value

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Model loading and inference')
    parser.add_argument('--target_pixel_value', type=validate_target_pixel_value, default=77, help='Targeted average pixel value (3-255)')
    parser.add_argument('--target_sensor_value', type=target_sensor_value, default=11000, help='Targeted ambient light sensor illuminance value (2072-20115)')

    args = parser.parse_args()

    # load the model
    model = keras.models.load_model('model_LSTM_85_units_22_dropout_500_epochs.h5')
    # load the scaler
    with open('scaler.pkl', 'rb') as file:
        scaler = load(file)

    #Use the scaler to normalize the input data
    Input_scaled = scaler.transform(np.array([float(args.target_pixel_value), float(args.target_sensor_value), float(0.0)]).reshape(1, 3))[0]
   
    #Create a window with a first value the input
    input_array = np.zeros((1, 167, 2))
    input_array[:, -1, :] = Input_scaled[:2]
    prediction = model.predict(input_array)

    #The output is a window and we take the average to rescale it to the original range
    Output_scaled = scaler.inverse_transform(np.array([float(args.target_pixel_value), float(args.target_sensor_value), np.mean(prediction[0, :, 0])]).reshape(1, 3))[0]

    print('Predicted exposure time:', Output_scaled[-1])