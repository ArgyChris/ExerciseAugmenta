# Augmenta exercise

This project is about analysing a .csv file from an experiment and training an ML model to perform predictions on the data. The experiment setup consists of a virtual camera and a dedicated ambient light sensor. During the experiment the camera exposure time is varied and an an image is captured, whose average pixel value is recorded. The value read by the ambient light sensor is recorded as well. The ambient light of the scene changes gradually throughout the experiment.

-----

### Prerequisites
    
        Python 
        Numpy
        Pandas
        Keras
        Tensorflow
        Seaborn
        Flask
        Scipy
        Scikit-learn
        Matplotlib
        Pickle

        Conda virtual environment

------
### Virtual environment

        I did the development in a linux environment.
        To install the required packages using the provided requirements.txt file:
        conda create --name test_camera_env
        conda activate test_camera_env

        To install the specific packages in the virtual environment and versions that I used:
        pip3 install -r requirements.txt
        For me Scipy package was not getting installed properly from the requirements.txt so I installed it manually.

------
### Running the scripts

To run the data analysis script, use the following command:

python data_analysis.py --input_csv *input_csv_file* --output_csv *output_csv_file* --output_plot *output_plot_file* --output_plot2 *output_plot2_file*

        --input_csv_file: Path to the input CSV file.
        --output_csv_file: Path to save the output CSV file.
        --output_plot_file: Path to save the 1st output plot file.
        --output_plot2_file: Path to save the 2nd output plot file.

To run the model loading and inference script:

python model_loading.py --target_pixel_value *target_pixel_value* --target_sensor_value *target_sensor_value*

    --target_pixel_value: Targeted average pixel value.
    --target_sensor_value: Targeted ambient light sensor illuminance value.

------
### Author

Argyrios Christodoulidis
29 June 2023

