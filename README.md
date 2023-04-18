# Face Recognition Boiler Plate Code

This repository contains a boilerplate code for getting started with the Face Recognition Udemy course by Anis Koubaa. The code provides a starting point for building your own face recognition application and ensures that you have all the necessary requirements to get started.

## Requirements

You can find the list of requirements in the following files:

- `requirements_face.txt`: A pip requirements file that includes all the necessary dependencies.
- `environment_face.yaml`: A conda environment file that includes the necessary packages.

Note that it is important to install the same version of TensorFlow and other packages to ensure that the face recognition application works correctly.

## Running the Code

To use the boilerplate code, follow these steps:

1. Clone the repository to your local machine.
2. Place the file `facenet_keras_128.h5` inside the `models` folder. This file is necessary to run the embedding function for the face.

`pip install -r requirements_face.txt`

or 

`conda env create -f environment_face.yaml`

3. Run the `test_environment.py` file to test that everything is working correctly.

## Note on Google Colab

Unfortunately, Google Colab no longer supports older versions of TensorFlow such as TensorFlow 2.0. Therefore, it is required to work locally on your machine to use this boilerplate code.

## Conclusion

We hope that this boilerplate code provides a helpful starting point for building your own face recognition application. If you have any questions or issues, please feel free to open an issue on GitHub.
