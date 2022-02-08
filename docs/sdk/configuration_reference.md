.. meta::
   :description: Basic configurations necessary to use the sdk: have a project in the Konfuzio Web App with some basic settings and initialize the konfuzio-sdk correctly.


# Install SDK

To test our SDK you need to have an account in the Konfuzio Server and initialize the package before using it. Just follow the instructions below to set yourself ready for the SDK.

## 1. Sign up in Konfuzio Server

[Register for free](https://app.konfuzio.com/accounts/signup/) in the Konfuzio Server.

### Some words about the data structure

The data structure of a project in Konfuzio Server includes:

* Labels  
A label is the name of a group of individual pieces of information found in a type of document.

* Label Set  
A Label Set combines labels.

* Training and Test documents  
Based on the data from the training documents the AI learns what to do. These newly acquired capabilities are then applied to the test documents to sample check the quality of the AI model.

* Annotations for documents  
An annotation is a single piece of information that a label has been assigned to.


## 2. Install `konfuzio_sdk` package

To use `konfuzio_sdk` on your machine you can install it via:

### Option 1

* Install the Python package directly in your working directory with:

  `pip install konfuzio_sdk`

*Notes*:
* Your coding environment should have a Python version >= 3.6.
* If you are not using a virtual environment, you may need to add the installation directory to your PATH.

## 3. Initialize the package

After the installation, initialize the package in your working directory with:

`konfuzio_sdk init`

This will require your credentials to access the Konfuzio Server and the project ID.
You can check your project ID by selecting the project in the Projects tab. The id of the project is shown in the URL.
It will also require a name of a folder where to allocate the data from your Konfuzio project.
At the end, two files will be created in your working directory: `.env` and `settings.py`.

The `.env` file contains the credentials to access the app and should not become public.
The `settings.py` file defines constant variables that will be available in the project, including the ones you defined in the `.env`. This file should not be modified.

## 4. Download the data

To download the data from your Konfuzio project, you can execute:

`konfuzio_sdk download_data`

The data from the documents that you uploaded in your Konfuzio project will be downloaded to the folder that you provided in the previous step.

*Note*:
Only documents in the training or test set are downloaded.
