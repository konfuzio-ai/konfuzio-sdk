## Install SDK

To test our SDK you need to have an account in the Konfuzio Server and initialize the package before using it. If you 
are using PyCharm have a look at [Quickstart with Pycharm](quickstart_pycharm.html).

### 1. Sign up in Konfuzio Server

[Register for free](https://app.konfuzio.com/accounts/signup/) in the Konfuzio Server.

### 2. Install `konfuzio_sdk` package

* Install the Python package directly in your working directory with:

  `pip install konfuzio_sdk`

* It is also possible to choose between the lightweight SDK and the SDK with the AI-related components (latter one is 
taking up more disk space). By default, the SDK is installed as a lightweight instance. To install the full instance,
run the following command:

  `pip install konfuzio_sdk[ai]`
  
  Currently, the full instance cannot be installed on MacOS machines with an ARM-based chip from the M-series. The `konfuzio_sdk` package can only be installed on MacOS on machines with an ARM chip if the lightweight instance is installed. However the Konfuzio SDK can be used on a hosted environment such as Google Colab. Follow the instructions in the next section to install the SDK in Colab.

### 2.1 Install the SDK in Google Colab
It is possible to use the SDK in [Google Colab](https://colab.research.google.com/) notebook regardless of the operating system of your computer. To install the SDK in Colab, run the following commands in a new Notebook cell:

  ```
  !pip install konfuzio_sdk[ai]
  !cd konfuzio-sdk && pip install .[ai]
  !sed -i 's/certifi==2022\.12\.7/certifi==2023.7.22/' konfuzio-sdk/setup.py
  ```
  In a new cell run `!cd konfuzio-sdk/ && cat setup.py` to check we have the right version of the `certifi` library. 
  
  Inspect the output of the command, at the bottom of it you should see the following:
  ```
  setuptools.setup(
    ...
    install_requires=[
        'certifi==2023.7.22',
        ...
    ],
  ``````
  Make sure the version of the `certifi` package is indeed `2023.7.22`. Now restart the Colab runtime by choosing `Runtime` -> `Restart runtime...` in the menu bar. After the restart, the SDK can be initialized in the Colab notebook, run in a new cell:
  ```
  import konfuzio_sdk
  !konfuzio_sdk init
  ```
  Follow the instructions in the terminal to initialize the SDK. The SDK is now ready to be used in the Colab notebook.

*Notes*:

* Supported Python environments are 3.8, 3.9, 3.10, 3.11.
* Please use Python 3.8 if you plan to upload your AIs to a self-hosted Konfuzio Server environment. 
* If you are not using a virtual environment, you may need to add the installation directory to your PATH.
* If you run this tutorial in Colab and experience any version compatibility issues when working with the SDK, restart 
the runtime and initialize the SDK once again; this will resolve the issue.

### 3. Initialize the package

After the installation, initialize the package in your working directory with:

`konfuzio_sdk init`

This will require your credentials to access the Konfuzio Server.
At the end, one file will be created in your working directory: `.env`.

The `.env` file contains the credentials to access the app and should not become public.

### 4. Download the data

To download the data from your Konfuzio project you need to specify the Project ID.
You can check your Project ID by selecting the project in the Projects tab in the Web App.
The ID of the Project is shown in the URL. Suppose that your Project ID is 123:

`konfuzio_sdk export_project 123`

The data from the documents that you uploaded in your Konfuzio project will be downloaded to a folder called `data_123`.

*Note*:
Only Documents in the Training and Test sets are downloaded.
