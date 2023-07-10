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
  

*Notes*:

* Supported Python environments are 3.8, 3.9, 3.10, 3.11.
* Please use Python 3.8 if you plan to upload your AIs to a self-hosted Konfuzio Server environment. 
* If you are not using a virtual environment, you may need to add the installation directory to your PATH.

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
