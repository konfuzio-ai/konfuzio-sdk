## Install SDK

To test our SDK you need to have an account in the Konfuzio Server and initialize the package before using it. \
If you are using PyCharm have a look at [Quickstart with Pycharm](quickstart_pycharm.html) and if you like to get started within a Google Colab, checkout the getting started notebook here. <a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Get_started_with_the_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


### 1. Sign up in Konfuzio Server

[Register for free](https://app.konfuzio.com/accounts/signup/) in the Konfuzio Server.

### 2. Install `konfuzio_sdk` package

* Install the Python package directly in your working directory with:

  `pip install konfuzio_sdk`

* It is also possible to choose between the lightweight SDK and the SDK with the AI-related components (latter one is 
taking up more disk space). By default, the SDK is installed as a lightweight instance. To install the full instance (`.[ai]`), the one with AI-related components, run the following command:

  `pip install konfuzio_sdk[ai]`
  
  Currently, the full instance cannot be installed on MacOS machines with an ARM-based chip from the M-series. The `konfuzio_sdk` package can only be installed on MacOS on machines with an ARM chip if the lightweight instance is installed. However the Konfuzio SDK can be used on a hosted environment such as [Google Colab](https://colab.research.google.com/). For the installation and usage within colab, you can follow the getting started notebook here. <a href="https://colab.research.google.com/github/konfuzio-ai/konfuzio-sdk/blob/master/notebooks/Get_started_with_the_Konfuzio_SDK.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

---

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

As an alternative you can pass your credentials via command line argument.

```bash
konfuzio_sdk init --user your.email@konfuzio-server.com --password YourPassword --host https://app.konfuzio.com
```

### 4. Usage

The following code snipped serves as a quick example for the SDK usage.

```python
from konfuzio_sdk.data import Project
# get project from server
project = Project(id_=YOUR_PROJECT_ID)
# list project documents
documents = project.documents
for doc in documents:
  print(doc)
```

To dive deeper, look at the [API Reference](sourcecode.html), which documents the SDKs functionality.
You can also checkout the [SDKs source code](https://github.com/konfuzio-ai/konfuzio-sdk) directly on Github.
 

### 5. CLI usage

Additional to the usage as python package a few function are also available from within the command line (CLI).

Run the following to get an overview of the CLI functionality:

```bash
konfuzio_sdk --help
```

#### Download data with `export_project`

To download data from your Konfuzio Project you need to specify the Project ID.
You can check your Project ID by selecting the Project in the Projects tab in the Web App.
The ID of the Project is shown in the URL. Suppose that your Project ID is 123:

`konfuzio_sdk export_project 123`

The data from the documents that you uploaded in your Konfuzio Project will be downloaded to a folder called `data_123`.

*Note*:
Only Documents in the Training and Test sets are downloaded.

> **TIP:**
  Your Project ID can be obtained by the web app URL when accessing Konfuzio from your browser. From your home page, navigate to `Projects` and pick the Project you want to work with. Then look at the URL in your browser. Your should see something like `https://app.konfuzio.com/admin/server/project/<project-id>/change/` where `<project-id>` is your Project ID.

#### Create Project with `create_project`

To create a new Project run the following by replacing PROJECT_NAME with your Project name of choice.

`konfuzio_sdk create_project PROJECT_NAME`