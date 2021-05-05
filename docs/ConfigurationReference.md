# Configuration Reference Konfuzio Python SDK

### 1. Have your project in Konfuzio App  

To test our SDK you need to set some basic configurations in our App and have one document of your choice at hand. Just follow the instructions below to set yourself ready for the SDK.
 
* #### Sign Up/Log In
Use web [log in](https://app.konfuzio.com/) or [register for free](https://app.konfuzio.com/accounts/signup/).
 
* #### Create Project
Create a project in your Account. Click HOME >Projects>[Add Project +](https://app.konfuzio.com/admin/server/project/add/) to create a new AI project. Name your project and click on “SAVE”.
 
* #### Upload Document
Upload documents to your project. Click on [DOCUMENTS](https://app.konfuzio.com/admin/server/document/). Via Drag&Drop or the browser window you can upload your local files. After all uploaded documents light up green, click the Reload button to reload the page. Now the OCR process begins. Depending on the file size and number of documents, this may take a moment.
 
* #### Create Label
Create a label. Click on HOME>[Templates](https://app.konfuzio.com/admin/server/sectionlabel/). Klick on the Template which has the same name as your project. Click on the green plus next to the "Chosen Labels" field. In the window that now opens, you can name your Label. Select your project in the "Project" tab and click on “SAVE” to close the window. Check if your label is displayed in the “Chosen Labels” field and click on “SAVE” again. 
 
* #### Add Training Data
Add your document to the training dataset. Click on [DOCUMENTS](https://app.konfuzio.com/admin/server/document/). Tick the box on the left side of your document name and select “Add to training dataset” in the action tab. Click on “Go”.  

### Some words about the training structure

The data structure of a project in Konfuzio App includes:

* Labels  
A label is the name of a group of individual pieces of information found in a type of document.

* Templates  
A template is a set of labels.

* Training and Test documents  
Based on the data from the training documents the ai learns what to do. These newly acquired capabilities are then applied to the test documents to sample check the quality of the ai model.

* Annotations for documents  
An annotation is a single piece of information that a label has been assigned to.


### 2. Install `konfuzio_sdk` package

To use `konfuzio_sdk` on your machine you can install it via:  

#### Option 1

* Install the Python package directly in your working directory with:  
  
  `python -m pip install --extra-index-url https://test.pypi.org/simple/ konfuzio-sdk==0.0.1`  

#### Option 2

Clone the project:

* clone the project in your working directory
  
  `git clone https://github.com/konfuzio-ai/Python-SDK.git`

* go inside the project folder
  
  `cd Python-SDK`

* install it
  
  `pip install -e .`


*Notes*:
* Your coding environment should have a Python version >= 3.6.  
* If you are not using a virtual environment, you may need to add the installation directory to your PATH. 

### 3. Initialize the package

After the installation, initialize the package in your working directory with:

`konfuzio_sdk init`

This will require your credentials to access the Konfuzio App and the project ID. 
You can check your project ID by selecting the project in the Projects tab. The id of the project is shown in the URL.
It will also require a name of a folder where to allocate the data from your Konfuzio project.
At the end, two files will be created in your working directory: `.env` and `settings.py`.  

The `.env` file contains the credentials to access the app and should not become public.  
The `settings.py` file defines constant variables that will be available in the project, including the ones you defined in the `.env`. This file should not be modified.

### 4. Download the data

To download the data from your Konfuzio project, you can execute:

`konfuzio_sdk download_data`

The data from the documents that you uploaded in your Konfuzio project will be downloaded to the folder that you provided in the previous step.  

*Note*:  
Only documents in the training or test set are downloaded.  

