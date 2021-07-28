### SDK Quickstart with PyCharm  

#### 1. PyCharm Setup
To start off with, please download the Community version of PyCharm for your respective operating system here: https://www.jetbrains.com/de-de/pycharm/download/#section=windows.  
Once it is dowloaded, create a new Project in PyCharm (File -> New Project).


#### 2. Create a Virtual Environment
As we want to execute our project/main file with our Python SDK package, we need a virtual enviroment with a python version > 3.6.   
To create such a virtual environment (venv), please refer to this documentation which describes the steps in detail: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env.  
(We add a new interpreter (bottom right corner: "Add Interpreter"). We choose "Virtualenv Environment" and "New Environment" and define the location to store this virtual environment. Please choose an empty folder as the location to store the environment in. If there is already a python version > 3.6 installed on your machine, choose your local python application. If not, we can select to download the most recent python version directly over PyCharm.)   
(Tick "Inherit global site-packages" (tbc!) to include all the packages on your local machine/ do not tick it: the virtual environment will just include the basic applications. Tick "Make available to all projects" to make the virtual environment reusable for further projects.   

To connect this virtual environment to your project, make sure that it is selected as the Python interpreter (see bottom right corner) and in the project setting preferences (top right corner -> Project -> Python Interpreter).  

![image](https://user-images.githubusercontent.com/85744792/127275314-e387ad14-5570-4963-b744-d2abe630ff08.png) 

After the installation of PyCharm and this setup of the virtual environment is completed, we can now start to install our Konfuzio SDK package (correct wording?). 

#### 3. Install the Python Konfuzio SDK package via pip install in the terminal:  
This will install the SDK package directly in your working directory:  
`pip install konfuzio-sdk`  

![image](https://user-images.githubusercontent.com/85744792/127275731-b730e743-0a90-4e5c-a454-3d74d047bd48.png)
  

#### 4. Define your working directory:
To store the Python SDK package, indicate the location by inputting the path of your working directory. This path should be the location of the "data" folder of the PyCharm project and can be found e.g. on the top left corner.  

`cd <your working directory>`     
![image](https://user-images.githubusercontent.com/85744792/127276445-9d95cc91-871c-4d1e-98c1-8781fec39e3a.png)


#### 5. Initialize the project with the required inputs:
After the installation, initialize the package in your working directory with:  
`konfuzio_sdk init`

This will require different inputs, starting with your **login credentials** to access the Konfuzio App.  
It will then ask for the **host** from where to get the data. If you are a business user, this might be the server url to access the Konfuzio application and different to app.konfuzio.com. In any other case, press "enter" to use the default url as the host address.  
As you are new to our Konfuzio application, there are no existing projects yet. To create a new one, enter your desired name of the project and then choose this certain project by inputting the respective **project ID** from the list of all available projects provided to you in the terminal. The ID of the project will also be shown in the URL.  
As a last input, please also enter the **folder** to which the data should be allocated to. If you have no specific preferences, you can use the proposed default folder name by pressing "enter" which will then be `data_<project_id>`.   
At the end, two files will be created in your working directory: .env and settings.py.
The .env file contains the credentials to access the app and should not become public.
The settings.py file defines constant variables that will be available in the project, including the ones you defined in the .env. This file should not be modified.  
![image](https://user-images.githubusercontent.com/85744792/127277914-a6a6da11-37e4-4871-9d13-e418b1740176.png)

You successfully initialized your project!  
Now you're Konfuzio SDK package is locally installed and will enable together with the usage of the API the usage of the Konfuzio web interface.


#### 6. Test your Setup:
To test whether everything worked as desired, we are going to execute a few basic commands in the main.py file.
Therefore, please remove the current sample Python script provided by PyCharm by deleting the code.  
We start off with initializing and updating the project, to retrieve the current version of the project from the web application. Once this is done, we define the documents as all of the documents allocated to the train (?) set in the project and count the total number of them. As we didn't upload any documents in our project yet, the total number should be 0.  
```
# Test your setup with the following code:
from konfuzio_sdk.data import Project

# Initialize the project:
my_project = Project()
my_project.update()

# Retrieve the documents from your project:
documents = my_project.get_documents_from_project()

# Receive the total number of documents in the project:
print("Number of documents in the project: {}.".format(len(documents)))
```


The data from the documents that you uploaded in your Konfuzio project will be downloaded to the folder that you provided in the previous step.
Note:
Only documents in the training or test set are downloaded.

----------------------------------------------
Create new document with 0, without any documents, project.update()
add screenshot folder structure: ls-la

Screenies größer, Eingaben ins markdown, Beispiel main.py, Screen recording steps? 

(PyCharm Projekt anlegen Dokumentation in Tutorial verlinken, PyCharm Installation verlinken, PyCharm installiert aktuellste Version von Python, was ist gewünschte Version? Projekt anlegen, main file muss ausgeführt werden, venv benötigt, env muss im terminal verfügbar sein zur Initialisierung
ab Terminalansicht, PyCharm venv aufsetzen verlinken mit Python >3.6

venv mit Terminal verbinden vor Initialisierung, 

Pycharm Community Installation, Link  
leeres Projekt initialisieren
final: 1 Command für main.py zum testen, project().documents = []) 







