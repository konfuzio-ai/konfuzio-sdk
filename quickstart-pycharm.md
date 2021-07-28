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

![image](https://user-images.githubusercontent.com/85744792/127024727-0a51303d-6e48-4692-8ece-0dea9dd9aaed.png)  

#### 4. Define your working directory:
Please indicate the location to store the Python SDK package by copying the desired path from your working directory:  
`cd <your working directory>`  

![image](https://user-images.githubusercontent.com/85744792/127024911-97d8f753-d96e-41bc-a66d-22a4455dad26.png)  

#### 5. Initialize the project with the required inputs:
After the installation, initialize the package in your working directory with:  
`konfuzio_sdk init`

This will require different inputs, starting with your **login credentials** to access the Konfuzio App. It will then ask for the **host** from where to get the data. If you are a business user, this will be the server url to access the Konfuzio application. In any other case, press "enter" to use the default url as the host address. As you are new to our Konfuzio application, there are no existing projects yet. To create a new one, enter your desired name of the project and then choose this certain project by inputting the respective **project ID** from the list provided to you in the terminal. The ID of the project will also be shown in the URL. As a last input, please also enter the **folder** to which the data should be allocated to. If you have no specific preferences, you can use the proposed default folder name by pressing "enter" which will then be `data_<project_id>`. At the end, two files will be created in your working directory: .env and settings.py.

The .env file contains the credentials to access the app and should not become public.
The settings.py file defines constant variables that will be available in the project, including the ones you defined in the .env. This file should not be modified.

##### 5.1. Input your login credentials from app.konfuzio:
![image](https://user-images.githubusercontent.com/85744792/127025052-1d37076c-1a5b-4ecc-a955-d9c8e8096dcc.png)  

##### 5.2. Input the host, press enter to use the default one:   
business user might be different host, which is used to log in into konfuzio
![image](https://user-images.githubusercontent.com/85744792/127025150-c7a0c1b0-3ff9-46b3-9a83-17c74a0ad33d.png)  

##### 5.3. Choose the project you want to initialize: 
![image](https://user-images.githubusercontent.com/85744792/127025273-58e55544-35c5-4831-9fa1-4581aa4b6a23.png)  

##### 5.4. Set the folder to which the data should be allocated to, press enter for default:   
![image](https://user-images.githubusercontent.com/85744792/127025370-ac4f0acf-afb2-4cef-abd2-82ef0f984173.png)

##### 5.5. You successfully initialized your project!
Now you're Konfuzio SDK package is locally installed and will enable together with the usage of the API to use the Konfuzio web interface.
![image](https://user-images.githubusercontent.com/85744792/127025449-80f049c8-b3e3-4e9f-8950-25ef01527c5e.png)


#### 6. Test your Setup:
To test whether everything worked as desired, we are going to execute a few commands in the main.py file.
Therefore, please remove the current sample Python script.
To download the data from your Konfuzio project, please execute:
konfuzio_sdk download_data (or project.update()?)
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







