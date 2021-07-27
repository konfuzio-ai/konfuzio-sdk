### SDK Quickstart with PyCharm  

To start off with, please download the Community version of PyCharm here for your respective operating system: https://www.jetbrains.com/de-de/pycharm/download/#section=windows.  
Once it is dowloaded, create a new Project in PyCharm (File -> New Project).

As we want to execute our project/main file with our Python SDK package, we need a virtual enviroment with a python version > 3.6.   
To create such a virtual environment (venv), please refer to this documentation which describes the steps in detail: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env.  
(We add a new interpreter (bottom right corner: "Add Interpreter"). We choose "Virtualenv Environment" and "New Environment" and define the location to store this virtual environment. Please choose an empty folder as the location to store the environment in. If there is already a python version > 3.6 installed on your machine, choose your local python application. If not, we can select to download the most recent python version directly over PyCharm.)   
(Tick "Inherit global site-packages" (tbc!) to include all the packages on your local machine/ do not tick it: the virtual environment will just include the basic applications. Tick "Make available to all projects" to make the virtual environment reusable for further projects.   

To connect this virtual environment to your project, make sure that it is selected as the Python interpreter (see bottom right corner).

After the installation of PyCharm and this setup of the virtual environment is completed, we can now start to install our Konfuzio SDK package (correct wording?). 

#### 1. Install the Python Konfuzio SDK package via pip install in the terminal:  
This will install the SDK package directly in your working directory:  
`pip install konfuzio-sdk`  

![image](https://user-images.githubusercontent.com/85744792/127024727-0a51303d-6e48-4692-8ece-0dea9dd9aaed.png)  

#### 2. Define your working directory:
Please indicate the location to store the Python SDK package by copying the desired path from your working directory:  
`cd <your working directory`  

![image](https://user-images.githubusercontent.com/85744792/127024911-97d8f753-d96e-41bc-a66d-22a4455dad26.png)  

#### 3. Initialize the project with the required inputs:
After the installation, initialize the package in your working directory with:  
`konfuzio_sdk init`

This will require your credentials to access the Konfuzio App and the project ID. You can check your project ID by selecting the project in the Projects tab. The id of the project is shown in the URL. It will also require a name of a folder where to allocate the data from your Konfuzio project. At the end, two files will be created in your working directory: .env and settings.py.

The .env file contains the credentials to access the app and should not become public.
The settings.py file defines constant variables that will be available in the project, including the ones you defined in the .env. This file should not be modified.

##### 3.1. Input your login credentials from app.konfuzio:
![image](https://user-images.githubusercontent.com/85744792/127025052-1d37076c-1a5b-4ecc-a955-d9c8e8096dcc.png)  

##### 3.2. Input the host, press enter to use the default one:   
business user might be different host, which is used to log in into konfuzio
![image](https://user-images.githubusercontent.com/85744792/127025150-c7a0c1b0-3ff9-46b3-9a83-17c74a0ad33d.png)  

##### 3.3. Choose the project you want to initialize: 
![image](https://user-images.githubusercontent.com/85744792/127025273-58e55544-35c5-4831-9fa1-4581aa4b6a23.png)  

##### 3.4. Set the folder to which the data should be allocated to, press enter for default:   
![image](https://user-images.githubusercontent.com/85744792/127025370-ac4f0acf-afb2-4cef-abd2-82ef0f984173.png)

##### 3.5. You successfully initialized your project!
![image](https://user-images.githubusercontent.com/85744792/127025449-80f049c8-b3e3-4e9f-8950-25ef01527c5e.png)


#### 4. Update your project:
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







