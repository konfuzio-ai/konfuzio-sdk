### SDK Quickstart with PyCharm  

#### 1. PyCharm Setup
PyCharm is a widespread common Integrated Development Environment (IDE) with strong debugging functionalities, which is why we choose PyCharm as our preferred IDE in this context.  
To start off with, please download the Community version of PyCharm for your respective operating system here: https://www.jetbrains.com/de-de/pycharm/download/#section=windows.  
Once it is dowloaded, create a new Project in PyCharm (File -> New Project). Please also refer to this documentation for further explanations: https://www.jetbrains.com/help/pycharm/creating-empty-project.html.


#### 2. Create a Virtual Environment
As we want to execute our project with our Konfuzio SDK package, we need a virtual enviroment which is based on the Python version 3.8. This Python version is the most widespread and commonly used version which still represents a good trade-off between newness and stability.  
To create such a virtual environment, please refer to this documentation which describes the steps in detail: https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html#python_create_virtual_env.  

To connect this virtual environment to your project, make sure that it is selected as the Python interpreter (see bottom right corner) and in the project setting preferences (top right corner -> Project -> Python Interpreter).  Nutzen lokales Venv, da Nutzer der community version kein Debugging von Containern oder SSH durchführen können.

![image](https://user-images.githubusercontent.com/85744792/127275314-e387ad14-5570-4963-b744-d2abe630ff08.png) 

After the installation of PyCharm and this setup of the virtual environment is completed, we can now start to install our Konfuzio SDK package.  

#### 3. Install the Konfuzio SDK package:  
Install the Konfuzio SDK package via pip install in the terminal. This will install the SDK package directly in your working directory:  
`pip install konfuzio-sdk`     

![image](https://user-images.githubusercontent.com/85744792/127275731-b730e743-0a90-4e5c-a454-3d74d047bd48.png)
  

#### 4. Define your working directory:
To store the Konfuzio SDK package, indicate the location by inputting the path of your working directory. This path should be the location of the "data" folder of the PyCharm project and can be found e.g. on the top left corner.  

`cd <your working directory>`   

![image](https://user-images.githubusercontent.com/85744792/127276445-9d95cc91-871c-4d1e-98c1-8781fec39e3a.png)


#### 5. Initialize the project with the required inputs:
After the installation, initialize the package in your working directory with:  
`konfuzio_sdk init`

This will require different inputs, starting with your **login credentials** to access the Konfuzio App.  
It will then ask for the **host** from where to get the data. If you are a business user, this might be different to app.konfuzio.com and will be the server url to access the Konfuzio application. In any other case, press "enter" to use the default url as the host address.  
If you are new to our Konfuzio application, there are no existing projects yet. To create a new one, enter your desired name of the project and then choose this certain project by inputting the respective **project ID** from the list of all available projects provided to you in the terminal. The ID of the project will then also be shown in the URL of the project once it is created.  
As a last input, please also enter the **folder** to which the data should be allocated to. If you have no specific preferences, you can use the proposed default folder name by pressing "enter" which will then be `data_<project_id>`.   
At the end, two files will be created in your working directory: .env and settings.py.
The .env file contains the credentials to access the app and should not become public.
The settings.py file defines constant variables that will be available in the project, including the ones you defined in the .env. This file should not be modified.  

![image](https://user-images.githubusercontent.com/85744792/127277914-a6a6da11-37e4-4871-9d13-e418b1740176.png)

You successfully initialized your project!  
Now you're Konfuzio SDK package is locally installed and will enable you together with the usage of the API the usage of the Konfuzio web interface.


#### 6. Test your Setup:
To test whether everything is working as desired, we are going to execute a few basic commands in the main.py file.
Therefore, please remove the current sample Python script provided by PyCharm by deleting the code.  
We start off with initializing the project to retrieve the current version of the project from the web application. Once this is done, we count the total number of all documents allocated to the train set in the project. To run the code, press "Run" and then "Run main". As we didn't upload any documents in our project yet, the total number should be 0.  
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
![image](https://user-images.githubusercontent.com/85744792/127280672-6e317a29-6731-4dbd-aca1-2cb45a68e6c9.png)  

#### 7. Test the Debugging
We also include a second test to check if debugging also works. Herefore, we include a breakpoint in the last line of code at the print statement and execute "Run" and then "Debug main". This should show no bugs and will also provide you with an overview of all available variables. As you can see in the code above, we only accessed the document element - however, all other elements can be found in the Debug console under "Variables".   

![image](https://user-images.githubusercontent.com/85744792/127323550-61690987-b705-4a23-82c6-9ffaf2aed661.png)




