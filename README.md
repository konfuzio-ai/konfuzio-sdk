![logo](docs/source/_static/docs__static_logo.png)

# Konfuzio SDK

The Konfuzio Software Development Kit (SDK) provides the tools to work with the data layer used by Konfuzio software.
Using the SDK you can communicate directly with the Konfuzio App and use directly the data structure in your projects.

The data structure of a project in Konfuzio App includes:

* Labels  
A label is the name of a group of individual pieces of information found in a type of document.

* Templates  
A template is a set of labels.

* Training and Test documents  
Based on the data from the training documents the ai learns what to do. These newly acquired capabilities are then applied to the test documents to sample check the quality of the ai model.

* Annotations for documents  
An annotation is a single piece of information that a label has been assigned to.

## Quickstart

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


### 2. Install `konfuzio_sdk` package

To use `konfuzio_sdk` on your machine you can install it via:  

#### Option 1

* Install the Python package directly in your working directory with:  
  
  `python -m pip install --extra-index-url https://test.pypi.org/simple/ konfuzio-sdk==0.0.1`  

#### Option 2

If access to the SDK GitLab project is granted:

* clone the project in your working directory
  
  `git clone https://gitlab.com/konfuzio/konfuzio-python-sdk.git`

* go inside the project folder
  
  `cd konfuzio-python-sdk`

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

## Example Usage

Let's see some simple examples of how can we use the `konfuzio_sdk` package to get information on a project and to post annotations.

To see which labels are available in the project:

```python
from konfuzio_sdk.data import Project

my_project = Project()
print(my_project.labels)
```

To post annotations of a certain word or expression in the first document uploaded, you can follow the example below:

```python
import re

from konfuzio_sdk.data import Project, Annotation, Label

my_project = Project()
my_project.update()

# Word/expression to annotate in the document
# should match an existing one in your document
input_expression = "John Smith"

# Label for the annotation
label_name = "Name"
# Creation of the Label in the project
my_label = Label(my_project, text=label_name)
# Saving it online
my_label.save()

# Template where label belongs
template_id = my_label.templates[0].id

# First document in the project
document = my_project.documents[0]

# Matches of the word/expression in the document
matches_locations = [(m.start(0), m.end(0)) for m in re.finditer(input_expression, document.text)]

# List to save the links to the annotations created
new_annotations_links = []

# Create annotation for each match
for offsets in matches_locations:
    annotation_obj = Annotation(
        document=document,
        document_id=document.id,
        start_offset=offsets[0],
        end_offset=offsets[1],
        label=my_label,
        template_id=template_id,
        accuracy=1.0,
    )
    new_annotation_added = annotation_obj.save()
    if new_annotation_added:
        new_annotations_links.append('https://app.konfuzio.com/a/' + str(annotation_obj.id))

print(new_annotations_links)

```

## Tutorial

An example of how Konfuzio SDK package can be used in a pipeline to have an easy feedback workflow can be seen in the tutorial bellow.

[Tutorial - Human in the loop of Machine Learning](https://colab.research.google.com/drive/1JaVL2L6MVUtl-x-8eGJ9FnSkAybHv3nh?usp=sharing)

## Documentation

The latest documentation can be accessed at https://konfuzio.gitlab.io/python-sdk/.

## Directory Structure

```
├── konfuzio-python-sdk         <- SDK project name
│   │
│   ├── docs                    <- Documentation to use konfuzio_sdk package in a project
│   │
│   ├── konfuzio_sdk            <- Source code of Konfuzio SDK
│   │  ├── __init__.py          <- Makes konfuzio_sdk a Python module
│   │  ├── api.py               <- Functions to interact with the Konfuzio App
│   │  ├── cli.py               <- Command Line interface to the konfuzio_sdk package
│   │  ├── data.py              <- Functions to handle data from the API
│   │  ├── settings_importer.py <- Meta settings loaded from the project
│   │  ├── urls.py              <- Endpoints of the Konfuzio host
│   │  └── utils.py             <- Utils functions for the konfuzio_sdk package
│   │
│   ├── tests                   <- Pytests: basic tests to test scripts based on a demo project
│   │
│   ├── .gitignore              <- Specify files untracked and ignored by git
│   ├── .gitlab-ci.yml          <- CI/CD configurations
│   ├── README.md               <- Readme to get to know konfuzio_sdk package
│   ├── pytest.ini              <- Configurations for pytests
│   ├── settings.py             <- Settings of SDK project
│   ├── setup.cfg               <- Setup configurations
│   ├── setup.py                <- Installation requirements

```


## Contact

Please add your questions or ideas in the [Issue Tracker](https://gitlab.com/konfuzio/konfuzio-python-sdk/-/issues).

## Contribute

If you would like to contribute, please use the development installation (`pip install -e .[dev]`) and open a PR with your contributions.  
Tests will automatically run for every commit you push.  
You can also run them locally by executing `pytest` in your terminal from the root of this project.

The files/folders listed below are ignored when you push your changes to the repository. 
- .env file
- .settings.py file
- data folder
- konfuzio_sdk.egg-info folder
- IDE settings files
- docs/build/ folder
- *.pyc files

*Note*:  
If you choose another name for the folder where you want to store the data being downloaded, please add 
the folder to *.gitignore*.
  
### Running tests locally

Some tests do not require access to the Konfuzio App. Those are marked as "local".

To run all tests, do:

`pytest`  

To run only local tests, do:

`pytest -m 'local'`  

To run tests from a specific file, do:

`pytest tests/<name_of_the_file>.py`  


## [License](LICENSE.md)
