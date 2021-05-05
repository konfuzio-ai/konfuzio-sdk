
# Contribute Gudie Konfuzio Python SDK

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
