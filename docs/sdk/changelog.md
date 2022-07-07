.. meta::
   :description: All notable changes in the konfuzio-sdk package will be documented in this file, chronologically ordered and with the correspondent tag version of the package.


# Changelog

Please find all updates on
<a href="https://github.com/konfuzio-ai/konfuzio-sdk/releases" target="_blank">Github</a>

# Archived Changelog

In addition to the GitHub Changelog, we keep the past Changelog records here.

## 2022-05-23 v.0.2.2

### Fixed
- Fix an issue that could affect the date normalization

## 2022-05-1 v.0.2.1

### Added
- Add the concept of a Tokenizer as class
- Add the concept of a Page of a Document as class

## 2022-03-11 v.0.2.0

### Fixed
- Renaming of "accuracy" to "confidence".

### Changed
- Initialization of the package without a specific project.
No longer necessary to specify a project to initialize the package.
- Now it is possible to load multiple projects in one Python process. 
For that, it is necessary to specify the ID of the desired project when creating the project object 
(my_project = Project(id_=project_ID)).
- The name of the folder where the data of the project can be downloaded cannot be customized anymore. 
The name is defined by "data_" followed by the project ID.
- The data of the documents is downloaded only when requested. This data refers to the file with the text of the 
document, the file with information about its pages, and the files with the annotations and annotation sets of the 
document. You can have documents in the project that are not used in your code.
Those files will be downloaded together when you:
  - access the text of the document (document.text)
  - access the pages of the document (document.pages)
  - access the annotations of the document (document.annotations())
  - access the annotation sets of the document (document.annotation_sets())
  - update the document (document.update())
  - call the method download_document_details (document.download_document_details())
- The data for the test and training documents in the project can be downloaded via CLI with "export_project" 
instead of "download_data".
- The import of environment variables happens directly from the running environment or .env. 
No longer necessary to have the file settings.py.
- Start and end offsets of an Annotation defined on a Span level.
- Expanded test coverage.
- Update documentation.

### Added
- Concept of evaluation. The evaluation of a document can be done by comparing two versions of it: version A, 
considered the correct one, and version B, the one to be evaluated. With this option you can:
  - compare a version of the document with annotations created by an AI with the version of the document on the server
  - compare versions of the document with annotations from different AIs.
- Concept of Span: sequence of characters or whitespaces without a line break.
An Annotation with multiple lines can now be described as an Annotation with multiple Spans.
- Introduction of eval_dict that returns any information necessary for evaluation.
The information is based on the Spans.
- Option to locally create empty data structures (for testing proposes).
- Normalization logic for a Span accordingly with the Annotation data type.
- Regex logic. Now it's possible to build regexes based on the Annotations of a certain Label.
- API endpoint to upload an AI model.

## 2021-11-24 v.0.1.15

### Fixed
- Issue when getting original file and ocr version of file.

## 2021-11-23 v.0.1.14

### Changed
- Option to pass the Konfuzio Host to the segmentation endpoint.

## 2021-11-18 v.0.1.13

### Changed
- Update URLs and API endpoints.
- Possibility to specify the category of a document when uploading it.
- Improve the performance of updating a project.

### Added
- Download original PDF version in download_data (cli).
- Function to split text into sentences.

### Fixed
- Loading of document related files.

## 2021-11-09 v.0.1.12

### Changed

- Option to download the original version of a file.

### Added

- Function to split text into paragraphs.
- Category class in the documentation for the source code.
- Add documentation for ParagraphModel in Konfuzio Trainer.

## 2021-11-02 v.0.1.11

### Changed

- Add legacy naming to Document class.

## 2021-10-26 v.0.1.10

### Changed

- Endpoints in API with option for custom host
- Increase timeout in get response
- Update tests

### Fixed

- Reading has_multiple_sections

## 2021-10-19 v.0.1.9

### Changed

- Option to define host in URLs

## 2021-10-18 v.0.1.8

### Changed

- Check for document being fully processed by the server

## 2021-10-12 v.0.1.7

### Changed

- Possibility of creating a multiline annotation

### Added

- Utilities functions for handling bounding boxes

## 2021-09-21 v.0.1.6

### Changed

- Renaming of classes to new Konfuzio Server version
- Improved documentation dev.konfuzio.com

### Added

- Category class
- Error message for missing .env

## 2021-08-16 v.0.1.5

### Changed

- Improved documentation dev.konfuzio.com

### Added

- Method to get the document text in the BIO scheme

## 2021-07-15 v.0.1.4

### Changed

- Verification of the creation of the data folder
- Maintain original file name in ocr pdf
- Get metadata from all documents in the project
- Download of document bboxes optional
- Improved documentation dev.konfuzio.com

### Fixed

- API endpoints usage

## 2021-06-30 v.0.1.3

### Changed

- Improved documentation dev.konfuzio.com
- Definition of the section class in the Project

## 2021-06-15 v.0.1.2

### Added

- Documentation for the Konfuzio Web App and Trainer package

### Changed

- Load sections that match the category of the document


## 2021-05-17 v.0.1.1

### Added

- Method to get link from Annotation
- Include readme file in the pip package

### Fixed

- Initialization of the package with new projects


## 2021-05-07 v.0.1.0

### Changed

- Cleaning of the repo.


## 2021-05-06 v.0.0.1

### Changed

- Updated documentation.
- Setup github actions.


## 2021-04-29

### Changed

- Update api endpoint.
- Methods for saving online Documents and Labels.


## 2021-03-29

### Changed

- Removed config directory from setup.py.
- Delete config directory.
- Reset the version of package in setup.py to 0.0.1



## 2021-03-26

### Fixed

- Fixes for the password input in the *cli.py* and *test_cli.py*.
- Quick code clean.
- Small fix flake8.



## 2021-03-25

### Added

- Defining tests in pipeline.
- Definition of the classes.
- Add str methods.

### Changed

- Split in pipeline
- Mark tests as local and split in pipeline.
- Replace test docs for zip.
- Clean definition of classes.

### Fixed

- Fix *test_get_file_type* method in *test_utils.py* .



## 2021-03-24

### Added

- Added test_utils.py class.
- Added test_cli.py class.

### Changed

- Removed functions from urls.py that are not being used.
- Allow to overwrite of classes in Project.



## 2021-03-18

### Changed

- Remove creation of releases folder.



## 2021-03-17

### Added

- Initialization of SDK.

### Changed

- New changes in the konfuzio_sdk package.



## 2021-03-15

### Changed

- Move data methods from trainer to SDK and update tests.

### Fixed

- Code clean.



## 2021-03-11

### Added

- Add test data.
- Add flake8 config .
- Add tests for images endpoints from API.
- Add test pdf for uploading.
- Add git pipeline.
- Add dockerfile.
- Add tests for api.py.

### Changed

- **does_not_raise** moved from wrapper in Trainer.
- Remove container.
- Update setup.py.

### Fixed

- Clean tests for API.



## 2021-03-10_

### Added

- Add other API endpoints and update data.py .

### Changed

- Update setup.py.



## 2021-03-3_

### Added

- Added new file for ReadMe.



## 2021-03-2_

### Added

- Add labels description to Label.



## 2021-02-23_

### Changed

- Update README.md.



## 2021-02-17_

### Added

- Added **get_bbox()** method to Document class.



## 2021-02-16_

### Changed

- Removed methods from each class in data.py that are not used for data downloading.



## 2021-02-15_

### Added

- Utils class added with utility functions for data.py.

### Changed

- Update README.md.
- Remove commented is_file function inside data.py.



## 2021-02-12_

### Added

- Add README.md file.

### Changed

- Refactor data.py.
