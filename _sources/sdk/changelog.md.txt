.. meta::
   :description: All notable changes in the konfuzio-sdk package will be documented in this file, chronologically ordered and with the correspondent tag version of the package.


# Changelog

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
