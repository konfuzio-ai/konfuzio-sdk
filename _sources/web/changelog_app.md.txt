.. meta::
   :description: Konfuzio Server Changlog to inform developers, partners and user about all notable changes.

.. _Server Changelog:

# Changelog

All notable changes in the Konfuzio Server will be documented according to the principles defined by [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

The changelog adheres to [Calendar Versioning](https://calver.org/overview.html) and the release tag relates to the date and time when those changes have been released to app.konfuzio.com. 

Self-hosted Konfuzio Server can be upgraded according to the [documentation](https://dev.konfuzio.com/web/on_premises.html#upgrade).

## Planned

You can think of the *Planned* section as a *Roadmap* that lists Konfuzio Server features our team is actively working on. This list covers a planning horizon of 12 weeks. 

- Add self-service customer center to manage contracts and invoices ([Internal Documentation](https://git.konfuzio.com/konfuzio/manual/-/merge_requests/23/diffs), [Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/8235)).
- Calculate and access Tokenizers via the web interface. The name Tokenizer might be revised. The SDK refers to this concept as Tokenizer. ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9271)).
- Add a filter to the list of Documents to find Documents that need to be revised by humans. ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9242)).
- Improved training time of Extraction AIs when using character detection mode (reduced up to 40%) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9271)).
- Train custom Page-based Categorization AI via [SDK](https://github.com/konfuzio-ai/konfuzio-sdk/) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9481)).
- First stable release of Capture Vue in version [0.1.0](#).
- Suggest page breaks if one file contains multiple Documents ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/7671)).
- Perfomance improvement of the Konfuzio Server interface ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9750)).
- Redirect https://app.konfuzio.com/api/ to stable API Version 3 ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9692)).
- Make all List Views sortable by click on column ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9470)).
- Sort Labels in Label-Sets to allow users to customize the UI per Category ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/8932)).
- Delta Training, Partial Fit an exisiting classifier, so that training documents used previously can be deleted ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9251))
- Allow administrators of Konfuzio on-premise installations to run a speedtest ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9870))
- Start automatic AI retraining after user confirms that he has finished a annotation review ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9166))
- [Show a warning if a user works on a Document that is not assigned to him](https://help.konfuzio.com/modules/documents/index.html#assignee) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9466)).

<!---
### Fixed
- A bug when training with character detection mode, which was tokenizing some labels incorrectly, causing them to be skipped during extraction ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9666))
- A bug during the extraction post-processing steps, which was causing the first line items of each page to be skipped ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9561))
-->

## Next Release

This version uses Konfuzio Trainer in version v.0.3.22, the Konfuzio Python SDK in version v.0.1.15 and Konfuzio Capture Vue in version [0.0.9](https://github.com/konfuzio-ai/konfuzio-capture-vue/releases/).

### Added
- [Allow to invite Members with different Roles to Projects. Available Roles are "Reviewer" and "Manager". All existing Members keep their current Permissions and will become Managers.](https://help.konfuzio.com/modules/members/index.html)
- [Superusers can define custom Roles for Members: Inviting Users can select from those Roles when inviting new Members to a Project](https://help.konfuzio.com/modules/superuserroles/index.html). ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/7364)).
- [Add the property 'has_multiple_top_candidates' to the Label API V3](https://app.konfuzio.com/v3/swagger/#/labels) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9687)).
- [Add the property 'has_multiple_annotation_sets' to the Label Set API V3](https://app.konfuzio.com/v3/swagger/#/labels) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9687)).
- [Save feedback that there are no Annotations for a Label/Label-Set combination in a document](http://localhost:8000/v3/swagger/#/documents/documents_missing_annotations_list) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9163)).
- [Add the property 'number' to the Page API V3](https://app.konfuzio.com/v3/swagger/#/documents) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9619)).
- [Include the Label Set name in the API V3 even when no Label Set is present](https://app.konfuzio.com/v3/swagger/#/documents) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9399)).
- [API v3 endpoint to sort and split Pages into Documents with different categories contained in one file](https://testing.konfuzio.com/v3/swagger/#/documents/documents_postprocess_create) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9452)).

### Changed
- [When a user creates a new Project, this user will become the default assignee for new Documents](https://help.konfuzio.com/modules/projects/index.html#default-assignee) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9705))
- [If a user rejects an Annotation, this user is tracked in the 'revised_by' attribute of the Annotation](https://help.konfuzio.com/modules/annotations/index.html#declined) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9479)).

### Fixed
- The numbering of Annotation Sets in the SmartView does not consider deleted Annotation Sets anymore ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9604)).
- In some situations a Project could not be deleted via API ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9706)).
- In specific scenarios, the deletion of the last remaining Annotation in a Document was not possible ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9736)).
- The SmartView did not use rotated pages due to a caching problem ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9830)).
- The arrow in the Project- and language selector was not clickable ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9714)).
- On the Annotation list Page the Category filter was not showing all Annotations ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9732)).

## released-2022-10-28_07-23-39

### Added
- [Allow on-premise users to customize timeouts of backgroud tasks](https://dev.konfuzio.com/web/on_premises.html#environment-variables-for-konfuzio-server) ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9812)).

### Fixed
- Accepting an Annotation was overwritting already existing custom offset strings ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9830)).
- In some cases an Extraction AI training was failing when detecton mode 'Character' was selected ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9935)).

## released-2022-09-21_12-00-31

This version uses Konfuzio Trainer in version v.0.3.22, the Konfuzio Python SDK in version [v.0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15) and Konfuzio Capture Vue in version [0.0.8](https://github.com/konfuzio-ai/konfuzio-capture-vue/releases/).

### Fixed
- Prevent an issue where a popup window could not be closed when using the SmartView ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9766)).
- Filtering for "feedback required" on Document overview ([Internal Ticket](https://git.konfuzio.com/konfuzio/objectives/-/issues/9696)).

## released-2022-09-04_09-11-18

This version uses Konfuzio Trainer in version v.0.3.21, the Konfuzio Python SDK in version [v.0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15) and Konfuzio Capture Vue in version [0.0.8](https://github.com/konfuzio-ai/konfuzio-capture-vue/releases/).

### Added
- [Auto-rotation for documents for all angles (until now only 90 degree angles have been supported).](https://help.konfuzio.com/modules/projects/index.html#automatically-rotate-documents)
- [Information about the embedded fonts of PDF documents.](https://help.konfuzio.com/modules/documents/index.html#fonts)
- [Superusers can inspect the logs of AI trainings and AI run.](https://help.konfuzio.com/modules/superuserdocuments/index.html#extraction-log)
- [Show minimum/medium/maximum loading time and runtime of AIs.](https://help.konfuzio.com/modules/extractions/index.html#loading-time-in-seconds)
- [Improve Swagger API definition for bounding boxes  (API V3).](https://testing.konfuzio.com/v3/swagger/#/documents/documents_retrieve)
- [Add threshold attribute to the Category endpoint (API V3).](https://testing.konfuzio.com/v3/swagger/#/categories/categories_retrieve)
- [Add the callback_url attribute to (API V3).](https://testing.konfuzio.com/v3/swagger/#/documents/documents_create)
- [Add /me Authentification endpoint (API V3).](https://app.konfuzio.com/v3/swagger/#/auth/auth_me_retrieve)

### Changed
- Improve visibility of the left navigation bar.
- [Create sandwich PDFs on demand (until now they have been created on document upload)](https://dev.konfuzio.com/web/on_premises.html#environment-variables-for-konfuzio-server).
- Add content-type header ("Content-Type": "application/json") to the callback response.
- [Update Power Query Excel Documentation](https://help.konfuzio.com/integrations/excel/index.html) with animated screencasts

### Fixed
- Top annotation filter in the SmartView now considers unrevised annotations.
- Fix an issue in which a Document cannot be processed because negative bounding boxes are detected. 
- Fix an issue which caused the processing time to be shown as negative.


## 2022-07-28_15-55-29

This version uses Konfuzio Trainer in version v.0.3.21, the Konfuzio Python SDK in version [v.0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15) and Konfuzio Capture Vue in version [0.0.6](https://github.com/konfuzio-ai/konfuzio-capture-vue/releases/).

### Changed
- Speed up runtime of Extraction AIs

### Fixed
- Fix an issue which causes some Extraction AIs to crash on multipage documents. 
- Fix an issue that prevents the calculation of bounding boxes for small or slightly rotated characters. 

## 2022-07-25_21-20-48

### Added
- Allow to set a default assignee for uploaded documents
- Allow to notify users via email when they get assigned to documents

### Fixed
- Top annotation filter in the SmartView now takes accepted Annotations into account
- Errors messages in case a document could not be processed are now displayed correctly

## 2022-07-19_16-30-46

### Changed
- New Extraction AIs are saved in a more efficient way

## 2022-07-05_19-35-21

This version uses Konfuzio Trainer in version v.0.3.15, the Konfuzio Python SDK in version [v.0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15) and Konfuzio Capture Vue in version [0.4.0](https://github.com/konfuzio-ai/konfuzio-capture-vue/releases/).

### Added 
- Show the user who started an AI training on the detail page of an AI
- Allow to set a time (in days) after which documents are automatically deleted
- Allow to rotate pages via API
- Add thumbnail images for document pages

### Changed
- Links to deleted annotation will now redirect to the respective document

## 2022-06-10_15-32-19

This version uses Konfuzio Trainer in version v.0.3.15 and the Konfuzio Python SDK in version [v.0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15).

### Added 
- Option to enforce running OCR even if text embeddings are present
- Improved error messages in case a document cannot be processed
- Option to exclude email content when using the email-integration
- Option to make document accessible via public link
- Beta Version of APIV3
- Beta Version of new document dashboard (bases on [Konfuzio-Capture-Vue](https://github.com/konfuzio-ai/konfuzio-capture-vue))
- Automatic rotation of pages (#8980)

### Changed
- For on-premise Users, now the Postgresql 10 is the minimum version
- Improved Extraction AI
- On-Premise container run now as non-root and using read-only fileystem
- Improved mouse pointer in the Smartview

### Fixed
- An issue where empty Annotation Sets could appear on Documents
- An issue where conflicting annotions could be created
- An issue where negative annotations where not correctly being deleted (#9127)
- Rare cases where OCR text included some characters mutiple times


## 2022-04-27_14-23-38

### Added
- Add assignee attribute of a Document to the API

## 2022-03-15_09-14-17

### Changed
- "Rerun extraction" via the user interface applies new annotations now also to training and test documents

## 2022-02-11_23-12-26

### Added 
- Add option to filter for related annotation sets

### Fixed
- Sorting of annotation sets in the csv export
- Document API endpoint returning declined annotations

## 2022-01-18_11-08-24

### Added
- Added api_name to Label API

### Fixed
- Link to documentation page
- Missing translation on document list page
- Evaluation did not complete for AIs with a large amount of training data

## 2021-12-11_14-33-57

### Added
- For on-premise installations, the OCR method for new projects is choosen based on the available OCR solutions.
- For on-premise installations, the project import considers now declined annotations
- For on-premise installations, Superusers can see the Konfuzio Server version and how many pages and documents have been processed.

## 2021-11-21_19-14-19

### Added
- Text summarization endpoint.
- Categorization AI parameters in the Project view

### Fixed
- An issue where the reload after uploading new documents does not happen

## 2021-11-26_08-11-36

This version uses Konfuzio Trainer in version [v.0.3.0](https://dev.konfuzio.com/training/changelog.html). We recommend to use the Konfuzio Python SDK in version [0.1.15](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-15)

### Added
- "Sentence" option to the available detection modes

### Fixed
- An error where an invalid date in the document text stoppped the training process

## 2021-11-23_18-14-28

### Fixed
- E-mails without an attachment have not been processed.

## 2021-11-16_23-02-22

### Fixed
- CSV export for [ProRis](https://www.inveos.com/proris-blue) by Inveos

## 2021-11-05_09-55-10

### Added
- Allow deletion of characters of an annotation without excluding it from the training process

## 2021-11-01_23-19-58

### Added
- An option to specify the category of a document when uploading it via API (and thereby skipping the categorization)

### Changed
- The GET document API endpoint now returns the annotation displayed in the SmartView (instead of only showing the extraction AI results)

## 2021-10-25_20-12-18

This version uses Konfuzio Trainer in version [2021-10-20_18-29-25](https://dev.konfuzio.com/training/changelog.html#id1). We recommend to use the Konfuzio Python SDK in version [0.1.10](https://dev.konfuzio.com/sdk/changelog.html#v-0-1-10)

### Added
- CSV export compatible with [ProRis](https://www.inveos.com/proris-blue) by Inveos

## 2021-10-16_13-20-12

### Added
- Improve detection of annotations which consist of multiple words
- Date filtering for project documents API endpoint
- Filtering of labels and label sets according to the category of a document (in the SmartView)

### Fixed
- Selection of characters in SmartView incomplete when editing an annotation
- Dark Mode setting of browser not compatible with Konfuzio Server
- Some case where the document list was not reloaded automatically

## 2021-10-07_11-42-29

### Added
- More advanced task priorities and improved worker ressource usage
- Auto-reload of new uploaded documents

## 2021-09-28_09-29-43

### Fixed
- Evaluation does not complete if no test documents are specified

## 2021-09-24_13-53-32

### Fixed
- Incompleted evaluation
- Formatting of the "Check your browser" page for logged out users.

## 2021-09-16_12-25-23

### Fixed
- Adding of categories to existing label sets

## 2021-09-08_16-01-46

### Added
- Migration scripts for user permissions and e-mail templates

## 2021-09-07_12-24-26

### Added
- Support for SMTP e-Mail backends via environment variables

### Fixed
- DOS protection prevents start of Konfuzio server

## 2021-09-05_20-57-31

### Added

- Autosave for any change on the document list page
- German language support
- Finetuning of exctraction AIs via parameters
- New fields AI quality and data quality
- More detailed evaluation
- Description Field for extraction AI, label sets, categorization AI, categories

### Changed

- Rename project inviations to members
- Rename the dataset status form "OCR Error" to "Excluded"
- Start training per extraction AI
- Get more insights via the document detail page

## 2021-08-10_17-08-11

### Changed
- Deactive adoption of template settings according to AI model if not explicitly allowed.

## 2021-08-10_11-19-33

### Added
- Maximum number of pages per document

### Fixed
- Slow processing of extraction tasks
- Evaluation when multiple annotations are present

## 2021-07-28_18-53-12

### Changed
- Make word-based tokenizer the default for new projects

## 2021-07-23_09-33-20

### Fixed
- Usage of word-base tokenizer
- Duplicated hints

## 2021-07-20_17-29-23

### Fixed
- Edited annotation were excluded from the training process

## 2021-07-15_17-29-25

### Added 
- Support to reuse label sets across categories

### Changed
- Allow "rerun extraction" on test and training documents
- Remove "project statistic csv export" as it is redundant to document csv export
- Include evaluation for training data in the AI model evaluation report

### Fixed
- Fixed a bug where the EXIF attribute orientation corrupted the bounding boxes images
- "accept top annotations" does not update human created annotations

## 2021-07-02_18-13-01

### Changed
- Rate limits for task system

## 2021-06-29_22-14-33

### Added
- HTTP codes to API interface

### Fixed
- Content type description for some API endpoints

## 2021-06-22_22-45-48

### Added
- A experimental version of a training health report

## 2021-06-20_15-14-31

### Fixed
- Failed retraninings for some projects
- Increased disk usage due to an cache deletion issue
- Filtering of project invotations according to currently selected project
- Clarify return types in API documentation

## 2021-05-26_20-16-02

### Added
- Show confidence for categorization results
- Show evaluation of categorization Ai models
- Track version (number of retrainings) for all Ai models
- Track project and template origin of AiModel

## 2021-05-24_13-42-45

### Changed
- Use business evaluation implementation from training package
- Loading time for CSV export evaluation reduced by saving it in the database.

## 2021-05-18_16-37-25

### Added
- Global project switcher
- "Top candidates" filter in SmartView
- "Change dataset" functionality in SmartView
- Landing page in case the user has no projects (i.e. just registered)
- Language switcher (not enabled yet)
- Initial support for German translations (not enabled yet)

### Fixed
- Label threshold is now limited from 0.0 to 1.0

### Changed
- New design for login/signup/reset password pages
- Design improvements in the control panel and SmartView
- New logo and favicon
- API documentation has been improved with types and examples; is now based on OpenAPI 3
- Updated frontend dependencies and tooling

### Removed
- `admin_importer`, `copy_extraction_as_annotation` and related functions have been removed

## 2021-05-04_12-37-16

### Fixed
- Calculation of true negative when using multiple templates.

## 2021-04-28_12-27-19

### Added
- Filter for top annotations in SmartView

### Changed
- Dont allow training if there are no training documents

## 2021-04-25_20-09-04

### Added
- Protect signup with captcha

### Fixed
- Editing of annotation if there are already declined annotations.

## 2021-04-19_22-32-19

### Added 
- Add label creation endpoint
- Token-based authentication for the API

## 2021-04-03_09-46-56

### Added 
- Show Django sidebar in Smartview and template view.

### Changed
- Save extraction results in a more efficient way.
- Show a warning if an annotation with a custom offset string is created
- Shwo loading indicator in the smartview search

### Fixed
- Default template dropdown sometimes disabled when creating a Template
- Rare case where the document list could not be loaded

## 2021-03-15_15-12-04

### Added
- Add option to accept all annotations.

## 2021-03-07_21-32-41

### Added
- Option to retrain project categorization model
- Improved OCR settings
- System check page https://app.konfuzio.com/check/

### Fixed
- Typo in privacy policy
- Confirmation message when deleting labels
- Performance of csv export

### Changed
- Delete old unrevised annotations when rerunning AiModel.

## 2021-02-25_09-28-07

### Added
- Option to select tokenizer for training (ProjectAdmin)
- Option to add training parameters (SuperuserProjectAdmin)

### Changed
- Set a documents category_template on new documents if there is only one category_template available
- Improved delete / accept performance of annotations

### Fixed
- Count of annotations on the LabelAdmin

## 2021-02-15_18-56-51

### Changed
- Show category template as empty when actual empty (instead of displaying the first available template)
- Improved Smartview performance by changing entity loading

### Added
- Project name added to SectionLabel in the AiModelAdmin
- Assign user to documents ("Assignee"). Can be enabled in the ProjectSuperuserAdmin
- Add status field to the AiModel ("Training", "Failed", "Done")
- Dont allow new retraining if there is a training in progress AiModel.


## 2021-02-13_18-18-52

### Changed
- Use annotation permalink in LabelAdmin

### Fixed
- OCR Read API did not use text embeddings when available
- Files with misssing fonts could not be processed
- Creation of small annotations when accepting or declining

## 2021-02-10_13-52-15

### Added
- Admin action for Microsoft Graph API / Planner API

### Fixed
- SuperUserDocumentAdmin performance
- OutOfMemory errors in the categorization

## 2021-02-03_17-07-23

### Added
- Permalink for annotations
- Add an additional routine to fix corrupted pds
- Improved frontend error tracking

### Fixed
- Validation when edting an annotation

### Changed
- Renamed option 'priority_ocr' to 'priority_processing'
- Allow rerun extraction for documents with revised annotations
- Allow deletion default templates

## 2021-01-26_18-07-11

### Added
- Add column 'category' to csv export

## 2021-01-20_11-17-24

### Added
- Show selection bounding boxes for automtic created annotations

## 2021-01-14_22-06-52

### Added
- Visual annotations: images and area can now be annotate

### Fixed
- Loading time for Smartview

## 2021-01-13_23-26-03

### Fixed
- Retraining now assigns AIModels to templates even if they was no before

### Added
- Add Message when doing evaluation which tells the user if test set is empty.

## 2021-01-12_21-13-48

### Fixed
- Google Analytics integration
- Empty Textextraction for ParagraphExtractions

## 2021-01-10_18-36-49

### Fixed
- Disable link formatting by sendgrid.

## 2021-01-08_22-30-10

### Fixed
- Bbox calculation in ParagraphModel
- Evaluation sometimes not running
- Speedup annotation creating

## 2021-01-05_11-53-22

### Changed
- Two column Annotation selection is now possible

### Added
- ParagraphModel introduced in addition to the Extraction- & CategoryModels, this is set per project via the SuperUserDocumentAdmin.
- Option to update the document document text, this is set per project via the SuperUserDocumentAdmin.
- Document Segmentation API Endpoint

## 2020-12-22_19-04-04

### Changed
- Email Template are now managed within the application.
- Major improvement and refactor in the underlying training package.

### Fixed
- Link to imprint on SignUp
- Smartview when scrolling horizontally

## 2020-12-16_20-17-30

### Added
- Search for Smartview

### Fixed
- TemplateCreationForm does not allow to select parent template

## 2020-12-16_09-44-30

### Added
- Searchbar for SuperuserProjectAdmin
- Add link to flower (task monitoring) for superusers
- Add support for GoogleTag Manager
- Create Support Ticket for Retraining and Invitation of new Users

### Changed
- Increase SoftTimeLimit for extraction (necessary for large documents >500 pages).

### Fixed
- Fix bbox generation fox Paragraph Annotations
- Fixed Evaluation not triggered for new AiModels

## 2020-12-10_13-15-14

### Added
- Sentry error reporting for Javascript Frontend (i.e. Smartview)
- Allow to add Project specific document CategorizationModel

### Changed
- Document Search now considers filenames and shows links to Dashhboard, Labeling and Smartview
- Allow deletion of Labels

### Fixed
- Allow "None" as confidence for rule-base ExtractionModels

## 2020-12-01_21-08-32

### Added
- Proof of Concept Microsoft Graph API connection (for logged in users): app.konfuzio.com/graph
- Button to upload demo Documents
- SuperuserProjectAdmin added (same like previous ProjectAdmin, however only accessible for Superusers only)
- Google Analytics Tag for app.konfuzio.com

### Changed
- Default permission Group "CanReadProject" replaced with "CanCreateReadUpdateProject". New users can now create new Projects.
- Project Page for "normal" user does not show technical fields like "ocr" and "text_layout" anymore.
- Dont show file endings like '.pkl' for AiModels

## 2020-11-26_19-43-14

### Fixed
- Missing bbox attribute in Document API (prevents retraining via training package)
- Running of proper ExtractionModel in Multi-Document-Template project
- Loading time for the Document page (still room for improvements)

### Added
- Slightly better Categorization model.

## 2020-11-20_20-05-47

### Added
- A public registration page: https://app.konfuzio.com/accounts/signup 
- A Internal registration page to create users manually and faster: `https://app.konfuzio.com/register/` (you need to be logged in to see this page)
- Users can invite new users to a project via "ProjectInvitations"
- Password reset functionality

### Fixed
- The Smartview is much faster
- Improved creation of Templates and additional validation logic template inconsistencies.

### Changed
- Save bbox and entity per page in order to improve performance

## 2020-11-09_18-04-28

### Added
- Support for more than one default Template in a project
- Categorization for multi Template projects
- Links to related models in the Project, AIModel, Label and Template view
- Internal user registration form, app.konfuzio.com/register

### Changed
- AiModel belongs now to DefaultTemplates instead of project

## 2020-10-27_10-37-15

### Changed
- Documents are now soft-deleted. There is a hard delete option in the SuperuserDocumentAdmin.
- AiModel are made active automatically for matching DefaultTemplates if the AIMode is better than before.

## 2020-10-21_08-53-42

### Fixed
- Loading time when updating a project.

## 2020-10-19_22-46-49

### Changed
- Increase max allowed workflow time from 90 to 180 seconds.

### Fixed
- sucess messages for 'rerun_workflow' admin action
- loading time of AiModel
- csv export

### Added
- add hocr fied to document api.
- add a project option to hide the Smartview and Labeling tool.

## 2020-10-14_11-39-17

### Changed
- AIModel can be uploaded and evaluted before setting active for a project

## 2020-10-13_15-10-22

### Added
- Multilanguage Support (DE/EN) in the backend (actuall translation are not included yet)

### Changed
- 'create_labels_and_templates' is now a project option (false by default).
- Gunicorn workers restart after 500 requests.
- Flower dashboard is running in separated container now

### Fixed
- Fix upload_ai_model to upload files larger than 2GB
- Loading speed for SequenceAnnotation Admin

## 2020-10-03_15-18-47

### Fixed
- Recover tasks in case celery worker crashes

## 2020-10-01_12-02-37

### Fixed
- Internet Explorer warning badge
- 'Not machine-readable' was not detecting 0 as proper value for normalization.

### Changed
- Remove extraction count from AiModel admin.
- Refactor annotation accept/delete buttons to separate components and SVG

## 2020-09-16_18-19-53

### Added
- Additional normalization formats
- Sentry message if retraining is triggered.
- Detectron (fully imlemented) and preparation for visual classification results in SuperUserDocumetAdmin

### Changed 
- Dont raise sentry error if document got deleted during workflow 

### Fixed
- Creation of Templates
[- Calculation of width and height dimension when creating sandwich pdf and when using azure](https://gitlab.com/konfuzio/training/-/blob/master/src/konfuzio/image.py#L78)

## 2020-09-11_13-47-51

### Added
- Add sentry message if project retraining is triggered.
- Fix cpu minute calculation.

## 2020-09-09_16-12-44

### Added 
- [Forbid Removing Labels from Temapltes (which still have Annotations)](https://gitlab.com/konfuzio/objectives/-/issues/1629)

### Changed
- Allow extractions which does not have an accuracy. 
- On the dashboard: Dont show section.position column if all extractions have the same. Dont show accuracy column if all extraction does not have one.
- Dont show retraining webhook url (on the project detail page). Display is with **** like it is password.

## 2020-09-08_22-38-19

### Added
- Per-project measuring of cpu time.
- Additional date-formats for normalization.
- First draft of boolean-formats for normalization.

## 2020-09-08_09-17-00

### Added
- Document Filter added for 'human feedback required' and '100% machine readable.
- Additional normalization formats for numbers.
- [Document Categorization Classifier](https://gitlab.com/konfuzio/meta-clf) added to DocumentSuperUserAdmin

### Changed
- For the document view and Smartview, rename 'possibly incorrect' to 'not machine-readable'
- For the document view and Smartview, rename 'pending review' to 'require feedback'
- For the document view, divide column NOTES into FEEDBACK REQUIRED and NOT MACHINE-READABLE

### Fixed
- Dont raise an error if ai_model predict section with a template that does not exist.

## 2020-09-07_17-48-22

### Fixed
- Filter for 'possibly incorrect' shows wrong number.
- [Missing username in top right corner.](https://gitlab.com/konfuzio/objectives/-/issues/2431)
- [TypeError when running extract() on a document](https://gitlab.com/konfuzio/objectives/-/issues/2428).
- Sorting in csv is now correctly ordered by document_id and template position.
