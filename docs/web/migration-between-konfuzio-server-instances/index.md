# Migrations between Konfuzio server instances

## Migrate an extraction AI to another Konfuzio server instance.

This feature is only available for on-prem customers.

### Download extraction AI from source instance

In a first step the extraction AI needs to be downloaded from the target Konfuzio instance. In order to download the extraction AI you need to be a superuser.
The extraction AI can be downloaded from the superuser AI page.

![select_AI.png](./select_AI.png)

Click on the extraction AI you want to migrate.

![download_AI.png](./download_AI.png)

Download the AI file.

### Upload Extraction or Category AI to target instance

Upload the downloaded extraction AI via superuser AI page.

![upload_AI.png](./upload_AI.png)

### Note:

In case of an Extraction AI, a target category needs to be chosen. A project relation is made by means of choosing a "project - category" relation in "Available Category". No project should be assigned in the shown "Available projects" select box.
In comparison for a Categorization AI the targe project has to be chosen from "Available projects".


![create_label_sets.png](./create_label_sets.png)

If you upload the extraction AI to a new project without the labels and label set, you need to enable "Create labels and templates" on the respective project. 


## Migrate projects between Konfuzio server instances.

Export the project data from the source Konfuzio server system.  
```
pip install konfuzio_sdk  
konfuzio_sdk init  
konfuzio_sdk export_project <PROJECT_ID>
```

The export will be saved in a folder with the name data_<project_id>. This folder needs to be transferred to the target system
The first argument is the path to the export folder, the second is the project name of the imported project on the target system.
```
python manage.py project_import "/konfuzio-target-system/data_123/" "NewProjectName"
```


