.. meta::
   :description: The FAQ section includes some of the most frequently asked questions regarding the Konfuzio SDK.


# FAQ

The FAQ section includes some of the most frequently asked questions regarding the Konfuzio SDK.

If you don't find what you are looking for, please add your question to the [Issue Tracker](https://github.com/konfuzio-ai/document-ai-python-sdk/issues).

## How to switch project?

If you have initialized your code with a project and now you want to switch to a different project, you have to rerun
`konfuzio_sdk init` in your working directory.

You will be asked the ID of the project you want to connect and you can also define the name of the folder where the new
data will be allocated (by default, the name of the folder will be built with the project ID).

To switch to another project **within a running session is not fully supported**.
The SDK was designed to connect to one project at a time.

However, if you need it to run tests in a specific project, for example, you can do it by previously downloading the
project data.

Let's say that you are running Konfuzio SDK with project A and you want to run a test with project B.

You need to:  
1) initialize the SDK with project B  
2) download the project data (`konfuzio_sdk download_data`)  
3) initialize the SDK with project A  
4) get project B with:  

```python
from konfuzio_sdk.data import Project

prj_B = Project(id_=id_prj_B, data_root=path_to_data_prj_B)
```
