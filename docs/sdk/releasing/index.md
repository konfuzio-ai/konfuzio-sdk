.. _architecture-sdk-to-server:
## Architecture SDK to Server

The following chart is automatically created by the version of the diagram on the branch master, see [source](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/tests/SDK%20and%20Server%20Integration.drawio).

If you hover over the image you can zoom or use the full page mode.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/docs/sdk/SDK%20and%20Server%20Integration.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Fdocs%2Fsdk%2FSDK%2520and%2520Server%2520Integration.drawio"></script>

If you want to edit the diagramm, please refer to the [GitHub Drawio Documentation](https://drawio-app.com/github-support/).

## Directory Structure

```
├── konfuzio-sdk      <- SDK project name
│   │
│   ├── docs                    <- Documentation to use konfuzio_sdk package in a project
│   │
│   ├── konfuzio_sdk            <- Source code of Konfuzio SDK
│   │  ├── __init__.py          <- Makes konfuzio_sdk a Python module
│   │  ├── api.py               <- Functions to interact with the Konfuzio Server
│   │  ├── cli.py               <- Command Line interface to the konfuzio_sdk package
│   │  ├── data.py              <- Functions to handle data from the API
│   │  ├── settings_importer.py <- Meta settings loaded from the project
│   │  ├── urls.py              <- Endpoints of the Konfuzio host
│   │  └── utils.py             <- Utils functions for the konfuzio_sdk package
│   │
│   ├── tests                   <- Pytests: basic tests to test scripts based on a demo project
│   │
│   ├── .gitignore              <- Specify files untracked and ignored by git
│   ├── README.md               <- Readme to get to know konfuzio_sdk package
│   ├── pytest.ini              <- Configurations for pytests
│   ├── settings.py             <- Settings of SDK project
│   ├── setup.cfg               <- Setup configurations
│   ├── setup.py                <- Installation requirements

```
