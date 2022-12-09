.. meta::
:description: Use this section for explaining how the software works, without completing a specific task (see Guides).

# Architecture SDK to Server

The following chart is automatically created by the version of the diagram on the branch master, see [source](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/tests/SDK%20and%20Server%20Integration.drawio).

If you hover over the image you can zoom or use the full page mode.

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000ff&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom layers tags lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;url&quot;:&quot;https://raw.githubusercontent.com/konfuzio-ai/konfuzio-sdk/master/tests/SDK%20and%20Server%20Integration.drawio&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/embed2.js?&fetch=https%3A%2F%2Fraw.githubusercontent.com%2Fkonfuzio-ai%2Fkonfuzio-sdk%2Fmaster%2Ftests%2FSDK%2520and%2520Server%2520Integration.drawio"></script>

If you want to edit the diagramm, please refer to the [GitHub Drawio Documentation](https://drawio-app.com/github-support/).

# Directory Structure

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

# Internal release process

Every day at 5:19 AM UTC (3:19 AM UTC+2, see code [here](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/.github/workflows/nightly.yml)) a new nightly release of the SDK (master branch) is released to <https://pypi.org/project/konfuzio-sdk/#history>.

Every day at 6:13 AM UTC a new nightly release of the Server using the latest nightly SDK is deployed at <https://nightly-sdk.branch.konfuzio.com/> as a Gitlab schedule from our server repository.

We get an early chance to find bugs with our integration of the SDK with the Konfuzio Server before the official release. During our internal development sprints (2 week periods) we follow the strategy summarized in the table below.

|  Key | Meaning       |
|------| ------------- |
| T    | Testing Time  |
| M    | Merge to next level |
| R    | Release       |
| B    | Bug Fixing    |


| Release  |                       | 1  |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10   | 
| -------- | --------------------- | -- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ |
| Server   | master Branch Server  |    |       |       |       |       |       |       |       |       | M      |
| Server   | testing Branch Server |    |       |       |       |       |       | T     | B     | B     | R      |
| Server   | nightly Build SDK     |    |       |       |       |       |       | T     | B     | B     | R      |
| SDK      | master Branch         |    |       |       |       |       | T     | T     | B     | B     | R      |
| SDK      | Pull Request          |    |       |       |       |       | M     |       |       |       |        |
| DVUI     | master Branch         |    |       |       |       |       | T     | T     | B     | B     | R      |
| DVUI     | Pull Request          |    |       |       |       |       | M     |       |       |       | --     |

The strategy follows a 2 weeks sprint schedule (10 work days). The SDK process is described in the following plan. The process with DVUI is completely analogous:

- On the first week we do development on the SDK / Server / DVUI side, and we open one pull request for each new SDK feature on Github (see the list of currently open SDK pull requests [here](https://github.com/konfuzio-ai/konfuzio-sdk/pulls)).
- On the Monday of the second week we merge pull requests to master, which triggers the creation of a SDK nightly release. This becomes available as a Konfuzio Server deployment the next day at <https://nightly-sdk.branch.konfuzio.com/>, as a consequence of a Konfuzio Server Gitlab schedule.
- We internally test the Konfuzio SDK/Server integration with the nightly deployment and collect any bugs that come up, either from the SDK side or the Server side. For the SDK side, these are scheduled as internal tickets for fixing until Friday, which marks the end of the sprint.
- On Friday the bug fixing is over, the associated pull requests are merged to master and a new SDK official release is created containing the new features and bugfixes.
- SDK Release Notes are automatically generated from our pull requests using [the Github’s feature](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes). Each pull request includes links to relevant documentation about how to use the new feature, see for example <https://github.com/konfuzio-ai/konfuzio-sdk/pull/124>.
- The new SDK features are available on Friday evening at the end of each sprint. As we internally test and integrated the Konfuzio Server with each nightly SDK release, a new Server release is also available at app.konfuzio.com on the same Friday evening. See the [changelog](https://dev.konfuzio.com/web/changelog_app.html) for full information about each Konfuzio Server release.
