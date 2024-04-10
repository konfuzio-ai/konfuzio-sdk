
## Internal release process

Every day when there is a verified and approved change a new pre-release of the SDK (master branch) is released to <https://pypi.org/project/konfuzio-sdk/#history> at 5:19 AM UTC (3:19 AM UTC+2, see code [here](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/.github/workflows/nightly.yml)).

Every day when there is a verified and approved change a new pre-release of the DVUI (main branch) is released to <https://www.npmjs.com/package/@konfuzio/document-validation-ui?activeTab=versions> at 5:19 AM UTC (3:19 AM UTC+2, see code [here](https://github.com/konfuzio-ai/document-validation-ui/blob/main/.github/workflows/dvui-prerelease.yml)).

Every day at 6:13 AM UTC a new nightly internal release of the Server using the latest pre-release of the SDK and the DVUI is deployed at <https://testing.konfuzio.com/> as a Gitlab schedule from our Server repository.

We get an early chance to find bugs with our integration of the SDK and the DVUI with the Konfuzio Server before the official release. During our internal development sprints (2 week periods) we follow the strategy summarized in the table below.

|  Key | Meaning       |
|------| ------------- |
| T    | Testing Time  |
| M    | Merge to next level |
| R    | Release       |
| B    | Bug Fixing    |
| CF    | Code-Freeze |


| Release  |                       | 1  |   2   |   3   |   4   |   5   |   6   |   7   |   8   |   9   |   10   |  +1  |   +2   |   +3   |   +4   |   +5   | 
| -------- | --------------------- | -- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ------ | ---- | ------ | ------ | ------ | ------ |
| Server   | testing Branch Server, using Server changes and latest official release of the SDK and DVUI |    |       |       |       |       |       |       |       |       | CF      | T    | B      | M      |        |        |
| SDK      | official release of SDK Master Branch|    |       |       |       |       |       |       |       |       |       |      |        |        |   R     |        |
|DVUI     | official release of DVUI Master Branch|    |       |       |       |       |       |       |       |       |       |      |        |        |   R     |        |
| Server   | master Branch Server  |    |       |       |       |       |       |       |       |       |        |      |        |       |   R     |        |

The strategy follows a 2 weeks sprint schedule (10 work days). The last five days from the diagram (+1 to +5) are not part of the sprint, but part of the final verification process). These additional days, in which final verification and validation is done, overlap with the first week of the next sprint. The SDK process is described in the following plan. The process with DVUI is completely analogous:

- During the sprint we do development on the SDK / Server / DVUI side, and we open one pull request for each new SDK feature on Github (see the list of currently open SDK pull requests [here](https://github.com/konfuzio-ai/konfuzio-sdk/pulls)).
- Once a pull request has passed the tests and has been reviewed it is merged to master, which triggers the creation of a SDK pre-release. This becomes available as a Konfuzio Server deployment the next day at <https://testing.konfuzio.com/>, as a consequence of a Konfuzio Server Gitlab schedule. This is an ongoing process and happens on demand.
- We internally test the Konfuzio SDK/Server integration with the latest pre-release and collect any bugs that come up, either from the SDK side or the Server side. These are scheduled as internal tickets for fixing until the second Friday, which marks the end of the sprint.
- The latest pre-release of the DVUI/SDK is used for testing the server after the end of the sprint. If there is a bug in the pre-release, it gets fixed and the final official release is published on the fourth day after the end of the sprint. If there is no bug, the pre-release is made to an official release.
- SDK Release Notes are automatically generated from our pull requests using [the Github’s feature](https://docs.github.com/en/repositories/releasing-projects-on-github/automatically-generated-release-notes). Each pull request includes links to relevant documentation about how to use the new feature, see for example <https://github.com/konfuzio-ai/konfuzio-sdk/pull/124>.
- All Server changes developed during the sprint are merged on the last day of the sprint - the second Friday. This final version of the Server, using the latest pre-releases of the SDK and DVUI, is deployed on <https://testing.konfuzio.com/>.
- The days after the sprint (+1 to +2) are used for Testing (happens on the Monday following the sprint - day '+1') and Bug Fixing (happens on the Tuesday following the sprint -day '+2') the version of the Server deployed on <https://testing.konfuzio.com/>. On the Wednesday after the sprint (day '+3') the testing branch is merged to the master branch. On the Thursday after the spint (day '+4') is time for the official release of the Server: the verified final version of the Server is delpoyed to the production environment on <https://app.konfuzio.com/>.
- Documenting the official release of the Server: see the [changelog](https://dev.konfuzio.com/web/changelog_app.html) for full information about each Konfuzio official Server release.

## How to release with GitHub to PyPI

1. Change the version number in the file VERSION use the format `v.X.X.X` without whitespaces.
   ![Update Python Package Version](/sdk/releasing/update-python-version.png)
2. Draft a Release [here](https://github.com/konfuzio-ai/konfuzio-sdk/releases/new).
   ![draft_new_release.png](/sdk/releasing/steps-to-draft-a-release.png)
   1. Create a new Tag on master, named as the version number in step 1.
   2. Add a title for the release
   3. Automatically generate the description using the Names of the merged Pull Requests
3. After you press publish release, a new Python Package will be uploaded to PyPI by a GitHub Action, see code
   [here](https://github.com/konfuzio-ai/konfuzio-sdk/blob/master/.github/workflows/release.yml). You can verify 
   that the Release was uploaded via having a look on [PyPI](https://pypi.org/project/konfuzio-sdk/#history)

## How to use nightly builds?

.. image:: /sdk/releasing/new-pypi-release.png

1. Install the latest pre-release `pip install --pre konfuzio_sdk` 
2. Force to pick the latest pre-release the version `pip install konfuzio_sdk>=0.2.3.dev0`. As PEP440 states: The 
   developmental release segment consists of the string .dev, followed by a non-negative integer value.  
   Developmental releases are ordered by their numerical component, immediately before the corresponding  release 
   (and before any pre-releases with the same release segment), and following any previous release (including any  
   post-releases)


.. Note:: 
   Pre-Releases don't use tags but reference commits. The version number of a pre-release relates to the 
   Year-Month-Date-Hour-Minute-Second of last commit date on branch master used to create this release.
   This process allows publish a new package if there are new commits on the master branch.

.. image:: /sdk/releasing/version-number-prerelease.png
