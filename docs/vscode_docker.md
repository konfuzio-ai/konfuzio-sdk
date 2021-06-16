# How to debug python code inside a docker container

<h2>1. Step: Download and Install VS Code on your machine</h2>

Either use the link (https://code.visualstudio.com/download) or if you are on Linux and have snap installed just run (for this tutorial v1.56.2 was used):

```ubuntu
sudo snap install --classic code
```
If you have not already installed docker, download and install it here (https://docs.docker.com/get-docker/).

<h2>2. Step: Pull/Create your project that includes the relevant docker file</h2>

In most cases you are going to be using git, so just set up a new git-repository via the terminal, VS Code’s built-in console or VS Code’s git extension. This project should of course include the docker file that is used for deployment and which behavior you want to mimic locally.
 


If you just want to try out how this all works, I have set up a sample repository using the Konfuzio’s SDK. You can clone our SDK from its GitHub page (https://github.com/konfuzio-ai/document-ai-python-sdk.git).

<h2>3. Step: Install the remote development extension</h2>

In VS Code open the extensions tab (its icon is located in the left sidebar) and search **Remote - Containers** (or for its ID: *ms-vscode-remote.remote-containers*). Install the extension.

<!--[extensions tab](images/vscode_docker/extensions.PNG)-->

![remote development extension](images/vscode_docker/remote_dev_extension.png)

<h2>4. Step: Set up your remote development environment</h2>

You should now be able to find the remote extension's symbol (arrows on a green background) in the bottom left corner of the VS Code window (picture below). Clicking on the symbol opens the extension's command pallet, which from now on is going to be our main access point to the extension.

![green arrows](images/vscode_docker/green_arrows.png)

In the Command Pallet (*'View' > 'Command Pallet'*) select *'Remote-Containers: Add Development Container Configuration Files' > 'From $your_dockerfile'*

![command pallet](images/vscode_docker/command_pallet.PNG)

Now you should see in the file explorer under .devcontainer your devcontainer.json file. Open it. These are the basic configurations of your devcontainer. Most of this can be left unchanged. Maybe give your container a name by changing the 'name' variable. Additionally, you should specify all the ports you need inside your docker container in 'forwardPorts'.
If you are working with the sample project you do not need to specify any ports.

![devcontainer.json](images/vscode_docker/devcontainer.png)

<h2>5. Step: Build and run your docker container</h2>

Open the extension’s command pallet by clicking on the arrows in the bottom left and search for *‘Reopen Folder in Container’*. If you are doing this the first time, this builds the docker container and thus can take quite a bit of time.
 


To confirm that you are now inside the container look again to the bottom left. You should now be able to see *'Dev Container: $your_name'* next to the two arrows.

![green arrows with text](images/vscode_docker/green_arrows_with_text.png)

<h2>6. Step: Install the python extension inside the docker container to debug and run python files</h2>

Again open up the extensions tab (now inside the docker container) and install the python extension (ID: *ms-python.python*).
 


Now you can debug/run any python file you want. Open up the chosen python file and the 'Run and Debug' tab by clicking the run/debug icon that should be now available on the left taskbar.
 
<!--[run and debug tab](images/vscode_docker/run_and_debug.PNG)-->

Click *‘Run and Debug’ > ‘Python File’* and you are good to go. Before make sure to set the needed breakpoints by clicking to the left of the line numbers.

![debug point](images/vscode_docker/debug_point.png)

If you want to evaluate certain expressions while debugging, open up the terminal (if it is not open already) by clicking *‘View’ > ‘Terminal’*. One of the terminal's tabs is the debug console, where you can evaluate any expression.

If you are in the sample project you can make sure that the docker container works as expected by entering the tests folder (*'cd tests'*) and executing:

```ubuntu
pytest -m local
```

![tests](images/vscode_docker/tests.png)


<h2>Additional Tips:</h2>
 


- If you want to switch back to your local machine (to e.g. switch branch), open the extension’s command pallet by clicking on the arrows and select *‘Reopen Folder Locally’*.
 


- If you want to rebuild the container, because e.g. a different branch uses different dependencies, open the extension’s command palette and click *'Rebuild Container'*.
(This of course means that you have to reinstall the python extension - if this becomes annoying you can specify its ID in the devcontainer.json file to be pre-installed with every rebuild).
