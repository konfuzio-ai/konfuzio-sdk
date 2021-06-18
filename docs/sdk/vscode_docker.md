.. meta::
   :description: Tutorial for how to debug Python code inside a Docker container. It uses Konfuzio SDk as example.

# How to debug Python code inside a Docker container

## 1. Step: Download and Install VS Code on your machine

Either use the link (https://code.visualstudio.com/download) or if you are on Linux and have snap installed just run 
(for this tutorial v1.56.2 was used):

```ubuntu
sudo snap install --classic code
```
If you have not already installed Docker, download and install it here (https://docs.docker.com/get-docker/).

## 2. Step: Pull/Create your project that includes the relevant Docker file

In most cases you are going to be using git, so just set up a new git-repository via the terminal, VS Code’s built-in 
console or VS Code’s git extension. This project should of course include the Docker file that is used for deployment 
and which behavior you want to mimic locally.
 


If you just want to try out how this all works, I have set up a sample repository using the Konfuzio’s SDK. You can 
clone our SDK from its GitHub page (https://github.com/konfuzio-ai/document-ai-python-sdk.git).

## 3. Step: Install the remote development extension

In VS Code open the extensions tab (its icon is located in the left sidebar) and search **Remote - Containers** (or for 
its ID: *ms-vscode-remote.remote-containers*). Install the extension.

<!--[extensions tab](images/vscode_docker/extensions.PNG)-->

![remote development extension](../_static/img/vscode_docker/remote_dev_extension.png)

## 4. Step: Set up your remote development environment

You should now be able to find the remote extension's symbol (arrows on a green background) in the bottom left corner of
the VS Code window (picture below). Clicking on the symbol opens the extension's command pallet, which from now on is 
going to be our main access point to the extension.

![green arrows](../_static/img/vscode_docker/green_arrows.png)

In the Command Pallet (*'View' > 'Command Pallet'*) select *'Remote-Containers: Add Development Container Configuration 
Files' > 'From $your_dockerfile'*

![command pallet](../_static/img/vscode_docker/command_pallet.png)

Now you should see in the file explorer under .devcontainer your devcontainer.json file. Open it. These are the basic 
configurations of your devcontainer. Most of this can be left unchanged. Maybe give your container a name by changing 
the 'name' variable. Additionally, you should specify all the ports you need inside your Docker container in 
'forwardPorts'.
If you are working with the sample project you do not need to specify any ports.

![devcontainer.json](../_static/img/vscode_docker/devcontainer.png)

## 5. Step: Build and run your Docker container

Open the extension’s command pallet by clicking on the arrows in the bottom left and search 
for *‘Reopen Folder in Container’*. If you are doing this the first time, this builds the Docker container and thus 
can take quite a bit of time.
 


To confirm that you are now inside the container look again to the bottom left. You should now be able to 
see *'Dev Container: $your_name'* next to the two arrows.

![green arrows with text](../_static/img/vscode_docker/green_arrows_with_text.png)

## 6. Step: Install the Python extension inside the Docker container to debug and run Python files

Again open up the extensions tab (now inside the Docker container) and install the Python extension (ID: *ms-python.python*).
 


Now you can debug/run any Python file you want. Open up the chosen Python file and the 'Run and Debug' tab by clicking 
the run/debug icon that should be now available on the left taskbar.
 
<!--[run and debug tab](images/vscode_docker/run_and_debug.PNG)-->

Click *‘Run and Debug’ > ‘Python File’* and you are good to go. Before make sure to set the needed breakpoints by 
clicking to the left of the line numbers.

![debug point](../_static/img/vscode_docker/debug_point.png)

If you want to evaluate certain expressions while debugging, open up the terminal (if it is not open already) by 
clicking *‘View’ > ‘Terminal’*. One of the terminal's tabs is the debug console, where you can evaluate any expression.

If you are in the sample project you can make sure that the Docker container works as expected by entering the tests 
folder (*'cd tests'*) and executing:

```ubuntu
pytest -m local
```

![tests](../_static/img/vscode_docker/tests.png)


## Additional Tips:
 


- If you want to switch back to your local machine (to e.g. switch branch), open the extension’s command pallet by 
  clicking on the arrows and select *‘Reopen Folder Locally’*.
 


- If you want to rebuild the container, because e.g. a different branch uses different dependencies, open the 
  extension’s command palette and click *'Rebuild Container'*.
(This of course means that you have to reinstall the Python extension - if this becomes annoying you can specify 
  its ID in the devcontainer.json file to be pre-installed with every rebuild).