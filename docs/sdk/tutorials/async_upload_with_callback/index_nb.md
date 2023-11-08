---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.2
  kernelspec:
    display_name: konfuzio
    language: python
    name: python3
---

## Asynchronously Uploading and Processing Multiple Files using Webhooks

---

**Prerequisites:**
- Understanding of the Konfuzio SDK and its usage, including authentication, [Project](https://dev.konfuzio.com/sdk/sourcecode.html#project) setup, and interacting with the Konfuzio API.
- Basic knowledge of web servers and how they handle incoming requests.
- Basic understanding of asynchronous programming concepts, including threads and event handling.
- Familiarity with ngrok and its setup process to expose a local server to the internet.
- Basic knowledge of working with files in Python, including reading and writing operations.
- Basic knowledge of HTTP protocols, particularly POST requests and JSON payloads.
- Comfortable navigating and executing commands in a terminal or command prompt.

**Difficulty:** Advanced

**Goal:**
This tutorial aims to guide users in efficiently uploading a large number of files to Konfuzio by utilizing the `Document.from_file` (see [docs](https://dev.konfuzio.com/sdk/sourcecode.html#document)) method in asynchronous mode. The primary objectives achieved through this tutorial are:

- Efficient File Upload
- Real-time Notifications
- Seamless Integration with ngrok
- Automated Update of Files
---

### Introduction

Uploading a large number of files to Konfuzio can be made highly convenient by employing the `Document.from_file` method in asynchronous mode. This approach allows for the simultaneous upload of multiple files without the need to wait for backend processing. However, a drawback of this method is the lack of real-time updates on processing status. This tutorial outlines a solution using a webhook callback URL to receive notifications once processing is complete, allowing for timely access to results.


### Preliminary Steps



1. **Set up Konfuzio**

    First, make sure that you have the Konfuzio SDK installed and that you have a Konfuzio account with a Project to use. 
    If you don't have this yet, please follow the instructions in the [Get Started guide](https://dev.konfuzio.com/sdk/get_started.html#get-started).

2. **Install Flask**

    Next, install Flask, which we will use to create a simple web server that will receive the callback from the Konfuzio
    Server. You can install Flask using pip:

    ```console
    pip install flask
    ```

3. **Set up ngrok**

    Then you will need to set up ngrok. If you already have a public web server able to receive post calls, you can 
    ignore this step and just use the callback URL to your web server's callback end point. To set up ngrok, first 
    create an account on the [ngrok website](https://ngrok.com/). It's free and you can use your GitHub or Google 
    account.

    Once logged into ngrok, simply follow the simple instructions available at https://dashboard.ngrok.com/get-started/setup
    On linux, all you need to do is:
    - Download ngrok
    - Follow the instructions to add the authentication token
    - Run this in a terminal:

    ```console
    ./ngrok http 5000
    ```
    This should give you the URL you can use as a callback URL. It should look something like 
    "https://abcd-12-34-56-789.ngrok-free.app".

Now that we have ngrok set up, we can see how to use it to pull the results of asynchronously uploaded files.


### Retrieving asynchronously uploaded files using a callback URL


1. **Import the necessary modules**

```python tags=["skip-execution"]
from flask import Flask, request
from konfuzio_sdk.data import Project, Document
import threading
from werkzeug.serving import run_simple
```

2. **Create a Project object**

You will find your Project id in the Konfuzio web interface.

```python tags=["skip-execution", "nbval-skip"]
project = Project(id_=YOUR_PROJECT_ID)
```

3. **Create a Flask application**

```python tags=["skip-execution", "nbval-skip"]
app = Flask(__name__)
```

4. **Set the callback URL**

You will find this callback url in the ngrok console where you ran `./ngrok http 5000`.

```python tags=["skip-execution", "nbval-skip"]
callback_url = YOUR_CALLBACK_URL  
# It should look something like "https://abcd-12-34-56-789.ngrok-free.app"
```

5. **Initialize data structures to share information between the threads**

We will use the main thread to host our Flask application and to receive the callback responses. We will use a separate thread to send the files to the Konfuzio Server. So, we will use the `callback_data_dict` to store the callback responses. The `data_lock` will be used to synchronize access to the `callback_data_dict` between the two threads, so that we can safely access it from both threads.

```python tags=["skip-execution", "nbval-skip"]
callback_data_dict = {}
data_lock = threading.Lock()
```

6. **Create a callback function**

Now we can create the callback function that will receive the callback responses from the Konfuzio server. We simply store the callback response in the `callback_data_dict` and set the `callback_received` event to notify the thread which is sending the files that the callback response has been received and that the files can be updated with the 
new OCR information.

```python tags=["skip-execution", "nbval-skip"]
@app.route('/', methods=['POST'])
def callback():
    data = request.json
    file_name = data.get('data_file_name')
    with data_lock:
        if file_name is not None and file_name in callback_data_dict:
            callback_data_dict[file_name]['callback_data'] = data
            callback_data_dict[file_name]['callback_received'].set()
    return '', 200
```

7. **Create a function to send your files asynchronously and update them once a callback response is received**

Now we can create the function that will send the files to the Konfuzio Server. We create a [Document](https://dev.konfuzio.com/sdk/sourcecode.html#document) object for each file and set the `sync` parameter to `False` to indicate that we want to upload the files asynchronously. We also set the `callback_url` parameter to the callback URL we created earlier.

We then start a thread for each Document to wait for the callback response to be received. Once the callback response for a Document has been received, we can update it with the OCR information.

```python tags=["skip-execution", "nbval-skip"]
def update_file(document, file_name):
    print(f'Waiting for callback for {document}')
    callback_data_dict[file_name]['callback_received'].wait()
    
    print(f'Received callback for {document}')

    # Once the callback is received we can update our Document with the OCR information    
    document.update()
    assert document.ocr_ready

    print(f'Updated {document} information with OCR results')

def send_files(file_names):
    for file_name in file_names:
        with data_lock:
            callback_data_dict[file_name] = {'callback_received': threading.Event(), 'callback_data': None, 'document': None}
        print(f'Sending {file_name} to Konfuzio servers...')
        document = Document.from_file(file_name, project=project, sync=False, callback_url=callback_url)
        with data_lock:
            callback_data_dict[file_name]['document'] = document

    # Wait for callbacks
    for file_name in callback_data_dict:
        threading.Thread(target=update_file, args=(callback_data_dict[file_name]['document'], file_name,)).start()
```

8. **Start the Flask application and upload the files**

Finally, we can start the Flask application and send the files. Simply add the path to all the files you want to upload. 

```python tags=["skip-execution", "nbval-skip"]
if __name__=='__main__':
    thread = threading.Thread(target=lambda: run_simple("0.0.0.0", 5000, app))
    thread.start()
    file_names = ['LIST.pdf', 'OF.jpg', 'FILES.tiff']
    threading.Thread(target=send_files, args=(file_names,)).start()
```

### Conclusion

In this tutorial, we've explored a powerful method for efficiently uploading a large number of files to Konfuzio using asynchronous mode and a webhook callback URL. By leveraging the Document.from_file method and ngrok for exposing a local server, we've enabled simultaneous file uploads without the need to wait for backend processing. Additionally, the implementation of a callback function ensures real-time notifications, allowing for timely access to results.

By following this tutorial, you've gained valuable insights into the seamless integration of Konfuzio with ngrok, optimizing your workflow for Document processing tasks. This approach not only enhances efficiency but also provides a foundation for building robust, automated solutions for Document management and analysis.
