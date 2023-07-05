## Using Ngrok to Pull Documents Uploaded Asynchronously

The most convenient way to upload a large number of files to the Konfuzio is to use the Document.from_file method in 
asynchronous mode. That way, you can upload multiple files without having to wait for them to be processed. The downside 
if this method is that you will not know when the processing is finished and you will be able to access the results. One 
solution to this problem is to use a callback URL. This URL will be called by the Konfuzio server when the processing is
finished. In this tutorial, we will show you how to upload multiple files and use ngrok to create a callback URL and how 
to use it to pull the OCR results.

### Preliminary Steps

First, make sure that you have the Konfuzio SDK installed and that you have a Konfuzio account with a Proejct to use. 
If you don't have this yet, please follow the instructions in the [Get Started guide](https://dev.konfuzio.com/sdk/get_started.html#get-started).

Next, install Flask, which we will use to create a simple web server that will receive the callback from the Konfuzio
server. You can install Flask using pip:

```console
pip install flask
```

Create ngrok account

Once logged into ngrok, simply follow the simple instructions available at https://dashboard.ngrok.com/get-started/setup
      
Download ngrok

Run 

```console
./ngrok http 5000
```


1. **Import the necessary modules**

    ```python
    from flask import Flask, request
    from konfuzio_sdk.data import Project, Document
    import threading
    ```

2. **Create a project object**

    ```python
    project = Project(id_=YOUR_PROJECT_ID)
    ```

3. **Create a Flask application**

    ```python
    app = Flask(__name__)
    ```

4. **Set the callback URL**

    ```python
    callback_url = YOUR_CALLBACK_URL  # It should look something like "https://abcd-12-34-56-789.ngrok-free.app"
    ```

    You will find this callback url in the ngrok console where you ran `./ngrok http 5000`.

5. **Initialize data structures to share information between the threads**

    ```python
    callback_data_dict = {}

    data_lock = threading.Lock()
    ```

    We will use the main thread to host our Flask application and to receive the callback responses. We will use a 
    separate thread to send the files to the Konfuzio server. 

    We will use the `callback_data_dict` to store the callback responses. The `data_lock` will be used to synchronize
    access to the `callback_data_dict` between the two threads, so that we can safely access it from both threads.


6. **Create a callback function**

    ```python
    @app.route('/', methods=['POST'])
    def callback():
        data = request.json
        file_name = data.get('data_file_name') # Adjust as necessary based on your actual callback data
        with data_lock:
            if thread_id is not None and thread_id in callback_data_dict:
                callback_data_dict[file_name]['callback_data'] = data
                callback_data_dict[file_name]['callback_received'].set()
        return '', 200
    ```

7. **Create a function to send your files asynchronously and update them once a callback response is received**

    ```python
    def send_files():
        file_names = ['pdf.pdf', 'test.pdf']
        docs = []
        for i, file_name in enumerate(file_names):
            with data_lock:
                callback_data_dict[file_name] = {'callback_received': threading.Event(), 'callback_data': None}
            print('Sending POST requests...')
            document = Document.from_file(file_name, project=project, sync=False, callback_url=callback_url)
            docs.append(document)

        # Wait for callbacks
        for file_name in callback_data_dict:
            callback_data_dict[thread_id]['callback_received'].wait()
            print(f'Received callback for {file_name}')
        
        for document in docs:
            document.update()

        print(docs)
    ```

8. **Start the Flask application and send the files**

    ```python
    if __name__=='__main__':
        threading.Thread(target=send_files).start()
        app.run(debug=True, use_reloader=False, port=5000)
    ```

