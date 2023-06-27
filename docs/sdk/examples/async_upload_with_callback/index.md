## Using Ngrok to Pull Documents Uploaded Asynchronously

The most convenient way to upload a large number of files to the Konfuzio is to use the Document.from_file method in 
asynchronous mode. 


Install Flask

```console
pip install flask
```

Create ngrok account

Once logged in, simply follow the simple instructions available at https://dashboard.ngrok.com/get-started/setup
      
Download ngrok

Run 

```console
./ngrok http 5000
```


```python
from flask import Flask, request
from konfuzio_sdk.data import Project, Document
import requests
import threading
```


project = Project(id_=YOUR_PROJECT_ID)

app = Flask(__name__)

callback_received = threading.Event()
callback_data = None

@app.route('/', methods=['POST'])
def callback():
    global callback_data
    callback_data = request.json  # this is your callback data
    callback_received.set()  # signal that the callback has been received
    return '', 200

callback_url = YOUR_CALLBACK_URL  # It should look something like "https://abcd-12-34-56-789.ngrok-free.app"

def send_post():
    print('Sending POST request...')
    doc = Document.from_file('pdf.pdf', project=project, sync=False, callback_url=callback_url)
    callback_received.wait()  # wait here until the callback is received
    print('Received callback:', callback_data)
    doc.update()
    print(doc.text)

if __name__=='__main__':
    threading.Thread(target=send_post).start()  # send the POST request in a separate thread
    app.run(debug=True, port=5000)
```