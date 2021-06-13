

## Konfuzio API Video Tutorial

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NZKUrKyFVA8/0.jpg)](https://www.youtube.com/watch?v=NZKUrKyFVA8)

## Konfuzio Train AI Tutorial

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/fMiK1xRsNzY/0.jpg)](https://youtu.be/p7P964DJmCc)



## How to make an API Call

The API documentation is available at: http://app.konfuzio.com/swagger. To access the documentation please [register](https://app.konfuzio.com/accounts/signup/) beforehand.


The API supports Password Authentication and Token Authentication.
An API Token can be obtained here: https://app.konfuzio.com/v2/swagger/#/token-auth/token_auth_create

"ENDPOINT" needs to be replaced with the respective API endpoint.Username, password and token are randomized examples.

### Using CURL

Password authentication:
```bash
curl -u john.doe@example.com:9XQw92GZsJB2Ti  -X GET "https://app.konfuzio.com/api/ENDPOINT/"
```

Token authentication:
```bash
curl -H "Authorization: Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b" -X GET "https://app.konfuzio.com/api/ENDPOINT/"
```

### Using Python

Password authentication:
```python
import requests 
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth('john.doe@example.com', '9XQw92GZsJB2Ti')
r = requests.get(url="https://app.konfuzio.com/api/ENDPOINT/", auth=auth)
```

Token authentication:
```python
import requests 

headers = {'Authorization': 'Token 9944b09199c62bcf9418ad846dd0e4bbdfc6ee4b'}
r = requests.get(url="https://app.konfuzio.com/api/ENDPOINT/", headers=headers)
```

### Document Categorization API

The API provides an endpoint that allows to upload a document and also get directly its metadata.

```python
import os
import requests 
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth('USERNAME', 'PASSWORD')

categories = {"xx": "Category A", "yy": "Category B"}

# filepath - path to your local file
with open(filepath, "rb") as f:
    file_data = f.read()

files_data = {
    "data_file": (os.path.basename(filepath), file_data, "multipart/form-data"),
}

# sync = True to have directly the metadata of the file
# PROJECT_ID - id of your project in app konfuzio
data = {'project': PROJECT_ID, 'sync': True}

r = requests.post(url="https://app.konfuzio.com/api/v2/docs/", auth=auth, files=files_data, data=data)

code_category = r.json()["category_template"]
print(categories[str(code_category)])
```

## Document Segmentation API

The API provides an endpoint that allows for detection of different elements in a document such as: text, title, table,
list, and figure. For each element it is possible to get a classification and bounding box.

The model used on the background for this endpoint is a Mask-RCNN [1] trained on the PubLayNet dataset [2].

This model is available in the Detectron2 [3] platform, which is a platform from Facebook AI Research that implements state-of-the-art object detection algorithms. The weights of the model trained with PubLayNet can also be found online.

To visualize the results of this endpoint (using a document uploaded in the Konfuzio app):

```python
import requests 
from requests.auth import HTTPBasicAuth

auth = HTTPBasicAuth('USERNAME', 'PASSWORD')
url = f'https://app.konfuzio.com/api/projects/{PROJECT_ID}/docs/{DOC_ID}/segmentation/'
r = requests.get(url=url, auth=auth)
result = r.json()

image_path = document.image_paths[i]
image = Image.open(image_path).convert('RGB')

color_code  = {'text': (255, 0, 0),
               'title': (0, 255, 0),
               'list': (0, 0, 255),
               'table': (255, 255, 0),
               'figure': (0, 255, 255)}

for bbox in result[i]:
    label = bbox['label']
    pp1 = (int(bbox["x0"]), int(bbox["y0"]))
    pp2 = (int(bbox["x1"]), int(bbox["y1"]))
    image = cv2.rectangle(np.array(image), pp1, pp2, color_code[label], 1)

plt.imshow(image)
plt.show()

```

![segmentation_endpoint](../_static/img/segmentation.png)

[1] Kaming H. et al., "Mask R-CNN", 2018  
[2] Zhong, X. et al., "PubLayNet: largest dataset ever for document layout analysis", 2019  
[3] Yuxin, W. et al., Detectron2, GitHub repository, https://github.com/facebookresearch/detectron2, 2019  

## Supported file types

Konfuzio supports various file types:

### PDFs   
Konfuzio supports PDF/A-1, PDF/A-2, PDF/A-3, PDF/1.7, PDF/2.0. An attempt will be made to repair corrupted PDFs.

### Images
Konfuzio supports JPEG and PNG (including support for alpha channel). _Support for TIFF is experimental._

### Office documents
Konfuzio offers limited support for common office documents like Microsoft® Word (.doc, .docx), Excel (.xls, .xlsx), PowerPoint (.ppt, .pptx) and Publisher as well as the Open Document Format (ODF). Uploaded office documents are converted to PDFs by Konfuzio. Libre Office is used for the PDF conversion. The layout of the converted office document may differ from the original. Office files can not be edited after they have been uploaded.

