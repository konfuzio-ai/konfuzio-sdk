.. meta::
   :description: Information about the Konfuzio Web Server including examples on how to use and communicate with it.

## Konfuzio API Video Tutorial

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/NZKUrKyFVA8/0.jpg)](https://www.youtube.com/watch?v=NZKUrKyFVA8)

## Konfuzio Train AI Tutorial

[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/fMiK1xRsNzY/0.jpg)](https://youtu.be/p7P964DJmCc)


## How to make an API Call

The API documentation is available at: https://app.konfuzio.com/swagger.

The API supports Password Authentication and Token Authentication.
An API Token can be obtained here: https://app.konfuzio.com/v2/swagger/#/token-auth/token_auth_create

"ENDPOINT" needs to be replaced with the respective API endpoint. Username, password, and token are randomized examples.

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

## Document Categorization API

The API provides an endpoint that allows to upload a document and also get directly its metadata.

```python
import os
import requests 
from requests.auth import HTTPBasicAuth
from konfuzio_sdk.data import Project

my_project = Project()
project_id = my_project.id

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
data = {'project': project_id, 'sync': True}

r = requests.post(url="https://app.konfuzio.com/api/v2/docs/", auth=auth, files=files_data, data=data)

code_category = r.json()["category_template"]
print(categories[str(code_category)])
```

## Document Segmentation API

The API provides an endpoint that allows the detection of different elements in a document such as text, title, table,
list, and figure. For each element, it is possible to get a classification and bounding box.

The model used on the background for this endpoint is a Mask-RCNN [1] trained on the PubLayNet dataset [2].

This model is available in the Detectron2 [3] platform, which is a platform from Facebook AI Research that implements state-of-the-art object detection algorithms. The weights of the model trained with PubLayNet can also be found online.

To visualize the results of this endpoint (using a document uploaded in the Konfuzio app):

```python
import cv2
import requests 
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from requests.auth import HTTPBasicAuth
from konfuzio_sdk.data import Project

my_project = Project()

project_id = my_project.id
# first training document uploaded in the project
document = my_project.documents[0]
doc_id = document.id

auth = HTTPBasicAuth('USERNAME', 'PASSWORD')
url = f'https://app.konfuzio.com/api/projects/{project_id}/docs/{doc_id}/segmentation/'
r = requests.get(url=url, auth=auth)
result = r.json()

# index of the page to test
page_index = 0

image_path = document.image_paths[page_index]
image = Image.open(image_path).convert('RGB')

color_code  = {'text': (255, 0, 0),
               'title': (0, 255, 0),
               'list': (0, 0, 255),
               'table': (255, 255, 0),
               'figure': (0, 255, 255)}

for bbox in result[page_index]:
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

## Supported OCR languages

The default OCR for https://app.konfuzio.com supports the following languages:

Afrikaans, Albanian, Arabic, Asturian, Basque, Bislama, Breton, Catalan, Cebuano, Chamorro, Chinese Simplified, Chinese Traditional, Cornish, Corsican, Crimean Tatar Latin, Czech, Danish, Dutch, English (incl. handwritten), Estonian, Fijian, Filipino, Finnish, French, Friulian, Galician, German, Gilbertese, Greek, Greenlandic, Haitian Creole, Hani, Hmong Daw Latin, Hungarian, Indonesian, Interlingua, Inuktitut Latin, Irish, Italian, Japanese, Javanese, K'iche', Kabuverdianu, Kachin Latin, Kara-Kalpak, Kashubian, Khasi, Korean, Kurdish Latin, Luxembourgish, Malay Latin, Manx, Neapolitan, Norwegian, Norwegian, Occitan, Polish, Portuguese, Romanian, Romansh, Russian, Scots, Scottish Gaelic, Serbian Cyrillic, Serbian Latin, Slovak, Slovenian, Spanish, Swahili Latin, Swedish, Tatar Latin, Tetum, Turkish, Upper Sorbian, Uzbek Latin, Volapük, Walser, Western Frisian, Yucatec Maya, Zhuang, Zulu.

## Supported file types

Konfuzio supports various file types:

### PDFs   
Konfuzio supports PDF/A-1a, PDF/A-1b, PDF/A-2a, PDF/A-2b, PDF/A-3a, PDF/A-3b, PDF/X-1a, PDF/1.7, PDF/2.0. An attempt will be made to repair corrupted PDFs.

### Images
Konfuzio supports JPEG and PNG (including support for alpha channel). _Support for TIFF is experimental._

### Office documents
Konfuzio offers limited support for common office documents like Microsoft® Word (.doc, .docx), Excel (.xls, .xlsx), PowerPoint (.ppt, .pptx) and Publisher as well as the Open Document Format (ODF). Uploaded office documents are converted to PDFs by Konfuzio. Libre Office is used for the PDF conversion. The layout of the converted office document may differ from the original. Office files can not be edited after they have been uploaded.

## Support

To qualify for minimum response times according to your SLA, you must report the issue using the email [support@konfuzio.com](emailto:support@konfuzio.com).
