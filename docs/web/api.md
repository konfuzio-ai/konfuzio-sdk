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

The default OCR engine for https://app.konfuzio.com supports the following languages:

Afrikaans, Albanian, Arabic, Asturian, Basque, Bislama, Breton, Catalan, Cebuano, Chamorro, Chinese Simplified, Chinese Traditional, Cornish, Corsican, Crimean Tatar Latin, Czech, Danish, Dutch, English (incl. handwritten), Estonian, Fijian, Filipino, Finnish, French, Friulian, Galician, German, Gilbertese, Greek, Greenlandic, Haitian Creole, Hani, Hmong Daw Latin, Hungarian, Indonesian, Interlingua, Inuktitut Latin, Irish, Italian, Japanese, Javanese, K'iche', Kabuverdianu, Kachin Latin, Kara-Kalpak, Kashubian, Khasi, Korean, Kurdish Latin, Luxembourgish, Malay Latin, Manx, Neapolitan, Norwegian, Norwegian, Occitan, Polish, Portuguese, Romanian, Romansh, Russian, Scots, Scottish Gaelic, Serbian Cyrillic, Serbian Latin, Slovak, Slovenian, Spanish, Swahili Latin, Swedish, Tatar Latin, Tetum, Turkish, Upper Sorbian, Uzbek Latin, Volapük, Walser, Western Frisian, Yucatec Maya, Zhuang, Zulu.

The availability of OCR languages depends on the selected OCR engine and might differ across configurations (e.g. on-premise installation).

## Supported file types

Konfuzio supports various file types:

### PDFs   
Konfuzio supports PDF/A-1a, PDF/A-1b, PDF/A-2a, PDF/A-2b, PDF/A-3a, PDF/A-3b, PDF/X-1a, PDF/1.7, PDF/2.0. An attempt will be made to repair corrupted PDFs.

### Images
Konfuzio supports JPEG and PNG (including support for alpha channel). _Support for TIFF is experimental._

### Office documents
Konfuzio offers limited support for common office documents like Microsoft® Word (.doc, .docx), Excel (.xls, .xlsx), PowerPoint (.ppt, .pptx) and Publisher as well as the Open Document Format (ODF). Uploaded office documents are converted to PDFs by Konfuzio. Libre Office is used for the PDF conversion. The layout of the converted office document may differ from the original. Office files can not be edited after they have been uploaded.

## Supported Data Normalization

Our Konfuzio application is able to normalize various data formats for different data types used in the annotations of the documents within your project.
It translates different formats of numbers, percentages, date and boolean values into a unique and machine-readable format. Down below you find an overview of the normalizations of the different data types with specific examples.

### 1. Numbers

Konfuzio is able to recognize numbers encoded in a written format to translate them into a data type that represents machine-readable numbers (i.e. float). Numbers from one to twelve are transformed, either from german or english language.
Given an annotation as an offset-string, this annotation will be translated into an (absolute) number.  
Starting with the first case, a dash or a series of dashes which also might include commas or decimal points will be recognized as Zero.   
For the specific context not relevant signs, like quotation marks or whitespaces, are completely removed form the annotation. If the annotation contains the letter "S" or "H" after the digits, it will also be removed.  

Negative expressions are recognized either by dashes in front or after the digits, the letter "S" after the digits or the number being framed into brackets.
For regular numbers, positive signs will be completely removed, while negative signs will be placed in front of the digits.
When dealing with absolute numbers, negative or positive signs are fully removed, as well as whitespaces or quotation marks for all number formats.

As there are varying ways to display float numbers, e.g. by using dots instead of commas to display the thousands mark or decimal numbers, Konfuzio also takes care of this to display it in a uniform format. 
The uniform chosen format uses a the english/american standard. Hence, dots are used as a separation for decimal numbers and commas to mark the thousands with two decimal places. 

**To give you specific examples:**  
1) Expressions only consisting of one or multiple dashes will be translated into Zero:    
e.g.: -,- = 0    

2) For regular numbers, negative signs will be placed in front of the digits and zeros after digits will be removed; floats are displayed with two decimal places:   
e.g.: 59,00- = -59    
e.g.: 786,71- = -786.71   
e.g.: (118.704) = -118704.0   

3) Absolute numbers are shown without negative or positive signs in the common format as described above:    
e.g.: 59,00- = 59  
e.g.: 786,71- = 786.71  
e.g.: -2.759,7° = 2759.7  
e.g.: +159,;03 = 159.03  

4) Irrelevant signs and whitespaces will be removed and it will be transformed into the unique format with a dot instead of a comma as the decimal separator:  
e.g.: :2.000, 08 = 2000.08  
e.g.: -2.759,7° = -2759.7    
e.g.: €1.010.296 = 1010296  
e.g.: 7,375,009+ = 7375009.0

5) Written numbers will be changed into a digits number format:  
e.g.: ein = 1  
e.g.: eleven = 11

5) Certain cases can't be normalized from Konfuzio:  
e.g.: 43.34.34 = None

6) The expression "NIL" meaning "nothing" will be translated into 0, however strings including this expression can't be normalized:  
e.g.: NIL = 0 

   
| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  -,- | yes   | 0.0    | 0|
|  59,00- | yes   | -59.0    | -59 |
|  786,71- | yes   | -786.71    | -786.71 |
| (118.704) | yes   | -118704.0    | -118704 |
|  absolute no.: 59,00- | yes   | 59.0    | 59 |
|  absolute no.: 786,71- | yes   | 786.71    | 786.71 |
|  absolute no.: -2.759,7° | yes   | 2759.7    | 2759.7 |
|  absolute no.: +159,;03 | yes   | 159.03    | 159.03 |
|  :2.000, 08 | yes   | 2000.08    | 2000.08 |
|  -2.759,7° | yes   | -2759.7    | -2759.7 |
|  €1.010.296 | yes   | 1010296.0    | 1010296 |
|  7,375,009+ | yes   | 7375009.0  | 7375009 |
|  ein | yes   | 1.0   | 1 |
|  eleven | yes   | 11.0   | 11 |
| 43.34.34 | no   | None    | null |
|  NIL | yes   | 0.0   | 0  |
| StringThatIncludesNIL | no  | None   | null  |


### 2. Percentage Numbers

Konfuzio also handels percentage numbers of different formats and brings them into a machine-readable one. Percentage expressions will not be displayed in the classic percentage format with the percentage sign %. The decimal number format without the percentage sign will be used, but with a dot as a decimal separator and 4 decimal places. 

**To give you specific examples:**  
1) Digits which are separated by commas with two decimal places will be easily converted into a uniform format, either with or without the percentage sign in its original format.  
e.g.: 12,34 = 0.1234  
e.g.: 12,34 % = 0.1234  
e.g.: 434,27% = 4.3427    
e.g.: 59,00- = 0.59  
e.g.: 123,45 = 12.345  
e.g.: 0,00 = 0  


| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  12,34 | yes   | 0.1234    | 0.1234 |
|  12,34 % | yes   | 0.1234    | 0.1234 |
|  434,27% | yes   | 4.3427  | 4.3427 |
|  59,00- | yes   | 0.59  | 0.59 |
|  123,45 | yes   | 1.2345  | 1.2345 |
|  0,00 | yes   | 0.0    | 0 |


### 3. Date Values

Konfuzio applies the so-called iso format (XXXX.XX.XX) for dates. It checks initially whether this format is already used which then won't be altered in this case. However, if another format is used, it will be adapted. In the next step, it checks if a format consisting only of a month and a year is used or even just a year without any other date information besides that.

Konfuzio recognizes written months either in german or english language and translates them into the iso format. 
                         
                              
**To give you specific examples:**
1) Dates where the month is posted in a written expression with the year and the day as digits can be transformed into the iso format without any problems.  
e.g.: 1. November 2019 = 2019-11-01   
e.g.: 13 Mar 2020 = 2020-03-13

2) If the year is indicated with just two digits, it will also be recognized, even if it's not separated with a sign like a dot or something similar.  
e.g.: 23.0919 = 2019-09-23  
e.g.: (29.03.2018) = 2018-03-29  

3) Given no information for the year, Konfuzio will assume 0000 by default.  
e.g.: /04.12. = 0000-12-04

4) If there is no information about the specific day, or even day and month, our application assumes the first day either of the respective year or year and month.  
e.g.: Oktober 2011 = 2011-10-01  
e.g.: 2001 = 2001-01-01

5) Date time values are translated into the iso format as well, but removing the time values:  
e.g.: 1993-02-05T00:00:00 = 1993-02-05

6) Some cases can't be identified correctly or uniquely. Thus, Konfuzio can't transfer it into a date format and will return "None".  
e.g.: 14132020 = None  
e.g.: 23.0K.2010 = None  
e.g.: 30.07.2.90 = None
   

| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  1. November 2019 | yes   | 2019-11-01   | 2019-11-01 |
|  13 Mar 2020 | yes   | 2020-03-13    | 2020-03-13 |
|  23.0919 | yes   | 2019-09-23    | 2019-09-23 |
|  (29.03.2018) | yes   | 2018-03-29    | 2018-03-29 |
|  /04.12. | yes   | 0000-12-04    | 0000-12-04 |
|  Oktober 2011 | yes   | 2011-10-01    | 2011-10-01 |
|  2001 | yes   | 2001-01-01    | 2001-01-01 |
|  1993-02-05T00:00:00| yes  | 1993-02-05  | 1993-02-05 |
|  14132020 | no   | None    | null |
|  23.0K.2010 | no   | None  | null |
|  30.07.2.90 | no   | None  | null |


### 4. Boolean values

Our application is also able to translate certain expressions into so-called boolean values, representing true or false values. This is based on certain pre-specified words. These words are representing positive or negative connotated responses with certain signal words which can be found down below in the _no_list_ and _yes_list_.


The pre-specified positive and negative (hence true and false) expressions:  
no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']  
yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

Given the following examples, you can recognize how certain expressions are clustered into either "False" or "True" boolean values. If the expression is including one of the "no" or "yes" words from above, Konfuzio allocates it to True or False. However, _only if yes/no word is not on last place in expression_!?

**To give you specific examples:** 
1) The word "nicht" ("no") will be assigned to "false".  
2) The expression "ja" ("yes") will be translated into "true".
3) Expressions including the no or yes signal words can be translated, but only if the expression is starting with this word:  
e.g.: nicht versichert = false  
e.g.: not insured = false  
e.g.: inkludiert: ja  = None  
e.g.: included: yes = None  
e.g.: inkludiert ja = None  
e.g.: included yes = None  
e.g.: ja inkludiert = true  
e.g.: yes included = true  
e.g.: alleinstehend ohne Kind = None  

5) Empty expressions, like " ", can't be translated. 


| Input      | Able to convert?     | Output Excel/CSV | Output API |
| :-------------: | :----------: | :-----------: | :-----------: |
|  nicht | yes   | false    | false |
|  no | yes   | false     | false |
|  ja | yes   | true    | true |
|  yes | yes   | true   | true |
|  nicht versichert| yes   | false    | false |
|  not insured | yes   | false     | false |
|  inkludiert: ja | no   | None    | null |
|  included: yes | no   | None   | null |
|  inkludiert ja | no   | None     | null |
|  included yes | no   | None   | null |
|  ja inkludiert | yes   | true    | true|
|  yes included | yes   | true  | true |
|  alleinstehend ohne Kind | no  | None  | null |
|   | no   |  not recognizable as annotation | not recognizable as annotation |


### 5. Known Issues and remarks

1) Limitations to other languages besides german: Konfuzio is optimized for the german language. It may partly support english expressions, but is limited in the usage in english.

2) Limitations to case sensitivity of boolean values: Classifying certain expressions as true or false values is based on pre-specified expressions. Thus, it is limited to these signal words as shown above in the lists when normalizing the inputs.

3) The non-normalizable values which can't be converted from Konfuzio will be marked as "not machine-readable" on app.konfuzio.com and can be viewed in the smart view of the documents. 
 
5) Your excel output might differ in its format from the one defined above. This is due to the default formats from excel, e.g. the default date format which might change with the version of excel of your respective country.  


## Support

To qualify for minimum response times according to your SLA, you must report the issue using the email [support@konfuzio.com](emailto:support@konfuzio.com).
