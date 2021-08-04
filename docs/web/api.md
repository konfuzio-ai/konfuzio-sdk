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

## Supported File Types

Konfuzio supports various file types:

### PDFs   
Konfuzio supports PDF/A-1a, PDF/A-1b, PDF/A-2a, PDF/A-2b, PDF/A-3a, PDF/A-3b, PDF/X-1a, PDF/1.7, PDF/2.0. An attempt will be made to repair corrupted PDFs.

### Images
Konfuzio supports JPEG and PNG (including support for alpha channel). _Support for TIFF is experimental._

### Office documents
Konfuzio offers limited support for common office documents like Microsoft® Word (.doc, .docx), Excel (.xls, .xlsx), PowerPoint (.ppt, .pptx) and Publisher as well as the Open Document Format (ODF). Uploaded office documents are converted to PDFs by Konfuzio. Libre Office is used for the PDF conversion. The layout of the converted office document may differ from the original. Office files can not be edited after they have been uploaded.

## Supported Data Normalization

Our Konfuzio application can normalize various data formats for different data types used in the annotations of the documents within your project.
It translates different formats of numbers, percentages, date, and boolean values into a unique and machine-readable format. Down below you find an overview of the normalizations of the different data types with specific examples.


### 1. Numbers

Konfuzio is able to recognize numbers encoded in a written format to translate them into a data type that represents machine-readable numbers (i.e. float). Numbers from one to twelve are transformed, either from German or English language.
Given an annotation as an offset string, this annotation will be translated into an (absolute) number.  
Starting with the first case, a dash or a series of dashes which also might include commas or decimal points will be recognized as Zero.   
For the specific context not relevant signs, like quotation marks or whitespaces, are completely removed from the annotation. If the annotation contains the letter "S" or "H" after the digits, it will also be removed.  

Negative expressions are recognized either by dashes in front or after the digits, the letter "S" after the digits, or the number being framed into brackets.
For regular numbers, positive signs will be completely removed, while negative signs will be placed in front of the digits.
When dealing with absolute numbers, negative or positive signs are fully removed, as well as whitespaces or quotation marks for all number formats.

As there are varying ways to display float numbers, e.g. by using dots instead of commas to display the thousands mark or decimal numbers, Konfuzio also takes care of this to display it in a uniform format. 
The uniform chosen format uses the English/American standard. Hence, dots are used as a separation for decimal numbers and commas to mark the thousands with two decimal places. 

**To give you specific examples with the outputs down below in the table:**  
1) Expressions only consisting of one or multiple dashes are translated into Zero:    
e.g.: -,-    

2) For regular numbers, negative signs are placed in front of the digits; floats are displayed with two decimal places:   
e.g.: 59,00-     
e.g.: 786,71-    
e.g.: (118.704) 

3) Absolute numbers are shown without negative or positive signs in the common format as described above:    
e.g.: 59,00-  
e.g.: 786,71-  
e.g.: -2.759,7°   
e.g.: +159,;03  

4) Irrelevant signs and whitespaces are removed and it will be transformed into the unique format with a dot instead of a comma as the decimal separator:    
e.g.: :2.000, 08   
e.g.: -2.759,7°    
e.g.: €1.010.296  
e.g.: 7,375,009+

5) Written numbers are changed into a digits number format:  
e.g.: ein  
e.g.: eleven 

5) Certain cases can't be normalized from Konfuzio:  
e.g.: 43.34.34 

6) The expression "NIL" meaning "nothing" is translated into 0; however, strings including this expression can't be normalized:  
e.g.: NIL   
e.g.: StringThatIncludesNIL

   
| Input      | Example no.| Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |:-----------:|:-----------:|:-----------:|
|  -,- |1 | yes   | 0.0    | 0| string | number |
|  59,00- | 2| yes   | -59.0    | -59 | string | number |
|  786,71- |2| yes   | -786.71    | -786.71 | string | number |
| (118.704) | 2|yes   | -118704.0    | -118704 | string | number |
|  absolute no.: 59,00- |3| yes   | 59.0    | 59 | string | number |
|  absolute no.: 786,71- |3| yes   | 786.71    | 786.71 | string | number |
|  absolute no.: -2.759,7° |3| yes   | 2759.7    | 2759.7 | string | number |
|  absolute no.: +159,;03 |3| yes   | 159.03    | 159.03 | string | number |
|  :2.000, 08 |4| yes   | 2000.08    | 2000.08 | string | number |
|  -2.759,7° |4| yes   | -2759.7    | -2759.7 | string | number |
|  €1.010.296 | 4| yes   | 1010296.0    | 1010296 | string | number |
|  7,375,009+ | 4|yes   | 7375009.0  | 7375009 | string | number |
|  ein |5| yes   | 1.0   | 1 | string | number |
|  eleven |5| yes   | 11.0   | 11 | string | number |
| 43.34.34 | 6|no   | None    | null | string | null |
|  NIL |6| yes   | 0.0   | 0  | string |number | 
| StringThatIncludesNIL |6| no  | None   | null  | string | null |


### 2. Percentage Numbers

Konfuzio also handles percentage numbers of different formats and brings them into a machine-readable one. Percentage expressions are not displayed in the classic percentage format with the percentage sign %. The decimal number format without the percentage sign is used, but with a dot as a decimal separator and 4 decimal places. 

**To give you specific examples with the outputs down below in the table::**  
1) Digits that are separated by commas with two decimal places will be easily converted into a uniform format, either with or without the percentage sign in its original format:  
e.g.: 12,34   
e.g.: 12,34 %  
e.g.: 434,27%    
e.g.: 59,00-  
e.g.: 123,45  
e.g.: 0,00 


| Input      | Example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |:-----------: | :-----------: |  :-----------: |
|  12,34 |1 | yes   | 0.1234    | 0.1234 |  string | number |
|  12,34 % |1 | yes   | 0.1234    | 0.1234 |  string | number |
|  434,27% | 1| yes   | 4.3427  | 4.3427 |string | number |
|  59,00- | 1| yes   | 0.59  | 0.59 |string | number |
|  123,45 | 1| yes   | 1.2345  | 1.2345 |string | number |
|  0,00 |  1| yes   | 0.0    | 0 |string | number |


### 3. Date Values

Konfuzio applies the so-called ISO 8601 format (YYYY-MM-DD) for dates. It checks initially whether this format is already used which then won't be altered in this case. However, if another format is used, it will be adapted. In the next step, it checks if a format consisting only of a month and a year is used or even just a year without any other date information besides that.

Konfuzio recognizes written months either in German or English language and translates them into the iso format. 
                         
                              
**To give you specific examples with the outputs down below in the table::**
1) Dates, where the month is posted in a written expression with the year and the day as digits, can be transformed into the iso format without any problems:  
e.g.: 1. November 2019   
e.g.: 13 Mar 2020

2) If the year is indicated with just two digits, it will also be recognized, even if it's not separated with a sign like a dot or something similar:  
e.g.: 23.0919  
e.g.: (29.03.2018)

3) Given no information for the year, Konfuzio will assume 0000 by default:  
e.g.: /04.12.

4) If there is no information about the specific day, or even day and month, our application assumes the first day either of the respective year or year and month:  
e.g.: Oktober 2011  
e.g.: 2001

5) Date time values are translated into the iso format as well, but time values are removed:  
e.g.: 1993-02-05T00:00:00

6) Some cases can't be identified correctly or uniquely. Thus, Konfuzio can't transfer it into a date format and will return "None":  
e.g.: 14132020  
e.g.: 23.0K.2010  
e.g.: 30.07.2.90
   

| Input      | Example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: | :-----------: | :-----------: |  :-----------: |
|  1. November 2019 | 1| yes   | 2019-11-01   | 2019-11-01 |string | string | 
|  13 Mar 2020 | 1 | yes   | 2020-03-13    | 2020-03-13 |string | string | 
|  23.0919 | 2| yes   | 2019-09-23    | 2019-09-23 |string |string | 
|  (29.03.2018) |  2| yes   | 2018-03-29    | 2018-03-29 |string |string | 
|  /04.12. |  3 | yes   | 0000-12-04    | 0000-12-04 |string |string | 
|  Oktober 2011 | 4 | yes   | 2011-10-01    | 2011-10-01 |string |string | 
|  2001 | 4 | yes   | 2001-01-01    | 2001-01-01 | string |string | 
|  1993-02-05T00:00:00| 5 |yes  | 1993-02-05  | 1993-02-05 |string |string | 
|  14132020 |6 | no   | None    | null | string |null| 
|  23.0K.2010 |  6 |no   | None  | null |string | null | 
|  30.07.2.90 | 6 | no   | None  | null |string |null | 


### 4. Boolean values

Our application is also able to translate certain expressions into so-called boolean values, representing true or false values. This is based on certain pre-specified words. These words are representing positive or negative connotated responses with certain signal words which can be found down below in the _no_list_ and _yes_list_.


The pre-specified positive and negative (hence true and false) expressions:  
no_list = ['NEIN', 'NICHT', 'KEIN', 'OHNE', 'NO']  
yes_list = ['VORHANDEN', 'JA', 'MIT', 'YES']

Given the following examples, you can recognize how certain expressions are clustered into either _False_ or _True_ boolean values. If the expression is including one of the _no_ or _yes_ words from above, Konfuzio allocates it to _True_ or _False_. However, normalization is only possible if the signal words are the first word of the annotation to avoid false positives with word combinations of words from the _no_list_ and _yes_list_, e.g. nicht vorhanden or mit ohne.

**To give you specific examples with the outputs down below in the table::** 
1) The word _nicht_ (_no_) will be assigned to _false_.    
3) The expression _ja_ (_yes_) will be translated into _true_.  
4) Expressions including the no or yes signal words can be translated, but only if the expression is starting with this word:    
e.g.: nicht versichert   
e.g.: not insured   
e.g.: ja versichert   
e.g.: yes insured   

4) If the expression is not starting with the signal word, it can't be translated as there is the possibility of a combination of positive and negative connotated words:  
e.g.: versichert: ja  
e.g.: insured: yes   
e.g.: versichert ja   
e.g.: insured yes     

5) Annotations with more than 2 words can't be normalized from Konfuzio:  
e.g.: alleinstehend ohne Kind   

| Input      |Example no. | Able to convert?     | Output Excel/CSV | Output API | Datatype CSV | Datatype JSON |
| :-------------: | :----------: | :-----------: | :-----------: |  :-----------: | :-----------: |  :-----------: |
|  nicht | 1  | yes   | false    | false | string | boolean |
|  no | 1  | yes   | false     | false |string | boolean |
|  ja | 2  | yes   | true    | true |string | boolean |
|  yes | 2  | yes   | true   | true |string | boolean |
|  nicht versichert| 3  | yes   | false    | false |string | boolean |
|  not insured | 3  | yes   | false     | false | string | boolean |
|  ja versichert | 3  | yes   | true    | true| string | boolean |
|  yes insured | 3  | yes   | true  | true | string | boolean |
|  versichert: ja | 4  | no   | None    | null | string| null|
|  insured: yes | 4  | no   | None   | null | string | null |
|  versichert ja | 4  | no   | None     | null | string | null |
|  insured yes | 4  | no   | None   | null | string | null |
|  alleinstehend ohne Kind | 5  | no  | None  | null | string | null |


### 5. Known Issues and remarks

1) Limitations to other languages besides German: Konfuzio is optimized for the German language. It may partly support English expressions but is limited in the usage in English.

2) Limitations to the case sensitivity of boolean values: Classifying certain expressions as true or false values is based on pre-specified expressions. Thus, it is limited to these signal words as shown above in the lists when normalizing the inputs.

3) The non-normalizable values which can't be converted from Konfuzio will be marked as "not machine-readable" on the Konfuzio server and can be viewed in the smart view of the documents. 
 
4) Your excel output might differ in its format from the one defined above. This is due to the default formats from excel, e.g. the default date format which might change with the version of excel of your respective country.  


## Support

To qualify for minimum response times according to your SLA, you must report the issue using the email [support@konfuzio.com](emailto:support@konfuzio.com).
