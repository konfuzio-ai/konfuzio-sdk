.. \_Server API v3:

# REST API v3

This document aims to provide developers with a high-level overview of what can be accomplished through the Konfuzio API
v3. For a more thorough description of the available endpoints and their parameters and response, we invite you to
browse our [Swagger documentation](http:/app.konfuzio.com/v3/swagger/), which also provides an OpenAPI specification
that can be used to generate language-specific API clients.

<style>
.video-container {
  position: relative;
  width: 100%;
  padding-bottom: 56.25%;
  margin: 15px 0px;
}
.video {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  border: 0;
}
</style>

<div class="video-container">
    <iframe class="video" src="https://www.youtube.com/embed/tSk4dCKIQBg" allowfullscreen></iframe>
</div>

.. contents:: Table of Contents

## General Information

The Konfuzio API v3 follows REST conventions and principles. Unless specified otherwise, all endpoints accept both
JSON-encoded and form-encoded request bodies, according to the specified content type. All endpoints return JSON-encoded
responses. We use standard HTTP verbs (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`) for actions, and return standard HTTP
response codes based on the success or failure of the request.

### Authentication

Most of our endpoints, excluding those that deal
with [public documents](http://help.konfuzio.com/integrations/public-documents/), strictly require authentication. We
support three types of authentication.

#### Basic HTTP authentication

Your Konfuzio username (email) and password are sent with every request as HTTP headers in the
format `Authorization: Basic <string>`, where `<string>` is a Base64-encoded string in the
format `<username>:<password>` (this is usually done automatically by the HTTP client).

.. warning::
  While this approach doesn't require additional setup and is useful for testing in the Swagger page, it is
  **discouraged** for serious/automated use, since it usually involves storing these credentials in plain text on the
  client side.

#### Cookie authentication

A `sessionid` is sent in the `cookie` field of every request.

.. warning::
This `sessionid` is generated and used by the Konfuzio website when you log in to avoid additional authentication in API
requests, and should **not** be relied upon by third parties.

#### Single sign-on (SSO) authentication

SSO authentication is available through [KeyCloak](https://www.keycloak.org/), which is an open source identity and
access management solution. This functionality is only offered to our on-prem users. Further documentation regarding our
KeyCloak integration can be found on our
[on-prem documentation page](https://dev.konfuzio.com/web/on_premises.html#keycloak-integration).

#### Token authentication

You send a `POST` request with your Konfuzio username (email) and password to our [authentication endpoint](link), which
returns a token string that you can use in lieu of your actual credentials for subsequent requests, providing it with a
HTTP header in the format `Authorization: Token <token>`.

This token doesn't currently expire, so you can use indefinitely, but you can delete it (and regenerated) via
the [authentication DELETE endpoint](link).

.. note::
  This is the authentication method you **should** use if you're building an external service that consumes the Konfuzio
  API.

An example workflow would look like:

1. User registers to app.konfuzio.com with email "example@example.org" and password "examplepassword".
2. A `POST` request is sent to `https://app.konfuzio.com/v3/auth/`. The request is JSON-encoded with the following
   body: `{"username": "example@example.org", "password": "examplepassword"}`.
3. The endpoint returns a JSON-encoded request like `{"token": "bf20d992c0960876157b53745cdd86fad95e6ff4"}`.
4. For any subsequent request, the user provides the HTTP
   header `Authorization: Token bf20d992c0960876157b53745cdd86fad95e6ff4`.

#### cURL example

To get a token:

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/auth/ \
  --header 'Content-Type: application/json' \
  --data '{"username": "example@example.org", "password": "examplepassword"}'
```

To use the token:

```
curl --request GET \
  --url https://app.konfuzio.com/api/v3/projects/ \
  --header 'Authorization: Token bf20d992c0960876157b53745cdd86fad95e6ff4'
```

#### Python example

To get a token:

```python
import requests

url = "https://app.konfuzio.com/api/v3/auth/"

payload = {
    "username": "example@example.org",
    "password": "examplepassword"
}

response = requests.post(url, json=payload)

print(response.json())
```

To use the token:

```python
import requests

url = "https://app.konfuzio.com/api/v3/projects/"

headers = {
    "Authorization": "Token bf20d992c0960876157b53745cdd86fad95e6ff4"
}

response = requests.get(url, headers=headers)

print(response.json())
```

### Response codes

All endpoints return an HTTP code that indicates the success of the request. Following the standard, codes starting
with `2` (`200`, `201`...) indicate success; codes starting with `4` (`400`, `401`...) indicate failure on the client
side, with the response body containing more information about what failed; codes starting with `5` (`500`, `502`...)
indicate failure on our side and are usually temporary (if they aren't, please
[contact us](https://konfuzio.com/support/)).

.. seealso::
  The `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ provides a more detailed breakdown of which response
  codes are expected for each endpoint.

### Pagination

All endpoints that list resources are paginated. Pagination is achieved by providing `offset` and `limit` as `GET`
parameters to the request. `limit` is the maximum amount of items that should be returned, and `offset` is the amount of
items that should be skipped from the beginning.

For example, if you wanted the first 50 items returned by an endpoint, you should pass `?limit=50`. If you wanted the
next 50 items, you should pass `?limit=50&offset=50`, and so on.

Paginated responses always have the same basic structure:

```json
{
  "count": 123,
  "next": "http://api.example.org/accounts/?offset=400&limit=100",
  "previous": "http://api.example.org/accounts/?offset=200&limit=100",
  "results": [
    ...
  ]
}
```

- `count` is the total number of available items.
- `next` is the API URL that should be called to fetch the next page of items based on the current `limit`.
- `previous` is the API URL that should be called to fetch the previous page of items based on the current `limit`.
- `results` is the actual list of returned items.

### Filtering

All endpoints that list resources support some filtering, based on the resource being fetched. These filters are passed
as `GET` parameters and can be combined.

Two filters that are usually available on all list endpoints are `created_at_after` and `created_at_before`, which
filters for items that have been created after or before the specified date. So you could
use `?created_at_before=2022-02-01&created_at_after=2021-12-01` to only return items that have been created between
December 1, 2021 and February 1, 2022 (specified dates excluded).

.. seealso::
  For more filtering options, refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for the endpoint
  that you want to filter.

### Ordering

Most endpoints that list resources support ordering on some fields. The ordering is passed as a single `GET` parameter
named `ordering` with the field name that you want to order by as the value.

You can combine multiple ordering fields by separating them with a `,`. For example: `?ordering=project,created_at`.

You can specify that you want the ordering to be reversed by prefixing the field name with a `-`. For
example: `?ordering=-created_at`.

.. seealso::
  For a list of fields that can be used for ordering, refer to
  the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for the endpoint that you want to order.

### Fields

Some endpoints allow you to override the default response schema and specify a subset of fields that you want to be
returned. You can specify the `fields` `GET` parameter with the field names separated by a `,`.

For example, you can specify `?fields=id,created_at` to only return the `id` and `created_at` fields in the response.

.. seealso::
  Refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for a specific endpoint to see if it
  supports using the `fields` parameter. When supported, any field in the response schema can be used in the `fields`
  parameter.

### Coordinates and bounding boxes

There are three concepts related to coordinates and bounding boxes that are used throughout the API v3:

- **Bounding boxes** (or **bboxes**). A bbox is a rectangle representing a subset of a document page. It has the
  following properties:
  - `x0`, `xy`, `y0`, `y1`: the four points representing the coordinates of the rectangle on the page.
  - `page_index`: the page of the document the bbox refers too.
- **Spans**. A span, like the bbox, is a rectangle representing a subset of a document page; unlike the bbox, it also
  represents the _text data_ contained inside the rectangle. So it has the same properties as the bbox, but it adds
  more:
  - `offset_string` (optional when user-provided): the text contained inside this span. This can be manually set by the
    user if the text existing at the specified coordinates is wrong.
  - `offset_string_original` (read-only): the text that was originally present at the specified coordinates. This is
    usually the same as `offset_string` unless it has been changed manually.
  - `start_offset`, `end_offset` (read-only): the start and end character of the text contained inside this span, in
    relation to the document's text.
- **Character bounding boxes** (or **char bboxes**). A char bbox is a rectangle representing a single character on the
  page of a document. This is always returned by the Konfuzio server and cannot be set manually. It has the same
  properties as the bbox, but it adds more:
  - `text` (read-only): the single character contained by this bbox.
  - `line_index` (read-only): the line the character is in, related to all the lines in the document.

If the endpoint you're working with uses a `span` or `bbox` field, refer to its Swagger schema and to the summary above
to understand which fields it needs.


## Supported OCR languages

The default OCR engine for [Konfuzio Server](https://app.konfuzio.com) supports the following languages:

Afrikaans, Albanian, Asturian, Azerbaijani (Latin), Basque, Belarusian (Cyrillic), Belarusian (Latin), Bislama,
Bosnian (Latin), Breton, Bulgarian, Buryat (Cyrillic), Catalan, Cebuano, Chamorro, Chinese Simplified, Chinese
Traditional, Cornish, Corsican, Crimean Tatar (Latin), Croatian, Czech, Danish, Dutch, English, Erzya (Cyrillic),
Estonian, Faroese, Fijian, Filipino, Finnish, French, Friulian, Gagauz (Latin), Galician, German, Gilbertese,
Greenlandic, Haitian Creole, Hani, Hawaiian, Hmong Daw (Latin), Hungarian, Icelandic, Inari Sami, Indonesian,
Interlingua, Inuktitut (Latin), Irish, Italian, Japanese, Javanese, K'iche', Kabuverdianu, Kachin (Latin), Kara-Kalpak (
Latin), Kara-Kalpak (Cyrillic), Karachay-Balkar, Kashubian, Kazakh (Cyrillic), Kazakh (Latin), Khasi, Korean, Koryak,
Kosraean, Kumyk (Cyrillic), Kurdish (Latin), Kyrgyz (Cyrillic), Lakota, Latin, Lithuanian, Lower Sorbian, Lule Sami,
Luxembourgish, Malay (Latin), Maltese, Manx, Maori, Mongolian (Cyrillic), Montenegrin (Cyrillic), Montenegrin (Latin),
Neapolitan, Niuean, Nogay, Northern Sami (Latin), Norwegian, Occitan, Ossetic, Polish, Portuguese, Ripuarian, Romanian,
Romansh, Russian, Samoan (Latin), Scots, Scottish Gaelic, Serbian (Cyrillic), Serbian (Latin), Skolt Sami, Slovak,
Slovenian, Southern Sami, Spanish, Swahili (Latin), Swedish, Tajik (Cyrillic), Tatar (Latin), Tetum, Tongan, Turkish,
Turkmen (Latin), Tuvan, Upper Sorbian, Uzbek (Cyrillic), Uzbek (Latin), Volapük, Walser, Welsh, Western Frisian, Yucatec
Maya, Zhuang, Zulu.

The detection of handwritten text is supported for the following languages:

English, Chinese Simplified, French, German, Italian, Portuguese, Spanish.

The availability of OCR languages depends on the selected OCR engine and might differ across configurations (e.g.
on-premise installation).

## Supported File Types

### File Types

Konfuzio supports the following Document types.

For information about file size and page limits, refer to the Content Limits, if you are using Konfuzio SaaS.

| Name                                    | File Extension(s) | [MIME Type](https://www.iana.org/assignments/media-types/media-types.xhtml) |
| --------------------------------------- | ----------------- | --------------------------------------------------------------------------- |
| Portable Document Format (PDF)          | `.pdf`            | `application/pdf`                                                           |
| Tag Image File Format (TIFF)            | `.tiff`, `.tif`   | `image/tiff`                                                                |
| Joint Photographic Experts Group (JPEG) | `.jpg`, `.jpeg`   | `image/jpeg`                                                                |
| Portable Network Graphics (PNG)         | `.png`            | `image/png`                                                                 |
| Excel                                   | `.xls`, `.xlsx`   |  several, see details below                                                 |
| PowerPoint                              | `.ppt`, `.pptx`   |  several, see details below                                                 |
| Word                                    | `.doc`, `.docx`   |  several, see details below                                                 |


Note that some of these image formats are "lossy" (for example, JPEG). Reducing file sizes for lossy formats may result in a degradation of image quality and accuracy of results from Konfuzio.

#### PDFs

Konfuzio supports PDF/A-1a, PDF/A-1b, PDF/A-2a, PDF/A-2b, PDF/A-3a, PDF/A-3b, PDF/X-1a, PDF/1.7, PDF/2.0. An attempt
will be made to repair corrupted PDFs. Konfuzio does not support AcroForms and AEM (Adobe Experience Manager) form
content.

#### Images

Konfuzio supports JPEG, TIFF and PNG (including support for alpha channel).

#### Office documents

Konfuzio offers limited support for common office documents like Microsoft® Word (.doc, .docx), Excel (.xls, .xlsx),
PowerPoint (.ppt, .pptx) and Publisher as well as the Open Document Format (ODF). Uploaded office documents are
converted to PDFs by Konfuzio. Libre Office is used for the PDF conversion. The layout of the converted office document
may differ from the original. Office files can not be edited after they have been uploaded.

### Content limits

The following content limits apply to Konfuzio SaaS.

| Content limit                                                | Default Value                   |
| ------------------------------------------------------------ | ------------------------------- |
| Maximum image resolution (limit does not apply to PDF files) | megapixels not limited per page |
| Maximum file size per request                                | not limited                     |
| Maximum number of Pages per Document (synchronous requests)  | 250 pages                       |
| Maximum number of Pages (batch/asynchronous requests)        | 250 pages                       |
| Concurrent processor version training requests               | one per Category                |
| Concurrent files processing per Project (Batch / Parallel)   | not limited                     |
| Requests per minute                                          | not limited                     |
| Synchronous requests process requests per minute             | not limited                     |
| Asynchronous requests process requests per minute            | not limited                     |
| Number of pages in active processing                         | not limited                     |
| Review document requests per minute                          | not limited                     |


If you would like to increase your content limits, submit request for your project as a [Support Ticket](https://konfuzio.com/en/support/).

### Document scan resolution

For most accurate OCR results from Konfuzio, document scans should be a minimum of 200 dpi [(dots per inch)](https://en.wikipedia.org/wiki/Dots_per_inch). 300 dpi and higher will generally produce the best results.

## OCR Processing

After [uploading a document](https://help.konfuzio.com/modules/documents/index.html#upload-new-documents), depending on
how your project was set up, the document is automatically queued and processed by our OCR. In the below breakdown we
try to demystify this process, and give you some insight into all the steps which happen during OCR processing.

### Project settings

We first look at the projects settings to see what base settings have been set up for your documents.

1. **Chosen OCR engine**
    1. easy
    2. precise

2. **Chosen processing type**
    1. OCR
    2. Embedding
    3. Embedding and OCR

3. **Chosen auto-rotation option**

   (Only available for precise OCR engine)

    1. None
    2. Rounded
    3. Exact

### File pre-processing

During file upload, after the Project settings have been evaluated, we look at the file:

1. We check if the filetype is [supported](https://dev.konfuzio.com/web/api-v3.html#supported-file-types).
2. We check if the file is valid and/or corrupted.
3. If the file is corrupted, some repairing is attempted.
4. We check if the filetype provides embeddings.
5. We check if the project enforces OCR.
6. We then conduct OCR on the file.
7. We check if the image is angled.
8. We create thumbnails per page.
9. We create images per page.


### OCR Text extraction

During evaluation of both project settings and file, we also process OCR extraction

1. We use the chosen engine on the pre-processed file.
    1. If "Embedding and OCR" is chosen, internally we check which processing type is the most suitable, and use either
       Embedding or OCR
    2. Depending on chosen processing type, some pre-processing may be done:
        1. Convert non-PDF Documents to a [PDF](https://dev.konfuzio.com/web/api.html#pdfs) 
           that is being used here
        2. Convert PDF to text (in case of embeddings)
    3. If some sort of PDF corruption is detected, within our ability we attempt to repair the PDF
    4. If the PDF or TIFF is multi page (and valid) we split the document in pages and process each page separately
2. We check whether auto-rotation was chosen when the precise OCR engine is used]
    1. If rounded angle correction was chosen, we rotate the image to the nearest 45/90 degrees.
    2. If exact angle rotation was chosen, we rotate the image at its exact angle rotation value.
3. We attempt to extract the text from (either ocr, embedded or both)
    1. OCR may fail because text on the document is technically unreadable, the file is corrupted or empty and cannot be
       repaired
    2. OCR may fail because engine does not support the text language

Finally, we return you the extracted text.

## Guides and How-Tos

These guides will teach you how to do common operations with the Konfuzio API. You can refer to the
[general information](#general-information) section above for a general overview of how the API works and to our
[Swagger documentation](https://app.konfuzio.com/v3/swagger/) for a full list of all the available endpoints.

The example snippets use cURL, but you can easily convert them to your preferred language manually or using tools
like [cURL Converter](https://curlconverter.com).

The guides assume you already have a [token](#token-authentication) that you will use in the headers of
every API call. If you're copy-pasting the snippets, remember to replace `YOUR_TOKEN` with the actual token value.

### Setup a project with labels, label sets and categories

This guide will walk you through the API-based initial setup of a project with all the initial data you need to start
uploading documents and training the AI.

#### Create a project

First you need to set up a [project](https://help.konfuzio.com/modules/projects/index.html). To do so, you will make a
call to our [project creation endpoint](https://app.konfuzio.com/v3/swagger/#/projects/projects_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/projects/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"name": "My Project"}'
```

`name` is the only required parameter. You can check the endpoint documentation for more available options.

This call will return a JSON object that, among other properties, will show the `id` of the created project. Take note
of it, as you will need it in the next steps.

#### Create a category

A [category](https://help.konfuzio.com/modules/categories/index.html) is used to group documents by type and can be
associated to an [extraction AI](https://help.konfuzio.com/modules/extractions/index.html). For example, you might want
to create a category called "Invoice". To do so, you will make a call to
our [category creation endpoint](https://app.konfuzio.com/v3/swagger/#/categories/categories_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/categories/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"project": PROJECT_ID, "name": "Invoice"}'
```

`name` and `project` are the only required parameters. Remember to replace `PROJECT_ID` with the actual `id` that you
got from the previous step. You can check the endpoint documentation for more available options.

This call will return a JSON object that, among other properties, will show the `id` of the created category. Take note
of it, as you will need it in the next steps. You can retrieve a list of your created categories by sending a `GET`
request to the same endpoint.

#### Create some labels

[Labels](https://help.konfuzio.com/modules/labels/index.html) are used to label annotations with their business context.
In the case of our invoice category, we might want to have labels such as "amount" and "product". For each label, we
need to make a different API request to
our [label creation endpoint](https://app.konfuzio.com/v3/swagger/#/labels/labels_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/labels/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"project": PROJECT_ID, "name": "Amount", "categories": [CATEGORY_ID]}'

curl --request POST \
  --url https://app.konfuzio.com/api/v3/labels/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"project": PROJECT_ID, "name": "Product", "categories": [CATEGORY_ID]}'
```

`name` and `project` are the only required parameters, however we also want to associate these labels to a category.
Since labels can be associated to multiple categories, the `categories` property is a list of integers. (We only have
one, so in this case it's going to be a list with a single integer). Remember to replace `PROJECT_ID` and `CATEGORY_ID`
with the actual values you got from the previous steps. You can check the endpoint documentation for more available
options.

These calls will return a JSON object that, among other properties, will show the `id` of the created labels. Take note
of it, as you will need it in the next steps. You can retrieve a list of your created labels by sending a `GET` request
to the same endpoint.

#### Create a label set

A [label set](https://help.konfuzio.com/modules/sets/index.html) is used to group labels that make sense together.
Sometimes these labels might occur multiple times in a document — in our "invoice" example, there's going to be one set
of "amount" and "product" for each line item we have in the invoice. We can call it "line item" and we can create it
with an API request to
our [label set creation endpoint](https://app.konfuzio.com/v3/swagger/#/label-sets/label_sets_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/label-sets/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"project": PROJECT_ID, "name": "Line Item", "has_multiple_sections": true, "categories": [CATEGORY_ID], "labels": [LABEL_IDS]}'
```

`name` and `project` are the only required parameters, however we also want to associate this label set to the category
and labels we created. Both `categories` and `labels` are lists of integers you need to fill with the actual ids of the
objects you created earlier. For example, if our `category id` was `1`, and our `label id`s were `2` and `3`, we would
need to change the data we send like this: `"categories": [1], "labels": [2, 3]`. With `has_multiple_sections` set
to `true`, we also specify that this label set can be repeating, i.e. you can have multiple line items in a single
invoice.

#### Next steps

Your basic setup is done! You're now ready to upload documents and train the AI.

### Upload a document

After your initial project setup, you can start uploading documents. To upload a document, you will make a call to
our [document creation endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_create).

.. note::
  Unlike most other endpoints, the document creation endpoint only supports `multipart/form-data` requests (to support
  file uploading), so you won't have to JSON-encode your request this time.

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/documents/ \
  --header 'Content-Type: multipart/form-data' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --form project=PROJECT_ID \
  --form category=CATEGORY_ID \
  --form sync=true \
  --form callback_url=https://callback.example.org \
  --form assignee=example@example.org \
  --form data_file='@LOCAL_FILE_NAME';type=application/pdf
```

In this request:

- `PROJECT_ID` should be replaced with the ID of your project.
- The `category` is optional. If present, `CATEGORY_ID` must be the ID of a category belonging to your project. If this
  is not set, the app will try to automatically detect the document category basaed on the available options.
- The `sync` parameter is optional. If set to `false` (the default), the API will immediately return a response after
  the upload, confirming that the document was received and is now queuing for extraction. If set to `true`, the server
  will wait for the document processing to be done before returning a response with the extracted data. This might take
  a long time with big documents, so it is recommended to use `sync=false` or set a high timeout for your request.
- The `callback_url` parameter is optional. If provided, the document details are sent to the specified URL via a POST
  request after the processing of the document has been completed. Future document changes via web interface or
  [API](https://app.konfuzio.com/v3/swagger/#/documents/documents_update) might also cause the callback URL to be
  called again if the changes trigger a re-extraction (for example when changing the category of the document).
- The `assignee` parameter is optional. If provided, it is the email of the user assigned to work on this document,
  which must be a member of the project you're uploading the document to.
- Finally, `data_file` is the document you're going to upload. Replace `LOCAL_FILE_NAME` with the path to the existing
  file on your disk, and remember to keep the `@` in front of it.

The API will return the uploaded document's ID and its current status. You can then use
the [document retrieve endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_retrieve) to check if the
document has finished processing, and if so, retrieve the extracted data.

### Create an annotation

[Annotations](https://help.konfuzio.com/modules/annotations/) are automatically created by the extraction process when
you upload a document, but if some data is missing you can annotate it manually to train the AI model to recognize it.

Creating an annotation via the API requires the client to provide the bounding box coordinates of the relevant text
snippet, which is usually done in a friendly user interface like our SmartView. The request to create an annotation
usually looks like this:

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/documents/DOCUMENT_ID/annotations \
  --header 'Authorization: Token YOUR_TOKEN' \
  --header 'Content-Type: application/json' \
  --data '{
	"label": LABEL_ID,
	"label_set_id": LABEL_SET_ID,
	"is_correct": true,
	"is_revised": true,
	"span": [
		{
			"page_index": 0,
			"x0": 59.52,
			"x1": 84.42,
			"y0": 708.31,
			"y1": 718.31
		}
	]
}'
```

In this request:

- You _must_ specify either `annotation_set` or `label_set`. Use `annotation_set` if an annotation set already exists.
  You can find the list of existing annotation sets by using the `GET` endpoint of the document. Using `label_set` will
  create a new annotation set associated with that label set. You can only do this if the label set
  has `has_multiple_sections` set to `true`.
- `label` should use the correct `LABEL_ID` for your annotation.
- `span` is a [list of spans](#coordinates-and-bounding-boxes).
- Other fields are optional.

To generate the correct `span` for your annotation, we also provide the
[document bbox retrieve endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_bbox_retrieve), which
can be called via `GET` to return a list of all the words in the document with their bounding boxes, that you can use to
create your annotations programmatically.

### Create training data and train the AI

Coming soon.

### Create your own document dashboard

In cases where our [public documents and iframes](https://help.konfuzio.com/integrations/public-documents/) are not
enough, you can build your own solution. Here we explain how you can easily build a read-only dashboard for your
documents.

#### Start from our Vue.js code

Our document dashboard is based on Vue.js and completely implemented with the API v3. You can check out our solution
[on GitHub](https://github.com/konfuzio-ai/konfuzio-capture-vue) and customize it to your needs. You will find a
technical overview and component description in the repository's `README`.

#### Start from scratch

If you're using React, Angular or other technologies, you can use API v3 to build your own solution.

For feature parity with our read-only document dashboard, you only need to use two endpoints, and if you're only
handling public documents, you don't need authentication for these two endpoints.

- The [document detail](https://app.konfuzio.com/v3/swagger/#/documents/documents_retrieve) endpoint provides general
  information about the document you're querying, as well as its extracted data. Most of the data you will need is
  inside the `annotation_sets` object of the response.
- The [document page detail](https://app.konfuzio.com/v3/swagger/#/documents/documents_pages_retrieve) endpoint
  provides information about a document's page, including its `entites` (a list of words inside the page, with their
  coordinates) and its `page_image` (a URL you can use to load the image version of the page).

For more advanced use cases, you can refer to our [Swagger documentation](http:/app.konfuzio.com/v3/swagger/) and/or
contact support for guidance.
