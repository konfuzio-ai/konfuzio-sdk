.. \_Server API:

# REST API

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
with [public documents](https://help.konfuzio.com/integrations/public-documents/), strictly require authentication. We
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
the [authentication DELETE endpoint](https://app.konfuzio.com/v3/swagger/#/auth/auth_destroy).

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

#### Konfuzio SDK example

To get a token, access it via `from konfuzio_sdk import KONFUZIO_TOKEN` (available only after `konfuzio_sdk init`).

To use a token:

```python
from konfuzio_sdk import KONFUZIO_TOKEN
from konfuzio_sdk.api import konfuzio_session

url = "https://app.konfuzio.com/api/v3/projects/"

# if you ran konfuzio_sdk init, you can run konfuzio_session() without explicitly specifying the token
session = konfuzio_session(KONFUZIO_TOKEN)  

response = session.get(url)

print(response.json())
```

To access this and other information via SDK's Data layer concepts, see [SDK Quickstart](https://dev.konfuzio.com/sdk/home/index.html) page.

### Response codes

All endpoints return an HTTP code that indicates the success of the request. Following the standard, codes starting
with `2` (`200`, `201`...) indicate success; codes starting with `4` (`400`, `401`...) indicate failure on the client
side, with the response body containing more information about what failed; codes starting with `5` (`500`, `502`...)
indicate failure on our side and are usually temporary (if they aren't, please
[contact us](https://konfuzio.com/support/)).

.. seealso::
The `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`\_ provides a more detailed breakdown of which response
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
For more filtering options, refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`\_ for the endpoint
that you want to filter.

### Ordering

Most endpoints that list resources support ordering on some fields. The ordering is passed as a single `GET` parameter
named `ordering` with the field name that you want to order by as the value.

You can combine multiple ordering fields by separating them with a `,`. For example: `?ordering=project,created_at`.

You can specify that you want the ordering to be reversed by prefixing the field name with a `-`. For
example: `?ordering=-created_at`.

.. seealso::
For a list of fields that can be used for ordering, refer to
the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`\_ for the endpoint that you want to order.

### Fields

Some endpoints allow you to override the default response schema and specify a subset of fields that you want to be
returned. You can specify the `fields` `GET` parameter with the field names separated by a `,`.

For example, you can specify `?fields=id,created_at` to only return the `id` and `created_at` fields in the response.

.. seealso::
Refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`\_ for a specific endpoint to see if it
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

## Guides and How-Tos

These guides will teach you how to do common operations with the Konfuzio API. You can refer to the
[general information](#general-information) section above for a general overview of how the API works and to our
[Swagger documentation](https://app.konfuzio.com/v3/swagger/) for a full list of all the available endpoints.

The example snippets use cURL, but you can easily convert them to your preferred language manually or using tools
like [cURL Converter](https://curlconverter.com).

The guides assume you already have a [token](#token-authentication) that you will use in the headers of
every API call. If you're copy-pasting the snippets, remember to replace `YOUR_TOKEN` with the actual token value.

### Setup a project with labels, label sets and categories

This guide will walk you through the API-based initial setup of a Project with all the initial data you need to start
uploading documents and training the AI.

#### Create a Project

First you need to set up a [Project](https://help.konfuzio.com/modules/projects/index.html). To do so, you will make a
call to our [Project creation endpoint](https://app.konfuzio.com/v3/swagger/#/projects/projects_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/projects/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"name": "My Project"}'
```

`name` is the only required parameter. You can check the endpoint documentation for more available options.

This call will return a JSON object that, among other properties, will show the `id` of the created Project. Take note
of it, as you will need it in the next steps.

#### Create a category

A [Category](https://help.konfuzio.com/modules/categories/index.html) is used to group Documents by type and can be
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

This call will return a JSON object that, among other properties, will show the `id` of the created Category. Take note
of it, as you will need it in the next steps. You can retrieve a list of your created Categories by sending a `GET`
request to the same endpoint.

#### Create some Labels

[Labels](https://help.konfuzio.com/modules/labels/index.html) are used to label Annotations with their business context.
In the case of our invoice Category, we might want to have Labels such as "amount" and "product". For each Label, we
need to make a different API request to
our [Label creation endpoint](https://app.konfuzio.com/v3/swagger/#/labels/labels_create):

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

`name` and `project` are the only required parameters, however we also want to associate these Labels to a Category.
Since Labels can be associated to multiple Categories, the `categories` property is a list of integers. (We only have
one, so in this case it's going to be a list with a single integer). Remember to replace `PROJECT_ID` and `CATEGORY_ID`
with the actual values you got from the previous steps. You can check the endpoint documentation for more available
options.

These calls will return a JSON object that, among other properties, will show the `id` of the created labels. Take note
of it, as you will need it in the next steps. You can retrieve a list of your created labels by sending a `GET` request
to the same endpoint.

#### Create a Label Set

A [Label Set](https://help.konfuzio.com/modules/sets/index.html) is used to group Labels that make sense together.
Sometimes these Labels might occur multiple times in a document — in our "invoice" example, there's going to be one set
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

`name` and `project` are the only required parameters, however we also want to associate this Label Set to the Category
and Labels we created. Both `categories` and `labels` are lists of integers you need to fill with the actual ids of the
objects you created earlier. For example, if our `category id` was `1`, and our `label id`s were `2` and `3`, we would
need to change the data we send like this: `"categories": [1], "labels": [2, 3]`. With `has_multiple_sections` set
to `true`, we also specify that this Label Set can be repeating, i.e. you can have multiple line items in a single
invoice.

#### Next steps

Your basic setup is done! You're now ready to upload Documents and train the AI.

### Upload a Document

After your initial project setup, you can start uploading Documents. To upload a Document, you will make a call to
our [Document creation endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_create).

.. note::
  Unlike most other endpoints, the Document creation endpoint only supports `multipart/form-data` requests (to support
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
- The `category` is optional. If present, `CATEGORY_ID` must be the ID of a Category belonging to your project. If this
  is not set, the app will try to automatically detect the Document category basaed on the available options.
- The `sync` parameter is optional. If set to `false` (the default), the API will immediately return a response after
  the upload, confirming that the Document was received and is now queuing for extraction. If set to `true`, the server
  will wait for the Document processing to be done before returning a response with the extracted data. This might take
  a long time with big documents, so it is recommended to use `sync=false` or set a high timeout for your request.
- The `callback_url` parameter is optional. If provided, the document details are sent to the specified URL via a POST
  request after the processing of the Document has been completed. Future Document changes via web interface or
  [API](https://app.konfuzio.com/v3/swagger/#/documents/documents_update) might also cause the callback URL to be
  called again if the changes trigger a re-extraction (for example when changing the Category of the Document).
- The `assignee` parameter is optional. If provided, it is the email of the user assigned to work on this Document,
  which must be a member of the Project you're uploading the Document to.
- Finally, `data_file` is the Document you're going to upload. Replace `LOCAL_FILE_NAME` with the path to the existing
  file on your disk, and if you're using the example code remember to keep the `@` in front of it.

The API will return the uploaded Document's ID and its current status. You can then use
the [Document retrieve endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_retrieve) to check if the
Document has finished processing, and if so, retrieve the extracted data.

### Create an Anotation

[Annotations](https://help.konfuzio.com/modules/annotations/) are automatically created by the extraction process when
you upload a Document, but if some data is missing you can annotate it manually to train the AI model to recognize it.

Creating an Annotation via the API requires the client to provide the bounding box coordinates of the relevant text
snippet, which is usually done in a friendly user interface like our SmartView. The request to create an Annotation
usually looks like this:

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/annotations/ \
  --header 'Authorization: Token YOUR_TOKEN' \
  --header 'Content-Type: application/json' \
  --data '{
  "document": DOCUMENT_ID,
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

- You _must_ specify either `annotation_set` or `label_set`. Use `annotation_set` if an Annotation Set already exists.
  You can find the list of existing Annotation Sets by using the `GET` endpoint of the Document. Using `label_set` will
  create a new Annotation Set associated with that Label Set. You can only do this if the Label Set
  has `has_multiple_sections` set to `true`. (See the note below for some examples.)
- `label` should use the correct `LABEL_ID` for your Annotation.
- `span` is a [list of spans](#coordinates-and-bounding-boxes).
- Other fields are optional.

To generate the correct `span` for your Annotation, we also provide the
[Document bbox retrieve endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_bbox_retrieve), which
can be called via `GET` to return a list of all the words in the Document with their bounding boxes, that you can use to
create your Annotations programmatically.

.. note::
  Annotation Sets are never created directly. When you create an Annotation, you can specify whether to re-use an
  existing Annotation Set, or to create a new one. You can refer to the following diagram to decide whether to use
  `annotation_set` or `label_set` in your request.

.. mermaid::
  graph TD
    A[Creating an Annotation in a Document<br>for Label <code>L</code> and Label Set <code>A</code>]
    A --> B[Can the Label Set <code>A</code> have multiple Annotation Sets?]
    B --> z[Yes] --> C[Is this Annotation<br>for a new Annotation Set<br>or an existing one <code>B</code>?]
    C --> x[New one] --> E["<code>label=L, label_set=A</code><br>(will create a new Annotation Set <code>C</code>)"]
    C --> y[Existing] --> D[<code>label=L, annotation_set=B</code>]
    B --> f[No] --> F[Does the Label Set <code>A</code><br> already have a single<br>corresponding Annotation Set <code>B</code>?]
    F --> G[Yes]
    G --> D
    G --> I["<code>label=L, label_set=A</code><br>(will reuse the existing Annotation Set <code>B</code>)"]
    F --> H[No]
    H --> E

### Create training data and train the AI

Once you have uploaded enough Documents and created enough Annotations, you can start training an
[extraction AI](https://help.konfuzio.com/modules/extractions/index.html). You will need at least one Document in the
"training" dataset for the Category you want to train, but more data is usually better (see our
[improving accuracy guide](https://help.konfuzio.com/tutorials/improve-accuracy/index.html)).

Then to train an AI you can simply call our
[Extraction AI create endpoint](https://app.konfuzio.com/v3/swagger/#/extraction-ais/extraction_ais_create) with the
ID of the Category the training Documents belong to:

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/extraction-ais/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"category": CATEGORY_ID}'
```

The training of the AI can take a while, depending on current server load and how large the training dataset is. You
will receive an email once the process is complete; you can also poll the
[Extraction AI detail endpoint](https://app.konfuzio.com/v3/swagger/#/extraction-ais/extraction_ais_detail) to see the
real time status of the process. The newly trained Extraction AI will then automatically be used to
extract machine-generated Annotations from newly uploaded Documents for that Category.

If you add new training/test Documents, or change existing ones, don't forget to train a new Extraction AI, otherwise
your modifications will not apply to the extraction process of new Documents. When training a new version of an AI, it
will be automatically set as the active one only if its
[evaluation results](https://help.konfuzio.com/modules/extractions/index.html?highlight=evaluation#evaluation) are
better than the previous AI's.

### Revise machine-generated annotations

You can revise the Annotations that are created automatically by an Extraction AI: this will help the next Extraction AI
training you create, as it will tell the system the points where the information it extract was correct and the points
where it was not.

To retrieve the list of Annotations for a document, you can use the Annotation list endpoint:

```
curl --request GET \
  --url https://app.konfuzio.com/api/v3/annotations/?document=DOCUMENT_ID \
  --header 'Authorization: Token YOUR_TOKEN'
```

A hierarchical list of Annotations in the context of Labels and Annotation Sets can also be found under the
`annotation_sets` property of the Document detail endpoint:

```
curl --request GET \
  --url https://app.konfuzio.com/api/v3/documents/DOCUMENT_ID/ \
  --header 'Authorization: Token YOUR_TOKEN'
```

.. note::
  The `annotation_sets` property contains both existing Annotation Sets and "potential" ones, i.e. Label Sets from the
  Document's Category which do not have a corresponding Annotation Set on the Document yet. These are easy to see
  because they don't have any Annotation and their `id` is `null`.

Whichever method you choose, you should be able to retrieve an ID for the Annotation(s) you want to revise. Unrevised
Annotations are easily filterable in the list because they have the properties `"revised": false` and
`"is_correct": false`. (See the
[Annotations documentation](https://help.konfuzio.com/modules/annotations/index.html#automated-annotations) for
more information.)

To mark an Annotation as _accepted_, you can then send a request like this one to the Annotation edit endpoint:

```
curl --request PATCH \
  --url https://app.konfuzio.com/api/v3/annotations/ANNOTATION_ID/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"revised": true, "is_correct": true}'
```

Conversely, to mark it as _declined_ you should send a request like this one:

```
curl --request PATCH \
  --url https://app.konfuzio.com/api/v3/annotations/ANNOTATION_ID/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"revised": true, "is_correct": false}'
```

Once there are no unrevised Annotations left in the document, the document is considered _reviewed_.

### Post-process a document: split, rotate and sort pages

We offer a [postprocess endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_postprocess_create)
that allows you to change uploaded Documents in three ways, which can be combined into a single API request:

- _Split_: divide a Document into two or more Documents, with the same total number of pages. Note: you cannot join
  documents that have been split, you will have to upload a new Document.
- _Rotate_: change the orientation of one or more pages in a Document, in multiple of 90 degrees.
- _Sort_: change the order of the pages in a document.

The endpoint accepts a list of objects, each one representing a single output Document. (If you're not using the
splitting functionality, this list should only contain one document). The `pages` property you send determines the
content of the Document.

### Download the OCR version of an uploaded Document

After uploading a Document, the Konfuzio server also creates a [PDF OCR version](#ocr-processing) of it with indexed and 
selectable text. This version is also used to generate images for each page for our SmartView functionality. If you
need it, you can download this OCR version of the Document: the `file_url` property of the
[document retrieve endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_retrieve) contains the URL to it
(relative to the Konfuzio installation: on the main server, `/doc/show/123/` would become
`https://app.konfuzio.com/doc/show/123/`); to access it, you need to be authenticated, so you would need a request like
this:

```
curl --request GET \
  --url https://app.konfuzio.com/doc/show/DOCUMENT_ID/ \
  --header 'Authorization: Token YOUR_TOKEN' \
  --remote-name --remote-header-name
```

This will save the file in the current directory.

### Create your own document dashboard

In cases where our [public documents and iframes](https://help.konfuzio.com/integrations/public-documents/) are not
enough, you can build your own solution. Here we explain how you can easily build a read-only dashboard for your
documents.

#### Start from our Vue.js code

Our document dashboard is based on Vue.js and completely implemented with the API v3. You can check out our solution
[on GitHub](https://github.com/konfuzio-ai/document-validation-ui) and customize it to your needs. You will find a
technical overview and component description [here](/dvui/index.html).

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
