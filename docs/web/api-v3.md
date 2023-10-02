.. \_Server API:

# REST API

This document aims to provide developers with a high-level overview of what can be accomplished through the Konfuzio API
v3. For a more thorough description of the available endpoints and their parameters and response, we invite you to
browse our [Swagger documentation](http:/app.konfuzio.com/v3/swagger/), which also provides an OpenAPI specification
that can be used to generate language-specific API clients.

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

You send a `POST` request with your Konfuzio username (email) and password to our [authentication endpoint](https://app.konfuzio.com/v3/swagger/#/auth), which
returns a token string that you can use in lieu of your actual credentials for subsequent requests, providing it with an
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

```bash
curl --request POST \
  --url https://app.konfuzio.com/api/v3/auth/ \
  --header 'Content-Type: application/json' \
  --data '{"username": "example@example.org", "password": "examplepassword"}'
```

To use the token:

```bash
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

#### Accessing and using the token via the Konfuzio SDK

To get an access token, simply run `konfuzio_sdk init` in the terminal and enter your login credentials. The token will
be stored in the `.env` file in your working directory. Then you are good to go and can use the SDK to access the API.

For more information on this and other information on what you can do with the SDK, see the [SDK Get Started](https://dev.konfuzio.com/sdk/get_started.html#initialize-the-package) page.

#### Authenticating with a Token as a Query Parameter (_Upcoming feature_)

.. Note::
  Please open a `support ticket` <https://konfuzio.com/support>`_ if you would like to have early access to this feature.

Sometimes you might need to use one of our endpoints with a third party service that doesn't allow you to specify
certain types of information (like authentication tokens) in the request. In these cases, you can use a special method
to authenticate your request using a token as a query parameter.

To do this, you'll first need to generate an authentication token using the steps we provided earlier. Then, you can
include the token in your request by adding it as a query parameter at the end of the URL.

Here's an example of what that might look like:

```html
http://app.konfuzio.com/api/v3/projects/3/export/all.csv?token=123456
```

Just replace 123456 with your own authentication token, and you'll be able to use this method to authenticate your
request.

.. Note::
  This functionality is offered as part of the Google-Sheet export function, as well as usable on
  any other csv importable software (SAP, Excel, etc).

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

### Coordinates and Bounding Boxes

There are three concepts related to coordinates and Bounding Boxes that are used throughout the API v3:

- **Bounding Boxes** (or **Bboxes**). A Bbox is a rectangle representing a subset of a Document Page. It has the
  following properties:
  - `x0`, `xy`, `y0`, `y1`: the four points representing the coordinates of the rectangle on the Page.
  - `page_index`: the Page of the Document the Bbox refers too.
- **Spans**. A Span, like the Bbox, is a rectangle representing a subset of a Document Page; unlike the Bbox, it also
  represents the _text data_ contained inside the rectangle. So it has the same properties as the Bbox, but it adds
  more:
  - `offset_string` (optional when user-provided): the text contained inside this Span. This can be manually set by the
    user if the text existing at the specified coordinates is wrong.
  - `offset_string_original` (read-only): the text that was originally present at the specified coordinates. This is
    usually the same as `offset_string` unless it has been changed manually. If the values of those two fields are different, the field `custom_offset_string` will be set to `true`.
  - `start_offset`, `end_offset` (read-only): the start and end character of the text contained inside this Span, in
    relation to the Document's text.
- **Character Bounding boxes** (or **char Bboxes**). A char Bbox is a rectangle representing a single character on the
  Page of a Document. This is always returned by the Konfuzio server and cannot be set manually. It has the same
  properties as the Dbox, but it adds more:
  - `text` (read-only): the single character contained by this Bbox.
  - `line_index` (read-only): the line the character is in, related to all the lines in the Document.

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

### Set up a Project with Labels, Label Sets and Categories

This guide will walk you through the API-based initial setup of a Project with all the initial data you need to start
uploading Documents and training the AI.

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

#### Create a Category

A [Category](https://help.konfuzio.com/modules/categories/index.html) is used to group Documents by type and can be
associated to an [extraction AI](https://help.konfuzio.com/modules/extractions/index.html). For example, you might want
to create a Category called "Invoice". To do so, you will make a call to
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

These calls will return a JSON object that, among other properties, will show the `id` of the created Labels. Take note
of it, as you will need it in the next steps. You can retrieve a list of your created Labels by sending a `GET` request
to the same endpoint.

#### Create a Label Set

A [Label Set](https://help.konfuzio.com/modules/sets/index.html) is used to group Labels that make sense together.
Sometimes these Labels might occur multiple times in a Document — in our "invoice" example, there's going to be one set
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

After your initial Project setup, you can start uploading Documents. To upload a Document, you will make a call to
our [Document creation endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_create).

.. note::
  Unlike most other endpoints, the Document creation endpoint only supports `multipart/form-data` requests (to support
  file uploading), so you won't have to JSON-encode your request this time.

#### Uploading a Document to Konfuzio

This tutorial will guide you on how to upload a Document to the Konfuzio platform.

Before starting, note that there are three possible ways to create a project on Konfuzio:

1.  Via the API (as shown in this tutorial)
2.  Using the web interface (for more information, watch the video tutorial at [help.konfuzio.com/quickstart/](https://chat.openai.com/c/help.konfuzio.com/quickstart/))
3.  By accepting an invitation to a ready Project (in this case, no setup is necessary on your part)

[After setting up your Project](https://dev.konfuzio.com/web/api-v3.html#create-a-project), you can start uploading
Documents. Document uploading is accomplished by making a POST request to our Document creation endpoint. Unlike most
endpoints, this one only supports multipart/form-data requests to accommodate file uploading, hence JSON-encoding your
request is not necessary this time.

#### Synchronous Document Upload

This tutorial will guide you on how to perform a synchronous Document upload to Konfuzio. In this mode, the server will
wait for the Document processing to finish before returning a response with the extracted data.

After setting up your Project, you can upload Documents. This is done by making a POST request to our Document creation
endpoint. This endpoint only supports multipart/form-data requests to facilitate file uploading, so you don't need to
JSON-encode your request.

```bash
curl --request POST\
  --url https://app.konfuzio.com/api/v3/documents/\
  --header 'Content-Type: multipart/form-data'\
  --header 'Authorization: Token YOUR_TOKEN'\
  --form project=PROJECT_ID\
  --form sync=true\
  --form data_file='@LOCAL_FILE_NAME';type=application/pdf
```

- Replace `YOUR_TOKEN` with your authorization token.
- `PROJECT_ID` should be replaced with the ID of your Project.
- `LOCAL_FILE_NAME` is the path to the file on your disk you wish to upload. Make sure to keep the `@` in front of it.
- The `sync` parameter is set to `true`, which means the server will wait for the Document processing to finish before
  returning a response.

Note that this synchronous mode might take a long time for large Documents, so it's recommended to set a high timeout
for your request. After the upload, the API will return the extracted data from the Document.

#### Asynchronous Document Upload with Assignee Notification

In this tutorial, we will guide you through the process of asynchronously uploading a Document to Konfuzio. This method
is particularly beneficial when dealing with large Documents or Documents with many Pages. We will also discuss how you
can use the `assignee` parameter to notify a specific user by email after the Document upload.

Here's an example of an asynchronous Document upload request, which includes the `assignee` parameter:

```bash
curl --request POST\
  --url https://app.konfuzio.com/api/v3/documents/\
  --header 'Content-Type: multipart/form-data'\
  --header 'Authorization: Token YOUR_TOKEN'\
  --form project=PROJECT_ID\
  --form sync=false\
  --form assignee=example@example.org\
  --form data_file='@LOCAL_FILE_NAME';type=application/pdf`
```

In this request:

- Replace `YOUR_TOKEN` with your authorization token.
- `PROJECT_ID` should be replaced with the ID of your Project.
- `CATEGORY_ID` is optional. If present, it must be the ID of a Category belonging to your Project. If this is not
  set, the app will try to automatically detect the Document category based on the available options.
- `sync` is set to false for asynchronous processing. With this, the API will immediately return a response after the
  upload, confirming that the Document was received and is now queuing for extraction.
- `callback_url` is optional. If provided, the Document details are sent to the specified URL via a POST request
  after the processing of the Document has been completed.
- `assignee` is optional. If provided, it is the email of the user assigned to work on this Document, which must be a
  member of the Project you're uploading the Document to.
- `LOCAL_FILE_NAME` is the path to the existing file on your disk you wish to upload. Keep the `@` in front of it.

Using the `assignee` parameter allows you to assign a specific user to work on the uploaded Document. The assignee, who
must be a member of your Project, will receive an email notification once the Document is uploaded. This feature is
beneficial when specific Documents require attention from specific team members, as it automatically informs them about
the new task.

After the upload, you can use the Document retrieve endpoint to check if the Document has finished processing, and if
so, retrieve the extracted data.

#### Document Upload by Disabling Automatic Categorization

This tutorial guides you on how to disable automatic categorization when uploading a Document to Konfuzio by specifying
the category of the Document manually. This can be useful when you want to ensure that a Document is assigned to a
specific Category, or when automatic categorization may not accurately categorize a Document due to its content or
layout.

Once your Project is set up, you can upload Documents. This is done by making a POST request to our Document creation
endpoint. This endpoint only supports multipart/form-data requests to facilitate file uploading, so you don't need to
JSON-encode your request.

```mermaid
sequenceDiagram
    Customer Software->>Konfuzio Server: Document POST (without category)
    Konfuzio Server-->>Konfuzio Server: Auto Categorization
    Konfuzio Server->>Konfuzio Server: Document Processing
    Konfuzio Server-->>Customer Software: Webhook*
    Customer Software->>Konfuzio Server: Document POST (with category)
    Konfuzio Server->>Konfuzio Server: Document Processing (No Categorization)
    Konfuzio Server-->>Customer Software: Webhook*
```

In this sequence diagram, the first flow represents the process when a Document is uploaded without a Category.
Konfuzio Server performs automatic categorization before processing the Document. A webhook is then sent to the
Customer Software after processing.

The second flow represents the process when a Document is uploaded with a Category. Since the Category is already
specified, Konfuzio Server skips the categorization step and directly processes the Document. After processing, a
webhook is sent to the Customer Software.

Here's an example of how to manually specify the Category of a Document:

```bash
curl --request POST\
  --url https://app.konfuzio.com/api/v3/documents/\
  --header 'Content-Type: multipart/form-data'\
  --header 'Authorization: Token YOUR_TOKEN'\
  --form project=PROJECT_ID\
  --form category=CATEGORY_ID\
  --form data_file='@LOCAL_FILE_NAME';type=application/pdf
```

- Replace `YOUR_TOKEN` with your authorization token.
- `PROJECT_ID` should be replaced with the ID of your Project.
- `CATEGORY_ID` is the ID of a Category that belongs to your Project. By specifying this, you are bypassing Konfuzio's
  automatic categorization process.
- `LOCAL_FILE_NAME` is the path to the file on your disk you wish to upload. Make sure to keep the `@` in front of it.

By manually specifying the Category of a Document, you ensure that the Document is categorized as you intend, without
relying on automatic categorization. This can be helpful in situations where the content or layout of a Document might
confuse the automatic categorization process. After the upload, the API will return the uploaded Document's ID and its
current status.

#### Asynchronous Document Processing with Webhook

This tutorial will guide you on how to use webhooks for asynchronous Document processing in Konfuzio.

Once your {p}roject is set up, you can upload Documents. This is done by making a POST request to our Document creation
endpoint. This endpoint only supports multipart/form-data requests to accommodate file uploading, so you won't have to
JSON-encode your request.

Webhooks allow you to receive a POST request from Konfuzio once the processing of your Document is complete. This
method is especially beneficial when dealing with large Documents or Documents with many Pages as it allows your
application to continue working on other tasks while the Document processing takes place. Here's an overview of the
process:

```mermaid
sequenceDiagram
    Customer Software->>Konfuzio Server: Document POST
    Konfuzio Server-->>Customer Software: Webhook*
```

In the sequence above, your software (Customer Software) sends a Document POST request to Konfuzio Server to upload a
Document. Once the processing of the Document is complete, Konfuzio Server sends a webhook (a POST request) back to
your software.

.. note::
  The webhook is currently only sent when the document is processed successfully. If the processing fails, the webhook
  is not sent. If you don't get a callback within 10 minutes of uploading a Document, use the
  `Document retrieve endpoint <https://app.konfuzio.com/v3/swagger/#/documents/documents_retrieve>`_ to see the status
  of the Document.

Here is how you can include a callback URL in your Document upload request to use webhooks:

```bash
curl --request POST\
  --url https://app.konfuzio.com/api/v3/documents/\
  --header 'Content-Type: multipart/form-data'\
  --header 'Authorization: Token YOUR_TOKEN'\
  --form project=PROJECT_ID\
  --form callback_url=https://callback.example.org\
  --form data_file='@LOCAL_FILE_NAME';type=application/pdf`
```

In this request:

- Replace `YOUR_TOKEN` with your authorization token.
- `PROJECT_ID` should be replaced with the ID of your Project.
- `callback_url` is the URL where the webhook should be sent once the Document is processed.
- `LOCAL_FILE_NAME` is the path to the file on your disk you wish to upload. Keep the `@` in front of it.

The `callback_url` parameter is optional. If provided, Konfuzio will send the Document details to the specified URL
via a POST request after the Document has been processed. This means your application can continue working on other
tasks and only needs to handle the Document data once it's ready.

This asynchronous approach is advantageous when dealing with Documents that contain many Pages or require a long
processing time, as it doesn't block your application while waiting for the response. Instead, your application is
notified via the webhook once the data is ready.

### Create an Annotation

[Annotations](https://help.konfuzio.com/modules/annotations/) are automatically created by the extraction process when
you upload a Document, but if some data is missing you can annotate it manually to train the AI model to recognize it.

Creating an Annotation via the API requires the client to provide the bounding box coordinates of the relevant text
snippet, which is usually done in a friendly user interface like our SmartView (see below for other options).
The [Annotations create endpoint](https://app.konfuzio.com/v3/swagger/#/annotations/annotations_create) accepts
requests that look like this:

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
- `span` is a [list of Spans](#coordinates-and-bounding-boxes).
- Other fields are optional.

As the `span` identifies a _position_ on the page, there are multiple ways to identify the correct one for the
Annotation you want to create:

1. The [Document Bbox endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_bbox_retrieve) returns an
   object with all the _characters_ from the Document with their coordinates. The characters can be identified by their
   offset (the keys in the object) and they can be easily converted in a list for the `span` attribute. You can also
   send a POST call to this endpoint with some coordinates to return a subset of the Document's characters that is
   completely contained into the sent coordinates.
2. The [Document Page endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_pages_retrieve) has an
   `entities` attribute that contains all the _words_ from the Document with their coordinates. These can be easily
   converted in a list for the `span` attribute.
3. The [Document search endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_search_create) takes a
   string as input and returns a list of all its occurrences in the Document. These can be fed directly to the `span`
   attribute.

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

### Review a Document

When working on a Document, the ultimate goal is to mark it as "reviewed", which means that all its Annotations have
been revised and the information inside them is correct.

To clarify how reviewing works, let's take a look at the statuses this data can go through:

.. mermaid::

  flowchart TD
    A(Feedback Required<br><small>Annotations created by AI)
    B(Unfilled<br><small>Potential Annotations that are<br>not found by the AI)
    C[Created by Human]
    D[Not Found<br><small>Missing Annotation instances</small>]
    E[Accepted]
    F[Declined]
    A --> E
    A --> F
    F --> B
    B --> C
    B --> D

- Annotations created by an AI extraction are initially marked as **Feedback Required**.
- They can be **Accepted**, which means that the information they contain is correct.
- They can be **Declined**, in case the information is wrong.
- Once an Annotation is Declined, or in case no Annotation was found for a specific Label, the Label (in the
  context of its Annotation Set) is considered **Unfilled**, and needs to be acted on.
- The user can manually select the part of the Document where the Unfilled Label is actually present to create an
  Annotation that is **Created by Human**.
- The user can signal that the Unfilled Label is **Not Found** in this Document by creating a Missing Annotation
  instance for this specific Label/Annotation Set combination.

This procedure will help the next Extraction AI training you create, as it will tell the system the where the
information it extracted was correct and the points where it was not. Once there are no "Feedback Required" and
"Unfilled" items, the Document can be marked as "reviewed".

To retrieve the list of Annotations for a Document, you can use the Annotation list endpoint:

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

If a specific Label does not exist at all in a Document, you can use the
[Missing Annotation endpoint](https://app.konfuzio.com/v3/swagger/#/missing-annotations) to tell the system about it:

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/missing-annotations/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"document": DOCUMENT_ID, "label": LABEL_ID, "label_set": LABEL_SET_ID}'
```

You can also see a list of all Missing Annotations that have been created for a document:

```
curl --request GET \
  --url https://app.konfuzio.com/api/v3/missing-annotations/?document=DOCUMENT_ID \
  --header 'Authorization: Token YOUR_TOKEN'
```

Once there are no Annotations left to be reviewed, and there are no Unfilled Labels, you can mark the Document as
"reviewed":

```
curl --request PATCH \
  --url https://app.konfuzio.com/api/v3/documents/DOCUMENT_ID/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token YOUR_TOKEN' \
  --data '{"is_reviewed": true}'
```

### Post-process a Document: split, rotate and sort Pages

We offer a [postprocess endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_postprocess_create)
that allows you to change uploaded Documents in three ways, which can be combined into a single API request:

- _Split_: divide a Document into two or more Documents, with the same total number of Pages. Note: you cannot join
  Documents that have been split, you will have to upload a new Document.
- _Rotate_: change the orientation of one or more Pages in a Document, in multiple of 90 degrees.
- _Sort_: change the order of the Pages in a Document.

The endpoint accepts a list of objects, each one representing a single output Document. (If you're not using the
splitting functionality, this list should only contain one Document). The `pages` property you send determines the
content of the Document.

### Document splitting suggestions

.. note::
  `Contact us <https://konfuzio.com/en/support/>`_ to enable this functionality.

The training data that was [previously created](#create-training-data-and-train-the-ai) can also be used to train a
Splitting AI to automatically propose splitting suggestions for uploaded documents.

To get started, you should "Enable Document splitting" in your
[Project settings](https://help.konfuzio.com/modules/projects/index.html#project-details); then, you can use our
[Splitting AI endpoints](https://app.konfuzio.com/v3/swagger/#/splitting-ais) to create a new Splitting AI, similar to
[how you train an Extraction AI](#create-training-data-and-train-the-ai).

Once this is done, when uploading a Document, you will notice an additional `proposed_split` field in the response.
This field contains a list of different Documents the AI thinks your original Document should be split into; each one
includes a Category, if it was found, and the list of Page IDs that should be part of that new Document. You can feed
this list, either as it is or after editing it and changing details, into the
[postprocess endpoint](https://app.konfuzio.com/v3/swagger/#/documents/documents_postprocess_create) to actualize
the AI's suggestions. You can also pass a list with one Document and all the page IDs to effectively reject the
suggestions and proceed with the original Document.

.. note::
  Once Document Splitting is enabled for a Project, newly uploaded Documents where splitting is detected will stay in
  the "Waiting for splitting confirmation" (`41`) status until the user takes action on the AI's suggestions. After
  that, extraction will run as usual on the resulting Documents.

After being split, the new Documents will keep a reference to the original "Document Set" via the `document_set`
property. Querying the [Document Sets endpoint](https://app.konfuzio.com/v3/swagger/#/document-sets) with that ID will
return all the existing Documents derived from the same original Document.

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
