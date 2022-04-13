# Guides and How-Tos

These guides will teach you how to do common operations with the Konfuzio API. You can refer to our [API guide](./api_v3.html) for a general overview of how the API works and to our [Swagger documentation](https://testing.konfuzio.com/v3/swagger/) for a full list of all the available endpoints.

The example snippets use cURL, but you can easily convert them to your preferred language manually or using tools like [cURL Converter](https://curlconverter.com).

The guides assume you already have a [token](./api_v3.html#token-authentication) that you will use in the headers of every API call. If you're copy-pasting the snippets, remember to replace `<your token>` with the actual token value.

## Setup a project with labels, label sets and categories

This guide will walk you through the API-based initial setup of a project with all the initial data you need to start uploading documents and training the AI.

### Create a project

First you need to set up a [project](https://help.konfuzio.com/modules/projects/index.html). To do so, you will make a call to our [project creation endpoint](https://testing.konfuzio.com/v3/swagger/#/projects/projects_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/projects/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token <your token>' \
  --data '{"name": "My Project"}'
```

`name` is the only required parameter. You can check the endpoint documentation for more available options.

This call will return a JSON object that, among other properties, will show the `id` of the created project. Take note of it, as you will need it in the next steps.

### Create a category

A [category](https://help.konfuzio.com/modules/categories/index.html) is used to group documents by type and can be associated to an [extraction AI](https://help.konfuzio.com/modules/extractions/index.html). For example, you might want to create a category called "Invoice". To do so, you will make a call to our [category creation endpoint](https://testing.konfuzio.com/v3/swagger/#/categories/categories_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/categories/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token <your token>' \
  --data '{"project": <project id>, "name": "Invoice"}'
```

`name` and `project` are the only required parameters. Remember to replace `<project id>` with the actual `id` that you got from the previous step. You can check the endpoint documentation for more available options.

This call will return a JSON object that, among other properties, will show the `id` of the created category. Take note of it, as you will need it in the next steps. You can retrieve a list of your created categories by sending a `GET` request to the same endpoint.

### Create some labels

[Labels](https://help.konfuzio.com/modules/labels/index.html) are used to label annotations with their business context. In the case of our invoice category, we might want to have labels such as "amount" and "product". For each label, we need to make a different API request to our [label creation endpoint](https://testing.konfuzio.com/v3/swagger/#/labels/labels_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/labels/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token <your token>' \
  --data '{"project": <project id>, "name": "Amount", "categories": [<category id>]}'

curl --request POST \
  --url https://app.konfuzio.com/api/v3/labels/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token <your token>' \
  --data '{"project": <project id>, "name": "Product", "categories": [<category id>]}'
```

`name` and `project` are the only required parameters, however we also want to associate these labels to a category. Since labels can be associated to multiple categories, the `categories` property is a list of integers. (We only have one, so in this case it's going to be a list with a single integer). Remember to replace `<project id>` and `<category id>` with the actual values you got from the previous steps. You can check the endpoint documentation for more available options. 

These calls will return a JSON object that, among other properties, will show the `id` of the created labels. Take note of it, as you will need it in the next steps. You can retrieve a list of your created labels by sending a `GET` request to the same endpoint.

### Create a label set

A [label set](https://help.konfuzio.com/modules/sets/index.html) is used to group labels that make sense together. Sometimes these labels might occur multiple times in a document — in our "invoice" example, there's going to be one set of "amount" and "product" for each line item we have in the invoice. We can call it "line item" and we can create it with an API request to our [label set creation endpoint](https://testing.konfuzio.com/v3/swagger/#/label-sets/label_sets_create):

```
curl --request POST \
  --url https://app.konfuzio.com/api/v3/label-sets/ \
  --header 'Content-Type: application/json' \
  --header 'Authorization: Token <your token>' \
  --data '{"project": <project id>, "name": "Line Item", "has_multiple_sections": true, "categories": [<category id>], "labels": [<label ids>]}'
```

`name` and `project` are the only required parameters, however we also want to associate this label set to the category and labels we created. Both `categories` and `labels` are lists of integers you need to fill with the actual ids of the objects you created earlier. For example, if our `category id` was `1`, and our `label id`s were `2` and `3`, we would need to change the data we send like this: `"categories": [1], "labels": [2, 3]`. With `has_multiple_sections` set to `true`, we also specify that this label set can be repeating, i.e. you can have multiple line items in a single invoice.

### Next steps

Your basic setup is done! You're now ready to upload documents and train the AI.

## Create an annotation

Coming soon.

## Create training data and train the AI

Coming soon.

## Create your own document dashboard

In cases where our [public documents and iframes](http://help.konfuzio.com/integrations/public-documents/) are not enough, you can build your own solution. Here we explain how you can easily build a read-only dashboard for your documents.

### Start from our Vue.js code

Our document dashboard is based on Vue.js and completely implemented via our [public API](https://app.konfuzio.com/api/). You can check out our solution [on GitHub](https://github.com/konfuzio-ai/konfuzio-capture-vue) and customize it to your needs. You will find a technical overview and component description in the repository's `README`.

### Start from scratch

If you're using React, Angular or other technologies, you can use our [public API](https://app.konfuzio.com/api/) to build your own solution.

For feature parity with our read-only document dashboard, you only need to use two endpoints, and if you're only handling public documents, you don't need authentication for these two endpoints.

* The **document detail** (link to swagger) endpoint provides general information about the document you're querying, as well as its extracted data. Most of the data you will need is inside the `annotation_sets` object of the response.
* The **document page detail** (link to swagger) endpoint provides information about a document's page, including its `entites` (a list of words inside the page, with their coordinates) and its `page_image` (a URL you can use to load the image version of the page).

For more advanced use cases, you can refer to our [API help](https://app.konfuzio.com/api/) and/or contact support for guidance.
