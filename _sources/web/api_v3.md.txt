# API Guide (v3)

This document aims to provide developers with a high-level overview of what can be accomplished through the Konfuzio API v3. For a more thorough description of the available endpoints and their parameters and response, we invite you to browse our [Swagger documentation](http:/app.konfuzio.com/v3/swagger/), which also provides an OpenAPI specification that can be used to generate language-specific API clients.

.. contents:: Table of Contents
   :depth: 1
   :local:
   :backlinks: none


## General Information

The Konfuzio API v3 follows REST conventions and principles. Unless specified otherwise, all endpoints accept both JSON-encoded and form-encoded request bodies, according to the specified content type. All endpoints return JSON-encoded responses. We use standard HTTP verbs (`GET`, `POST`, `PUT`, `PATCH`, `DELETE`) for actions, and return standard HTTP response codes based on the success or failure of the request.


## Authentication

Most of our endpoints, excluding those that deal with [public documents](http://help.konfuzio.com/integrations/public-documents/), strictly require authentication. We support three types of authentication.

### Basic HTTP authentication

Your Konfuzio username (email) and password are sent with every request as HTTP headers in the format `Authorization: Basic <string>`, where `<string>` is a Base64-encoded string in the format `<username>:<password>` (this is usually done automatically by the HTTP client). 

.. warning::
  While this approach doesn't require additional setup and is useful for testing in the Swagger page, it is **discouraged** for serious/automated use, since it usually involves storing these credentials in plain text on the client side.

### Cookie authentication

A `sessionid` is sent in the `cookie` field of every request.

.. warning::
  This `sessionid` is generated and used by the Konfuzio website when you log in to avoid additional authentication in API requests, and should **not** be relied upon by third parties.

### Token authentication

You send a `POST` request with your Konfuzio username (email) and password to our [authentication endpoint](link), which returns a token string that you can use in lieu of your actual credentials for subsequent requests, providing it with a HTTP header in the format `Authorization: Token <token>`.

This token doesn't currently expire, so you can use indefinitely, but you can delete it (and regenerated) via the [authentication DELETE endpoint](link).

.. note::
  This is the authentication method you **should** use if you're building an external service that consumes the Konfuzio API.

An example workflow would look like:

1. User registers to app.konfuzio.com with email "example@example.org" and password "examplepassword".
2. A `POST` request is sent to `https://app.konfuzio.com/v3/auth/`. The request is JSON-encoded with the following body: `{"username": "example@example.org", "password": "examplepassword"}`.
3. The endpoint returns a JSON-encoded request like `{"token": "bf20d992c0960876157b53745cdd86fad95e6ff4"}`.
4. For any subsequent request, the user provides the HTTP header `Authorization: Token bf20d992c0960876157b53745cdd86fad95e6ff4`.

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

url = "https://testing.konfuzio.com/api/v3/auth/"

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

url = "https://testing.konfuzio.com/api/v3/projects/"

headers = {
    "Authorization": "Token bf20d992c0960876157b53745cdd86fad95e6ff4"
}

response = requests.get(url, headers=headers)

print(response.json())
```


## Response codes

All endpoints return an HTTP code that indicates the success of the request. Following the standard, codes starting with `2` (`200`, `201`...) indicate success; codes starting with `4` (`400`, `401`...) indicate failure on the client side, with the response body containing more information about what failed; codes starting with `5` (`500`, `502`...) indicate failure on our side and are usually temporary (if they aren't, please [contact us](link)).

.. seealso::
  The `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ provides a more detailed breakdown of which response codes are expected for each endpoint.


## Pagination

All endpoints that list resources are paginated. Pagination is achieved by providing `offset` and `limit` as `GET` parameters to the request. `limit` is the maximum amount of items that should be returned, and `offset` is the amount of items that should be skipped from the beginning.

For example, if you wanted the first 50 items returned by an endpoint, you should pass `?limit=50`. If you wanted the next 50 items, you should pass `?limit=50&offset=50`, and so on.

Paginated responses always have the same basic structure:

```json
{
    "count": 123,
    "next": "http://api.example.org/accounts/?offset=400&limit=100",
    "previous": "http://api.example.org/accounts/?offset=200&limit=100",
    "results": [...]
}
```

* `count` is the total number of available items.
* `next` is the API URL that should be called to fetch the next page of items based on the current `limit`.
* `previous` is the API URL that should be called to fetch the previous page of items based on the current `limit`.
* `results` is the actual list of returned items.


## Filtering

All endpoints that list resources support some filtering, based on the resource being fetched. These filters are passed as `GET` parameters and can be combined.

Two filters that are usually available on all list endpoints are `created_at_after` and `created_at_before`, which filters for items that have been created after or before the specified date. So you could use `?created_at_before=2022-02-01&created_at_after=2021-12-01` to only return items that have been created between December 1, 2021 and February 1, 2022 (specified dates excluded).

.. seealso::
  For more filtering options, refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for the endpoint that you want to filter.


## Ordering

Most endpoints that list resources support ordering on some fields. The ordering is passed as a single `GET` parameter named `ordering` with the field name that you want to order by as the value.

You can combine multiple ordering fields by separating them with a `,`. For example: `?ordering=project,created_at`.

You can specify that you want the ordering to be reversed by prefixing the field name with a `-`. For example: `?ordering=-created_at`.

.. seealso::
  For a list of fields that can be used for ordering, refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for the endpoint that you want to order.


## Fields

Some endpoints allow you to override the default response schema and specify a subset of fields that you want to be returned. You can specify the `fields` `GET` parameter with the field names separated by a `,`.

For example, you can specify `?fields=id,created_at` to only return the `id` and `created_at` fields in the response.

.. seealso::
  Refer to the `Swagger documentation <http:/app.konfuzio.com/v3/swagger/>`_ for a specific endpoint to see if it supports using the `fields` parameter. When supported, any field in the response schema can be used in the `fields` parameter.


## Guides and How-Tos

Check our [API guides section](./api_guides.html) for practical examples on how to interact with the Konfuzio API
