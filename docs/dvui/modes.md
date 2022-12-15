.. meta::
:description: description of the two possible modes to run the app

# Read Only Mode vs Full Mode

The Document Validation UI can be configured to be run as Read Only or Full Mode:

## Read Only Mode

This is the default mode of the app. In this mode, you will have a sample Document with annotations that you can only preview. Unless configured, it uses the default API endpoint at https://app.konfuzio.com and no user account is needed.

## Full Mode

If you want to run the widget in full mode to be able to interact with the Document by editing Annotations, Document pages and other functionalities, you will need to have a user account created (more information in our [Managing users](./users.md) section). Then, you should generate a user Token by accessing the [Konfuzio API version 3 Auth Request](https://app.konfuzio.com/v3/swagger/) and making a request with your username and password. If the provided credentials are correct, then a Token will be generated that you can copy and add to the `.env` file (see below for more details).

You will also need a [Document uploaded](https://app.konfuzio.com/v3/swagger/#/documents/documents_create) and a Document id, and will need to be logged in to [Konfuzio](https://app.konfuzio.com/)) before being able to upload the Document. After successfully uploading it, if you want to show it on the Document Validation UI, you can copy the Document id from the URL, as shown in the image below:

![docid.png](./images/docid.png)

To complete the setup, create an environment variables file `.env` on the root of the repository based on the [`.env.example`](https://github.com/konfuzio-ai/konfuzio-capture-vue/blob/main/.env.example) for specifying:

- The API URL
- The images URL
- The user Token
- The Document ID
- The default language of the app

Please be aware that any variable in the `.env` will have priority from the variables defined in the `index.html`.
