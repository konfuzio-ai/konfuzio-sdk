.. meta::
:description: this section explains the different user and mode options available in the Document Validation UI

# Explanations

## Managing users in the Konfuzio Document Validation UI

There are two options to manage users in the Document Validation UI:

- The first option is to [create a single user](https://app.konfuzio.com/accounts/signup/) with limited permissions (i.e. with the [Reader role](https://help.konfuzio.com/modules/superuserroles/index.html)) and [user token](https://app.konfuzio.com/v3/swagger/#/auth/auth_create), that anyone who interacts with the user interface can use in a Project. Since the token will be public, this option is best suited for a read-only view.
- The second one is to create as many users as needed, each with their own token. This is best suited for Projects that use [single sign-on authentication](https://dev.konfuzio.com/web/api-v3.html#single-sign-on-sso-authentication) and/or where users will be interacting with the Documents (annotating and revising), since it allows for fine-grained control over who can access what, and it allows the application to keep a record of the different actions performed by them.

## Read Only Mode vs Full Mode

The Konfuzio Document Validation UI can be configured to be run as Read Only or Full Mode:

### Read Only Mode

This is the default mode of the app. In this mode, you will have a sample Document with Annotations that you can only preview. Unless configured, it uses the default API endpoint at https://app.konfuzio.com and no user account is needed.

### Full Mode

If you want to run the widget in full mode to be able to interact with the Document by editing Annotations, Document Pages and other functionalities, you will need to have a user account created (more information in our [Managing users](/dvui/explanations.html#managing-users-in-the-konfuzio-document-validation-ui) section). Then, you should generate a user Token by accessing the [Konfuzio API version 3 Auth Request](https://app.konfuzio.com/v3/swagger/) and making a request with your username and password. If the provided credentials are correct, then a Token will be generated that you can copy and add to the `.env` file (see below for more details).

You will also need a [Document uploaded](https://app.konfuzio.com/v3/swagger/#/documents/documents_create) and a Document id, and will need to be logged in to [Konfuzio](https://app.konfuzio.com/)) before being able to upload the Document. After successfully uploading it, if you want to show it on the Document Validation UI, you can copy the Document ID from the URL, as shown in the image below:

.. image:: ./images/docid.png

To complete the setup, create an environment variables file `.env` on the root of the repository based on the [.env.example](https://github.com/konfuzio-ai/document-validation-ui/blob/main/.env.example) for specifying the following values:

- The user Token
- The Document ID

Some other optional variables you can include are:

- The API URL
- The images URL
- The default language of the app
- The Category ID

Please be aware that any variable in the `.env` will have priority from the variables defined in the `index.html`.
