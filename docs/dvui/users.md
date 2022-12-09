# Managing users in the Document Validation UI

There are two options to manage users in the Document Validation UI:

- The first option is to [create a single user](https://app.konfuzio.com/accounts/signup/) with limited permissions (i.e. with the [Reader role](https://help.konfuzio.com/modules/superuserroles/index.html?highlight=role)) and [user token](https://app.konfuzio.com/v3/swagger/#/auth/auth_create), that anyone who interacts with the user interface can use in a project. Since the token will be public, this option is best suited for a read-only view.
- The second one is to create as many users as needed, each with their own token. This is best suited for projects that use [single sign-on authentication](https://dev.konfuzio.com/web/api-v3.html#single-sign-on-sso-authentication) and/or where users will be interacting with the Documents (annotating and revising), since it allows for fine-grained control over who can access what and it allows the application to keep a record of the different actions performed by them.
