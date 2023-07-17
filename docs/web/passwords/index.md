# Password Management

Before we begin, it's important to mention that Konfuzio, an AI platform, provides a [Single Sign-On (SSO)](https://dev.konfuzio.com/web/on_premises.html#sso-via-openid-connect-oidc) option.
SSO can simplify password management, improve user experience, and potentially enhance security.

With SSO, users only need to remember a single set of credentials, and this primary authentication can give users access to other services without needing to authenticate again.

However, this tutorial focuses on how you can set up comprehensive password management, including password conventions, reusability, and expiry within the default Konfuzio framework, [Django](https://docs.djangoproject.com/en/4.2/releases/).

## Password Conventions

Django comes with a built-in `django.contrib.auth.password_validation` module that includes a range of password validation options. This can be customised to fit your specific password conventions.

Here's how you can do this:

### Set a Minimum Length

Django's `MinimumLengthValidator` helps you set a minimum length for passwords. You can add this to your settings like so:

```python
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
        'OPTIONS': {
            'min_length': 9,
        }
    },
]
```

### Prevent Common Passwords:

The `CommonPasswordValidator` ensures that passwords aren't among the most common ones. Add this to your settings:

```python
AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
]
```

### Ensure Password Complexity:

If you want to enforce certain complexities in your passwords, such as including a mix of letters, numbers, and special characters, you can use the `ComplexityValidator`.

## Preventing Password Reusability:

To prevent password reusability, you can use the `PasswordHistory` module from the `django-passwords` library. This module keeps a history of a user's passwords and checks if a new password is already in the user's password history when changing passwords.

Here's how to add it:

```python
PASSWORD_STORE = {
    'PASSWORD_HISTORY_COUNT': 5,
}
```

This stores the last five passwords of each user and prevents their reuse.

## Password Expiry

For password expiry, you can use the `ExpiringPasswordStorage` from the `django-passwords` module. This allows you to set a maximum validity period for passwords, after which users are asked to change their password.

Here's how to add it:

```python
PASSWORD_STORE = {
    'PASSWORD_EXPIRY': timedelta(days=90),  # Passwords expire after 90 days
}
```

## Summary

In conclusion, Konfuzio smartly leverages the robust capabilities of Django, opting to stand on the shoulders of this established framework instead of creating its own password management procedures. This approach not only underscores the flexibility and adaptability of Konfuzio but also allows developers to utilize the tried-and-tested security procedures inherent within Django.

The absence of built-in password management in Konfuzio should not be seen as a shortcoming, but rather a thoughtful design decision. It offers an opportunity for developers to delve into Django's rich and secure functionalities for managing passwords, enabling the creation of more personalized, secure, and efficient systems.

With this tutorial, you now have a roadmap to seamlessly integrate advanced password management strategies into your Django application, fully aligned with the stringent PS951 exam's requirements. This knowledge empowers you to provide your users with a more secure, reliable, and user-friendly experience, enhancing the overall effectiveness of your application. Happy coding!
