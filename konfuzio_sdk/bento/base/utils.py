"""Utils shared by container services."""
import functools
import traceback
import typing as t

from pydantic import PlainSerializer, PlainValidator, WithJsonSchema, errors
from starlette.responses import JSONResponse
from typing_extensions import Annotated


def add_credentials_to_project(project, ctx):
    """Add credentials from the request headers to the Project object."""
    # Add credentials from the request headers to the Project object, but only if the SDK version supports this.
    # Older SDK versions do not have the credentials attribute on Project.
    if hasattr(project, 'credentials'):
        for key, value in ctx.request.headers.items():
            if key.startswith('env_'):
                key = key.replace('env_', '', 1)
                project.credentials[key.upper()] = value
    return project


def cleanup_project_after_document_processing(project, document):
    """Remove the document and its copies from the Project object to avoid memory leaks."""
    project._documents = [d for d in project._documents if d.id_ != document.id_ and d.copy_of_id != document.id_]
    return project


def get_error_details(exc: Exception) -> str:
    error_details = type(exc).__name__
    error_message = str(exc)
    if error_message:
        error_details = f'{error_details}: {error_message}'
    return error_details


def handle_exceptions(func: t.Callable) -> t.Callable:
    """
    Decorator to handle exceptions in service API endpoints and return a JSON response with error details.
    Pydantic errors are not handled here, as they are handled by Bento automatically.
    """

    @functools.wraps(func)
    async def wrapper(*args: t.Any, **kwargs: t.Any) -> t.Any:
        try:
            return await func(*args, **kwargs)
        except Exception as exc:
            tb = traceback.format_exception(type(exc), exc, exc.__traceback__)
            error_details = get_error_details(exc)
            # Override the default status code, otherwise it will be 200.
            if 'ctx' in kwargs:
                ctx = kwargs['ctx']
                ctx.response.status_code = 500
            return JSONResponse(status_code=500, content={'error': error_details, 'traceback': tb})

    return wrapper


def hex_bytes_validator(o: t.Any) -> bytes:
    """
    Custom validator to be able to correctly serialize and unserialize bytes.
    See https://github.com/pydantic/pydantic/issues/3756#issuecomment-1654425270
    """
    if isinstance(o, bytes):
        return o
    elif isinstance(o, bytearray):
        return bytes(o)
    elif isinstance(o, str):
        return bytes.fromhex(o)
    raise errors.BytesError


HexBytes = Annotated[
    bytes, PlainValidator(hex_bytes_validator), PlainSerializer(lambda b: b.hex()), WithJsonSchema({'type': 'string'})
]
