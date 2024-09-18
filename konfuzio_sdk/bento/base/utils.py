"""Utils shared by container services."""


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
