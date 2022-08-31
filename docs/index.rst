.. meta::
   :description: The Konfuzio Developer's Guide includes the technical documentation as well as usage examples for the different modules.

.. image:: _static/docs__static_logo.png
    :alt: Konfuzio Logo
    :align: right
    :width: 30%

|

##########################
Konfuzio Developer Center
##########################

Konfuzio is a data-centric Document AI tool to build intelligent document workflows. It provides
enterprise ready Web, API and Python interfaces to easily customize document workflows with AI and integrate it in any
IT infrastructure. Find out more on our `Homepage <https://www.konfuzio.com>`_.

**Konfuzio SDK:** The Open Source Konfuzio Software Development Kit (Konfuzio SDK) provides a Python API to build custom
document processes. Review the release notes and and the source code on
`GitHub <https://github.com/konfuzio-ai/konfuzio-sdk/releases>`_.

**Konfuzio Server:** Register a trial account via our SaaS hosting on `app.konfuzio.com <https://app.konfuzio.com>`_ or
:ref:`install it on your own infrastructure <Server Installation>` (see :ref:`Server Changelog`).
To then access the REST API via :ref:`Server API v2` and  :ref:`Server API v3` or
use further integrations, see `help.konfuzio.com <https://help.konfuzio.com/integrations/index.html>`_.

.. Note::
    Konfuzio Trainer will no longer be available.
    The functionality of Trainer will be moved to Server or become open source in SDK.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Konfuzio SDK

   sdk/configuration_reference.md
   sdk/sourcecode.rst
   sdk/examples/examples.rst
   sdk/contribution.md
   sdk/coordinates_system.md

.. toctree::
   :caption: Konfuzio Server
   :hidden:
   :maxdepth: 3

   web/api.md
   web/api-v3.md
   web/frontend/index.md
   web/on_premises.md
   web/changelog_app.md
