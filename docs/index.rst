.. image:: _static/docs__static_logo.png
    :alt: Konfuzio Logo
    :align: right
    :width: 30%

|

##########################
Konfuzio Developer’s Guide
##########################

Konfuzio is a cloud and on-premises B2B platform used thousands of times a day to train and run AI.
SMEs and large companies train their AI to understand and process documents, e-mails and texts like human beings.
Find out more on our `Homepage <https://konfuzio.com>`_.

Our **API Documentation** is available online via https://app.konfuzio.com/api. Have a look at our `YouTube API Tutorial <https://www.youtube.com/watch?v=NZKUrKyFVA8>`_ too.

The Konfuzio Software Development Kit, the **Konfuzio SDK**, can be installed via `pip install konfuzio-sdk <https://pypi.org/project/konfuzio-sdk/>`_.
Find examples in Python and review the source code on `GitHub <https://github.com/konfuzio-ai/document-ai-python-sdk>`_.

In addition, enterprise clients do have access to our `Python Konfuzio Training Module <./training/training_documentation.html>`_ to define, train and run custom Document AI.

Download the **On-Prem and Private Cloud** documentation to deploy Konfuzio on a single VM or on your cluster. Use `this link <./_static/pdf/konfuzio_on_prem.pdf>`_ to access the latest version.

The `Changelog of app.konfuzio.com Server <./web/changelog_app.html>`_ provides you with insights about any (future) release.

**This technical documentation is updated frequently. Please feel free to share your feedback via info@konfuzio.com**.


SDK
###

The Konfuzio Software Development Kit (SDK) provides the tools to work with the data layer used by Konfuzio software.
Using the SDK you can communicate with the Konfuzio App and use the data structure in your projects.

Training
########

The training module is available for enterprise clients and allows to define, train and run custom Document AI.

Web Server
##########

The Konfuzio Web Server allows the interaction with the Konfuzio projects.

.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: SDK


   sdk/configuration_reference.md
   sdk/helloworld.md
   sdk/coordinates_system.md
   sdk/sourcecode.rst
   sdk/changelog.md


.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Training

   training/training_documentation.md

.. toctree::
   :caption: Web Server
   :hidden:
   :maxdepth: 3

   web/api.md
   web/changelog_app.md
