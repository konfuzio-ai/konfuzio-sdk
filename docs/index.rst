.. meta::
   :description: The Konfuzio Developer's Guide includes the technical documentation as well as usage examples for the different modules.

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

This Developer's Guide compiles the resources that can be used to expand Konfuzio.

The documentation for the Konfuzio Software Development Kit, the **Konfuzio SDK**, can be found in the SDK section.

The documentation for the **Trainer** module, only available for enterprise clients, can be found in the Trainer section.

In addition, the **Server** section provides documentation and examples for the Konfuzio REST API and Konfuzio on premises.

**This technical documentation is updated frequently. Please feel free to share your feedback via info@konfuzio.com**.


Konfuzio Python SDK
###################

The Konfuzio Software Development Kit (Konfuzio SDK) provides a Python API to interact with the Konfuzio Server.
Find in this section the documentation of the package, the installation instructions and code examples that will help
you in starting using it.

Konfuzio Trainer
################

The Trainer module allows to define, train and run custom Document AI.
In the Trainer section you find the documentation of the features of this module, including the contents structure
and examples of how to use it.

Konfuzio Server
###############

The Konfuzio Server allows the interaction with the Konfuzio projects.
In this section you can find information regarding how to use it. Check here the latest updates, information in how to
have Konfuzio on premises and examples on how to use the API for communicate directly with the Konfuzio Server.


.. toctree::
   :hidden:
   :maxdepth: 3
   :caption: Konfuzio Python SDK


   sdk/configuration_reference.md
   sdk/quickstart_pycharm.md
   sdk/examples/examples.rst
   sdk/sourcecode.rst
   sdk/contribution.md
   sdk/coordinates_system.md
   sdk/changelog.md


.. toctree::
   :maxdepth: 6
   :hidden:
   :caption: Konfuzio Trainer

   training/training_documentation.md

.. toctree::
   :caption: Konfuzio Server
   :hidden:
   :maxdepth: 3

   web/api.md
   web/on_premises.md
   web/changelog_app.md
