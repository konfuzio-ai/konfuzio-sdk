# What is the Konfuzio SDK?

## Overview of the SDK

The Open Source Konfuzio Software Development Kit (Konfuzio SDK) provides a Python API to build custom document processes. For a quick introduction to the SDK, check out the [Get Started](get_started.html) section. Review the release notes and the source code on [GitHub](https://github.com/konfuzio-ai/konfuzio-sdk/releases).

| Section                                                           | Description                                                                                                 |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| [Get Started](get_started.html)                                   | Learn more about the Konfuzio SDK and how it works.                                                         |
| [Tutorials](tutorials.html)                                       | Learn how to build your first document extraction pipeline, speed up your annotation process and many more. |
| [Explanations](explanations.html)                                 | Here are links to teaching material about the Konfuzio SDK.                                                 |
| [API Reference](sourcecode.html)                                  | Get to know all major Data Layer concepts of the Konfuzio SDK.                                              |
| [Contribution Guide](contribution.html)                           | Learn how to contribute, run the tests locally, and submit a Pull Request.                                  |
| [Changelog](https://github.com/konfuzio-ai/konfuzio-sdk/releases) | Review the release notes and the source code of the Konfuzio SDK.                                           |

### Customizing document processes with the Konfuzio SDK

For documentation about how to train and evaluate document understanding AIs, as well as use a trained AI in the
Konfuzio Server web interface, please see our [Konfuzio Guide](https://help.konfuzio.com/tutorials/quickstart/index.html).

If you need to **add custom functionality** to the document processes of the Konfuzio Server, the Konfuzio SDK 
is the tool for you. You can customize pipelines for automatic document :ref:`Categorization<document-categorization-tutorials>`,
:ref:`File Splitting<file-splitting-tutorials>`, and :ref:`Extraction<information-extraction-tutorials>`.
These processes allow to split stack scans, categorize and extract information from the Documents.

.. note::
  Customizing document AI pipelines with the Konfuzio SDK requires a self-hosted installation of the Konfuzio Server.

### Other use cases around documents

You can also use the Konfuzio SDK for other Document-related purposes like filling in the PDF forms via the generator. 
For further information, check out our tutorial on :ref:`how to create a PDF form generator<pdf-form-generator>`.


