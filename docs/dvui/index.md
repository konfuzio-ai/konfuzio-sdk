.. meta::
:description: overview of the Document Validation UI

.. \_Konfuzio Document Validation UI overview:

# What is the Konfuzio Document Validation UI?

The Konfuzio Document Validation UI is a collection of open-source components, developed with [Vue.js](https://v2.vuejs.org/), that lets you preview and interact with Documents uploaded to [Konfuzio Server](https://konfuzio.com/).

![dvui.png](./images/dvui.png)

Chrome (from version 51), Firefox (from version 54), Edge (from version 15), and Safari (from version 10) are supported in the latest version.

If you want to read the full user guide visit [this link](https://help.konfuzio.com/document-validation-ui/index.html).

## Components

The app has four main components:

### Document Thumbnails

![document_thumbnails.svg](./images/document_thumbnails.svg)

This is a simple component that loads the Document images and displays them in a vertical list to act as a thumbnail selector. It is fully clickable and it updates its state automatically.

### Document Page

![document_page.svg](./images/document_page.svg)

This component is responsible for drawing the Document and handling any interaction with it. It uses [Konva JS](https://konvajs.org/docs/vue/index.html) to render a canvas element that has the Document image as background and shapes on top. Most of the shapes are Annotations from the Konfuzio API.

### Document Annotations

![document_annotations.svg](./images/document_annotations.svg)

This component is responsible for creating a list to interact with each Annotation on the Document. You can accept, decline or edit the Annotation, edit empty Labels, and change the Annotation bounding box, among other things.
