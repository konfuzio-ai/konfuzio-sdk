.. _pdf-form-generator:
## Build your own PDF Form Generator

A simple implementation of a PDF form generator can look like the following.

### 1. Upload a PDF Form and create empty Annotations where you want to fill in data

.. image:: /sdk/home/pdf_form_gen_example.png
   :scale: 40%

### 2. Use the SDK to fill in values in the PDF

Example implementation:
```python
import io

from PyPDF2 import PdfFileWriter, PdfFileReader
from reportlab.pdfgen import canvas
from reportlab.lib import colors

from konfuzio_sdk.data import Project, Document, Page, Annotation


def render(document_id, project_id, values):
    my_project = Project(id_=project_id, update=True)
    document: Document = my_project.get_document_by_id(document_id)

    # read your existing PDF
    existing_pdf = PdfFileReader(open(document.get_file(), "rb"))
    output = PdfFileWriter()

    for page_index, page in enumerate(document.pages()):
        packet = io.BytesIO()
        my_canvas = canvas.Canvas(packet)
        my_canvas.setPageSize((page.width, page.height))
        for annotation: Annotation in document.annotations():
            if annotation.selection_bbox.get('page_index') == page_index:
                text_value = values.get(annotation.label.name, '')
                textobject = my_canvas.beginText()
                # Set text location (x, y)
                textobject.setTextOrigin(annotation.x0, annotation.y0)
                # Change text color
                textobject.setFillColor(colors.black)
                # Write text
                textobject.textLine(text=text_value)
                # Write text to the canvas
                my_canvas.drawText(textobject)
        my_canvas.save()

        # move to the beginning of the StringIO buffer
        packet.seek(0)

        # create a new PDF with Reportlab
        new_pdf = PdfFileReader(packet)

        page = existing_pdf.getPage(page_index)
        page.mergePage(new_pdf.getPage(0))
        output.addPage(page)

    return output


if __name__ == '__main__':
    values = {'Name': 'Valerie', 'Kurs': 'Pilates', 'Startdatum': '15.01.2022', 'Enddatum': '30.03.2022'}
    # To find your Project ID and Document ID see https://dev.konfuzio.com/sdk/get_started.html#example-usage
    project_id = YOUR_PROJECT_ID
    document_id = YOUR_DOCUMENT_ID

    output = render(document_id=document_id, project_id=project_id, values=values)
    # finally, write "output" to a real file
    outputStream = open(f"{document_id}.pdf", "wb")
    output.write(outputStream)
    outputStream.close()
```

The resulting filled in PDF form will look like this:

.. image:: /sdk/home/pdf_form_gen_example_compiled.png
   :scale: 40%