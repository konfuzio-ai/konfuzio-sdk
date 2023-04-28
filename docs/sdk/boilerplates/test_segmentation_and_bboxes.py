"""Test code examples for visualizing segmentation and bboxes in the documentation."""
from PIL import Image, ImageDraw
from konfuzio_sdk.data import Project
from konfuzio_sdk.api import get_results_from_segmentation
from variables import YOUR_PROJECT_ID, TEST_DOCUMENT_ID

my_project = Project(id_=YOUR_PROJECT_ID, strict_data_validation=False)
# first Document uploaded
document = my_project.get_document_by_id(TEST_DOCUMENT_ID)
# index of the Page to test
page_index = 0

doc_bbox = document.get_bbox()
document.get_images()
page = document.pages()[page_index]
img = page.get_image(update=True)
image_path = document.pages()[page_index].image_path

factor_x = page.width / img.width
factor_y = page.height / img.height
assert 0.42 < factor_x < 0.43
assert 0.42 < factor_y < 0.43

image = Image.open(image_path).convert('RGB')
image_factor = image.copy()

image = image.resize((int(image.size[0] * factor_x), int(image.size[1] * factor_y)))
height = image.size[1]

image_characters_bbox = [char_bbox for _, char_bbox in page.get_bbox().items()]
assert len(image_characters_bbox) == 2249

draw = ImageDraw.Draw(image)

for bbox in image_characters_bbox:
    image_bbox = (
        int(bbox["x0"]),
        int((height - bbox["y1"])),
        int(bbox["x1"]),
        int((height - bbox["y0"])),
    )
    draw.rectangle(image_bbox, outline='blue', width=1)

image_segmentation_bboxes = get_results_from_segmentation(document.id_, my_project.id_)

for bbox in image_segmentation_bboxes[page_index]:
    image_bbox = (
        int(bbox["x0"]),
        int(bbox["y0"]),
        int(bbox["x1"]),
        int(bbox["y1"]),
    )
    draw.rectangle(image_bbox, outline='blue', width=1)

for bbox in image_segmentation_bboxes[page_index]:
    image_bbox = (
        int(bbox["x0"] * factor_x),
        int(bbox["y0"] * factor_y),
        int(bbox["x1"] * factor_x),
        int(bbox["y1"] * factor_y),
    )
    draw.rectangle(image_bbox, outline='blue', width=1)

image

# Note: cv2 has the origin of the y coordinates in the upper left corner. Therefore, for visualization, the height of
# the image is subtracted to the y coordinates.
