"""Test code examples for visualizing the bounding boxes in the documentation."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from konfuzio_sdk.data import Project
from PIL import Image
from variables import YOUR_PROJECT_ID

my_project = Project(id_=YOUR_PROJECT_ID)
# first document uploaded
document = my_project.documents[0]
# index of the page to test
page_index = 0

doc_bbox = document.get_bbox()
page = document.pages()[page_index]
img = page.get_image(update=True)
image_path = document.pages()[page_index].image_path

factor_x = page.width / img.width  # determines difference between sizes of page and actual document images
factor_y = page.height / img.height
assert 0.42 < factor_x < 0.43
assert 0.42 < factor_y < 0.43

image = Image.open(image_path).convert('RGB')
image = image.resize((int(image.size[0] * factor_x), int(image.size[1] * factor_y)))
height = image.size[1]

image_characters_bbox = [char_bbox for _, char_bbox in doc_bbox.items() if char_bbox["page_number"] - 1 == page_index]
assert len(image_characters_bbox) == 2249

for bbox in image_characters_bbox:
    pp1 = (int(bbox["x0"]), int((height - bbox["y0"])))
    pp2 = (int(bbox["x1"]), int((height - bbox["y1"])))
    image = cv2.rectangle(np.array(image), pp1, pp2, (0, 255, 0), 1)

plt.imshow(image)
plt.show()

# Note: cv2 has the origin of the y coordinates in the upper left corner. Therefore, for visualization, the height of
# the image is subtracted to the y coordinates.
