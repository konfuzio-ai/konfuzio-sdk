"""Test code examples for segmentation in the documentation."""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from konfuzio_sdk.api import get_results_from_segmentation
from konfuzio_sdk.data import Project
from variables import YOUR_PROJECT_ID

my_project = Project(id_=YOUR_PROJECT_ID)
# first Document uploaded
document = my_project.documents[0]
# index of the Page to test
page_index = 0

document.get_images()
image_path = document.pages()[0].image_path
image = Image.open(image_path).convert('RGB')
image_segmentation_bboxes = get_results_from_segmentation(document.id_, my_project.id_)

for bbox in image_segmentation_bboxes[page_index]:
    pp1 = (int(bbox["x0"]), int(bbox["y0"]))
    pp2 = (int(bbox["x1"]), int(bbox["y1"]))
    image = cv2.rectangle(np.array(image), pp1, pp2, (255, 0, 0), 1)

plt.imshow(image)
plt.show(block=False)
plt.close('all')
