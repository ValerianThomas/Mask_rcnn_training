import os
import matplotlib.pyplot as plt
import numpy as np

from mrcnn.utils import Dataset, extract_bboxes
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from xml.etree import ElementTree



# define a configuration for the model
class KangarooConfig(Config):
	# Give the configuration a recognizable name
	NAME = "kangaroo_cfg"
	# Number of classes (background + kangaroo)
	NUM_CLASSES = 1 + 1
	# Number of training steps per epoch
	STEPS_PER_EPOCH = 143
 
# prepare config
config = KangarooConfig()

class KangarooDataset(Dataset):

  def extract_boxes(self,filename):
	  # load and parse the file
	  tree = ElementTree.parse(filename)
	  # get the root of the document
	  root = tree.getroot()
	  # extract each bounding box
	  boxes = list()
	  for box in root.findall('.//bndbox'):
	  	xmin = int(box.find('xmin').text)
	  	ymin = int(box.find('ymin').text)
	  	xmax = int(box.find('xmax').text)
	  	ymax = int(box.find('ymax').text)
	  	coors = [xmin, ymin, xmax, ymax]
	  	boxes.append(coors)
	  # extract image dimensions
	  width = int(root.find('.//size/width').text)
	  height = int(root.find('.//size/height').text)
	  return boxes, width, height

  def load_dataset(self,root_dir, isTraining=True):
    self.add_class("dataset", 1, "kangaroo") # internal function from mrcnn

    IMAGE_DIR =f'{root_dir}/images'
    ANNOT_DIR = f'{root_dir}/annots'
    list_of_images = os.listdir(IMAGE_DIR)
    
    if isTraining == True :
      list_of_images = list_of_images[:-20]
    else :
      list_of_images = list_of_images[-20:]
    for image in list_of_images:
      image_id = image[:-4]
      if image_id in ['00090']:
			    continue
      # bug with image 90 so we get it out of the process
      
      self.add_image('dataset', image_id = image_id, path = f'{IMAGE_DIR}/{image}', annotation = f'{ANNOT_DIR}/{image_id}.xml')
      # this method will save the images in the an array called image_info
  
  def load_mask(self, image_id): # here we rewrite the function load_mask build in Dataset class
    # get details the path of an image, the path are insinde image_info[image_id][annotation] in order to extract the bounding boxes xml out the right image
    info = self.image_info[image_id]
    path = info['annotation']
    boxes, w, h = self.extract_boxes(path)

    """  We can now define a mask for each bounding box, and an associated class.

      A mask is a two-dimensional array with the same dimensions as the photograph with all zero values where the object isn’t and all one  values where the object is in the photograph.

      We can achieve this by creating a NumPy array with all zero values for the known size of the image and one channel for each bounding box., in our case we will have a 3D numpy array as we may have more than one box per pictures (2 Kangaroos in one pic) thus the len(boxes)
    """

    masks = np.zeros([h, w, len(boxes)], dtype='uint8')
    """Each bounding box is defined as min and max, x and y coordinates of the box.

      These can be used directly to define row and column ranges in the array that can then be marked as 1
    """
    class_ids = list()
    for i in range(len(boxes)):
      box = boxes[i]
      row_s, row_e = box[1], box[3] # Ys are in rows as we set h in the numpy array i in the row position [h, w, len(boxes)]
      cols_s, col_e = box[0], box[2]
      masks[row_s:row_e,cols_s:col_e, i] = 1 # and this is how we do iteration in XD arrays (look at the i ), here we replace the 0 value on the box pixels
      class_ids.append(self.class_names.index('kangaroo'))  # here we add the class name for each box, as we've selected kangaroo it will be mapped to the value 1 as we have define kangaroo as a class value to 1 in the add_class method

      
    return masks, np.asarray(class_ids, dtype='int32')  # does 2 values must always be return in this method which are the mask and the class value for each mask


  """
  Finally, we must implement the image_reference() function.

  This function is responsible for returning the path or URL for a given ‘image_id‘, which we know is just the ‘path‘ property on the ‘image info‘ dict.
  """

  def image_reference(self, image_id) :
    info = self.image_info[image_id]
    path = info['path']

    return path
    


if __name__ == '__main__':
  data = KangarooDataset()
  data.load_dataset(f'{os.getcwd()}/kangaroo', isTraining=True)
  data.prepare()
  print(f'Number of entries: {len(data.image_ids)}')

  test_data = KangarooDataset()
  test_data.load_dataset(f'{os.getcwd()}/kangaroo', isTraining=False)
  test_data.prepare()
  print(f'Number of test entries: {len(test_data.image_ids)}')
  

    
# test the data_set
image_id = 0
image = data.load_image(image_id)
print(image.shape)
mask, class_ids = data.load_mask(image_id)
print(mask.shape)

""" for one image
# check image
# plot image
plt.imshow(image)

# plot mask

plt.imshow(mask[:,:,0],cmap='gray',alpha=0.5)
plt.show()
"""
# for 9 images and x masks per image
"""
for i in range(9):
  image = data.load_image(i)
  plt.imshow(image)
  mask, class_ids = data.load_mask(i)
  for j in range(mask.shape[2]):
    plt.imshow(mask[:,:,j],cmap='gray', alpha=0.5)
  plt.show()
"""

# for 9 images using the built in function of mrcnn
"""
for i in range(9):
  image = data.load_image(i)
  mask, class_ids = data.load_mask(i)

  # extract bounding boxes from the masks
  bbox =  extract_bboxes(mask)
  # display image with masks and bounding boxes
  display_instances(image, bbox, mask, class_ids, data.class_names)
"""

config = KangarooConfig()
model = MaskRCNN(mode='training',model_dir='./', config=config)
model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])
# train weights (output layers or 'heads')
model.train(data,test_data, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')

