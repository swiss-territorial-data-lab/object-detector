#!/bin/python
# -*- coding: utf-8 -*-

import os, sys
import json
import numpy as np
import logging

from datetime import datetime, date
from PIL import Image

pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.INFO)


class MissingImageIdException(Exception):
    "Raised when an annotation is lacking the image ID field"
    pass


class MissingCategoryIdException(Exception):
    "Raised when an annotation is lacking the category ID field"
    pass


class LicenseIdNotFoundException(Exception):
    "Raised when a given license ID is not found"
    pass


class COCO:
    # cf. http://cocodataset.org/#format-data

    def __init__(self):

        self.info = ""
        self.images = []
        self.annotations = []
        self.licenses = []
        self.categories = []
        self.images = []

        self._licenses_dict = {}
        self._categories_dict = {}
        self._annotations_dict = {}
        self._images_dict = {}

        return None

    def set_info(self, 
                 year: int, 
                 version: str, 
                 description: str, 
                 contributor: str,
                 url: str,
                 date_created: datetime=None):
    
        if date_created == None:
            date_created = date.today()
        
        info = {"year": year,
                "version": version,
                "description": description,
                "contributor": contributor,
                "url": url,
                "date_created": date_created,
        }
    
        self.info = info

        return self


    def annotation(self,
                   image_id: int, 
                   category_id: int, 
                   segmentation: list,
                   iscrowd: int,
                   annotation_id: int=None):
    
        _annotation = {
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": segmentation,
            #"area": area,
            #"bbox": bbox, #[x,y,width,height],
            "iscrowd": iscrowd,
        }

        if annotation_id != None:
            _annotation['id'] = annotation_id

        # init
        _annotation['area'] = 0
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        for seg in segmentation:

            xx = [x for idx, x in enumerate(seg) if idx % 2 == 0]
            yy = [x for idx, x in enumerate(seg) if idx % 2 == 1]

            xmin = np.min([xmin, np.min(xx)])
            xmax = np.max([xmax, np.max(xx)])

            ymin = np.min([ymin, np.min(yy)])
            ymax = np.max([ymax, np.max(yy)])

            _annotation['area'] += self._PolyArea(xx, yy)

        _annotation['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
        
        return _annotation


    def insert_annotation(self, annotation):

        # let's perform some checks...
        if 'image_id' not in annotation.keys():
            raise MissingImageIdException(f"Missing image ID = {annotation['image_id']}")

        if 'category_id' not in annotation.keys():
            raise MissingCategoryIdException(f"Missing category ID = {annotation['category_id']}")

        if 'id' not in annotation:
            annotation['id'] = len(self.annotations) + 1
        
        self.annotations.append(annotation)

        self._annotations_dict[annotation['id']] = annotation

        return annotation['id']
    
    
    def license(self, name: str, url: str, id: int=None):

        _license = {
            "name": name,
            "url": url
        }

        if id != None:
            _license['id'] = id 

        return _license

    
    def insert_license(self, license):

        if 'id' not in license:
            license['id'] = len(self.licenses) + 1

        self.licenses.append(license)
        self._licenses_dict[license['id']] = license
        
        return license['id']


    def category(self, name: str, supercategory: str, id: int=None):

        _category = {
            "name": name,
            "supercategory": supercategory
        }

        if id != None:
            _category['id'] = id

        return _category


    def insert_category(self, category):

        if 'id' not in category:
            category['id'] = len(self.categories) + 1

        self.categories.append(category)
        self._categories_dict[category['id']] = category
    
        return category['id']

    
    def image(self, 
              path: str, 
              filename: str, 
              license_id: int,
              id: int=None,
              date_captured: datetime=None,
              flickr_url: str=None, 
              coco_url: str=None):


        full_filename = os.path.join(path, filename)
        img = Image.open(full_filename) # this was checked to be faster than skimage and rasterio
        width, height = img.size

        image = {
            "width": width, 
            "height": height, 
            "file_name": filename,
            "license": license_id
        }

        if id != None:
            image['id'] = id

        if flickr_url != None:
            image['flickr_url'] = flickr_url

        if coco_url != None:
            image['coco_url'] = coco_url

        if date_captured != None:
            image['date_captured'] = date_captured
        else:
            dc = os.stat(full_filename).st_ctime
            image['date_captured'] = datetime.utcfromtimestamp(dc)

        return image


    def insert_image(self, image):

        # check whether the license_id is valid
        if image['license'] not in self._licenses_dict.keys():
            raise LicenseIdNotFoundException(f"License ID = {image['license']} not found.")

        if 'id' not in image:
            image['id'] = len(self.images)+1

        self.images.append(image)
        self._images_dict[image['id']] = image
        
        return image['id']


    def to_json(self):

        out = {}
        out['info'] = self.info
        out['images'] = self.images
        out['annotations'] = self.annotations
        out['licenses'] = self.licenses
        out['categories'] = self.categories

        return json.loads(json.dumps(out, default=self._default))

    # cf. https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    def _PolyArea(self, x, y):
        return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

    
    def _default(self, o):
        if isinstance(o, (date, datetime)):
            return o.isoformat()

    def __str__(self):

        return json.dumps(self.to_json())

    def __repr__(self):

        return json.dumps(self.to_json())


if __name__ == '__main__':

    from pprint import pprint
    coco = COCO()
    coco.set_info(2020, 'the version', 'the description', 'the contributor', 'the url')


    segmentation = [[214.59, 205.04, 218.39, 203.27, 
        218.39, 198.97, 221.18, 195.42, 
        225.73, 193.9, 228.77, 192.39, 
        241.17, 193.4, 243.45, 212.13, 
        252.57, 213.65, 252.06, 199.98, 
        256.87, 201.25, 260.92, 204.03, 
        263.45, 206.56, 267.75, 223.27, 
        259.91, 230.86, 249.78, 256.68, 
        253.58, 261.24, 243.39, 262.67, 
        241.78, 258.9, 236.94, 258.1, 
        237.21, 252.45, 239.9, 252.45, 
        240.17, 236.05, 237.48, 224.49, 
        233.17, 219.92, 225.11, 219.11, 
        219.73, 216.42, 214.62, 210.77, 
        213.81, 206.47, 215.43, 205.13], 
        [247.96, 237.39, 246.89, 254.87, 248.77, 238.2, 248.77, 238.2]]

    license = coco.license('test license', 'test url')
    coco.insert_license(license)

    license = coco.license('test license', 'test url', 100)
    coco.insert_license(license)

    cat = coco.category('test cat', 'the supercat')
    coco.insert_category(cat)

    cat = coco.category('test cat', 'the supercat', 3)
    coco.insert_category(cat)

    try:
        ann = coco.annotation(image_id=1, category_id=1, segmentation=segmentation, iscrowd=0, annotation_id=0)
        coco.insert_annotation(ann)
    except Exception as e:
        print(f"Failed to insert annotation. Exception: {e}")
        sys.exit(1)

    ann = coco.annotation(image_id=1, category_id=1, segmentation=segmentation, iscrowd=0, annotation_id=123)
    coco.insert_annotation(ann)

    pprint(coco.to_json())