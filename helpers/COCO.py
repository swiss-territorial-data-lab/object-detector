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
                 the_year: int, 
                 the_version: str, 
                 the_description: str, 
                 the_contributor: str,
                 the_url: str,
                 the_date_created: datetime=None):
    
        if the_date_created == None:
            the_date_created = date.today()
        
        info = {"year": the_year,
                "version": the_version,
                "description": the_description,
                "contributor": the_contributor,
                "url": the_url,
                "date_created": the_date_created,
        }
    
        self.info = info

        return self


    def annotation(self,
                   the_image_id: int, 
                   the_category_id: int, 
                   the_segmentation: list,
                   the_iscrowd: int,
                   the_annotation_id: int=None):
    
        _annotation = {
            "image_id": the_image_id,
            "category_id": the_category_id,
            "segmentation": the_segmentation,
            #"area": the_area,
            #"bbox": the_bbox, #[x,y,width,height],
            "iscrowd": the_iscrowd,
        }

        if the_annotation_id != None:
            _annotation['id'] = the_annotation_id

        # init
        _annotation['area'] = 0
        xmin = np.inf
        xmax = -np.inf
        ymin = np.inf
        ymax = -np.inf

        for seg in the_segmentation:

            xx = [x for idx, x in enumerate(seg) if idx % 2 == 0]
            yy = [x for idx, x in enumerate(seg) if idx % 2 == 1]

            xmin = np.min([xmin, np.min(xx)])
            xmax = np.max([xmax, np.max(xx)])

            ymin = np.min([ymin, np.min(yy)])
            ymax = np.max([ymax, np.max(yy)])

            _annotation['area'] += self._PolyArea(xx, yy)

        _annotation['bbox'] = [xmin, ymin, xmax-xmin, ymax-ymin]
        
        return _annotation


    def insert_annotation(self, the_annotation):

        # let's perform some checks...
        if 'image_id' not in the_annotation.keys():
            raise MissingImageIdException(f"Missing image ID = {the_annotation['image_id']}")

        if 'category_id' not in the_annotation.keys():
            raise MissingCategoryIdException(f"Missing category ID = {the_annotation['category_id']}")

        if 'id' not in the_annotation:
            the_annotation['id'] = len(self.annotations) + 1
        
        self.annotations.append(the_annotation)

        self._annotations_dict[the_annotation['id']] = the_annotation

        return the_annotation['id']
    
    
    def license(self, the_name: str, the_url: str, the_id: int=None):

        _license = {
            "name": the_name,
            "url": the_url
        }

        if the_id != None:
            _license['id'] = the_id 

        return _license

    
    def insert_license(self, the_license):

        if 'id' not in the_license:
            the_license['id'] = len(self.licenses) + 1

        self.licenses.append(the_license)
        self._licenses_dict[the_license['id']] = the_license
        
        return the_license['id']


    def category(self, the_name: str, the_supercategory: str, the_id: int=None):

        _category = {
            "name": the_name,
            "supercategory": the_supercategory
        }

        if the_id != None:
            _category['id'] = the_id

        return _category


    def insert_category(self, the_category):

        if 'id' not in the_category:
            the_category['id'] = len(self.categories) + 1

        self.categories.append(the_category)
        self._categories_dict[the_category['id']] = the_category
    
        return the_category['id']

    
    def image(self, 
              the_path: str, 
              the_filename: str, 
              the_license_id: int,
              the_id: int=None,
              the_date_captured: datetime=None,
              the_flickr_url: str=None, 
              the_coco_url: str=None):


        full_filename = os.path.join(the_path, the_filename)
        img = Image.open(full_filename) # this was checked to be faster than skimage and rasterio
        width, height = img.size

        image = {
            "width": width, 
            "height": height, 
            "file_name": the_filename,
            "license": the_license_id
        }

        for el in ['id', 'flickr_url', 'coco_url']:
            if eval('the_' + el) != None:
                image[el] = eval('the_' + el)

        if the_date_captured != None:
            image['date_captured'] = the_date_captured
        else:
            dc = os.stat(full_filename).st_ctime
            image['date_captured'] = datetime.utcfromtimestamp(dc)

        return image


    def insert_image(self, the_image):

        # check whether the license_id is valid
        if the_image['license'] not in self._licenses_dict.keys():
            raise LicenseIdNotFoundException(f"License ID = {the_image['license']} not found.")

        if 'id' not in the_image:
            the_image['id'] = len(self.images)+1

        self.images.append(the_image)
        self._images_dict[the_image['id']] = the_image
        
        return the_image['id']


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
        ann = coco.annotation(the_image_id=1, the_category_id=1, the_segmentation=segmentation, the_iscrowd=0,the_annotation_id=0)
        coco.insert_annotation(ann)
    except Exception as e:
        print(f"Failed to insert annotation. Exception: {e}")
        sys.exit(1)

    ann = coco.annotation(the_image_id=1, the_category_id=1, the_segmentation=segmentation, the_iscrowd=0,the_annotation_id=123)
    coco.insert_annotation(ann)

    pprint(coco.to_json())