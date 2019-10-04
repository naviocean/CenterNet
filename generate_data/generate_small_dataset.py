import json
import os
import argparse

TYPES = ['instances', 'person_keypoints']
TYPE = 0
CATEGORIES = ['person']


class TinyCOCO:
    train_annotation = {}
    val_annotation = {}

    def __init__(self, data_annotation=os.path.abspath("."), data_output='./', num_train=2000, num_val=1000,
                 split_by='annotation'):
        self.num_train = num_train
        self.num_val = num_val
        self.data_annotation = data_annotation
        self.data_output = data_output
        self.split_by = split_by
        self._load_annotation()

    def _load_annotation(self):
        with open(
                os.path.join(self.data_annotation, '{}_train2017.json'.format(TYPES[TYPE]))) as train_json_file:
            self.train_annotation = json.load(train_json_file)
        with open(os.path.join(self.data_annotation, '{}_val2017.json'.format(TYPES[TYPE]))) as val_json_file:
            self.val_annotation = json.load(val_json_file)

    def _get_categories(self):
        categories = []
        for cat in self.train_annotation['categories']:
            if cat['name'] in CATEGORIES:
                categories.append(cat)
        return categories

    def _get_data(self, type='train'):
        data = {}
        data_by_cat_id = {}
        cat_ids = [int(cat['id']) for cat in self._get_categories()]
        annotation_image_mapping = {}
        source_annotations = self.train_annotation if type == 'train' else self.val_annotation

        # Get all annotations of each category
        for item in source_annotations['annotations']:
            if item['category_id'] in cat_ids:
                if item['category_id'] not in data_by_cat_id.keys():
                    data_by_cat_id[item['category_id']] = {
                        'annotations': [item],
                        'images': [],
                    }

                else:
                    data_by_cat_id[item['category_id']]['annotations'].append(item)

        # Get a range (num_train || num_val) annotations
        num_range = self.num_train if type == 'train' else self.num_val

        # if self.split_by == 'annotation':
        for cat_id, item in data_by_cat_id.items():
            item['annotations'] = item['annotations'][0:num_range]
            for anno in item['annotations']:
                if anno['image_id'] not in annotation_image_mapping.keys():
                    annotation_image_mapping[anno['image_id']] = [cat_id]
                else:
                    if cat_id not in annotation_image_mapping[anno['image_id']]:
                        annotation_image_mapping[anno['image_id']].append(cat_id)

        for image in source_annotations['images']:
            if image['id'] in annotation_image_mapping.keys():
                if len(annotation_image_mapping[image['id']]) > 1:
                    for cat_id in annotation_image_mapping[image['id']]:
                        data_by_cat_id[cat_id]['images'].append(image)
                else:
                    data_by_cat_id[annotation_image_mapping[image['id']][0]]['images'].append(image)

        data = {
            'info': source_annotations['info'],
            'licenses': source_annotations['licenses'],
            'images': [],
            'annotations': [],
            'categories': self._get_categories()
        }
        for id, item in data_by_cat_id.items():
            data['annotations'] += item['annotations']
            data['images'] += item['images']
        # else:
        #     # split by images
        #     for cat_id, item in data_by_cat_id.items():
        #         for anno in item['annotations']:
        #             annotation_image_mapping[anno['image_id']] = cat_id
        #
        #     # get a range images
        #     annotation_image_mapping = annotation_image_mapping[0:num_range]
        #
        #     for image in source_annotations['images']:
        #         if image['id'] in annotation_image_mapping.keys():
        #             data_by_cat_id[annotation_image_mapping[image['id']]]['images'].append(image)

        return data

    def process(self):
        train_dataset = self._get_data(type='train')
        val_dataset = self._get_data(type='val')
        folder = os.path.join(self.data_output, 'annotations')
        if not os.path.exists(folder):
            os.makedirs(folder)

        train_json_name = os.path.join(self.data_output, 'annotations', 'traintinycoco.json')
        val_json_name = os.path.join(self.data_output, 'annotations', 'valtinycoco.json')
        print(len(train_dataset['images']))
        with open(train_json_name, 'w') as f:
            json.dump(train_dataset, f)

        with open(val_json_name, 'w') as f:
            json.dump(val_dataset, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-annotation', type=str, required=True, help='path to coco dataset')
    parser.add_argument('--data-output', type=str, default='./', help='path to coco dataset')
    parser.add_argument('--num-train', type=int, default=2000, help='number of images in train set')
    parser.add_argument('--num-val', type=int, default=1000, help='number of images in val set')
    args = parser.parse_args()
    args = vars(args)

    tiny = TinyCOCO(**args)
    tiny.process()
