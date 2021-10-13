# Load raw COCO annotations
# Modify COCO annotations to only include id subset
#
import json
import os
import shutil


def find_coco_images_pth(anno_file):
    if 'train2017' in anno_file:
        path_to_coco_images = 'data/datasets/coco/train2017/'
    elif 'val2017' in anno_file:
        path_to_coco_images = 'data/datasets/coco/val2017/'
    else:
        raise Exception('I can\'t auto-find the path to images based on the anno file')
    assert os.path.exists(path_to_coco_images), \
        f"I thought coco images would be {path_to_coco_images}, but it doesn't exist :("
    return path_to_coco_images


def construct_new_dataset_name(image_ids):
    assert image_ids is not None
    return 'cocosubset_' + '-'.join(str(i) for i in image_ids)


def setup_new_dataset(dataset_parentpth, dataset_name, overwrite=False):
    dataset_pth = os.path.join(dataset_parentpth, dataset_name)
    if os.path.exists(dataset_pth):
        if overwrite is False:
            raise ValueError(f"Please remove {dataset_pth}, add --overwrite, "
                             f"or suggest a new --dataset_name")
        else:
            print(f"Writing inside existing path (probably overwriting!!) {dataset_pth}")
            os.makedirs(dataset_pth, exist_ok=True)
    os.makedirs(os.path.join(dataset_pth, 'annotations'), exist_ok=True)
    os.makedirs(os.path.join(dataset_pth, 'images'), exist_ok=True)
    return dataset_pth


def main(anno_file, image_ids, dataset_name=None, dataset_parentpth='data/datasets/custom/',
         overwrite=False, path_to_coco_images=None):
    path_to_coco_images = path_to_coco_images or find_coco_images_pth(anno_file=anno_file)
    dataset_name = dataset_name or construct_new_dataset_name(image_ids)
    dataset_pth = setup_new_dataset(dataset_parentpth, dataset_name, overwrite=overwrite)
    d = json.load(open(anno_file, 'rb'))
    d['images'] = [i for i in d['images'] if i['id'] in image_ids]
    ids_found = [i['id'] for i in d['images']]
    if len(ids_found) != len(image_ids):
        missing_ids = set(image_ids) - set(ids_found)
        print(f"Missing {len(missing_ids)}/{len(image_ids)}.")
        raise Exception(f"Missing ids: {missing_ids}.")
    else:
        print(f"Found all {len(image_ids)} image ids.")

    d['annotations'] = [i for i in d['annotations'] if i['image_id'] in image_ids]
    imfilenames = [i['file_name'] for i in d['images']]
    for fname in imfilenames:
        impath = os.path.join(path_to_coco_images, fname)
        assert os.path.exists(impath), FileNotFoundError(f"{impath} does not exist.")
    for fname in imfilenames:
        shutil.copyfile(os.path.join(path_to_coco_images, fname),
                        os.path.join(dataset_pth, 'images', fname))
    json.dump(d, open(os.path.join(dataset_pth, 'annotations', 'cocosubset.json'), 'w'))
    print(f"Images and annotations for new dataset written to {dataset_pth}")


if __name__ == '__main__':
    image_ids = [776]
    coco_annotations_file = 'data/datasets/coco/annotations/instances_val2017.json'
    main(coco_annotations_file, image_ids, overwrite=True)
