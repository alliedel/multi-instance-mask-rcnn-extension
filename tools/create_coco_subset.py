import os
import json

COCO_DIRECTORY = 'datasets/coco'


def coco_ids_match(anno_id, image_id):
    return int(anno_id) == int(image_id)


def main():
    for split in ['train', 'val'][::-1]:
        name = f"coco_2017_debug_{split}"
        subset_list_file = os.path.join(COCO_DIRECTORY, f"{name}.txt")
        full_annos_file = os.path.join(COCO_DIRECTORY, 'annotations', f"instances_{split}2017.json")
        subset_annos_file = os.path.join(COCO_DIRECTORY, 'annotations', f"instances_{name}.json")
        assert os.path.isfile(subset_list_file), subset_list_file
        assert os.path.isfile(full_annos_file), full_annos_file
        annotations = create_subset_annotations(full_annos_file, subset_list_file)
        json.dump(annotations, open(subset_annos_file, 'w'))


def create_subset_annotations(full_annos_file, subset_list_file):
    image_list = []
    with open(subset_list_file, 'r') as f:
        s = f.readline().strip()
        while s:
            image_list.append(s)
            s = f.readline()
    image_id_list = [os.path.splitext(os.path.basename(s))[0] for s in image_list]
    annotations = json.load(open(full_annos_file, 'r'))
    new_annotations = prune_annotations(annotations['annotations'], image_id_list)
    del annotations['annotations']
    annotations['annotations'] = new_annotations
    return annotations


def prune_annotations(annotations_list, image_id_list):
    annotation_idxs = []
    for imid in image_id_list:
        i = 0
        while not coco_ids_match(annotations_list[i]['image_id'], imid):
            i += 1
            if i == len(annotations_list):
                print(f"{imid} not found")
                break
        if i < len(annotations_list):
            annotation_idxs.append(i)
    n_not_found = len(image_id_list)-len(annotation_idxs)
    if n_not_found > 0:
        print(f"{n_not_found}/{len(image_id_list)} not found")
    else:
        print(f"All {len(image_id_list)} annotations found")
    return [annotations_list[i] for i in annotation_idxs]


if __name__ == '__main__':
    main()
