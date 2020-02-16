import os
import argparse
import json
import cv2


def load_annotation(annotation_path):
    annotation_list = []
    class_list = []
    with open(annotation_path, "r") as f:
        for line in f:
            tmp_line = line.strip()
            one_annotation = tmp_line.split(" ")#image filename, class name, subset of class, x1, y1, x2, y2
            annotation_list.append(one_annotation)
            class_list.append(one_annotation[1])

    class_list = list(set(class_list))#remove duplicated class
    return annotation_list, sorted(class_list)


def SplitAnnotationData(logo_annotation_list, testID=6):
    train_annotation_list = []
    test_annotation_list = []
    for logo_annotation in logo_annotation_list:
        if(int(logo_annotation[2]) == testID):
            test_annotation_list.append(logo_annotation)
        else:
            train_annotation_list.append(logo_annotation)

    return train_annotation_list, test_annotation_list


def Flickrlogo2JSON(logo_annotation_list, class_list, images_dir, outname):
    attrDict = dict()
    categories_dict_list = []
    for i, logoclass in enumerate(class_list):
        category_dict = {"supercategory":"logo", "id":i+1,"name":logoclass}
        categories_dict_list.append(category_dict)

    attrDict["categories"] = categories_dict_list
    #print(attrDict["categories"])

    images_list = list()
    annotations_list = list()

    current_image = ""
    image_id = 0
    annotation_id = 1
    for i, logo_annotation in enumerate(logo_annotation_list):
        #There is probability of having multiple annotations in same image.
        if(current_image != logo_annotation[0]):
            image_id += 1
            current_image = logo_annotation[0]
            imagepath = os.path.join(images_dir, logo_annotation[0])
            image = cv2.imread(imagepath)
            height, width, _ = image.shape
            image_dict = {"file_name":logo_annotation[0], "height":height, "width":width, "id":image_id}
            images_list.append(image_dict)
        
            annotation_id = 1 # initialize annotation_id

        annotation_dict = dict()
        annotation_dict["iscrowd"] = 0
        annotation_dict["image_id"] = image_id
        #bbox : sx, sy, w, h
        sx = int(logo_annotation[3])
        sy = int(logo_annotation[4])
        ex = int(logo_annotation[5])
        ey = int(logo_annotation[6])
        bbox_w = ex - sx + 1
        bbox_h = ey - sy + 1
        annotation_dict["bbox"] = [sx, sy, bbox_w, bbox_h]
        annotation_dict["area"] = float(bbox_w * bbox_h)
        annotation_dict["category_id"] = class_list.index(logo_annotation[1]) + 1
        annotation_dict["id"] = annotation_id
        annotation_dict["segmentation"] = [[sx,sy, sx,ex, ex,ey, sx,ey]]

        annotation_id += 1
        annotations_list.append(annotation_dict)

        
    attrDict["images"] = images_list   
    attrDict["annotations"] = annotations_list
    attrDict["type"] = "instances"

    jsonString = json.dumps(attrDict)
    with open(outname, "w") as f:
        f.write(jsonString)
        

def main():
    parser = argparse.ArgumentParser("flickr Logo Converter")
    parser.add_argument("input", type=str, help="flickr Logo Path")
    parser.add_argument("-testID", default=6, type=int, help="subset of class for test data.")
    parser.add_argument("-trainout", default="train.json", type=str, help="json name for train.")
    parser.add_argument("-testout", default="test.json", type=str, help="json name for test.")

    args = parser.parse_args()

    images_dir = os.path.join(args.input, "flickr_logos_27_dataset_images")
    annotation_path = os.path.join(args.input, "flickr_logos_27_dataset_training_set_annotation.txt")

    logo_annotation_list, class_list = load_annotation(annotation_path)

    train_annotation_list, test_annotation_list = SplitAnnotationData(logo_annotation_list, testID=args.testID)

    Flickrlogo2JSON(train_annotation_list, class_list, images_dir, args.trainout)
    if(len(test_annotation_list) > 0):
        Flickrlogo2JSON(test_annotation_list, class_list, images_dir, args.testout)


if __name__ == '__main__':
    main()