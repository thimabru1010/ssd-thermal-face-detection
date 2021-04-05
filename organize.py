import cv2
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

def organize():
    folder_path = 'data/Thermal'

    set_folders = os.listdir(folder_path)

    faces_count = 0

    new_folder = folder_path+'_organized'

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
    for set_single in set_folders:
        set_path = os.path.join(folder_path, set_single)
        participant_folders = os.listdir(set_path)
        print(set_single)
        for participant_folder in participant_folders:
            print(participant_folder)
            person_path = os.path.join(set_path, participant_folder)
            if os.path.isdir(person_path):
                image_names = os.listdir(person_path)

                for image_name in image_names:
                    image_path = os.path.join(person_path, image_name)
                    print(image_name)
                    img = cv2.imread(image_path)

                    cv2.imwrite(os.path.join(new_folder, set_single+'-'+participant_folder+'-'+image_name), img)

def split(folder_path, new_folder):
    def save_images(data, folder_path, new_folder, image_names, dataset_type):
        csv_list = []
        for d in data:
            image_id = str(d[0])+'-'+str(d[1])+'-'+str(d[2])
            #image_id = str(d[0])
            img = cv2.imread(os.path.join(folder_path, image_id))
            # Salvando como Xmin, Xmax, Ymin, Ymax
            #csv_list.append([d[0], d[1], d[2], d[3], d[4], 'Face'])
            #csv_list.append([image_id, d[3], d[4], d[5], d[6], 'Face'])
            h, w, c = img.shape
            # csv_list.append([image_id, d[3]/w, d[4]/h, (d[3]+d[5])/w, (d[4]+d[6])/h, 'Face'])
            csv_list.append([image_id, d[3]/w, (d[3]+d[5])/w, d[4]/h, (d[4]+d[6])/h, 'Face'])
            cv2.imwrite(os.path.join(new_folder, dataset_type, image_id), img)

        df = pd.DataFrame(csv_list, columns= ['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
        df.to_csv(os.path.join(new_folder,'sub-'+dataset_type+'-annotations-bbox.csv'), index=False)

    if not os.path.exists(new_folder):
        os.mkdir(new_folder)
        os.mkdir(os.path.join(new_folder, 'train'))
        os.mkdir(os.path.join(new_folder, 'val'))
        os.mkdir(os.path.join(new_folder, 'test'))

    image_names = os.listdir(folder_path)

    csv_folder = 'data'
    #csv_path = os.path.join(csv_folder,'sub-'+'all'+'-annotations-bbox.csv')
    csv_path = os.path.join(csv_folder,'annotations.csv')
    csv = pd.read_csv(csv_path)
    headers = list(csv)
    print(headers)
    data = csv.values.tolist()

    # print(type(data['Set']))
    print(data[0])

    #data1, data2 = train_test_split(data, test_size=0.8, random_state=42)

    train_val_data, test_data = train_test_split(data, test_size=0.1, random_state=42)

    train_data, val_data = train_test_split(train_val_data, test_size=0.1, random_state=42)

    save_images(train_data, folder_path, new_folder, image_names, 'train')

    save_images(val_data, folder_path, new_folder, image_names, 'val')

    save_images(test_data, folder_path, new_folder, image_names, 'test')

def convert_txt2csv(txt_path, csv_folder):
    f = open(txt_path)
    lines = f.readlines()
    # print(type(lines))
    # print(lines[2])
    # print(type(lines[1]))
    data = []
    for line in lines:
        #print(line)
        line_replace= line.replace(' ', '-')
        line_split = line_replace.split('-')
        #print(data_dirty)
        # Filtra a string
        data_dirty = []
        for l in line_split:
            if l != '':
                data_dirty.append(l)
        #print(data_dirty)
        d = [data_dirty[0], int(data_dirty[1]), int(data_dirty[2]), int(data_dirty[1])+int(data_dirty[3]), int(data_dirty[2])+int(data_dirty[4][:-1]), 'Face']
        #print(d)
        data.append(d)

    f.close()
    print(len(data))
    df = pd.DataFrame(data, columns= ['ImageID', 'XMin', 'XMax', 'YMin', 'YMax', 'ClassName'])
    df.to_csv(os.path.join(csv_folder,'sub-'+'all'+'-annotations-bbox.csv'), index=False)

# txt_path = 'data/CelebA/list_bbox_celeba.txt'
# csv_folder = 'data/CelebA'
# convert_txt2csv(txt_path, csv_folder)

#folder_path = 'data/CelebA/img_celeba'

#new_folder = 'data/CelebA/img_celeba_splitted'
folder_path = 'data/Thermal_organized'

new_folder = 'data/Thermal_organized_splitted'
split(folder_path, new_folder)
#organize()
    #train_folder = folder_path+'train'
