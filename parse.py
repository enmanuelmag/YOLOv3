"""
Create function to get all filenames on USD_DIVISA/labels and save into a csv
"""
import os
import re
import random

def get_filenames(path):
  filenames = []
  for filename in os.listdir(path):
    #if file ends with .jpg or .png or .jpeg
    if re.search(r'\.(jpg|png|jpeg)$', filename):
      #replace .jpg or .png or .jpeg with .txt
      txt_filename = filename.replace(re.search(r'\.(jpg|png|jpeg)$', filename).group(0), '.txt')

      exits_txt = os.path.isfile(os.path.join(
        path.replace('images', 'labels'),
        txt_filename
      ))
      if exits_txt:
        filenames.append({ 'txt': txt_filename, 'img': filename })
  return filenames

def save_csv(path, save_path_train, save_path_test):
  filenames = get_filenames(path)
  #shuffle filenames
  random.shuffle(filenames)
  length = len(filenames)

  train_filenames = filenames[:int(length*0.8)]
  test_filenames = filenames[int(length*0.8):]

  if os.path.isfile(save_path_train):
    os.remove(save_path_train)
  if os.path.isfile(save_path_test):
    os.remove(save_path_test)
    
  with open(save_path_test, 'w') as csvfile:
    for filename in train_filenames:
      csvfile.write(filename['img'] + ',' + filename['txt'] + '\n')
  with open(save_path_train, 'w') as csvfile:
    for filename in test_filenames:
      csvfile.write(filename['img'] + ',' + filename['txt'] + '\n')

save_csv('./USD_DIVISA/images', './USD_DIVISA/train.csv', './USD_DIVISA/test.csv')