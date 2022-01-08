"""
Create function to get all filenames on USD_DIVISA/labels and save into a csv
"""
import os

def get_filenames(path):
  filenames = []
  for filename in os.listdir(path):
    if filename.endswith(".txt"):
      exits_image = os.path.isfile(os.path.join(
        path.replace('labels', 'images'),
        filename.replace('txt', 'png')
      ))
      exits_image = None
      for img in os.listdir(path.replace('labels', 'images')):
        if img.split('.')[0] == filename.split('.')[0]:
          exits_image = img
      if exits_image:
        filenames.append({ 'txt': filename, 'img': exits_image })
  return filenames

def save_csv(path, save_path_train, save_path_test):
  import random
  filenames = get_filenames(path)
  #shuffle filenames
  random.shuffle(filenames)
  length = len(filenames)

  train_filenames = filenames[:int(length*0.8)]
  test_filenames = filenames[int(length*0.8):]
  #delete old csv
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

save_csv('./USD_DIVISA/labels', './USD_DIVISA/train.csv', './USD_DIVISA/test.csv')