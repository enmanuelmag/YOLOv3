"""
Create function to get all filenames on USD_DIVISA/labels and save into a csv
"""
import os

def get_filenames(path):
  filenames = []
  for filename in os.listdir(path):
    if filename.endswith(".txt"):
      filenames.append(filename)
  return filenames

def save_csv(path, save_path_train, save_path_test):
  import random
  filenames = get_filenames(path)
  #shuffle filenames
  random.shuffle(filenames)
  length = len(filenames)

  train_filenames = filenames[:int(length*0.8)]
  test_filenames = filenames[int(length*0.8):]
  
  with open(save_path_test, 'w') as csvfile:
    for filename in train_filenames:
      csvfile.write(str(filename) + ',' +str(filename) + '\n')
  with open(save_path_train, 'w') as csvfile:
    for filename in test_filenames:
      csvfile.write(str(filename) + ',' +str(filename) + '\n')

save_csv('./USD_DIVISA/labels', './USD_DIVISA/train.csv', './USD_DIVISA/test.csv')