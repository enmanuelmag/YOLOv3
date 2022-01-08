import os

PATH = './USD_DIVISA/labels/'
PATHT = './USD_DIVISA/labels_trans/'

#remove all file on PATHT
for file in os.listdir(PATHT):
  os.remove(PATHT + file)


for file in os.listdir('./USD_DIVISA/labels'):
  trans = []
  with open(PATH + file) as f:
   for line in f.readlines():
    line = line.strip()
    a, b, c, d, e = line.split(' ')

    b = float(b) + ( float(d) * 0.5 )
    c = float(c) + ( float(e) * 0.5 )
    #d = float(d) * 1.5
    #e = float(e) * 1.5
    trans.append(f'{a} {b} {c} {d} {e}\n')
  
  with open(PATHT + file, 'w') as ft:
    ft.writelines(trans)