import os

PATH = './USD_DIVISA/labels.old/'
PATHT = './USD_DIVISA/labels/'

#remove all file on PATHT
for file in os.listdir(PATHT):
  os.remove(PATHT + file)


for file in os.listdir(PATH):
  trans = []
  with open(PATH + file) as f:
   for line in f.readlines():
    line = line.strip()
    a, b, c, d, e = line.split(' ')
    d = float(d)
    e = float(e)
    b = float(b) + ( d * 0.5 )
    c = float(c) + ( e * 0.5 )
    if b > 1:
      b = 0.95
    if c > 1:
      c = 0.95
    if b < 0:
      b = 0.05
    if c < 0:
      c = 0.05
    if d > 1:
      d = 0.95
    if e > 1:
      e = 0.95
    if d < 0:
      d = 0.05
    if e < 0:
      e = 0.05
    trans.append(f'{a} {b} {c} {d} {e}\n')
  
  with open(PATHT + file, 'w') as ft:
    ft.writelines(trans)