import time
import torch
import torch.nn as nn
import config

if torch.cuda.is_available():
  torch.cuda.empty_cache()
  torch.cuda.set_device(0)


if torch.cuda.is_available():  
  dev = "cuda:0" 
else:  
  dev = "cpu"  
device = torch.device(dev) 

types = { 'U': 'UPSAMPLING', 'B': 'BLOCK_CONVS', 'S': 'SKIP_PREDICTIONS' }
"""
This list has the config for the DarkNet35 layers and Scale predictions layers:

- Tuple : (out_channel, kernel_size, stride)

- List: ["B", numer_of_repeats], B is for a block that consisting of:
  1. One Conv (32 filtersd, 1x1 size, stride 1)
  2. One Conv (64 filters, 3x3, stride 1)
  3. One Conv (2x2 size and stride 2)

- String: "S" is for Skip predictions branch and "U" is for Upsampling layer
"""
config_layers = [
  #Begin the Darknet53
  (32, 3, 1),
  (64, 3, 2),
  [types.get('B'), 1],
  (128, 3, 2),
  [types.get('B'), 2],
  (256, 3, 2),
  [types.get('B'), 8],
  (512, 3, 2),
  [types.get('B'), 8],
  (1024, 3, 2),
  [types.get('B'), 4],
  #End the Darknet53
  #Begin scales predictions
  (512, 1, 1),
  (1024, 3, 1),
  types.get('S'),
  (256, 1, 1),
  types.get('U'),
  (256, 1, 1),
  (512, 3, 1),
  types.get('S'),
  (128, 1, 1),
  types.get('U'),
  (128, 1, 1),
  (256, 3, 1),
  types.get('S'),
  #End scale predictions
]

class CNNBlock(nn.Module):
  """
  This block is used to group a layers that repeat a number of times
  """
  def __init__(self, in_chns, out_chns, bn=True, drop=0.10, **kwargs):
    super().__init__()
    #bn is for batch normalization
    self.conv = nn.Conv2d(in_chns, out_chns, bias=not bn,**kwargs)
    self.dropout = nn.Dropout2d(drop if config.USE_DROPOUT else 0)
    self.bn = nn.BatchNorm2d(out_chns) if bn else None
    self.leaky = nn.LeakyReLU(0.1)
    self.use_bn = bn

  def forward(self, x):
    if self.use_bn:
      return self.leaky(self.bn(self.dropout(self.conv(x))))
    
    return self.dropout(self.conv(x))


class ResidualBlock(nn.Module):
  """
  This block is used to group a layers that repeat a number of times
  """
  def __init__(self, chns, residual=True, start_drop=0.10, end_drop=0.20, repeats=1):
    super().__init__()
    self.layers = nn.ModuleList()
    for i in range(repeats):
      drop = (start_drop + (end_drop - start_drop) * i) / (repeats - 1)
      self.layers += [
        nn.Sequential(
          CNNBlock(in_chns=chns, out_chns=chns//2, drop=drop, kernel_size=1),
          CNNBlock(in_chns=chns//2, out_chns=chns, drop=drop, kernel_size=3, padding=1)
        )
      ]

    self.residual = residual
    self.repeats = repeats

  def forward(self, x):
    for layer in self.layers:
      x = layer(x) + x if self.residual else layer(x)
    return x


class ScalePrediction(nn.Module):
  """
  This block is used for create a branch to get the scale prediction.
  This layers show the prediction that is a vector that contains:
    - The object clas score
    - The x and y coordinates of the center of the object
    - The width and height of the boundary of the object
  """
  def __init__(self, in_chns, num_classes):
    super().__init__()
    """"""
    self.num_classes = num_classes
    self.pred = nn.Sequential(
      CNNBlock(in_chns, 2 * in_chns, kernel_size=3, drop=0.25, padding=1),
      CNNBlock(2 * in_chns, 3 * (num_classes + 5), drop=0.30, bn=False, kernel_size=1)
    )

  def forward(self, x):
    return (
      self.pred(x) 
        .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
        .permute(0, 1, 3, 4, 2) 
    )


class YOLOv3(nn.Module):
  def __init__(self, in_chns=3, num_classes=6):
    super().__init__()
    self.num_classes = num_classes
    self.in_chns = in_chns
    self.layers = self._create_layers()
  
  def forward(self, x):
    outputs = []
    routes_cons = []

    for layer in self.layers:
      if isinstance(layer, ScalePrediction):
        outputs.append(layer(x))
        continue

      x = layer(x)

      if isinstance(layer, ResidualBlock) and layer.repeats == 8:
        routes_cons.append(x)
      
      elif isinstance(layer, nn.Upsample):
        x = torch.cat([x, routes_cons.pop()], dim=1)
    
    return outputs

  def _create_layers(self):
    print("Creating YOLOv3 layers...")
    layers = nn.ModuleList()
    num_classes = self.num_classes
    in_chans = self.in_chns

    skips_pred_num = 1
    up_samples_num = 1

    for module in config_layers:
      if isinstance(module, tuple):
        out_chans, kernel_size, stride = module
        layers.append(
          CNNBlock(
            in_chans,
            out_chans,
            stride=stride,
            kernel_size=kernel_size,
            padding=1 if kernel_size == 3 else 0
          )
        )
        in_chans = out_chans

      elif isinstance(module, list):
        num_repeats = module[1]
        layers.append(
          ResidualBlock(in_chans, repeats=num_repeats)
        )

      elif isinstance(module, str):

        if module == types.get('S'):
          skips_pred_num += 1
          layers += [
            ResidualBlock(in_chans, residual=False, repeats=1),
            CNNBlock(in_chans, in_chans//2, kernel_size=1),
            ScalePrediction(in_chans//2 , num_classes)
          ]
          in_chans = in_chans // 2

        elif module == types.get('U'):
          up_samples_num += 1
          layers.append(
            nn.Upsample(scale_factor=2, mode='nearest'),
          )
          in_chans = in_chans * 3
    
    return layers

if __name__ == "__main__":
  num_classes = 6
  IMAGE_SIZE = 416

  model = YOLOv3(num_classes=num_classes)
  model
  x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
  x.to(device)
  start = time.time()
  out = model(x)
  print(f'Time: {time.time() - start}')

  assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
  assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
  assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)

  print("All done without errors")
