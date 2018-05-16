from keras.layers import Cropping2D
from keras.models import Model


def crop( output1 , output2 , input  ):
  _, w1, h1, _ = Model( input  , output1 ).output_shape
  _, w2, h2, _ = Model(input, output2).output_shape

  if w1 == w2 and h1 == h2:
    return output1, output2
    
  cx = abs(h1 - h2)
  cy = abs(w1 - w2)

  if w1 > w2:
    output1 = Cropping2D(cropping=((0, 0), (0, cx)))(output1)
  else:
    output2 = Cropping2D(cropping=((0, 0), (0, cx)))(output2)

  if h1 > h2:
    output1 = Cropping2D(cropping=((0, cy), (0, 0)))(output1)
  else:
    output2 = Cropping2D(cropping=((0, cy), (0, 0)))(output2)


  return output1, output2
