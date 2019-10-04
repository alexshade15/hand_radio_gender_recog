import numpy
import pymorph as m
import mahotas

def hsv_from_rgb(image):
    image = image/255.0
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    m, M = numpy.min(image[:,:,:3], 2), numpy.max(image[:,:,:3], 2)
    d = M - m

    # Chroma and Value
    c = d
    v = M

    # Hue
    h = numpy.select([c ==0, r == M, g == M, b == M], [0, ((g - b) / c) % 6, (2 + ((b - r) / c)), (4 + ((r - g) / c))], default=0) * 60

    # Saturation
    s = numpy.select([c == 0, c != 0], [0, c/v])

    return h, s, v

import os

print (os.getcwd())

image = mahotas.imread('hand.jpg')

#downsample for speed
image = image[::10, ::10, :]

h, s, v = hsv_from_rgb(image)

# binary image from hue threshold
b1 = h<35

# close small holes
b2 = m.closerec(b1, m.sedisk(5))

# remove small speckle
b3 = m.openrec(b2, m.sedisk(5))

# locate space between fingers
b4 = m.closeth(b3, m.sedisk(10))

# remove speckle, artifacts from image frame
b5 = m.edgeoff(m.open(b4))

# find intersection of hand outline with 'web' between fingers
b6 = m.gradm(b3)*b5

# reduce intersection curves to single point (assuming roughly symmetric, this is near the center)
b7 = m.thin(m.dilate(b6),m.endpoints('homotopic'))

# overlay marker points on binary image
out = m.overlay(b3, m.dilate(b7, m.sedisk(3)))

mahotas.imsave('output.jpg', out)