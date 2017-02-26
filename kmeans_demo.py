# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 22:33:31 2017

@author: James
"""

'''
This is an example script using KMeansImage

The KMeansImage library can be used to create KMeansImage objects, but can also be used by calling methods in the KMeansImage file


to create an object, initialize as kmi.KMeansImage
from here you have access to its attributes and can plot or save the image by calling .plot()


to just call and return a KMeansImage you can use the function makeKmeansImage





'''






import time
import KMeansImage as kmi

image_names = ['millie','civil']
image_types = ['jpg','png']


#using classes
print('using class objects:\n')

t_start0 = time.time()
kmi0 = kmi.KMeansImage(image_name = image_names[0], image_type = image_types[0], n_colors=3)
kmi0.plot(save = True, output_image_name = 'test')
print('RGB values:\n',kmi0.center_rgb_values)
time0 = round((time.time()-t_start0),1)
print('K-Means image took',time0,'seconds')


'''
t_start1 = time.time()
kmi1 = kmi.KMeansImage(image_name = image_names[1], image_type = image_types[1], n_colors=3)
print(kmi1.image_file_name)
kmi1.plot()
print('RGB values:\n',kmi1.center_rgb_values)
time1 = round((time.time()-t_start1),1)
print('K-Means image took',time1,'seconds')

#using methods
print('using method calls:\n')

t_start2 = time.time()
kmi2 = kmi.makeKMeansImage(image_name = image_names[0],image_type = image_types[0])
kmi.plotKMeansImage(kmi2)
time2 = round((time.time()-t_start2),1)
print('K-Means image took',time2,'seconds')

t_start3 = time.time()
kmi3 = kmi.makeKMeansImage(image_name = image_names[1],image_type = image_types[1])
kmi.plotKMeansImage(kmi3)
time3 = round((time.time()-t_start3),1)
print('K-Means image took',time3,'seconds')
'''







