# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 19:16:16 2017

@author: James
"""


class KMeansImage:
    
    def __init__(self, image_name, image_type = 'jpg', n_colors = 3,method = 'k-means++'):
        
        #importing necessary libraries
        import numpy as np
        from sklearn.cluster import KMeans
        import matplotlib.pyplot as plt    
        
        self.image_name = image_name
        self.n_colors = n_colors
        self.image_type = image_type
        
        #.jpg images have RGB values from (0-1) and .png images have RGB values from (0-255). We need to account for this.
        normalize = 255
        if (image_type == 'png'):
            normalize = 1
        
        
        self.image_file_name = '{0}.{1}'.format(self.image_name,self.image_type) 
        input_image_array = plt.imread(self.image_file_name)/normalize
        self.image_shape = input_image_array.shape
        self.height = self.image_shape[0]
        self.width = self.image_shape[1]
        self.dimensions = self.image_shape[2]
        height_range = range(self.height)   
        width_range = range(self.width)        

        image_array_reshape = np.reshape(input_image_array,(self.height*self.width,self.dimensions))    
        
        image_clustering = KMeans(n_clusters = self.n_colors ,init=method) #can also be random
        image_clustering.fit(image_array_reshape)
        
        self.color_centers = image_clustering.cluster_centers_ 
        self.center_rgb_values = self.color_centers*255
        
        rgb_predict = image_clustering.predict(image_array_reshape) 

        new_image_array = np.zeros((self.height,self.width,self.dimensions)) 

        i_label =0 
        for i in height_range:
            for j in  width_range:
                new_image_array[i][j] = self.color_centers[rgb_predict[i_label]]
                i_label = i_label + 1 
        #end loop
   
        self.output_image_array = new_image_array
        
        
        
        
    def plot(self, save = False, output_image_name = 'default'):
        import matplotlib.pyplot as plt    

        image = self.output_image_array
        plt.clf()
        plt.figure()
        plt.imshow(image)
        plt.tick_params(axis = 'both', which = 'both', bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off') #gets rid of ticks and tick labels
        plt.show()        
        
        if(save == True):
            output_file_name = '{0}.{1}'.format(output_image_name,self.image_type)
            
            plt.imsave(output_file_name,self.output_image_array)
        
        
        
    #this method isn't needed since center_rgb_values is now an attribute (self.center_rgb_values)
    #def getRGBvalues(self):
    #    return(self.color_centers*255)





#using only plt instead of skio. 
def makeKMeansImage(image_name, image_type = 'jpg', colors = 3, method = 'k-means++'):
    
    import numpy as np
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt    
    
    normalize = 255
    if(image_type =='png'):
        normalize = 1
    
    image_file = '{0}.{1}'.format(image_name,image_type)
    image_array = plt.imread(image_file)/normalize #normalizing is necessary since we need to go from 0-255 for .png, but only (0-1) for .jpg
    image_shape = image_array.shape #should be (height, width, 3)
    print('image shape:', image_shape) #printed to make sure the image array is set correctly - should return (height,width,3)
    height = image_shape[0]
    width= image_shape[1]
    dimensions = image_shape[2]
    height_range = range(height)
    width_range = range(width)

    #we need to reshape as two dimensional ((h x w),3) so it can be handled by scikit-learn
    image_array_new = np.reshape(image_array,(height*width,dimensions))
    clusters = colors #number of colors to reduce to
    

    image_clustering = KMeans(n_clusters = clusters ,init=method) #here, you can change initialization to random, kmeans++, etc.  
    image_clustering.fit(image_array_new) #MOST IMPORTANT LINE. This effectively chooses which n-colors to use in reconstruction
    centers = image_clustering.cluster_centers_ #this is an array of the n-colors RGB values
        
    rgb_predict = image_clustering.predict(image_array_new) #this is an array of indexed values (0,n-colors - 1) for corresponding colors

    #reconstructing image using n-colors (clusters)
    new_image = np.zeros((height,width,dimensions)) #initializes the new images with all zeros


    i_label =0 #initialize counter for pixels and rgb_predict (which is indexed), while new_image is a two-dimensional array
    for i in height_range:
        for j in width_range:
            new_image[i][j] = centers[rgb_predict[i_label]] #break down later? a little hard to understand looking at this line
            i_label = i_label + 1 #we need to keep an index counter since rgb_predict is an array of (0, #pixels - 1)
    #end reconstruction loop. new_image is now ready
                
    return(new_image)
            

def plotKMeansImage(k_means_image, save = False, output_name = 'default',output_type = 'jpg'):
    import matplotlib.pyplot as plt   

    image = k_means_image #since k_means_image returns new_image and centers, we just want the image
    plt.clf()
    plt.figure()
    plt.imshow(image)
    plt.tick_params(axis = 'both', which = 'both', bottom = 'off', left = 'off', labelbottom = 'off', labelleft = 'off') #gets rid of ticks and tick labels
    plt.show()
    
    if (save ==True):
        output_file_name = '{0}.{1}'.format(output_name, output_type)
        plt.imsave(output_file_name,k_means_image)


'''
#start of agglomerative clustering
#
def agglomerativeImage(image_name, ext = 'jpg', output_name = 'none', colors = 16,linkage_type = 'average'):
    #same as kmeans initialization
    image_file = '{0}.{1}'.format(image_name,ext)
    image_array = plt.imread(image_file)/255 
    image_shape = image_array.shape 
    print('image shape:', image_shape)
    height = image_shape[0]
    width= image_shape[1]
    dimensions = image_shape[2]
    height_range = range(height)
    width_range = range(width)    
    
    image_array_new = np.reshape(image_array,((height*width),dimensions))
    print(image_array_new.shape)
    clusters = colors
    
    image_clustering = AgglomerativeClustering(linkage=linkage_type, n_clusters=colors)
    image_clustering.fit(image_array_new)
    
    labels = image_clustering.labels_
    children = image_clustering.children_
    p = image_clustering.get_params
    
    return(children, labels)
     
#end agglomerative (doesn't work)

'''
