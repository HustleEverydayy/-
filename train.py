import dlib

options = dlib.simple_object_detector_training_options()
options.add_left_right_image_flips = False  
options.C = 5  
options.num_threads = 4 
options.be_verbose = True
print("開始训練:")
dlib.train_simple_object_detector('mydataset.xml', 'mydataset.svm', options)
print("訓練結束!")
