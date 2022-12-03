def feature_detection(fname):
    import numpy as np
    import cv2
    from matplotlib import pyplot as plt
    import os

    if not os.path.exists('output'):
        os.makedirs('output')

    # read image from file and convert to RGB
    img_bgr = cv2.imread(fname)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_RGB2BGR)
    img1 = img_rgb.copy()
    img2 = img_rgb.copy()
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Tomasi/Shi/Kanade features
    corners = cv2.goodFeaturesToTrack(img_gray, 65, 0.1, 50)
    corners = np.int0(corners)

    h_fp = []
    v_fp = []

    #draw features as circles
    for i in corners:
        x, y = i.ravel()
        h_fp.append(x)
        v_fp.append(y)
        cv2.circle(img1, (x, y), 10, (0, 255, 0), 5)

    return h_fp, v_fp    

#save x and y in their own arrays    
#print(h_fp)
#print(v_fp)

# plot figure
#plt.figure()
#plt.imshow(img1)  # ,plt.show()
#cv2.imwrite('./output/statue_tomasi_features.png', cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))

# FAST features
#fast = cv2.FastFeatureDetector_create()

# find and draw the keypoints
#kp = fast.detect(img2, None)
#img2 = cv2.drawKeypoints(img2, kp, None, color=(0, 255, 0))

# print all default params
#print("Threshold: {}".format(fast.getThreshold()))
#print("nonmaxSuppression:{}".format(fast.getNonmaxSuppression()))
#print("neighborhood: {}".format(fast.getType()))
#print("Total Keypoints with nonmaxSuppression: {}".format(len(kp)))

#plt.figure()
#plt.imshow(img2)
#cv2.imwrite('./output/statue_fast_features.png', cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))

#plt.show()
