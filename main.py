import sys
import os
import imghdr
import time
import datetime
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import configparser

from moviepy.editor import VideoFileClip
from skimage import morphology
from collections import deque

## colors definitions ##
_ColorBlue = (255, 0, 0)
_ColorGreen = (0, 255, 0)
_ColorRed = (0, 0, 255)
_ColorYellow = (0, 255, 255)
_ColorWhite = (255, 255, 255)
_ColorBlack = (0, 0, 0)

def parse_camera_configuration():
  config = configparser.ConfigParser()
  config.read('camera.cfg')
  # setup section
  setup_screen = config.getboolean('setup', 'setup_screen')
  # parameters section
  scale_percent = config.getint('parameters', 'scale_percent')
  debug_mode = config.getboolean('parameters', 'debug_mode')
  video_fps = config.getint('parameters', 'video_fps')
  # mask section
  bottom_base = config.getfloat('mask', 'bottom_base')
  bottom_left_w_mask = config.getfloat('mask', 'bottom_left_w_mask')
  bottom_right_w_mask = config.getfloat('mask', 'bottom_right_w_mask')
  upper_left_w_mask = config.getfloat('mask', 'upper_left_w_mask')
  upper_left_h_mask = config.getfloat('mask', 'upper_left_h_mask')
  upper_right_w_mask = config.getfloat('mask', 'upper_right_w_mask')
  upper_right_h_mask = config.getfloat('mask', 'upper_right_h_mask')
  # calibration
  perform_calibration = config.getboolean('calibration', 'perform_calibration')
   
  return {\
    'cfg_bottom_base':bottom_base,\
    'cfg_setup_screen':setup_screen,\
    'cfg_scale_percent':scale_percent,\
    'cfg_debug_mode':debug_mode,\
    'cfg_video_fps':video_fps,\
    'cfg_bottom_left_w_mask':bottom_left_w_mask,\
    'cfg_bottom_right_w_mask':bottom_right_w_mask,\
    'cfg_upper_left_w_mask':upper_left_w_mask,\
    'cfg_upper_left_h_mask':upper_left_h_mask,\
    'cfg_upper_right_w_mask':upper_right_w_mask,\
    'cfg_upper_right_h_mask':upper_right_h_mask,\
    'cfg_perform_calibration':perform_calibration,\
  }

#Display
def display(img,title,color=1):
    '''
    func:display image
    img: rgb or grayscale
    title:figure title
    color:show image in color(1) or grayscale(0)
    '''
    if color:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()

def camera_calibration(folder,nx,ny,showMe=0):
    objpoints = []#3D
    imgpoints = []
    objp = np.zeros((nx*ny,3),np.float32)
    #print(objp.shape)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)   
    assert len(folder)!=0
    #print(len(folder))
    for fname in folder:
        img = cv2.imread(fname)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        ret,corners = cv2.findChessboardCorners(gray,(nx,ny))
        img_sz = gray.shape[::-1]
        #print(corners)
        #if ret is True , find corners
        if ret:
            imgpoints.append(corners)
            objpoints.append(objp)
            if showMe:
                draw_corners = cv2.drawChessboardCorners(img,(nx,ny),corners,ret)
                #display(draw_corners,'Found all corners:{}'.format(ret))
    if len(objpoints)==len(imgpoints) and len(objpoints)!=0:
        ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_sz,None,None)
        return {'ret':ret,'cameraMatrix':mtx,'distorsionCoeff':dist,\
               'rotationVec':rvecs,'translationVec':tvecs}
    else:
        raise  Error('Camera Calibration failed')

def correction(image,calib_params,showMe=0):
    corrected = cv2.undistort(image,calib_params['cameraMatrix'],calib_params['distorsionCoeff'],\
                              None,calib_params['cameraMatrix'])
    #if showMe:
        #display(image,'Original',color=1)
        #display(corrected,'After correction',color=1)
    return corrected

def directional_gradient(img,direction='x',thresh=[0,255]):
    '''
    pencv Sobel
    img:Grayscale
    direction:x or y axis
    thresh:apply threshold on pixel intensity of gradient image
    output is binary image
    '''
    if direction=='x':
        sobel = cv2.Sobel(img,cv2.CV_64F,1,0)
    elif direction=='y':
        sobel = cv2.Sobel(img,cv2.CV_64F,0,1)
    sobel_abs = np.absolute(sobel)#absolute value
    scaled_sobel = np.uint8(sobel_abs*255/np.max(sobel_abs))
    binary_output = np.zeros_like(sobel)
    binary_output[(scaled_sobel>=thresh[0])&(scaled_sobel<=thresh[1])] = 1
    return binary_output

def color_binary(img,dst_format='HLS',ch=2,ch_thresh=[0,255]):
    '''
    Color thresholding on channel ch
    img:RGB
    dst_format:destination format(HLS or HSV)
    ch_thresh:pixel intensity threshold on channel ch
    output is binary image
    '''
    if dst_format =='HSV':
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    else:
        img = cv2.cvtColor(img,cv2.COLOR_RGB2HLS)
        ch_binary = np.zeros_like(img[:,:,int(ch-1)])
        ch_binary[(img[:,:,int(ch-1)]>=ch_thresh[0])&(img[:,:,int(ch-1)]<=ch_thresh[1])] = 1
    return ch_binary

def birdView(img,M):
    '''
    Transform image to birdeye view
    img:binary image
    M:transformation matrix
    return a wraped image
    '''
    img_sz = (img.shape[1],img.shape[0])
    img_warped = cv2.warpPerspective(img,M,img_sz,flags = cv2.INTER_LINEAR)
    return img_warped

def perspective_transform(src_pts,dst_pts):
    '''
    perspective transform
    args:source and destination points
    return M and Minv
    '''
    M = cv2.getPerspectiveTransform(src_pts,dst_pts)
    Minv = cv2.getPerspectiveTransform(dst_pts,src_pts)
    return {'M':M,'Minv':Minv}

def find_centroid(image,peak_thresh,window,showMe):
    '''
    find centroid in a window using histogram of hotpixels
    img:binary image
    window with specs {'x0','y0','width','height'}
    (x0,y0) coordinates of bottom-left corner of window
    return x-position of centroid ,peak intensity and hotpixels_cnt in window
    '''
    #crop image to window dimension 
    mask_window = image[round(window['y0']-window['height']):round(window['y0']),
                        round(window['x0']):round(window['x0']+window['width'])]
    histogram = np.sum(mask_window,axis=0)
    centroid = np.argmax(histogram)
    hotpixels_cnt = np.sum(histogram)
    peak_intensity = histogram[centroid]
    if peak_intensity<=peak_thresh:
        centroid = int(round(window['x0']+window['width']/2))
        peak_intensity = 0
    else:
        centroid = int(round(centroid+window['x0']))
    '''
    if showMe:
        plt.plot(histogram)
        plt.title('Histogram')
        plt.xlabel('horzontal position')
        plt.ylabel('hot pixels count')
        plt.show()
    '''
    return (centroid,peak_intensity,hotpixels_cnt)

def find_starter_centroids(image,x0,peak_thresh,showMe):
    '''
    find starter centroids using histogram
    peak_thresh:if peak intensity is below a threshold use histogram on the full height of the image
    returns x-position of centroid and peak intensity
    '''
    window = {'x0':x0,'y0':image.shape[0],'width':image.shape[1]/2,'height':image.shape[0]/2}  
    # get centroid
    centroid , peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    if peak_intensity<peak_thresh:
        window['height'] = image.shape[0]
        centroid,peak_intensity,_ = find_centroid(image,peak_thresh,window,showMe)
    return {'centroid':centroid,'intensity':peak_intensity}

def run_sliding_window(image,centroid_starter,sliding_window_specs, peak_thresh, showMe):
    '''
    Run sliding window from bottom to top of the image and return indexes of the hotpixels associated with lane
    image:binary image
    centroid_starter:centroid starting location sliding window
    sliding_window_specs:['width','n_steps']
        width of sliding window
        number of steps of sliding window alog vertical axis
    return {'x':[],'y':[]}
        coordiantes of all hotpixels detected by sliding window
        coordinates of alll centroids recorded but not used yet!        
    '''
    #Initialize sliding window
    window = {'x0':centroid_starter-int(sliding_window_specs['width']/2),
              'y0':image.shape[0],'width':sliding_window_specs['width'],
              'height':round(image.shape[0]/sliding_window_specs['n_steps'])}
    hotpixels_log = {'x':[],'y':[]}
    centroids_log = []
    if showMe:
        out_img = (np.dstack((image,image,image))*255).astype('uint8')
    for step in range(sliding_window_specs['n_steps']):
        if window['x0']<0: window['x0'] = 0
        if (window['x0']+sliding_window_specs['width'])>image.shape[1]:
            window['x0'] = image.shape[1] - sliding_window_specs['width']
        centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)
        if step==0:
            starter_centroid = centroid
        if hotpixels_cnt/(window['width']*window['height'])>0.6:
            window['width'] = window['width']*2
            window['x0']  = round(window['x0']-window['width']/2)
            if (window['x0']<0):window['x0']=0
            if (window['x0']+window['width'])>image.shape[1]:
                window['x0'] = image.shape[1]-window['width']
            centroid,peak_intensity,hotpixels_cnt = find_centroid(image,peak_thresh,window,showMe=0)          
            
        #if showMe:
        #    print('peak intensity{}'.format(peak_intensity))
        #    print('This is centroid:{}'.format(centroid))
        mask_window = np.zeros_like(image)
        mask_window[window['y0']-window['height']:window['y0'],
                    window['x0']:window['x0']+window['width']]\
            = image[window['y0']-window['height']:window['y0'],
                window['x0']:window['x0']+window['width']]
        
        hotpixels = np.nonzero(mask_window)
        #print(hotpixels_log['x'])
        
        hotpixels_log['x'].extend(hotpixels[0].tolist())
        hotpixels_log['y'].extend(hotpixels[1].tolist())
        # update record of centroid
        centroids_log.append(centroid)
        
        if showMe:
          cv2.rectangle(out_img,
                          (window['x0'],window['y0']-window['height']),
                          (window['x0']+window['width'],window['y0']),(0,255,0),2)
          if step == (sliding_window_specs['n_steps'] - 1):
            plt.imshow(out_img)
            plt.show()
        
        # set next position of window and use standard sliding window width
        window['width'] = sliding_window_specs['width']
        window['x0'] = round(centroid-window['width']/2)
        window['y0'] = window['y0'] - window['height']

    if showMe:
      print('sliding window y count:' + str(len(hotpixels_log['y'])))

    return hotpixels_log

def add_virtual_points(lineLeft, lineRight, showMe):
  # compute mean x left and right to use as default when missing detection
  mean_left = int(np.mean(lineLeft['y']))
  mean_right = int(np.mean(lineRight['y']))
  map_y = {}
  # set in the map with all y coordinates the left x points and default right
  for count in range(len(lineLeft['x'])):
    map_y[lineLeft['x'][count]] = [lineLeft['y'][count], mean_right]
  # add the right x point
  for count in range(len(lineRight['x'])):
    left_x = mean_left
    if lineRight['x'][count] in map_y:
      left_x = map_y.get(lineRight['x'][count])[0]
    map_y[lineRight['x'][count]] = [left_x, lineRight['y'][count]]

  # apply virtual point to the input lanes  
  for key, value in map_y.items(): 
    lineLeft['x'].append(key)
    lineLeft['y'].append(value[0])
    lineRight['x'].append(key)
    lineRight['y'].append(value[1])

  if showMe == True:
    print(map_y.items())
    plt.plot(lineLeft['y'], lineLeft['x'])
    plt.plot(lineRight['y'],lineRight['x'])
    plt.show()

def MahalanobisDist(x,y):
    '''
    Mahalanobis Distance for bi-variate distribution
    
    '''
    covariance_xy = np.cov(x,y,rowvar=0)
    inv_covariance_xy = np.linalg.inv(covariance_xy)
    xy_mean = np.mean(x),np.mean(y)
    x_diff = np.array([x_i-xy_mean[0] for x_i in x])
    y_diff = np.array([y_i-xy_mean[1] for y_i in y])
    diff_xy = np.transpose([x_diff,y_diff])
    
    md = []
    for i in range(len(diff_xy)):
        md.append(np.sqrt(np.dot(np.dot(np.transpose(diff_xy[i]),inv_covariance_xy),diff_xy[i])))
    return md
    
def MD_removeOutliers(x,y,MD_thresh):
    '''
    remove pixels outliers using Mahalonobis distance
    '''
    MD = MahalanobisDist(x,y)
    threshold = np.mean(MD)*MD_thresh
    nx,ny,outliers = [],[],[]
    for i in range(len(MD)):
        if MD[i]<=threshold:
            nx.append(x[i])
            ny.append(y[i])
        else:
            outliers.append(i)
    return (nx,ny)

def update_tracker(tracker,new_value, allx, ally):
    '''
    update tracker(self.bestfit or self.bestfit_real or radO Curv or hotpixels) with new coeffs
    new_coeffs is of the form {'a2':[val2,...],'a1':[va'1,...],'a0':[val0,...]}
    tracker is of the form {'a2':[val2,...]}
    update tracker of radius of curvature
    update allx and ally with hotpixels coordinates from last sliding window
    '''
    if tracker =='bestfit':
        bestfit['a0'].append(new_value['a0'])
        bestfit['a1'].append(new_value['a1'])
        bestfit['a2'].append(new_value['a2'])
    elif tracker == 'bestfit_real':
        bestfit_real['a0'].append(new_value['a0'])
        bestfit_real['a1'].append(new_value['a1'])
        bestfit_real['a2'].append(new_value['a2'])
    elif tracker == 'radOfCurvature':
        radOfCurv_tracker.append(new_value)
    elif tracker == 'hotpixels':
        allx.append(new_value['x'])
        ally.append(new_value['y'])

#fit to polynomial in pixel space
def polynomial_fit(data):
    '''
    a0+a1 x+a2 x**2
    data:dictionary with x and y values{'x':[],'y':[]}
    '''
    a2,a1,a0 = np.polyfit(data['x'],data['y'],2)
    return {'a0':a0,'a1':a1,'a2':a2}

def predict_line(x0,xmax,coeffs):
    '''
    predict road line using polyfit cofficient
    x vaues are in range (x0,xmax)
    polyfit coeffs:{'a2':,'a1':,'a2':}
    returns array of [x,y] predicted points ,x along image vertical / y along image horizontal direction
    '''
    x_pts = np.linspace(x0,xmax-1,num=xmax)
    pred = coeffs['a2']*x_pts**2+coeffs['a1']*x_pts+coeffs['a0']
    return np.column_stack((x_pts,pred))

def compute_radOfCurvature(coeffs,pt):
    return ((1+(2*coeffs['a2']*pt+coeffs['a1'])**2)**1.5)/np.absolute(2*coeffs['a2'])

# read kitty configuration file and read the calibration matrix parameters
def read_kitti_calibration(file_path):
  params = {}
  with open(file_path) as kitti:
    for line in kitti:
      name, values = line.partition(":")[::2]  # every other element
      params[name.strip()] = values.strip()

  #K_xx: 3x3 calibration matrix of camera xx before rectification
  K_00_list = params['K_00'].split(' ')
  K_00 = np.array(K_00_list, dtype='float')
  mtx = np.reshape(K_00, (3, 3))
  #print(mtx)

  #D_xx: 1x5 distortion vector of camera xx before rectification
  D_00_list = params['D_00'].split(' ')
  dist = np.array(D_00_list, dtype='float')
  #print(dist)

  #R_xx: 3x3 rotation matrix of camera xx (extrinsic)
  R_00_list = params['R_00'].split(' ')
  R_00 = np.array(R_00_list, dtype='float')
  rvecs = np.reshape(R_00, (3, 3))
  #print(rvecs)

  #T_xx: 3x1 translation vector of camera xx (extrinsic)
  T_00_list = params['T_00'].split(' ')
  T_00 = np.array(T_00_list, dtype='float')
  tvecs = np.reshape(T_00, (3, 1))
  #print(tvecs)

  return {'ret':True,'cameraMatrix':mtx,'distorsionCoeff':dist,\
     'rotationVec':rvecs,'translationVec':tvecs}

def mask_vertices(cfg, width, height):
  vertices = np.array([[\
    (width * cfg['cfg_bottom_left_w_mask'], height - height * cfg['cfg_bottom_base']),\
    (width/2 - width * cfg['cfg_upper_left_w_mask'],height * cfg['cfg_upper_left_h_mask']),\
    (width/2 + width * cfg['cfg_upper_right_w_mask'], height * cfg['cfg_upper_right_h_mask']),\
    (width - width * cfg['cfg_bottom_right_w_mask'], height - height * cfg['cfg_bottom_base'])]],dtype=np.int32)
  return vertices

def transform_src_vertices(vertices):
  #print('transform_src_vertices:' + str(vertices[0][0][0]) + ' ' + str(vertices[0][0][1]))
  # first point: vertices[0][0] x: [0] y: [1]
  bottom_left = vertices[0][0]
  top_left = vertices[0][1]
  top_right = vertices[0][2]
  bottom_right = vertices[0][3]
  # width margin uses the right points as reference also for the left (same measure)
  src = np.float32([ \
                    [bottom_left[0] + bottom_right[0] * 0.1, bottom_left[1]], \
                    [top_left[0] + top_right[0] * 0.1, top_left[1]], \
                    [top_right[0] - top_right[0] * 0.1, top_right[1]], \
                    [bottom_right[0] - bottom_right[0] * 0.1, bottom_right[1]] \
                    ])
  #src_pts = np.float32([[width*0.2,height],[width/2 - width*0.1,height*0.6],[width/2 + width*0.1,height*0.6],[width - width*0.1,height]])
  return src

def transform_dst_vertices(cfg, vertices, width, height):
  bottom_left = vertices[0][0]
  bottom_right = vertices[0][3]
  dst = np.float32([ \
                    [bottom_left[0] + bottom_left[0] * 0.1, bottom_left[1]], \
                    [bottom_left[0] + bottom_left[0] * 0.1, height * 0.25], \
                    [bottom_right[0] - bottom_right[0] * 0.1, height * 0.25], \
                    [bottom_right[0] - bottom_right[0] * 0.1, bottom_right[1]] \
                    ])
  #dst_pts = np.float32([[width*0.2,height],[width*0.2,0],[width - width*0.1,0],[width - width*0.1,height]])
  return dst

def get_scaled_dimensions(cfg, image):
  scale_percent = cfg['cfg_scale_percent'] # percent of original size
  height = int(image.shape[0] * scale_percent / 100)
  width = int(image.shape[1] * scale_percent / 100)
  return { 'width':width, 'height':height }

'''
# mouse callback handling
class SetupMasking:
  def __init__(self, image):
    self.image = image
  def track_left(self, event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONUP:
      print('Mouse click: x=%, y=%', x, y)
      cv2.circle(self.image,(x,y),1,_ColorRed,10)
'''  

def trigger_alarm():
  # ring the system bell
  sys.stdout.write('\a')
  sys.stdout.flush()
  print('Alarm!')

def setup_screen(cfg, image):
  new_dim = get_scaled_dimensions(cfg, image)
  vertices = mask_vertices(cfg = cfg, width = new_dim['width'], height = new_dim['height'])
  scaled_img = cv2.resize(image, (new_dim['width'], new_dim['height']), interpolation = cv2.INTER_AREA)
  cv2.polylines(scaled_img, vertices, True, _ColorRed, 2)
  src_pts = np.array(transform_src_vertices(vertices), dtype=np.int32)
  print('src points arr:' + str(src_pts))
  for pt in src_pts:
    cv2.circle(scaled_img, tuple(pt), 3, _ColorYellow, -1)
  cv2.imshow('Setup', scaled_img)
  # wait for ESC key to quit
  # delay each frame display
  k = cv2.waitKey(0) & 0xFF
  if k == 27: # ESC
    sys.exit(0)

## DO THE JOB ##
def find_lanes(cfg, input_img, delay = 0):

  # display setup mask and quit
  if(cfg['cfg_setup_screen'] == True):
    setup_screen(cfg, input_img)
    sys.exit(0)
  
  cfg_debug_mode = cfg['cfg_debug_mode']

  corr_img = input_img

  if cfg['cfg_perform_calibration'] == True:
    #folder_calibration = glob.glob("camera_cal/calibration[1-3].jpg")
    #calib_params = camera_calibration(folder_calibration,nx,ny,showMe=1)
    calib_params = read_kitti_calibration('test_images/kitti/calib_cam_to_cam.txt')
    corr_img = correction(input_img,calib_params, cfg_debug_mode)

  # resize image
  new_dim = get_scaled_dimensions(cfg, corr_img)
  width = new_dim['width']
  height = new_dim['height']
  corr_img = cv2.resize(corr_img, (width, height), interpolation = cv2.INTER_AREA)

  gray_ex = cv2.cvtColor(corr_img,cv2.COLOR_RGB2GRAY)
  if cfg_debug_mode == True:
    print('scaled width: %d heigth: %d' % (width, height))
    display(gray_ex,'Apply Camera Correction',color=0)

  gradx_thresh = [25,255]
  gradx = directional_gradient(gray_ex,direction='x',thresh = gradx_thresh)
  if cfg_debug_mode == True:
    display(gradx,'Gradient x',color=0)

  #ch_thresh = [50,255]
  #ch3_hls_binary = color_binary(corr_img,dst_format='HLS',ch=3,ch_thresh=ch_thresh)
  ##display(ch3_hls_binary,'HLS to Binary S',color=0)

  combined_output = np.zeros_like(gradx)
  #combined_output[((gradx==1)|(ch3_hls_binary==1))] = 1
  #if cfg_debug_mode == True:
  #  display(combined_output,'Combined output',color=0)

  # original image to bird view (transformation)
  mask = np.zeros_like(combined_output)

  vertices = mask_vertices(cfg = cfg, width = new_dim['width'], height = new_dim['height'])
  cv2.fillPoly(mask,vertices,1)
  masked_image = cv2.bitwise_and(gradx,mask)
  if cfg_debug_mode == True:
    print('vertices:\n' + str(vertices))
    display(masked_image,'Masked',color=0)

  min_sz = 18
  cleaned = morphology.remove_small_objects(masked_image.astype('bool'),min_size=min_sz,connectivity=2)
  if cfg_debug_mode == True:
    display(cleaned,'cleaned',color=0)

  src_pts = transform_src_vertices(vertices)
  dst_pts = transform_dst_vertices(cfg, vertices, width = width, height = height)

  transform_matrix = perspective_transform(src_pts,dst_pts)
  warped_image = birdView(cleaned*1.0,transform_matrix['M'])
  if cfg_debug_mode == True:
    print('src_pts:\n' + str(src_pts))
    print('dst_pts:\n' + str(dst_pts))
    display(warped_image,'BirdViews',color=0)
  bottom_crop = -40
  warped_image = warped_image[0:bottom_crop,:]
  
  # if number of histogram pixels in window is below 10,condisder them as noise and does not attempt to get centroid
  peak_thresh = 10
  centroid_starter_right = find_starter_centroids(warped_image,x0=warped_image.shape[1]/2,
                                                peak_thresh=peak_thresh,showMe=(cfg_debug_mode == True))
  centroid_starter_left = find_starter_centroids(warped_image,x0=0,peak_thresh=peak_thresh,
                                                showMe=(cfg_debug_mode == True))
  #
  sliding_window_specs = {'width':120,'n_steps':10}
  log_lineLeft = run_sliding_window(warped_image,centroid_starter_left['centroid'],sliding_window_specs, peak_thresh=peak_thresh, showMe=(cfg_debug_mode == True))
  log_lineRight = run_sliding_window(warped_image,centroid_starter_right['centroid'],sliding_window_specs,peak_thresh=peak_thresh, showMe=(cfg_debug_mode == True))

  MD_thresh = 1.8
  log_lineLeft['x'],log_lineLeft['y'] = \
  MD_removeOutliers(log_lineLeft['x'],log_lineLeft['y'],MD_thresh)
  log_lineRight['x'],log_lineRight['y'] = \
  MD_removeOutliers(log_lineRight['x'],log_lineRight['y'],MD_thresh)

  if cfg_debug_mode == True:
    print('Outliers y Left count:' + str(len(log_lineLeft['y'])))
    print('Outliers y Right count:' + str(len(log_lineRight['y'])))
    plt.plot(log_lineLeft['y'],log_lineLeft['x'])
    plt.plot(log_lineRight['y'],log_lineRight['x'])
    plt.show()

  display_warning = False
  if len(log_lineLeft['x']) == 0 and len(log_lineLeft['y']) == 0 or \
    len(log_lineRight['x']) == 0 and len(log_lineRight['y']) == 0:
    # missing lane detection - alarm
    trigger_alarm()
    display_warning = True
  else:
    # add virtual points to increase polyfit precision
    add_virtual_points(log_lineLeft, log_lineRight, showMe=(cfg_debug_mode == True))
  
  # if left and right points are the same trigger alarm
  if log_lineLeft['y'] == log_lineRight['y']:
    trigger_alarm()
    display_warning = True

  fit_lineRight_singleframe = {'a0':0,'a1':0,'a2':0}
  if log_lineRight['x'] and log_lineRight['y']:
    fit_lineRight_singleframe = polynomial_fit(log_lineRight)

  fit_lineLeft_singleframe = {'a0':0,'a1':0,'a2':0}
  if log_lineLeft['x'] and log_lineLeft['y']:
    fit_lineLeft_singleframe = polynomial_fit(log_lineLeft)

  var_pts = np.linspace(0,corr_img.shape[0]-1,num=corr_img.shape[0])
  pred_lineLeft_singleframe = predict_line(0,corr_img.shape[0],fit_lineLeft_singleframe)
  pred_lineRight_sigleframe = predict_line(0,corr_img.shape[0],fit_lineRight_singleframe)

  center_of_lane = (pred_lineLeft_singleframe[:,1][-1]+pred_lineRight_sigleframe[:,1][-1])/2

  #merters per pixel in y or x dimension
  ym_per_pix = 12/450
  xm_per_pix = 3.7/911
  offset = (corr_img.shape[1]/2 - center_of_lane)*xm_per_pix

  side_pos = 'right'
  if offset <0:
    side_pos = 'left'

  if abs(offset) > 0.7:
    trigger_alarm()
    display_warning = True

  wrap_zero = np.zeros_like(gray_ex).astype(np.uint8)
  color_wrap = np.dstack((wrap_zero,wrap_zero,wrap_zero))
  
  left_fitx = fit_lineLeft_singleframe['a2']*var_pts**2 + fit_lineLeft_singleframe['a1']*var_pts + fit_lineLeft_singleframe['a0']
  pts_left = np.array([np.transpose(np.vstack([left_fitx,var_pts]))])

  right_fitx = fit_lineRight_singleframe['a2']*var_pts**2 + fit_lineRight_singleframe['a1']*var_pts+fit_lineRight_singleframe['a0']
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx,var_pts])))])

  pts = np.hstack((pts_left,pts_right))
  cv2.fillPoly(color_wrap,np.int_([pts]), _ColorGreen)
  cv2.putText(color_wrap,'|',(int(center_of_lane),corr_img.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1, _ColorBlue,8)
  newwrap = cv2.warpPerspective(color_wrap,transform_matrix['Minv'],(corr_img.shape[1],corr_img.shape[0]))
  result = cv2.addWeighted(corr_img,1,newwrap,0.3,0)
  if display_warning == True:
      cv2.putText(result,'Warning!', (int(result.shape[1]/3), int(result.shape[0]/2)),cv2.FONT_HERSHEY_SIMPLEX,1,_ColorRed,thickness=2)
  else:
      cv2.putText(result,'Vehicle is: ' + str(round(abs(offset),1))+' m ' + side_pos + ' of center',
            (50,100),cv2.FONT_HERSHEY_SIMPLEX,1,_ColorRed,thickness=1)
 
  cv2.imshow("Result", result)
  # wait for ESC key to quit
  # delay each frame display
  k = cv2.waitKey(delay) & 0xFF
  if k == 27: # ESC
    sys.exit(0)

def play_video(cfg, capture):
  media_capture = cv2.VideoCapture(capture)
  # get the frames per second of the input video
  fps = media_capture.get(cv2.CAP_PROP_FPS)
  print('Original video at  %d fps' % fps)
  video_fps = cfg['cfg_video_fps']
  if(video_fps == 0):
    video_fps = fps
  print('Playing at %d fps' % video_fps)
  
  # initialize fps timestamps
  delay_each_frame_ms = 1000/video_fps
  dt_next = datetime.datetime.utcnow() 

  while True:
    # Capture frame-by-frame
    ret, image = media_capture.read()
    dt_now = datetime.datetime.utcnow()

    if not ret:
      sys.exit(2)
  
    # delay each frame display
    #find_lanes(cfg, image, int(1000/video_fps))
    if(dt_now >= dt_next):
      dt_next = dt_now + datetime.timedelta(microseconds=delay_each_frame_ms*1000)
      print('Now: ' + dt_now.strftime("%H:%M:%S.%f"))
      print('Nxt: ' + dt_next.strftime("%H:%M:%S.%f"))
      find_lanes(cfg, image, 1)
    else:
      time.sleep(fps/1000) # sleep original tps to skip to the correct next frame

  media_capture.release()

def process_input(argv): 
  # get configuration params
  cfg = parse_camera_configuration()

  if len(argv) == 1: # open input camera
    play_video(cfg, 0)
  elif len(argv) != 3:
    print('Specify the input type: picture/video/directory path')
    sys.exit(2)
  else:
    if argv[1] == 'picture':
      print('Processing file: ' + str(argv[2]))
      media_capture = cv2.imread(argv[2])
      find_lanes(cfg, media_capture)
    elif argv[1] == 'directory':
      files = [f.path for f in os.scandir(argv[2])]
      files.sort()
      for file in files:
        if(imghdr.what(file) != None):
          print('Processing file: ' + str(file))
          media_capture = cv2.imread(file)
          find_lanes(cfg, media_capture, 1)
    elif argv[1] == 'video':
      play_video(cfg, argv[2])

if __name__ == "__main__":
  process_input(sys.argv)

