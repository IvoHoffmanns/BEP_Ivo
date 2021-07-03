# -*- coding: utf-8 -*-
import PySimpleGUI as sg
import os
from PIL import Image, ImageTk
import io
import numpy as np
import cv2 
import pandas as pd
import time
import sys
from matplotlib.ticker import NullFormatter  # useful for `logit` scale
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# =============================================================================
# import matplotlib
# matplotlib.use('TkAgg')
# =============================================================================
import matplotlib
matplotlib.use('Qt5Agg')  # YES - Qt!

"""
Simple Image Browser based on PySimpleGUI
--------------------------------------------
There are some improvements compared to the PNG browser of the repository:
1. Paging is cyclic, i.e. automatically wraps around if file index is outside
2. Supports all file types that are valid PIL images
3. Limits the maximum form size to the physical screen
4. When selecting an image from the listbox, subsequent paging uses its index
5. Paging performance improved significantly because of using PIL
Dependecies
------------
Python3
PIL
"""
k=0
pvalue=200 #pressure value constant for test
D_tip = 0.175 # Tip diameter in mm. Maybe an idea to take it as an input 
D_cap = 0.680 # Capillary inner diameter. Take as an input 
L_taper = 2.3 # Length of taper in mm. Maybe an idea to take it as an input 
count =0    #counts the amount of points selected
number=0    #counts the number of images processed (slice)
error=True
Firstsave=True
pos=np.zeros((6,2),np.int)  #initialize the position matrix for the coords
folder='videos/test/24-06'
img_types = (".png", ".jpg", "jpeg", ".tiff", ".bmp")
#fnames=[]

# ------------------------------------------------------------------------------
# draw figure without toolbar
# ------------------------------------------------------------------------------

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg
# ------------------------------------------------------------------------------
# Draw figure with toolbar
# ------------------------------------------------------------------------------
def draw_figure_w_toolbar(canvas, fig, canvas_toolbar):
    if canvas.children:
        for child in canvas.winfo_children():
            child.destroy()
    if canvas_toolbar.children:
        for child in canvas_toolbar.winfo_children():
            child.destroy()
    figure_canvas_agg = FigureCanvasTkAgg(fig, master=canvas)
    figure_canvas_agg.draw()
    toolbar = Toolbar(figure_canvas_agg, canvas_toolbar)
    toolbar.update()
    figure_canvas_agg.get_tk_widget().pack(side='right', fill='both', expand=1)
    
class Toolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    # toolitems = [t for t in NavigationToolbar2Tk.toolitems if
    #              t[0] in ('Home', 'Pan', 'Zoom')]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    toolitems = [t for t in NavigationToolbar2Tk.toolitems]
                # t[0] in ('Home', 'Pan', 'Zoom','Save')]
    def __init__(self, *args, **kwargs):
        super(Toolbar, self).__init__(*args, **kwargs)
# ------------------------------------------------------------------------------
# use PIL to read data of one image
# ------------------------------------------------------------------------------
def get_img_data(f, maxsize=(1440, 1024), first=False):
    """Generate image data using PIL
    """
    img = Image.open(f)
    img.thumbnail(maxsize)
    if first:                     # tkinter is inactive the first time
        bio = io.BytesIO()
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
    return ImageTk.PhotoImage(img)
# ------------------------------------------------------------------------------
# initialize to the first file in the list


#image_elem = sg.Image(data=get_img_data('defaultimg.png',maxsize=(1440,1024),first=True),key='-tab1-image-')
graph_elem = sg.Graph(
            canvas_size=(1440,1024),
            graph_bottom_left=(0, 0),
            graph_top_right=(1440, 1024),
            key="graph",
            change_submits=True, 
            drag_submits=False
        )
#filename_display_elem = sg.Text(filename, size=(80, 3))
#file_num_display_elem = sg.Text('File 1 of {}'.format(num_files), size=(15, 1))
n_points_text_elem=sg.Text('Press save to store coords in textfile', size=(25,6))
# define layout, show and read the form
#col = [[filename_display_elem],
#       [graph_elem]]
col = [[graph_elem]]

col_files = [[sg.Button('browse',key='-folder-')],
             [sg.Listbox([['pick a folder first']],key='listbox', change_submits=True, size=(60, 30))],
             [sg.Button('Prev', size=(8, 2)), sg.Button('Next', size=(8, 2))], #file_num_display_elem],
             [sg.Button('Reset', key='-RESET-'),sg.Button('Save', key='-SAVE-')],[n_points_text_elem],
             [sg.Button('Exit',key='-EXIT1-')],
             ]
########### pointpicker tab ###########################
tab_layout1=[[sg.Column(col_files), sg.Column(col)]]
########### live feed tab #############################
tab_2_col1=[[sg.Button('Start feed',key=('-LivefeedButton-')),sg.Button('Stop feed', key=('-Stopfeed-')),sg.Button('Exit',key='-EXIT2-')],
             [sg.Image(filename='', key='image',size=(800,500))]]

tab_2_col2=[[sg.Text('Current pressure value:',size=(30,1)), sg.Text(size=(30,1), key='-OUTPUTpressure-')],
            [sg.Input('Manual  pressure input mbar',key='-inputpressure-'),sg.Button('Update',key='-pbutton-')],
            [sg.Text('Specify capillary dimensions')],
            [sg.Text('Taper length:')],
            [sg.InputText(do_not_clear=True,key='-Taper_Length-')],
            [sg.Text('Tip diameter:')],
            [sg.InputText(do_not_clear=True,key='-Tip_Diameter-')],
            [sg.Text('Inside diameter')],
            [sg.InputText(do_not_clear=True,key='-Inside_Diameter-')],
            [sg.Button('Update params',key='-Update_Params-')],
            [sg.Text('Measurement Parameters')],
            [sg.Text('Max pressure value(mbar):')],
            [sg.Input(key='-maxP-')],
            [sg.Text('Step size')],
            [sg.Input(key='stepsize')],
            [sg.Button('Start Measurement',key='-Startm-')]
            ]
tab_layout2=[[sg.Column(tab_2_col1),sg.Column(tab_2_col2)]]
########### Plots tab #################################
# =============================================================================
col1=[      [sg.Text('Figure 1')],
            [sg.Canvas(key='-CANVAS1-',size=(200 * 2, 200))],
            [sg.Canvas(key='controls_cv1')],
            [sg.Text('Figure 3')],
            [sg.Canvas(key='-CANVAS3-',size=(200 * 2, 200))],
            [sg.Canvas(key='controls_cv3')],
        ]
col2=[      [sg.Text('Figure 2')],
            [sg.Canvas(key='-CANVAS2-',size=(200 * 2, 200))],
            [sg.Canvas(key='controls_cv2')],
            [sg.Text('Figure 4')],
            [sg.Canvas(key='-CANVAS4-',size=(200 * 2, 200))],
            [sg.Canvas(key='controls_cv4')],
        ]
tab3_layout=[[sg.Button('Ok',key='-Graphbutton-'),sg.Button('Exit',key='-EXIT3-')],
    [sg.Column(col1),sg.Column(col2)]]
# =============================================================================
# general layout of the entire applet
layout=[[
    sg.TabGroup(
        [[sg.Tab('Point picker',tab_layout1),sg.Tab('Live feed', tab_layout2),sg.Tab('Plots',tab3_layout)]])
    ]]
window = sg.Window('Image Browser', layout, return_keyboard_events=True,
                   location=(0, 0), use_default_focus=False).Finalize()
window.Maximize() 

# loop reading the user input and displaying image, filename
i = 0
cap=cv2.VideoCapture("videos/P_steps_100_Trim.mp4") #initiate videocapture
while True: #Main event loop
    event, values = window.read()     #Read values like button presses etc from the GUI
    mouse=values['graph']   #Get the mouse coords
    print(event, values)    #Prints the interactions in the console
###########################################################################
########## Point picker tab ##########################
    if event == '-folder-':
        # Get the folder containin:g the images from the user
        folder = sg.popup_get_folder('Image folder to open', default_path='')   #ask to select a folder
        if not folder:
            sg.popup_cancel('Cancelling')
            raise SystemExit()
        n_points= sg.popup_get_text('Enter the number of points you want to select per image',
                                    'Number of points')
        while error==True:  #make sure the input is an integer
            try:
                n_points=int(n_points)
                error=False
            except:
                n_points=sg.popup_get_text('Must be an integer','Number of points')
                continue
        # PIL supported image types
        # get list of files in folder
        
        flist0 = os.listdir(folder)

        # create sub list of image files (no sub folders, no wrong file types)
        fnames = [f for f in flist0 if os.path.isfile(
        os.path.join(folder, f)) and f.lower().endswith(img_types)]

        num_files = len(fnames)                # number of iamges found
        if num_files == 0:
            sg.popup('No files in folder')
            raise SystemExit()
        filename = os.path.join(folder, fnames[0])  # name of first file in list

        del flist0                             # no longer needed
        window['listbox'].update(values=fnames)
        #window['-tab1-image-'].update(data=get_img_data(filename, first=False))
        window.refresh()
        
###########################################################################
############## VIDEO STUFF (Tab 2) #########################
# Update image in window
    if event == '-LivefeedButton-': #Start button
        counter=0
        pvalue=10
        while True: #Event loop to keep refreshing frames
            event, values = window.read(timeout=20)
            ret, frame = cap.read()
            if not ret:     #If a frame is not collected, move to start of loop
                break
            else:           #If a fram is succesfully read, apply all opencv functions to it
                print(ret)
                ## add text to image
                font                   = cv2.FONT_HERSHEY_SIMPLEX
                bottomLeftCornerOfText = (10,600)
                fontScale              = 1
                fontColor              = (255,255,255)
                lineType               = 2
                #get video properties, 535 frames   
                # =============================================================================
                # property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
                # length = int(cv2.VideoCapture.get(cap, property_id))
                # print( length )
                # =============================================================================
                nmilisec=1000/2 # time interval between detection in ms
                cap.set(cv2.CAP_PROP_POS_MSEC,(counter*nmilisec)) #only select the frames at nmilisec intervals
                #print('Read a new frame:%d '% count, ret)
                h,w,layers=frame.shape #get height and width
                timepassed=counter*nmilisec/1000
                frame=frame[:h,:w-400] #select only the capillary part
                gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # =============================================================================
                #     edges = cv2.Canny(gray,90,180,apertureSize = 3)
                # =============================================================================
                output=gray.copy()
                circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,8, param1=100,param2=50,minRadius=80,maxRadius=160) 
                # ensure at least some circles were found
                if (circles is not None): 
                    # convert the (x, y) coordinates and radius of the circles to integers
                    circles = np.round(circles[0, :]).astype("int")
                    print("Circle detected after:%.1f seconds" % timepassed)
                    # loop over the (x, y) coordinates and radius of the circles
                    for (x, y, r) in circles:
                         #draw the circle in the output image, then draw a rectangle
		                 # corresponding to the center of the circle
                         cv2.circle(output, (x, y), r, (255, 0, 255), 2)
                         cv2.rectangle(output, (x - 2, y - 2), (x + 2, y + 2), (0, 128, 255), -1)
                         
                         #cv2.imwrite('videos/test/frameanalyze.jpg', frame)
                         #stop after first circle is detected
                         
                    cv2.putText(output,'Circle detected after: {} seconds'.format(timepassed),
                                     bottomLeftCornerOfText,
                                     font, 
                                     fontScale,
                                     fontColor,
                                     lineType)
                    cv2.imwrite('videos/test/frame_%d_%s.jpg' % (counter, pvalue), output) #save the frames with the circles drawn        
                    counter+=1      #Count the frames displayed to deduce the time passed
                    imgbytes = cv2.imencode('.png', output)[1].tobytes()  # Change the matrix image to .png image such that PySimpleGUI can read it
                    window['image'].update(data=imgbytes)   #Update the image
                    window['-OUTPUTpressure-'].update(pvalue)
                    break
                
                cv2.imwrite('videos/test/frame_%d_%s.jpg' % (counter, pvalue), output) #save the frames with the circles drawn         
                counter+=1      #Count the frames displayed to deduce the time passed
                pvalue+=2
                imgbytes = cv2.imencode('.png', output)[1].tobytes()  # Change the matrix image to .png image such that PySimpleGUI can read it
                window['image'].update(data=imgbytes)   #Update the image
                window['-OUTPUTpressure-'].update(pvalue)
                if event == '-pbutton-':
                    try:
                        pvalue=int(values['-inputpressure-'])
                        window['-OUTPUTpressure-'].update(pvalue)
                    except: 
                        sg.popup('not an integer')
                        continue
                if event in [sg.WIN_CLOSED,'-Stopfeed-']:   #Pause the video or live feed
                    break
    if event == '-Startm-':
        try:
            psteps=int(values['stepsize'])
            pmax=int(values['-maxP-'])
        except:
            sg.popup('not an integer')
            continue
        flist0 = os.listdir(folder)
        fnames = [f for f in flist0 if os.path.isfile(
            os.path.join(folder, f)) and f.lower().endswith(img_types)]
# =============================================================================
#         for i in range(len(fnames)):
#              fnames[i] = fnames[i].rsplit('.',1)[0]
#              print(fnames)
# =============================================================================
 
        fnames.sort(key = lambda x: int(x.rsplit('_',2)[1]))
        pvalue=50
        for i in list(range(1,round(pmax/psteps)-1)):
            pvalue=pvalue+psteps    #current pressuer value
            window['-OUTPUTpressure-'].update(pvalue)   #update the pressure displayed
            time.sleep(6)   #time delay
            filenames=frame.copy()
            cv2.imwrite('videos/test/frame_%d_%s.jpg' % (k, pvalue), filenames)
            testimg=Image.open(filenames)
            testimg.thumbnail((800,500))    #resize the image
            bio=io.BytesIO()
            testimg.save(bio,format='PNG')  #save as bytestring          
            window['image'].update(data=bio.getvalue()) #update the image
            window.refresh()    #refresh the window
            k=k+1   #frame counter
           
            
                
############# END OF VIDEO STUFF
##########################################################################
# perform button and keyboard operations in tab 1
    if event in [None, '-EXIT1-','-EXIT2-','-EXIT3-']:  # always,  always give a way out!
        plt.close('all')
        window.Close()
        break
    if event == sg.WIN_CLOSED:
        break
# =============================================================================
#     elif event in ('Next', 'MouseWheel:Down', 'Down:40', 'Next:34'):
#         i += 1
#         if i >= num_files:
#             i -= num_files
#         filename = os.path.join(folder, fnames[i])
#         count=0
#         pos=np.zeros((n_points,2),np.int) #reset position matrix when changing images
#         graph_elem.draw_image(data=get_img_data(filename,first=True),location=(50,900))
#         n_points_text_elem.update('Press save to store coords in textfile')
#     elif event in ('Prev', 'MouseWheel:Up', 'Up:38', 'Prior:33'):
#         i -= 1
#         if i < 0:
#             i = num_files + i
#         filename = os.path.join(folder, fnames[i])
#         count=0
#         pos=np.zeros((n_points,2),np.int) #reset position matrix when changing images
#         graph_elem.draw_image(data=get_img_data(filename,first=True),location=(50,900))
#         n_points_text_elem.update('Press save to store coords in textfile')
    elif event == 'listbox':            # something from the listbox
        f = values["listbox"][0]            # selected filename
        filename = os.path.join(folder, f)  # read this file
        i = fnames.index(f)                 # update running index
        count=0
        pos=np.zeros((n_points,2),np.int)   #reset position matrix when changing images
        graph_elem.draw_image(data=get_img_data(filename,first=True),location=(0,1024))
        n_points_text_elem.update('Press save to store coords in textfile')
#     else:
#         filename = os.path.join(folder, fnames[i])
#         #graph_elem.draw_image(data=get_img_data(filename,first=True),location=(50,900))
# =============================================================================
    if event == 'graph' and count<n_points:
        if mouse == (None,None) :
            continue
        xcoord=mouse[0]
        ycoord=mouse[1]
        pos[count]= [xcoord,ycoord]
        count+=1
        graph_elem.DrawCircle((xcoord,ycoord),5,fill_color='green',line_color='green')
        window.refresh()
    if event == '-RESET-':
        graph_elem.draw_image(data=get_img_data(filename,first=True),location=(50,900))
        count=0
        pos=np.zeros((7,2),np.int)
    if event == '-SAVE-' and count==n_points:
        if Firstsave==True:
            file = open("txtfile/test4.txt" ,"w") # "w" will overwrite the text file, "a" will append to the text file
            pos=np.insert(pos,0,np.arange(n_points),axis=1)   #insert the slice and number columns in pos
            pos=np.insert(pos,3,np.ones(n_points),axis=1)
            p=filename.rsplit('.',1)[0]
            p=int(p.rsplit('_',1)[1])
            pos=np.insert(pos,4,p*np.ones(n_points),axis=1)
            #newpos=np.insert(pos,0,np.arange(7)+1+number)
            np.savetxt(file,pos.astype(int), fmt='%i',header='Point x y Slice Pressure', comments="")
            file.close()    
            Firstsave=False
            number+=1   
            print(pos)
        else:
            file = open("txtfile/test4.txt" ,"a") # "w" will overwrite the text file, "a" will append to the text file
            pos=np.insert(pos,0,np.arange(n_points)+number*n_points,axis=1)   #insert the slice and number columns in pos
            pos=np.insert(pos,3,np.ones(n_points)+number,axis=1)
            p=filename.rsplit('.',1)[0]
            p=int(p.rsplit('_',1)[1])
            pos=np.insert(pos,4,p*np.ones(n_points),axis=1)
            np.savetxt(file,pos.astype(int), fmt='%i')
            file.close()    
            number+=1
            print(pos)
        n_points_text_elem.update('Points saved succesfully. Go to the next image.')
    elif event== '-SAVE-' and count !=n_points:
        n_points_text_elem.update('Select {} points before you save'.format(n_points))
    if event == '-Update_Params-':
        try:
            L_taper=float(values['-Taper_Length-'])
            D_cap=float(values['-Inside_Diameter-'])
            D_tip=float(values['-Tip_Diameter-'])
            #print(D_tip,D_cap,L_taper)
        except:
            #print(':/')
            sg.popup('Please enter a number')
########################################################################################
######### ALL THE PLOTS #################
    if event == '-Graphbutton-':    #OK button in plotting tab
        data = pd.read_csv("Useful_1.txt",header=0,delim_whitespace=True)
        data = pd.read_csv("txtfile/test2.txt",header=0,delim_whitespace=True)
        print(data)
        print(data.shape)
        
        data_arr = data.to_numpy()
        #print(data_arr)
        data_arr[7*3+1,1]
        # Calculation of L_band, R_band and Volume and p_wall
        n_rows = np.size(data_arr,0)
        size_L_band = n_rows//7 #7 points on each image so '//' used to remove floating point result 
        print(size_L_band)
        size_R_band = n_rows//7
        #print(size_L_band)
        # L_band is the upper L_band and L_band_1 is the lower L_band
        
        argument_alpha = ((D_cap-D_tip)/2)/L_taper # Argument for the arctan function. Half taper angle
        alpha = np.arctan(argument_alpha) 
        #print(alpha)
        L_band = np.empty(size_L_band)
        L_shape = np.empty(size_L_band)
        L_band_1 = np.empty(size_L_band) # L_band is the upper L_band and L_band_1 is the lower L_band
        R_upper = np.empty([size_R_band,2])
        R_bottom = np.empty([size_R_band,2])
        R_band = np.empty(size_R_band)
        a_left_mid = np.empty([size_L_band,2])
        a_right_mid = np.empty([size_L_band,2])
        a_left = np.empty(size_L_band)
        a_right = np.empty(size_L_band)
        h_left = np.empty(size_L_band)
        h_right = np.empty(size_L_band)
        h_mid = np.empty(size_L_band)
        V_cap_left = np.empty(size_L_band)
        V_cap_right = np.empty(size_L_band)
        V_frustum = np.empty(size_L_band)
        #print(L_band) 
        #print(R_upper)
        #print(R_bottom)
        #data_arr[size_L_band*2+6,1]
        #Pressure = np.array([100,150,200,250,300,350,400]) #Pressure in mbar for Useful_1.txt
        mylist = os.listdir('videos/test/24-06-2/')
        
        Pressure=np.zeros(int(len(data_arr)/7))
        for i in range(int((len(data))/7)):
            Pressure[i]=data_arr[1+i*7,-1]
        print(Pressure)
       
        # =============================================================================
#         for i in range(len(mylist)):
#             mylist[i] = mylist[i].rsplit('.',1)[0]
#             print(mylist)
# 
#         mylist.sort(key = lambda x: int(x.rsplit('_',1)[-1]))
#         for i in range(len(mylist)):
#             Pressure[i]=int(mylist[i].rsplit('_',1)[-1])
#         Pressure=Pressure[0:19]
# =============================================================================
        #Pressure=np.array([50,100,150,200,250,300,350,400])
        #P_pa = Pressure*0.1
        #Pressure = np.array([200,250,300,350,400,450,500,550,600,650,700,750,800,850,900,950,1000,1050,1100,1150]) #Pressure in mbar for 06_04_Run_1.txt
        P_pa = Pressure*100 
        #Pressure = np.array([0,1,2,3,4,5,10,20,30,40,50,60,70]) #Pressure in mbar
        #P_pa = Pressure*98.06
        
        # Every 7th point (number 6 in first column) is the last point for that pressure_data 
        for n in range(size_L_band):
            # L_band[n] = ((data_arr[size_L_band*n+6,1]-data_arr[size_L_band*n+5,1])**2+(data_arr[size_L_band*n+6,2]-data_arr[size_L_band*n+5,2])**2)**0.5
            L_band[n] = ((data_arr[7*n+2,1]-data_arr[7*n+1,1])**2+(data_arr[7*n+2,2]-data_arr[7*n+1,2])**2)**0.5 # L_band is the upper L_band and L_band_1 is the lower L_band
            L_band_1[n] = ((data_arr[7*n+4,1]-data_arr[7*n+3,1])**2+(data_arr[7*n+4,2]-data_arr[7*n+3,2])**2)**0.5 # L_band is the upper L_band and L_band_1 is the lower L_band
            R_upper[n,0] = (data_arr[7*n+1,1]+data_arr[7*n+2,1])/2
            R_upper[n,1] = (data_arr[7*n+1,2]+data_arr[7*n+2,2])/2
            R_bottom[n,0] = (data_arr[7*n+3,1]+data_arr[7*n+4,1])/2
            R_bottom[n,1] = (data_arr[7*n+3,2]+data_arr[7*n+4,2])/2
            R_band[n] = ((R_upper[n,0]-R_bottom[n,0])**2+(R_upper[n,1]-R_bottom[n,1])**2)**0.5
            # L_shape for the calculation of strain, shape change in the z direction
            L_shape[n] = ((data_arr[7*n+6,1]-data_arr[7*n+5,1])**2+(data_arr[7*n+6,2]-data_arr[7*n+5,2])**2)**0.5
            #Volume calculations
            #Volume calculation for spherical cap
            a_left_mid[n,0] = (data_arr[7*n+1,1]+data_arr[7*n+3,1])/2
            a_left_mid[n,1] = (data_arr[7*n+1,2]+data_arr[7*n+3,2])/2
            a_left[n] =  ((data_arr[7*n+1,1]-a_left_mid[n,0])**2+(data_arr[7*n+1,2]-a_left_mid[n,1])**2)**0.5
            h_left[n] =   ((data_arr[7*n+5,1]-a_left_mid[n,0])**2+(data_arr[7*n+5,2]-a_left_mid[n,1])**2)**0.5
            a_left[n] =  ((data_arr[7*n+1,1]-a_left_mid[n,0])**2+(data_arr[7*n+1,2]-a_left_mid[n,1])**2)**0.5 
            a_right_mid[n,0] = (data_arr[7*n+2,1]+data_arr[7*n+4,1])/2
            a_right_mid[n,1] = (data_arr[7*n+2,2]+data_arr[7*n+4,2])/2 
            a_right[n] =  ((data_arr[7*n+2,1]-a_right_mid[n,0])**2+(data_arr[7*n+2,2]-a_right_mid[n,1])**2)**0.5
            h_right[n] =   ((data_arr[7*n+6,1]-a_right_mid[n,0])**2+(data_arr[7*n+6,2]-a_right_mid[n,1])**2)**0.5
            V_cap_left[n] = (1/6)*np.pi*h_left[n]*(3*a_left[n]**2+h_left[n]**2)
            V_cap_right[n] = (1/6)*np.pi*h_right[n]*(3*a_right[n]**2+h_right[n]**2)
            # Volume calculation for conical frustum 
            h_mid[n] = ((a_left_mid[n,0]-a_right_mid[n,0])**2+(a_left_mid[n,1]-a_right_mid[n,1])**2)**0.5
            V_frustum[n] = (np.pi*h_mid[n]/3)*(a_right[n]**2+a_left[n]**2+a_right[n]*a_left[n])
        #print(L_band*1.2)
        #print(L_band_1*1.2)
        #print(R_band*1.2)
        V_total = V_cap_left+V_cap_right+V_frustum
        print(V_total*1.2**3)
        p_wall = np.empty(size_L_band)
        D_particle = 2*R_band[0]
        L_particle = L_shape[0] 
        #V_original = (np.pi*D_particle**3)/6
        V_original = V_cap_left[0]+V_cap_right[0]+V_frustum[0]
        Delta_V = V_original-V_total 
        # Calculation for p_wall
        R_by_L = np.divide(R_band,L_band)
        R_by_L_p = np.multiply(R_by_L,P_pa)
        #print(np.sin(alpha))
        p_wall = R_by_L_p/(2*np.sin(alpha))
        #print(2*R_band*1.2)
        #print(L_shape*1.2)
        fig1=plt.figure(0)
        #plt.subplot(221)
        plt.plot(P_pa,L_band,'go--',label='Upper $L_{band}$')
        plt.plot(P_pa,L_band_1,'bo--',label='lower $L_{band}$')
        plt.plot(P_pa,R_band,'ro--',label='$R_{band}$')
        plt.plot(P_pa,L_shape,'ko--',label="$L_{shape}$")
        plt.title('Lband and Rband as function of pressure')
        plt.xlabel("Pressure in Pa")
        plt.legend()
        [m,b] = np.polyfit(P_pa,V_total,1)
        fig2=plt.figure(1)
        #plt.subplot(222)
        plt.plot(P_pa,V_total,'ko--',label='Volume')
        plt.plot(P_pa,m*P_pa+b)
        plt.title('Volume as a function of pressure')
        plt.xlabel("Pressure in Pa")
        plt.legend()
######################################################################        
        #print(P_pa)
        epsilon_r = (D_particle-2*R_band)/D_particle
        print(epsilon_r)
        epsilon_z = (L_particle-L_shape)/D_particle
        print(epsilon_z)
        #print(epsilon_r-epsilon_z)
        #print(p_wall-P_pa)
        [m,b] = np.polyfit((epsilon_r-epsilon_z),((p_wall-P_pa)/2),1)
        fig3=plt.figure(2)
        #plt.subplot(223)
        plt.plot((epsilon_r-epsilon_z),((p_wall-P_pa)/2),'o')
        x = epsilon_r-epsilon_z
        plt.plot(x,m*x+b)
        plt.title('Shear modulus- $\sigma_{shear}$ as a function of $\epsilon_{shear}$')
        plt.xlabel('$\epsilon_r$-$\epsilon_z$')
        plt.ylabel('$1/2\cdot (P_{wall}-P$)')
        print(m)

        [m1,b1] = np.polyfit((Delta_V/V_original),(1/3)*(2*p_wall+P_pa),1)
        fig4=plt.figure(3)
        #plt.subplot(224)
        plt.plot((Delta_V/V_original),((1/3)*(2*p_wall+P_pa)),'o')
        x1 = (Delta_V/V_original)
        plt.plot(x1,m1*x1+b1)
        plt.title('Compression modulus- $\sigma_{compr}$ as a function of Volumetric strain')
        plt.xlabel('$\Delta_V/V_{original}$')
        plt.ylabel('$2/3\cdot P_{wall}+1/3\cdot P$')
        print(m1)
        E=9*m*m1/(3*m1+m)
        print(E)
        #print(R_upper)
        #print(R_bottom)
        #L_band[n] = ((data_arr[6,1]**2-data_arr[5,1]**2)+(data_arr[6,2]**2-data_arr[5,2]**2))**0.5
        #plt.plot(P_pa,L_band*1.2,P_pa,L_band_1*1.2)  
        #plt.plot(P_pa,R_band*1.2)
##################### END OF ALL THE PLOTS ############################
#######################################################################

        # add the plot to the window without toolbar
# =============================================================================
#         fig_canvas_agg = draw_figure(window['-CANVAS1-'].TKCanvas, fig1)
#         fig_canvas_agg = draw_figure(window['-CANVAS2-'].TKCanvas, fig2)
#         fig_canvas_agg = draw_figure(window['-CANVAS3-'].TKCanvas, fig3)
#         fig_canvas_agg = draw_figure(window['-CANVAS4-'].TKCanvas, fig4)
# =============================================================================
###########################################################################
        # add plots with toolbars
        draw_figure_w_toolbar(window['-CANVAS1-'].TKCanvas, fig1, window['controls_cv1'].TKCanvas)
        draw_figure_w_toolbar(window['-CANVAS2-'].TKCanvas, fig2, window['controls_cv2'].TKCanvas)
        draw_figure_w_toolbar(window['-CANVAS3-'].TKCanvas, fig3, window['controls_cv3'].TKCanvas)
        draw_figure_w_toolbar(window['-CANVAS4-'].TKCanvas, fig4, window['controls_cv4'].TKCanvas)
        window.Maximize()
##################################################################################################
    # update window with new image
    #image_elem.update(data=get_img_data(filename, first=True))
    #graph_elem.draw_image(data=get_img_data(filename,first=True),location=(50,900))     #update the image in the first tab
    window.refresh()
    # update window with filename
    #filename_display_elem.update(filename)
    # update page display
    #file_num_display_elem.update('File {} of {}'.format(i+1, num_files))

window.close()  # close everything if main loop is broken
plt.close('all')