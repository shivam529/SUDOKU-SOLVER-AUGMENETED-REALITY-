# Load opencv module
import cv2
from PIL import Image
import numpy as np
import math
import sys
from scipy import misc
from scipy import fftpack,ndimage
import cv2
import solver as sl
import imutils
import math
import project2 as dis


def run_all():
    


    def sort_grid_points(points):
        """
        Given a flat list of points (x, y), this function returns the list of
        points sorted from top to bottom, then groupwise from left to right.
        We assume that the points are nearly equidistant and have the form of a
        square.
        """
        w, _ = points.shape
        sqrt_w = int(np.sqrt(w))
        # sort by y
        points = points[np.argsort(points[:, 1])]
        # put the points in groups (rows)
        points = np.reshape(points, (sqrt_w, sqrt_w, 2))
        # sort rows by x
        points = np.vstack([row[np.argsort(row[:, 0])] for row in points])
        # undo shape transformation
        points = np.reshape(points, (w, 1, 2))
        return points

    def gett(temp_image,f):
        gray = cv2.cvtColor(temp_image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
        blurred = cv2.medianBlur(binary, ksize=3)

        print("en")
        #
        # 2. try to find the sudoku
        #
        contours, _ = cv2.findContours(image=cv2.bitwise_not(blurred),
                                       mode=cv2.RETR_LIST,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
        sudoku_area = 0
        sudoku_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            if (0.7 < float(w) / h < 1.3     # aspect ratio
                    and area > 150 * 150     # minimal area
                    and area > sudoku_area   # biggest area on screen
                    and area > .5 * w * h):  # fills bounding rect
                sudoku_area = area
                sudoku_contour = cnt

        #
        # 3. separate sudoku from background
        #
        if sudoku_contour is not None:

            # approximate the contour with connected lines
            perimeter = cv2.arcLength(curve=sudoku_contour, closed=True)
            approx = cv2.approxPolyDP(curve=sudoku_contour,
                                      epsilon=0.1 * perimeter,
                                      closed=True)

            if len(approx) == 4:
                # successfully approximated
                # we now transform the sudoku to a fixed size 450x450
                # plus 50 pixel border and remove the background

                # create empty mask image
                mask = np.zeros((gray.shape), np.uint8)
                # fill a the sudoku-contour with white
                cv2.drawContours(mask, [sudoku_contour], 0, 255, -1)
                # invert the mask
                mask_inv = cv2.bitwise_not(mask)
                # the blurred picture is already thresholded so this step shows
                # only the black areas in the sudoku
                separated = cv2.bitwise_or(mask_inv, blurred)
                # if args.debug:
                # cv2.imshow('separated', separated)
                # cv2.imwrite('separated.png',separated)
                print("#####",gray.shape,separated.shape)

              ##WARPING##
                square = np.float32([[0, 0], [450, 0], [0, 450], [450, 450]])

                approx = np.float32([i[0] for i in approx]) 
                # sort the approx points to match the points defined in square
                approx = sort_grid_points(approx)

                m = cv2.getPerspectiveTransform(approx, square)
                m_inv=cv2.getPerspectiveTransform(square,approx)
                if(f==1):
                    return separated,m_inv
                transformed_orginal_color=cv2.warpPerspective(temp_image, m, (450, 450))
                transformed = cv2.warpPerspective(separated, m, (450, 450))
                cv2.imwrite("y.png",transformed)

                transformed=transformed[4:446,4:446]
                transformed=cv2.resize(transformed,(306,324))
                transformed=cv2.copyMakeBorder(transformed,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
                transformed=cv2.copyMakeBorder(transformed,10,10,10,10,cv2.BORDER_CONSTANT,value=255)
                kernel = np.ones((3,3),np.uint8)
                transformed= cv2.morphologyEx(transformed, cv2.MORPH_OPEN, kernel)
                # transformed = cv2.adaptiveThreshold(transformed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 3)
                transformed = cv2.GaussianBlur(transformed,(3,3),0)
                ret3,transformed = cv2.threshold(transformed,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                transformed1 = cv2.adaptiveThreshold(transformed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)
                ## creating a rectange again to be added after removing grid lines
                lines_img=np.zeros(transformed.shape).astype(np.uint8)
                rectangle_extracted=np.full(transformed1.shape,255).astype(np.uint8)
                cont, _ = cv2.findContours(image=transformed1,
                                       mode=cv2.RETR_LIST,
                                       method=cv2.CHAIN_APPROX_SIMPLE)
                area=[cv2.contourArea(item) for item in cont]
                max_index=area.index(max(area))
                cnt = cont[max_index]
                x,y,w,h = cv2.boundingRect(cnt)
                cv2.rectangle(rectangle_extracted,(x,y),(x+w,y+h),0,1)


                edges = cv2.Canny(transformed,60,180,apertureSize = 3)
                lines = cv2.HoughLinesP(edges,rho=1,theta=1*np.pi/180,threshold=60,minLineLength=30,maxLineGap=100)
                for line in lines:
                    for x1,y1,x2,y2 in line:
                        cv2.line(lines_img,(x1,y1),(x2,y2),255,4)
                transformed[np.where(lines_img==255)]=255
                transformed[np.where(rectangle_extracted==0)]=0
                # cv2.imshow("lines_img",lines_img)
                # cv2.imshow('transformed', transformed)
                # cv2.imshow('rectangle',rectangle_extracted)
                # cv2.imshow("original",transformed_orginal_color)
                
                
                answer_retreived=dis.print_answer(transformed,transformed_orginal_color)
                print(answer_retreived)
                if(isinstance(answer_retreived, int)):
                    return 0,0
                else:
                    return answer_retreived,m_inv
                

             
            # cv2.imshow("Input",temp_image)


           

    got=None
    flag=0
    found=0
    last=None
    # Create a VideoCapture object
    cap = cv2.VideoCapture(0)
    # total_frames=cap.get(7)
    # cap.set(1,1000)

    # Check if camera opened successfully
    if (cap.isOpened() is False):
        print("Unable to read camera feed")

    # Default resolutions of the frame are obtained.
    # The default resolutions
    # are system dependent.
    # We convert the resolutions from float to integer.
    ctr=0
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    # out = cv2.VideoWriter('output1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
    temp_image=None
    while(True):
        ret, frame = cap.read()
        if ret:
            ctr+=1
            print(ctr)
            # cv2.imshow('frame', frame)
            if(found==0):
                if(got is None and ctr%100==0):
                    a,b=gett(frame,found)
                    if(isinstance(a, int)):
                        pass
                    else:
                        flag=1
                if(flag==0):
                    cv2.imshow('frame',frame)
                    # out.write(frame)
                else:
                    got=a
                if(got is not None):
                    final=cv2.warpPerspective(got, b, (frame.shape[1], frame.shape[0]))
                    rest=np.where(final==(0,0,0))
                    final[rest]=frame[rest]
                    found=1
                    cv2.imshow('frame',final)
                    # out.write(final)
            if(found==1):
                try:
                    a,b=gett(frame,found)
                    final=cv2.warpPerspective(got, b, (frame.shape[1], frame.shape[0]))
                    rest=np.where(final==(0,0,0))
                    final[rest]=frame[rest]
                    cv2.imshow('frame',final)
                    # out.write(final)
                    last=final
                    cv2.waitKey(200)
                    if 0xFF == ord('q'):
                        break
                except:
                    cv2.imshow('frame',last)
                    # out.write(last)
                    if 0xFF == ord('q'):
                        break





            # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  
        else:
            break


    cap.release()

    cv2.destroyAllWindows()

run_all()