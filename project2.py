from PIL import Image
import numpy as np
import math
import sys
from scipy import misc
from scipy import fftpack,ndimage
import cv2
import solver as sl
from copy import deepcopy
##############################################################TRAIN IMAGES##############################################################
one=cv2.imread("train/sudo_one.png",0)
two=cv2.imread("train/sudo_two.png",0)
three=cv2.imread("train/sudo_three.png",0)
four=cv2.imread("train/sudo_four.png",0)
five=cv2.imread("train/sudo_five.png",0)
six=cv2.imread("train/sudo_six.png",0)
seven=cv2.imread("train/sudo_seven.png",0)
eight=cv2.imread("train/sudo_eight.png",0)
nine=cv2.imread("train/sudo_nine.png",0)
train_images=[one,two,three,four,five,six,seven,eight,nine]
one2=cv2.imread("train/sudo2_one.png",0)
two2=cv2.imread("train/sudo2_two.png",0)
three2=cv2.imread("train/sudo2_three.png",0)
four2=cv2.imread("train/sudo2_four.png",0)
five2=cv2.imread("train/sudo2_five.png",0)
six2=cv2.imread("train/sudo2_six.png",0)
seven2=cv2.imread("train/sudo2_seven.png",0)
eight2=cv2.imread("train/sudo2_eight.png",0)
nine2=cv2.imread("train/sudo2_nine.png",0)
train_images2=[one2,two2,three2,four2,five2,six2,seven2,eight2,nine2]
one3=cv2.imread("train/sudo3_one.png",0)
two3=cv2.imread("train/sudo3_two.png",0)
three3=cv2.imread("train/sudo3_three.png",0)
four3=cv2.imread("train/sudo3_four.png",0)
five3=cv2.imread("train/sudo3_five.png",0)
six3=cv2.imread("train/sudo3_six.png",0)
seven3=cv2.imread("train/sudo3_seven.png",0)
eight3=cv2.imread("train/sudo3_eight.png",0)
nine3=cv2.imread("train/sudo3_nine.png",0)
train_images3=[one3,two3,three3,four3,five3,six3,seven3,eight3,nine3]
one4=cv2.imread("train/sudo4_one.png",0)
two4=cv2.imread("train/sudo4_two.png",0)
three4=cv2.imread("train/sudo4_three.png",0)
four4=cv2.imread("train/sudo4_four.png",0)
five4=cv2.imread("train/sudo4_five.png",0)
six4=cv2.imread("train/sudo4_six.png",0)
seven4=cv2.imread("train/sudo4_seven.png",0)
eight4=cv2.imread("train/sudo4_eight.png",0)
nine4=cv2.imread("train/sudo4_nine.png",0)
train_images4=[one4,two4,three4,four4,five4,six4,seven4,eight4,nine4]
one5=cv2.imread("train/sudo5_one.png",0)
two5=cv2.imread("train/sudo5_two.png",0)
three5=cv2.imread("train/sudo5_three.png",0)
four5=cv2.imread("train/sudo5_four.png",0)
five5=cv2.imread("train/sudo5_five.png",0)
six5=cv2.imread("train/sudo5_six.png",0)
seven5=cv2.imread("train/sudo5_seven.png",0)
eight5=cv2.imread("train/sudo5_eight.png",0)
nine5=cv2.imread("train/sudo5_nine.png",0)
train_images5=[one5,two5,three5,four5,five5,six5,seven5,eight5,nine5]
one6=cv2.imread("train/sudo6_one.png",0)
two6=cv2.imread("train/sudo6_two.png",0)
three6=cv2.imread("train/sudo6_three.png",0)
four6=cv2.imread("train/sudo6_four.png",0)
five6=cv2.imread("train/sudo6_five.png",0)
six6=cv2.imread("train/sudo6_six.png",0)
seven6=cv2.imread("train/sudo6_seven.png",0)
eight6=cv2.imread("train/sudo6_eight.png",0)
nine6=cv2.imread("train/sudo6_nine.png",0)
train_images6=[one6,two6,three6,four6,five6,six6,seven6,eight6,nine6]
one7=cv2.imread("train/sudo7_one.png",0)
two7=cv2.imread("train/sudo7_two.png",0)
three7=cv2.imread("train/sudo7_three.png",0)
four7=cv2.imread("train/sudo7_four.png",0)
five7=cv2.imread("train/sudo7_five.png",0)
six7=cv2.imread("train/sudo7_six.png",0)
seven7=cv2.imread("train/sudo7_seven.png",0)
eight7=cv2.imread("train/sudo7_eight.png",0)
nine7=cv2.imread("train/sudo7_nine.png",0)
train_images7=[one7,two7,three7,four7,five7,six7,seven7,eight7,nine7]
one8=cv2.imread("train/sudo8_one.png",0)
two8=cv2.imread("train/sudo8_two.png",0)
three8=cv2.imread("train/sudo8_three.png",0)
four8=cv2.imread("train/sudo8_four.png",0)
five8=cv2.imread("train/sudo8_five.png",0)
six8=cv2.imread("train/sudo8_six.png",0)
seven8=cv2.imread("train/sudo8_seven.png",0)
eight8=cv2.imread("train/sudo8_eight.png",0)
nine8=cv2.imread("train/sudo8_nine.png",0)
train_images8=[one8,two8,three8,four8,five8,six8,seven8,eight8,nine8]
one9=cv2.imread("train/sudo9_one.png",0)
two9=cv2.imread("train/sudo9_two.png",0)
three9=cv2.imread("train/sudo9_three.png",0)
four9=cv2.imread("train/sudo9_four.png",0)
five9=cv2.imread("train/sudo9_five.png",0)
six9=cv2.imread("train/sudo9_six.png",0)
seven9=cv2.imread("train/train/sudo9_seven.png",0)
eight9=cv2.imread("train/sudo9_eight.png",0)
nine9=cv2.imread("train/sudo9_nine.png",0)
train_images9=[one9,two9,three9,four9,five9,six9,seven9,eight9,nine9]
one10=cv2.imread("train/sudo10_one.png",0)
two10=cv2.imread("train/sudo10_two.png",0)
three10=cv2.imread("train/sudo10_three.png",0)
four10=cv2.imread("train/sudo10_four.png",0)
five10=cv2.imread("train/sudo10_five.png",0)
six10=cv2.imread("train/sudo10_six.png",0)
seven10=cv2.imread("train/sudo10_seven.png",0)
eight10=cv2.imread("train/sudo10_eight.png",0)
nine10=cv2.imread("train/sudo10_nine.png",0)
train_images10=[one10,two10,three10,four10,five10,six10,seven10,eight10,nine10]
one11=cv2.imread("train/sudo11_one.png",0)
two11=cv2.imread("train/sudo11_two.png",0)
three11=cv2.imread("train/sudo11_three.png",0)
four11=cv2.imread("train/sudo11_four.png",0)
five11=cv2.imread("train/sudo11_five.png",0)
six11=cv2.imread("train/sudo11_six.png",0)
seven11=cv2.imread("train/sudo11_seven.png",0)
eight11=cv2.imread("train/sudo11_eight.png",0)
nine11=cv2.imread("train/sudo11_nine.png",0)
train_images11=[one11,two11,three11,four11,five11,six11,seven11,eight11,nine11]





## Grayscale and then gaussian blurring using FFT convolution
def print_answer(a,b):
	
	# img1=a ## colour version
	img=a ## grayscale version
	img2=b
	lines_img=np.zeros(img.shape,dtype=np.uint8)
	
	#
	## Binary thresholding
	#
	img=cv2.blur(img, (3,3))
	thresholded = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY,11,2)


	#
	## Hough Lines
	#
	edges = cv2.Canny(thresholded,100,200,apertureSize = 3)
	lines = cv2.HoughLinesP(edges,rho=1,theta=1*np.pi/180,threshold=100,minLineLength=120,maxLineGap=50)
	for line in lines:
		for x1,y1,x2,y2 in line:
			cv2.line(lines_img,(x1,y1),(x2,y2),255,1)
	

    #
	## contours 
	#
	contours,hierarchy = cv2.findContours(lines_img,  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	area=[cv2.contourArea(item) for item in contours]
	max_index=area.index(max(area))
	cnt = contours[max_index]
	x,y,w,h = cv2.boundingRect(cnt)
	
	#
	## to find perespective transformation
	#
	pts1=np.float32([[x,y],[x,y+w],[x+h,y+w],[x+h,y]])
	pts2 = np.float32([[0,0],[0,w],[h,w],[h,0]])




	# perspective transform
	M = cv2.getPerspectiveTransform(pts1,pts2)
	# Inverse perspective transform
	M_inv = cv2.getPerspectiveTransform(pts2,pts1)


	## inverse perspective transform
	# original warped image

	dst = cv2.warpPerspective(img,M,(w,h))
	# gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
	gray=dst
	width,height=gray.shape
	gray=gray[1:width-1,1:height-1]
	width1,height1=gray.shape
	width_dist=width1/9
	height_dist=height1/9

	sudoku_numbers=[]
	print(gray.shape,width_dist,height_dist)
	def find_match(detect):
		values1=[]
		values2=[]
		values3=[]
		values4=[]
		values5=[]
		values6=[]
		values7=[]
		values8=[]
		values9=[]
		values10=[]
		values11=[]
		values12=[]
		for images in train_images:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values1.append(ssd)
		for images in train_images2:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values2.append(ssd)
		for images in train_images3:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values3.append(ssd)
		for images in train_images4:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values4.append(ssd)
		for images in train_images5:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values5.append(ssd)
		for images in train_images6:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values6.append(ssd)
		for images in train_images7:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values7.append(ssd)
		for images in train_images8:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values8.append(ssd)
		for images in train_images9:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values9.append(ssd)
		for images in train_images10:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values10.append(ssd)
		for images in train_images11:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values11.append(ssd)
		for images in train_images11:
			diff=images-detect
			square=diff*diff
			ssd=np.sum(square)
			values12.append(ssd)
		values=[values1[i]+values2[i]+values3[i]+values4[i]+values5[i]+values6[i]+values7[i]+values8[i]+values9[i]+values10[i]+values11[i]+values12[i] for i in range(len(values1))]
		return values.index(min(values))+1

	for i in range(9):
		temp=[]
		for j in range(9):
			boxes=gray[int(i*(width_dist)):int((i+1)*(width_dist)),int(j*(height_dist)):int((j+1)*(height_dist))]
			t,boxes1=cv2.threshold(boxes, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
			stand=cv2.resize(boxes,(30,30))
			M1= cv2.moments(stand)
			cx1= int(M1['m10']/M1['m00'])
			cy1 = int(M1['m01']/M1['m00'])
			rows,cols = stand.shape
			tx=12-cx1
			ty=14-cy1
			M = np.float32([[1,0,tx],[0,1,ty]])
			stand = cv2.warpAffine(stand,M,(cols,rows))
			# (thresh, final_box) = cv2.threshold(stand, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
			final_box=cv2.adaptiveThreshold(stand,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
	            cv2.THRESH_BINARY_INV,11,3)
			final_box=final_box[4:26,4:26]
			final_box=cv2.copyMakeBorder(final_box,2,2,2,2,cv2.BORDER_CONSTANT,value=0)
			
			if(np.count_nonzero(final_box)>19):
				M1= cv2.moments(final_box)
				cx1= int(M1['m10']/M1['m00'])
				cy1 = int(M1['m01']/M1['m00'])
				rows,cols = final_box.shape
				tx=15-cx1
				ty=15-cy1
				M = np.float32([[1,0,tx],[0,1,ty]])
				dstt = cv2.warpAffine(final_box,M,(cols,rows))
				# kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
				# dstt= cv2.morphologyEx(dstt, cv2.MORPH_CLOSE, kernel)
				sudoku_numbers.append(str(find_match(dstt)))
			else:
				sudoku_numbers.append('.')
	print("##########################Detected Board###########################")
	detceted_baord_grid = np.array(sudoku_numbers).reshape(-1, 9)
	print(detceted_baord_grid)
	print("### SOLVING ###")
	ans=sl.solve_and_return(sudoku_numbers)
	print(ans)
	if(ans==0):
		return 0
	solved_baord_grid = np.array(ans).reshape(-1, 9)
	print(solved_baord_grid)
	warped_width=img2.shape[1]/9
	warped_height=img2.shape[0]/9
	# adding digit text on warped image
	for row,(i,j) in enumerate(zip(detceted_baord_grid,solved_baord_grid)):
		for col,(item1,item2) in enumerate(zip(i,j)):
			# print(row,col,item2)
			if(item1=='.'):
			
				cv2.putText(img2,item2, (int(col*(warped_width-1.5)+20),int(row*(warped_height))+35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,255,0),thickness=3)
	
	if(ans!=0):
		return img2
	
	