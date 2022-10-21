def emptydir(dirname):  # 清空資料夾
    if os.path.isdir(dirname):  # 資料夾存在就两除
        shutil.rmtree(dirname)
        sleep(2) #否則會出諾
    os.mkdir(dirname)  # 建立資料夾
def dirResize(src, dst):
    myfiles = glob.glob(src + '/*.JPG')  
    emptydir(dst)
    print(src + ' 資料夾:' )
    print('開始轉換圖形尺寸!')
    for f in myfiles:
        fname=f.split("\\")[-1]
        img = Image.open(f)
        img_new= img.resize((300, 225), PIL.Image.ANTIALIAS)  # 尺寸300x225
        img_new.save(dst + '/' + fname)
    print('轉換圖形尺寸完成!\n')
def area(row, col):
    global nn
    if bg[row][col] != 255:
        return
    bg[row][col]=lifearea  
    if col > 1:
        if bg[row][col-1] == 255:
            nn += 1
            area(row, col-1)
    if col < w-1:
        if bg[row][col+1] == 255:
            nn += 1
            area(row, col+1)
    if row > 1:
        if bg[row-1][col] == 255:
            nn += 1
            area(row-1, col)
    if row < h-1:
        if bg[row+1][col]==255:
            nn+=1 
            area(row+1,col)
            
import cv2
import PIL
from PIL import Image
import glob
import shutil, os
from time import sleep
import numpy as np
import sys
import pyocr
import pyocr.builders
import re
import pytesseract
import dlib
dirResize('predictPlate_sr','predictPlate2')
print('取車牌!')
dstdir = 'cropPlate3'
myfiles = glob.glob('predictPlate2\*.JPG')
# print(myfiles)
emptydir(dstdir)
detector = dlib.simple_object_detector("mydataset.svm")
for imgname in myfiles:
    filename = (imgname.split("\\"))[-1] #取得檔察名稱
    img = cv2.imread(imgname)
    # print(img)
    # detector = cv2.CascadeClassifier("C:\\Users\\YEE\\Desktop\\小作品\\car\\plateimg.xml")
    # signs = detector.detectMultiScale(img,scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
    dets = detector(img)
    # print(dets)
    # if len(signs) > 0 :
    print("偵測到的車牌數:{}".format(len(dets)))
    for index ,face in enumerate(dets):
        # for (x , y , w, h) in signs:
            x = face.left()
            y = face.top()
            w = face.right()-x
            h = face.bottom()-y
            image1 = Image.open(imgname)
            image2 = image1.crop((x, y, x+w, y+h)) #做放本牌運形
            image3 = image2.resize((140, 40), Image.ANTIALIAS) #轉接尺寸13140X49
            img_gray =np.array(image3.convert('L'))
            _,img_thre = cv2.threshold(img_gray,127,255, cv2.THRESH_BINARY)
            cv2.imwrite(dstdir + '/' +  filename, img_thre)
    else:
        print(filename)
print('擷取車牌結束')

myfiles = glob.glob('cropPlate3\*.jpg')
for file in myfiles:
    image = cv2.imread(file)
    basename = os.path.basename(file)
    # print(basename)
    filename = (file.split("\\"))[-1] #取得檔察名稱
    tools = pyocr.get_available_tools()
    if len(tools) == 0:
        print("NO OCR tool found")
        sys.exit(1)
    tool = tools[0]
    gray = cv2.cvtColor (image, cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray, 127,255, cv2.THRESH_BINARY_INV)#轉办黑白(上刚化)
    cv2.imwrite('cropPlate4' + '/' +  filename, img_thre)
    contours1 = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#院
    contours = contours1[0]       

    letter_image_regions = []
    for contour in contours:
        (x, y, w, h)= cv2.boundingRect(contour) 
        letter_image_regions.append((x, y, w, h)) 
    letter_image_regions = sorted(letter_image_regions, key =lambda x: x[0])

    count = 0
    for box in letter_image_regions:
        x, y, w, h = box
        if x>=2 and x<=125 and w>=5 and w<=26 and h>=20 and h<40:
           count += 1 
    if count<6:
         wmax=35
    else:
         wmax=26
    nChar = 0
    letterlist = []
    for box in letter_image_regions:
        x, y, w, h = box

        if x>=2 and x<=125 and w>=5 and w<=wmax and h>=20 and h<40:
            nChar += 1
            letterlist.append((x, y, w, h))
        else:
            nChar += 1
            letterlist.append((x, y, w, h))

    for i in range(len(thresh)):#度
        for j in range(len(thresh[i])):#j3点度
             if thresh[i][j] == 255: #面色為白色
                count = 0
                for k in range(-2, 3):
                    for l in range(-2, 3):
                        try:
                            if thresh[i+k][j+l]==255:#若屋白%count加
                                count += 1
                        except IndexError:
                            pass
                    if count <= 6: #巡圖少終等6個白點
                        thresh[i][j] = 0 #將白點去除
                    
    real_shape=[]
    for i , box in enumerate(letterlist):
        x,y,w,h = box 
        bg = thresh[y:y+h, x:x+w]

        if i == 0 or i == nChar:
            lifearea = 0
            nn = 0
            life = []   
            for row in range(0,h):
                for col in range(0,w):
                    if bg[row][col] == 255:
                        nn = 1
                        lifearea = lifearea + 1 
                        area(row,col)
                        life.append(nn)
            maxlife = max(life)
            indexmaxlife = life.index(maxlife)

            for  row in range(0,h):
                for col in range(0,w):
                    if bg[row][col] == indexmaxlife+1:
                        bg[row][col]=255
                    else:
                        bg[row][col]=0
        real_shape.append(bg)

        image2 = thresh.copy()
        newH, newW = image2.shape
        space = 10 
        bg = np.zeros((newH+space*2 , newW+space*2+30 , 1) , np.uint8)
        bg.fill(0)

        for i,letter in enumerate(real_shape):
            h=letter.shape[0] #原火文字
            w=letter.shape[1]
            x=letterlist[i][0] #原火文
            y=letterlist[i][1]
            bw2 = np.ones((50, 50, 1), np.uint8)*255 #逃立許景
            bw=~letter
            for row in range(h): #將文字圖片加人對景
                for col in range(w):
                    bw2(int((50-h)/2)+row)[int((50-w)/2)*col] = bw[row][col]
            bw1=bw>0
            ratio = sum(sum(bw1))/(h*w)
            if ratio > 0.15:
                cv2.imwrite("alphaNum3a\\"+basename.split(".")[0]+"_"+str(i)+".jpg",bw2)
                cv2.imwrite("alphaNum3\\"+basename .split(".")[0]+"_"+str(i)+"["+str(ratio)+"].jpg" , bw2)
            else:
                print("切不了")
            for row in range(h):
                for col in range(w):
                    if ratio>0.15:
                        bg[space+y+row][space+x+col+i*4] = letter[row][col]
        _, bg = cv2.threshold(bg, 127, 255, cv2.THRESH_BINARY_INV)
            #轉為白色背景·黑色文字
        # cv2.imwrite('result.jpg', bg)#存檔
        cv2.imwrite("result\\"+basename.split(".")[0]+".jpg",bg) #存位
        #OCR辨識車牌
    img=Image.open("result\\"+basename.split(".")[0]+".jpg")


    # tools = pyocr.get_available_tools()
    # if len(tools) == 0:
    #     print("No OCR tool found")
    #     sys.exit(1)
    # tool=tools[0]#取得可用工具

    # result = tool.image_to_string( 
    #     Image.open('result2.jpg'),
    #     builder=pyocr.builders.TextBuilder()
    # )   
    result = pytesseract.image_to_string(img , lang = 'eng')

    print(result)

    txt=result.replace("!","1") #如果是!字元,更改為字元1
    real_txt=re.findall(r'[A-Z]+|[\d]+',txt)#只取數字和大窝英文字母
    #組合真正的車牌
    txt_Plate=""
    for char in real_txt:
        txt_Plate += char
    print("ocr辨識結果:",result)
    basename=os.path.basename(file)
    if basename.split(".")[0]==txt_Plate:
        mess="V"
    else:
        mess="X"
    print("優化後:{} 檔名:{} 辨識結果:{}".format(txt_Plate,basename,mess))
    # cv2.imshow('image',image) #顯示原始圖形
    # cv2.imshow('bg', bg) #顯示組合的字元
    # cv2.moveWindow("image",500,250)#將视窗移到指定位置
    # cv2.moveWindow("bg" ,500,350) #將祝窗移到指定位置
    # key = cv2.waitkey(이) #按任意鍵結束
    # cv2.destroyAllwindows()
    # if key == 113 or key==81:#按9鍵結束
    #     break