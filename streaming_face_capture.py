from PIL import ImageGrab
import cv2
import numpy as np
import win32gui


def proc(hwnd, ar):
    title = win32gui.GetWindowText(hwnd)
    if ar[0] in title:
        ar[1].append(hwnd)
    return 1


def getid(title, n=0):
    hwnds = []
    win32gui.EnumWindows(proc, [title, hwnds])
    return hwnds[n]


# 画像の名前
filename = 'Face'

# カスケード分類器読み込み
# Xml保存場所
xmlpath = '/xmls/'
folderpath = '/capture_face/'

# 顔判定で使うxmlファイルを指定する。
cascade_path = xmlpath + "lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_path)

Count = 0
while 1:
    handle = getid('Google Chrome')
    rect = win32gui.GetWindowRect(handle)

    img = ImageGrab.grab(rect)
    img = np.asarray(img)
    img_src = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_result = img_src

    faces = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=1, minSize=(100, 100))

    # 顔があった場合
    if len(faces) > 0:
        # 複数の顔があった場合、１つずつ四角で囲っていく
        for face in faces:
            # faceには(四角の左上のx座標, 四角の左上のy座標, 四角の横の長さ, 四角の縦の長さ) が格納されている。
            # 囲う四角の左上の座標
            coordinates = tuple(face[0:2])
            # (囲う四角の横の長さ, 囲う四角の縦の長さ)
            length = tuple(face[0:2] + face[2:4])
            # 四角で囲う処理
            zeros1 = np.zeros((face[2 - 0], face[4 - 2]))
            face_img = np.stack([zeros1, zeros1, zeros1], 2)
            face_img_shape = face_img.shape
            print('face' + str(face) + str(face_img_shape))
            for x in range(face_img_shape[0]):
                for y in range(face_img_shape[1]):
                    try:
                        face_img[x, y] = img_src[face[1] + x, face[0] + y]
                    except:
                        print("error")
            cv2.imwrite(folderpath + str(filename) + str(Count) + '.png', face_img)
            Count += 1

    # 顔があった場合
    if len(faces) > 0:
        # 顔認識の枠の色
        color = (255, 0, 0)
        # 複数の顔があった場合、１つずつ四角で囲っていく
        for face in faces:
            # faceには(四角の左上のx座標, 四角の左上のy座標, 四角の横の長さ, 四角の縦の長さ) が格納されている。
            # 囲う四角の左上の座標
            coordinates = tuple(face[0:2])
            # (囲う四角の横の長さ, 囲う四角の縦の長さ)
            length = tuple(face[0:2] + face[2:4])
            # 四角で囲う処理
            cv2.rectangle(img_result, coordinates, length, color, thickness=3)
    cv2.imshow("Face", cv2.resize(img_result, dsize=(1600, 900)))
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
