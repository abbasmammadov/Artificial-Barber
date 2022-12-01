import numpy as np
from PIL import Image
import os
import dlib
from torchvision import transforms
from utils.drive import open_url

ROOT = os.path.dirname(os.path.abspath(__file__))
print(ROOT)
def get_face(image_path, detector, output_size, transform_size=4096, save_path=None):
    # pred_path = os.path.join('/'.join(ROOT.split('/')[:]) + '/shape_predictor_68_face_landmarks.dat')
    pred_path_dir = f'{ROOT}/pretrained_models/' 
    pred_path = pred_path_dir + 'shape_predictor_68_face_landmarks.dat'
    # check whether the file exists
    if not os.path.exists(pred_path):
        print('Downloading the pretrained model...')
        pretrained_face_model = open_url("https://drive.google.com/uc?id=1huhv8PYpNNKbGCLOaYUjOgR1pY5pmbJx", cache_dir=pred_path_dir, return_path=True)
        os.rename(pretrained_face_model, pred_path)
    predictor = dlib.shape_predictor(pred_path)
    # print(image_path)
    img = dlib.load_rgb_image(image_path)
    detections = detector(img, 1)
    # lm = np.array([[p.x, p.y] for p in predictor(img, detections[0]).parts()])
    lm = np.array([[p.x, p.y] for p in predictor(img, detections[0]).parts()])
    # transform_size = 4096
    lm_left_eye = lm[36:42]
    lm_right_eye = lm[42:48]
    lm_outer_lip = lm[48:60]

    # get the center of the eyes and the mouth
    left_eye = lm_left_eye.mean(axis=0)
    right_eye = lm_right_eye.mean(axis=0)
    average_eye = (left_eye + right_eye) / 2
    eye_to_eye = right_eye - left_eye

    left_mouth = lm_outer_lip[0]
    right_mouth = lm_outer_lip[6]
    average_mouth = (left_mouth + right_mouth) / 2

    eye_to_mouth = average_mouth - average_eye

        # Choose oriented crop rectangle.
    x = eye_to_eye - eye_to_mouth[::-1] * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = x[::-1] * [-1, 1]
    c = average_eye + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # read image
    target_face = Image.open(image_path)

    # Shrink.
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(target_face.size[0]) / shrink)), int(np.rint(float(target_face.size[1]) / shrink)))
        target_face = target_face.resize(rsize, Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink


    # Crop.
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))),
            int(np.ceil(max(quad[:, 1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, target_face.size[0]),
            min(crop[3] + border, target_face.size[1]))
    if crop[2] - crop[0] < target_face.size[0] or crop[3] - crop[1] < target_face.size[1]:
        target_face = target_face.crop(crop)
        quad -= crop[0:2]
    target_face = target_face.transform((transform_size, transform_size), Image.QUAD, (quad + 0.5).flatten(),
                Image.BILINEAR)
    if output_size < transform_size:
        target_face = target_face.resize((output_size, output_size), Image.ANTIALIAS)
    
    # save the output image
    face_tensor = transforms.ToTensor()(target_face)
    face_tensor_lr = face_tensor.clamp(0, 1)
    target_face = transforms.ToPILImage()(face_tensor_lr)
    if (1024//output_size) != 1:
        target_face = target_face.resize((output_size, output_size), Image.LANCZOS)
    
    if save_path:
        target_face.save(save_path)
    return target_face # this is a PIL image

# get_face('high_res.png', dlib.get_frontal_face_detector(), 1024, save_path='high_res_face.png')