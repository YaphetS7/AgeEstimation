import torch
import cv2
from scipy.spatial.distance import euclidean
from PIL import Image
from torchvision import transforms
from model import resnet34


model = resnet34(pretrained=True, progress=True)
fc_in_features = model.fc.in_features
model.fc = torch.nn.Linear(fc_in_features, 101)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(post_process=False, device=device)


state = torch.load('/content/drive/MyDrive/AgeEstimation/best.pth') # insert here ur path to .pth model
model.to(device)

model.load_state_dict(state)
model.eval()


def preprocess(img):
  img = Image.open(img).convert('RGB')
  imgs = [img, img.transpose(Image.FLIP_LEFT_RIGHT)]
  transform_list = [
      transforms.Resize((224, 224), interpolation=3),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ]
  transform = transforms.Compose(transform_list)
  imgs = [transform(i) for i in imgs]
  imgs = [torch.unsqueeze(i, dim=0) for i in imgs]

  return imgs


def alignment_procedure(img, left_eye, right_eye):
    #this function aligns given face in img based on left and right eye coordinates
    
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    
    #-----------------------
    #find rotation direction
    
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock
    
    #-----------------------
    #find length of triangle edges
    
    a = euclidean(left_eye, point_3rd)
    b = euclidean(right_eye, point_3rd)
    c = euclidean(right_eye, left_eye)
    
    #-----------------------
    
    #apply cosine rule
    
    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
    
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
    
        #-----------------------
        #rotate base image
    
        if direction == -1:
            angle = 90 - angle
    
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    
    #-----------------------
    
    return img #return img anyway



def detect_face(input_path):
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bbox, _, landmarks = mtcnn.detect(img, landmarks=True)

    if bbox is None:
        return "can't detect"
    if landmarks is None:
        return "can't detect"


    bbox = list(map(int, bbox[0]))
    bbox = [max(0, int(x)) for x in bbox]
    img = img[bbox[1]: bbox[3], bbox[0]: bbox[2], :]
    align = alignment_procedure(img, (int(landmarks[0][0][0]),
                    int(landmarks[0][0][1])),
                    (int(landmarks[0][1][0]), 
                    int(landmarks[0][1][1])))
    
    return align, img


def inference(img_path, uid):
    
    rank = torch.Tensor([i for i in range(101)]).to(device)

    age = 20
    age = torch.IntTensor([int(age)])
    age = age.to(device)

    p = detect_face(img_path)

    if type(p) == str:
        return "MTCNN can't detect face"
    else:
        align, not_align = p
        align = cv2.cvtColor(align, cv2.COLOR_BGR2RGB)
        not_align = cv2.cvtColor(not_align, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(uid) + 'align.jpg', align) 
        cv2.imwrite(str(uid) + 'not_align.jpg', not_align)
        imgs = preprocess(str(uid) + 'align.jpg')
        imgs2 = preprocess(str(uid) + 'not_align.jpg')
        predict_age_align = 0
        predict_age_not_align = 0
        prototype = np.zeros([101, 512], dtype=np.float32)
        instance_num = np.zeros([101, 1], dtype=np.float32)
        intra = np.zeros([101, 1], dtype=np.float32)
        inter = np.zeros([101, 101], dtype=np.float32)
        pro = [prototype, instance_num]

        for img in imgs:
            img = img.to(device)
            output, pro, intra, inter = model(img, age, pro, intra, inter)
            predict_age_align += torch.sum(output * rank, dim=1).item() / 2

        for img in imgs2:
            img = img.to(device)
            output, pro, intra, inter = model(img, age, pro, intra, inter)
            predict_age_not_align += torch.sum(output * rank, dim=1).item() / 2


    return predict_age_align, predict_age_not_align