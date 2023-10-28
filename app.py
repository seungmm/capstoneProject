import os
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename

app = Flask(__name__)

# ResNet-50 모델 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'secondmodel.pth')
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()  # 모델을 평가 모드로 설정

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 클래스 레이블 (원하는 클래스 레이블로 변경)
class_labels = ['파손', '긁힘', '분리', '찌그러짐']

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':

        # 클라이언트로부터 파일을 받습니다.
        if 'image' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['image']
        parts = request.form.get('parts')

        # 유효한 파일인지 확인합니다.
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        filename = secure_filename(file.filename)

        # 이미지를 PIL Image 객체로 변환합니다.
        img = Image.open(file)

        # 이미지 전처리
        img = preprocess(img)
        img = img.unsqueeze(0)  # 배치 차원 추가

        # 모델로 예측 수행
        with torch.no_grad():
            output = model(img)

        # 예측 결과 해석
        _, predicted_class = output.max(1)
        predicted_damage = class_labels[predicted_class.item()]

        predicted_repair = "도색"
        predicted_cost = "100만원"

        return jsonify({"parts" : parts,
                        "damage" : predicted_damage,
                        "repair" : predicted_repair,
                        "cost" : predicted_cost
                        })
if __name__ == '__main__':
    app.run()
