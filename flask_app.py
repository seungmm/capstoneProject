import os
import torch
from flask import Flask, jsonify, request
from PIL import Image
from torchvision import transforms
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})
# 손상 분류 모델 불러오기
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model', 'damage_resnet_11_24.pth')
model_damage = torch.load(model_path, map_location=torch.device('cpu'))
model_damage.eval()

# 수리 방법 분류 모델 불러오기
model_path = os.path.join(current_dir, 'model', 'repair_resnet_11_8.pth')
model_repair = torch.load(model_path, map_location=torch.device('cpu'))
model_damage.eval()

# 이미지 전처리 함수
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 클래스 레이블 (원하는 클래스 레이블로 변경)
damage_labels = ['파손', '스크레치', '이격', '찌그러짐']
repair_labels = ['도색', '교체', '판금']
cost_dic = {
    '휠': {
        '교체': '15 ~ 25만원',
        '도색': '7 ~ 10만원',
        '판금': '5 ~ 10만원'},
    '도어': {
            '교체': '40 ~ 60만원',
            '도색': '20 ~ 25만원',
            '판금': '20 ~ 25만원'},
    '휀더': {
            '교체' : '30 ~ 35만원',
            '도색' : '19 ~ 25만원',
            '판금' : '19 ~ 25만원'},
    '앞 범퍼': {
            '교체': '30 ~ 40만원',
            '도색': '21 ~ 26만원',
            '판금':  '21 ~ 26만원'},
    '뒷 범퍼': {
            '교체': '30 ~ 40만원',
            '도색': '21 ~ 26만원',
            '판금': '21 ~ 25만원'}
}
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
            output_damage = model_damage(img)
            output_repair = model_repair(img)
        # 예측 결과 해석
        _, predicted_class_damage = output_damage.max(1)
        _, predicted_class_repair = output_repair.max(1)
        predicted_damage = damage_labels[predicted_class_damage.item()]
        predicted_repair = repair_labels[predicted_class_repair.item()]


        return jsonify({"parts" : parts,
                        "damage" : predicted_damage,
                        "repair" : predicted_repair,
                        "cost" : cost_dic[parts][predicted_repair]
                        })
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
