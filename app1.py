from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
from flask import Flask, request, render_template
from markupsafe import Markup
from werkzeug.utils import secure_filename
import pandas as pd
from utils.disease import disease_dic
import requests
import pickle
import io
import sys, glob, re
import torch
from torchvision import transforms
from PIL import Image
from utils.model import ResNet9


disease_classes = ['Apple___Apple_scab',
                   'Apple___Black_rot',
                   'Apple___Cedar_apple_rust',
                   'Apple___healthy',
                   'Blueberry___healthy',
                   'Cherry_(including_sour)___Powdery_mildew',
                   'Cherry_(including_sour)___healthy',
                   'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                   'Corn_(maize)___Common_rust_',
                   'Corn_(maize)___Northern_Leaf_Blight',
                   'Corn_(maize)___healthy',
                   'Grape___Black_rot',
                   'Grape___Esca_(Black_Measles)',
                   'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                   'Grape___healthy',
                   'Orange___Haunglongbing_(Citrus_greening)',
                   'Peach___Bacterial_spot',
                   'Peach___healthy',
                   'Pepper,_bell___Bacterial_spot',
                   'Pepper,_bell___healthy',
                   'Potato___Early_blight',
                   'Potato___Late_blight',
                   'Potato___healthy',
                   'Raspberry___healthy',
                   'Soybean___healthy',
                   'Squash___Powdery_mildew',
                   'Strawberry___Leaf_scorch',
                   'Strawberry___healthy',
                   'Tomato___Bacterial_spot',
                   'Tomato___Early_blight',
                   'Tomato___Late_blight',
                   'Tomato___Leaf_Mold',
                   'Tomato___Septoria_leaf_spot',
                   'Tomato___Spider_mites Two-spotted_spider_mite',
                   'Tomato___Target_Spot',
                   'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
                   'Tomato___Tomato_mosaic_virus',
                   'Tomato___healthy']

app = Flask(__name__)

# ✅ All models set to None — loaded lazily on first request
model = None
ferti = None
SoilNet = None
svm = None
disease_model = None

model_path = "SoilNet_93_86.h5"
disease_model_path = 'plant_disease_model.pth'

def load_models():
    global model, ferti, SoilNet, svm, disease_model
    if svm is None:
        file = open('cropmodel2.pkl', 'rb')
        svm = pickle.load(file)
        file.close()
    if model is None:
        model = pickle.load(open('classifier.pkl', 'rb'))
    if ferti is None:
        ferti = pickle.load(open('fertilizer.pkl', 'rb'))
    if SoilNet is None:
        SoilNet = load_model(model_path)
    if disease_model is None:
        disease_model = ResNet9(3, len(disease_classes))
        disease_model.load_state_dict(torch.load(
            disease_model_path, map_location=torch.device('cpu')))
        disease_model.eval()


def predict_image(img):
    load_models()
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    prediction = disease_classes[preds[0].item()]
    return prediction


classes = {
    0: "Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",
    1: "Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",
    2: "Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",
    3: "Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"
}


def model_predict(image_path, model):
    print("Predicted")
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image / 255
    image = np.expand_dims(image, axis=0)
    result = np.argmax(model.predict(image))

    if result == 0:
        return "Alluvial", "Alluvial.html"
    elif result == 1:
        return "Black", "Black.html"
    elif result == 2:
        return "Clay", "Clay.html"
    elif result == 3:
        return "Red", "Red.html"


mapper = {1: 'rice', 2: 'maize', 3: 'chickpea', 4: 'kidneybeans',
          5: 'pigeonpeas', 6: 'mothbeans', 7: 'mungbean', 8: 'blackgram',
          9: 'lentil', 10: 'pomegranate', 11: 'banana', 12: 'mango',
          13: 'grapes', 14: 'watermelon', 15: 'muskmelon', 16: 'apple',
          17: 'orange', 18: 'papaya', 19: 'coconut', 20: 'cotton',
          21: 'jute', 22: 'coffee'}

fertilizer_dic = {
    'NHigh': """The N value of soil is high and might give rise to weeds.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i> Manure </i> – adding manure is one of the simplest ways to amend your soil with nitrogen.
        <br/> 2. <i>Coffee grinds </i> – rich in nitrogen and helps drainage.
        <br/>3. <i>Plant nitrogen fixing plants</i> – like peas, beans and soybeans.
        <br/>4. Plant 'green manure' crops like cabbage, corn and brocolli.
        <br/>5. <i>Use mulch (wet grass) while growing crops</i>""",

    'Nlow': """The N value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/> 1. <i>Add sawdust or fine woodchips to your soil</i>
        <br/>2. <i>Plant heavy nitrogen feeding plants</i> – tomatoes, corn, broccoli, cabbage and spinach.
        <br/>3. <i>Water</i> – soaking your soil with water will help leach the nitrogen deeper.
        <br/>4. Add composted manure to the soil.
        <br/>5. Plant Nitrogen fixing plants like peas or beans.
        <br/>6. <i>Use NPK fertilizers with high N value.</i>""",

    'PHigh': """The P value of your soil is high.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Avoid adding manure</i>
        <br/>2. <i>Use only phosphorus-free fertilizer</i>
        <br/>3. <i>Water your soil</i> – soaking will aid in driving phosphorous out.
        <br/>4. Plant nitrogen fixing vegetables like beans and peas.
        <br/>5. Use crop rotations to decrease high phosphorous levels""",

    'Plow': """The P value of your soil is low.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Bone meal</i> – rich in phosphorous.
        <br/>2. <i>Rock phosphate</i> – a slower acting source.
        <br/>3. <i>Phosphorus Fertilizers</i> – high phosphorous content in NPK ratio.
        <br/>4. <i>Organic compost</i>
        <br/>5. <i>Manure</i>
        <br/>6. <i>Ensure proper soil pH</i> – 6.0 to 7.0 range is optimal.""",

    'KHigh': """The K value of your soil is high.
        <br/> Please consider the following suggestions:
        <br/><br/>1. <i>Loosen the soil</i> deeply and water thoroughly.
        <br/>2. <i>Sift through the soil</i> and remove rocks.
        <br/>3. Stop applying potassium-rich commercial fertilizer.
        <br/>4. Mix crushed eggshells or wood ash to add calcium.
        <br/>5. Use NPK fertilizers with low K levels.""",

    'Klow': """The K value of your soil is low.
        <br/>Please consider the following suggestions:
        <br/><br/>1. Mix in muricate of potash or sulphate of potash
        <br/>2. Try kelp meal or seaweed
        <br/>3. Try Sul-Po-Mag
        <br/>4. Bury banana peels an inch below the soils surface
        <br/>5. Use Potash fertilizers since they contain high values potassium"""
}


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    load_models()  # ✅ lazy load
    print("Entered")
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        print("@@ Input posted = ", filename)
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path, SoilNet)
        return render_template(output_page, pred_output=pred, user_image=file_path)


@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    load_models()  # ✅ lazy load
    if request.method == 'POST':
        mydict = request.form
        nitrogen = mydict.get('nitrogen')
        phosphorus = mydict.get('phosphorus')
        potassium = mydict.get('potassium')
        temperature = mydict.get('temperature')
        humidity = mydict.get('humidity')
        ph = mydict.get('ph')
        rainfall = mydict.get('rainfall')

        input_features = [nitrogen, phosphorus, potassium,
                          temperature, humidity, ph, rainfall]

        inf = svm.predict([input_features])
        inf = inf[0]
        value = mapper[inf]

        df = pd.read_csv('fertilizer.csv')
        nitro = df[df['Crop'] == value]['N'].iloc[0]
        phos = df[df['Crop'] == value]['P'].iloc[0]
        pota = df[df['Crop'] == value]['K'].iloc[0]

        n = int(nitro) - int(nitrogen)
        p = int(phos) - int(phosphorus)
        k = int(pota) - int(potassium)

        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_val = temp[max(temp.keys())]

        if max_val == 'N':
            key = 'NHigh' if n < 0 else 'Nlow'
        elif max_val == 'P':
            key = 'PHigh' if p < 0 else 'Plow'
        else:
            key = 'KHigh' if k < 0 else 'Klow'

        response = Markup(str(fertilizer_dic[key]))
        value = value.capitalize()
        return render_template('result.html', inf=response, value=value)

    return render_template('predict1.html')


@app.route('/Model1')
def Model1():
    return render_template('Model1.html')


@app.route('/predict2', methods=['POST'])
def predict2():
    load_models()  # ✅ lazy load
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')

    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(
            val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values for all fields.')

    input = [int(temp), int(humi), int(mois), int(soil),
             int(crop), int(nitro), int(pota), int(phosp)]
    res = ferti.classes_[model.predict([input])]
    return render_template('Model1.html', x=res)


@app.route('/disease-predict')
def disease_predict():
    title = 'Harvestify - Disease Detection'
    return render_template('disease.html', title=title)


@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    load_models()  # ✅ lazy load
    title = 'Harvestify - Disease Detection'
    if 'file' not in request.files:
        return render_template('disease.html', title=title, error="No file uploaded")
    file = request.files.get('file')
    if not file:
        return render_template('disease.html', title=title, error="No file selected")
    try:
        img = file.read()
        prediction = predict_image(img)
        prediction = Markup(str(disease_dic.get(prediction, "Unknown disease")))
        return render_template('disease-result.html', prediction=prediction, title=title)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return render_template('disease.html', title=title, error="Error processing image")


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, threaded=False)