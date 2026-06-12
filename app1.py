import threading
import os
import io
import numpy as np
import pickle
import torch
from torchvision import transforms
from PIL import Image
from flask import Flask, request, render_template
from markupsafe import Markup
import pandas as pd
from utils.disease import disease_dic
from utils.model import ResNet9

app = Flask(__name__)

# All models None initially
model = None
ferti = None
SoilNet = None
svm = None
disease_model = None
models_loaded = False

disease_classes = ['Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy','Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']

classes = {0:"Alluvial Soil:-{ Rice,Wheat,Sugarcane,Maize,Cotton,Soyabean,Jute }",1:"Black Soil:-{ Virginia, Wheat , Jowar,Millets,Linseed,Castor,Sunflower} ",2:"Clay Soil:-{ Rice,Lettuce,Chard,Broccoli,Cabbage,Snap Beans }",3:"Red Soil:{ Cotton,Wheat,Pilses,Millets,OilSeeds,Potatoes }"}

mapper = {1:'rice',2:'maize',3:'chickpea',4:'kidneybeans',5:'pigeonpeas',6:'mothbeans',7:'mungbean',8:'blackgram',9:'lentil',10:'pomegranate',11:'banana',12:'mango',13:'grapes',14:'watermelon',15:'muskmelon',16:'apple',17:'orange',18:'papaya',19:'coconut',20:'cotton',21:'jute',22:'coffee'}

fertilizer_dic = {
    'NHigh': "The N value of soil is high. Use manure, coffee grinds, or plant nitrogen-fixing plants.",
    'Nlow': "The N value is low. Use NPK fertilizers with high N, add composted manure, plant peas or beans.",
    'PHigh': "The P value is high. Avoid manure, use phosphorus-free fertilizer, water the soil well.",
    'Plow': "The P value is low. Use bone meal, rock phosphate, or phosphorus fertilizers.",
    'KHigh': "The K value is high. Stop potassium-rich fertilizers, loosen and water soil deeply.",
    'Klow': "The K value is low. Use potash fertilizers, kelp meal, or bury banana peels."
}

def load_all_models():
    global model, ferti, SoilNet, svm, disease_model, models_loaded
    print("Loading models in background...")
    from tensorflow.keras.models import load_model
    f = open('cropmodel2.pkl', 'rb')
    svm = pickle.load(f)
    f.close()
    model = pickle.load(open('classifier.pkl', 'rb'))
    ferti = pickle.load(open('fertilizer.pkl', 'rb'))
    SoilNet = load_model("SoilNet_93_86.h5")
    disease_model = ResNet9(3, len(disease_classes))
    disease_model.load_state_dict(torch.load('plant_disease_model.pth', map_location=torch.device('cpu')))
    disease_model.eval()
    models_loaded = True
    print("All models loaded!")

# Start loading in background thread immediately
threading.Thread(target=load_all_models, daemon=True).start()

def model_predict(image_path):
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    image = load_img(image_path, target_size=(224, 224))
    image = img_to_array(image) / 255
    image = np.expand_dims(image, axis=0)
    result = np.argmax(SoilNet.predict(image))
    if result == 0: return "Alluvial", "Alluvial.html"
    elif result == 1: return "Black", "Black.html"
    elif result == 2: return "Clay", "Clay.html"
    elif result == 3: return "Red", "Red.html"

def predict_image(img):
    transform = transforms.Compose([transforms.Resize(256), transforms.ToTensor()])
    image = Image.open(io.BytesIO(img))
    img_t = transform(image)
    img_u = torch.unsqueeze(img_t, 0)
    yb = disease_model(img_u)
    _, preds = torch.max(yb, dim=1)
    return disease_classes[preds[0].item()]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if not models_loaded:
        return "Models are still loading, please wait 1-2 minutes and refresh...", 503
    if request.method == 'POST':
        file = request.files['image']
        filename = file.filename
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)
        pred, output_page = model_predict(file_path)
        return render_template(output_page, pred_output=pred, user_image=file_path)

@app.route('/predict1', methods=['GET', 'POST'])
def predict1():
    if not models_loaded:
        return "Models are still loading, please wait 1-2 minutes and refresh...", 503
    if request.method == 'POST':
        mydict = request.form
        nitrogen = mydict.get('nitrogen')
        phosphorus = mydict.get('phosphorus')
        potassium = mydict.get('potassium')
        temperature = mydict.get('temperature')
        humidity = mydict.get('humidity')
        ph = mydict.get('ph')
        rainfall = mydict.get('rainfall')
        input_features = [nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]
        inf = svm.predict([input_features])[0]
        value = mapper[inf]
        df = pd.read_csv('fertilizer.csv')
        nitro = df[df['Crop'] == value]['N'].iloc[0]
        phos = df[df['Crop'] == value]['P'].iloc[0]
        pota = df[df['Crop'] == value]['K'].iloc[0]
        n = int(nitro)-int(nitrogen)
        p = int(phos)-int(phosphorus)
        k = int(pota)-int(potassium)
        temp = {abs(n): "N", abs(p): "P", abs(k): "K"}
        max_val = temp[max(temp.keys())]
        if max_val == 'N': key = 'NHigh' if n < 0 else 'Nlow'
        elif max_val == 'P': key = 'PHigh' if p < 0 else 'Plow'
        else: key = 'KHigh' if k < 0 else 'Klow'
        response = Markup(str(fertilizer_dic[key]))
        return render_template('result.html', inf=response, value=value.capitalize())
    return render_template('predict1.html')

@app.route('/Model1')
def Model1():
    return render_template('Model1.html')

@app.route('/predict2', methods=['POST'])
def predict2():
    if not models_loaded:
        return "Models are still loading, please wait 1-2 minutes and refresh...", 503
    temp = request.form.get('temp')
    humi = request.form.get('humid')
    mois = request.form.get('mois')
    soil = request.form.get('soil')
    crop = request.form.get('crop')
    nitro = request.form.get('nitro')
    pota = request.form.get('pota')
    phosp = request.form.get('phos')
    if None in (temp, humi, mois, soil, crop, nitro, pota, phosp) or not all(val.isdigit() for val in (temp, humi, mois, soil, crop, nitro, pota, phosp)):
        return render_template('Model1.html', x='Invalid input. Please provide numeric values.')
    input = [int(temp), int(humi), int(mois), int(soil), int(crop), int(nitro), int(pota), int(phosp)]
    res = ferti.classes_[model.predict([input])]
    return render_template('Model1.html', x=res)

@app.route('/disease-predict', methods=['GET'])
def disease_predict():
    return render_template('disease.html', title='Disease Detection')

@app.route('/disease-predict', methods=['POST'])
def disease_prediction():
    if not models_loaded:
        return "Models are still loading, please wait 1-2 minutes and refresh...", 503
    if 'file' not in request.files:
        return render_template('disease.html', title='Disease Detection', error="No file uploaded")
    file = request.files.get('file')
    if not file:
        return render_template('disease.html', title='Disease Detection', error="No file selected")
    try:
        img = file.read()
        prediction = predict_image(img)
        prediction = Markup(str(disease_dic.get(prediction, "Unknown disease")))
        return render_template('disease-result.html', prediction=prediction, title='Disease Detection')
    except Exception as e:
        return render_template('disease.html', title='Disease Detection', error="Error processing image")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)