# AI Soil Crop System

This project is a Flask application for soil type classification, crop recommendation, fertilizer advice, and plant disease detection.

## Run locally

1. Create a Python environment and install dependencies:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

2. Run the app:

```bash
python app1.py
```

3. Open `http://127.0.0.1:5000/` in your browser.

## Deploy to Heroku

This repo includes a `Procfile` and `requirements.txt` for deployment.

```bash
heroku create
git push heroku main
```

## Notes

- Ensure `plant_disease_model.pth` and `SoilNet_93_86.h5` are included in the repository.
- Netlify is not suitable for Flask backend apps; use Heroku, Render, Railway, or another Python hosting service instead.
