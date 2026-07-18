# Weather Temperature Prediction — Django + LSTM

A Django web application that forecasts temperature in 3-hour steps using an
LSTM model trained on historical weather data.

🔗 **Live demo:** [djangowebapp-mu.vercel.app](https://djangowebapp-mu.vercel.app)

![prediction plot](prediction_plot.png)

## What it does

The user requests a forecast through the web interface. The app loads the
trained LSTM, applies the saved scaler to the input window, and returns
predicted temperatures for the next 3-hour steps.

The scaler is persisted alongside the model so that inference applies exactly
the same normalisation as training — a common source of silent errors when a
model is served outside the notebook it was trained in.

Built during a project at 3D Smart Factory.

## Stack

Python · Django · TensorFlow/Keras · pandas · NumPy · scikit-learn · HTML/CSS
Deployed on Vercel.

## Repository structure

| Path | Description |
|---|---|
| `PFA_3dSmartFactory/` | Django project settings, URLs, WSGI |
| `predict_meteo/` | Django app — views, templates, prediction logic |
| `static/images/` | Static assets |
| `data_normalized.csv` | Preprocessed training data |
| `lstm_model.h5` | Trained Keras model |
| `scaler.pkl` | Fitted scaler, required at inference |
| `prediction_plot.png` | Predicted vs actual temperature |
| `requirements.txt` | Python dependencies |
| `manage.py` | Django entry point |

## Running locally

```bash
git clone https://github.com/cdywolf/Django_Web_App_For_Weather_Prediction
cd Django_Web_App_For_Weather_Prediction

python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python manage.py migrate
python manage.py runserver
```

Then open `http://127.0.0.1:8000`.

## Limitations

- Trained on a single location; the model does not generalise to other climates.
- Forecast quality degrades beyond a few steps ahead, as errors compound when
  predictions are fed back as inputs.
- No confidence intervals — the app returns a point estimate with no measure of
  uncertainty.
- The model is not retrained on new data, so performance drifts over time.
- Both `.h5` and `.pkl` copies of the model are committed; only one is needed.
