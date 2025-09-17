from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("battery_health_model.joblib")

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def index():
    prediction = None
    input_summary = {}
    bar_color = ""
    status_msg = ""
    bar_width = "0%"
    battery_class = ""

    if request.method == 'POST':
        # Collect input values from form
        device_type = request.form['device_type']
        battery_capacity = int(request.form['battery_capacity'])
        charge_cycles = int(request.form['charge_cycles'])
        avg_temp = float(request.form['avg_temp'])
        charge_time = float(request.form['charge_time'])
        fast_charging = request.form['fast_charging']
        screen_on_time = float(request.form['screen_on_time'])
        age_months = int(request.form['age_months'])

        # Encode inputs
        device_type_map = {'Smartphone': 0, 'Tablet': 1, 'Laptop': 2, 'Other': 3}
        fast_charging_val = 1 if fast_charging == 'Yes' else 0
        device_type_val = device_type_map.get(device_type, 0)

        # Create input array
        features = np.array([[device_type_val, battery_capacity, charge_cycles,
                              avg_temp, charge_time, fast_charging_val,
                              screen_on_time, age_months]])
        prediction = model.predict(features)[0]
        bar_width = f"{prediction:.2f}%"

        # Decision logic
        if prediction >= 80:
            bar_color = "green"
            status_msg = "✅ Your battery is in excellent condition. Keep using it wisely!"
            battery_class = "high-battery flash-high"
        elif prediction >= 50:
            bar_color = "orange"
            status_msg = "⚠️ Your battery is in moderate condition. Consider optimizing usage."
            battery_class = "medium-battery"
        else:
            bar_color = "red"
            status_msg = "🔻 Your battery health is poor. You may need a replacement soon."
            battery_class = "low-battery flash-low"

        # Show input summary
        input_summary = {
            "Device Type": device_type,
            "Battery Capacity (mAh)": battery_capacity,
            "Charge Cycles": charge_cycles,
            "Avg. Temperature (°C)": avg_temp,
            "Avg. Charge Time (hrs)": charge_time,
            "Fast Charging": fast_charging,
            "Screen On Time (hrs)": screen_on_time,
            "Device Age (months)": age_months
        }

    return render_template("index.html",
                           prediction=prediction,
                           input_summary=input_summary,
                           bar_color=bar_color,
                           status_msg=status_msg,
                           bar_width=bar_width,
                           battery_class=battery_class)

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
