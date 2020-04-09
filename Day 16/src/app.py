import numpy as np
from flask import Flask
from flask import render_template,request
from sklearn.externals import joblib
from forms import CarForm
from config import Config


app = Flask("Car Price Prediction")
app.config.from_object(Config)


@app.route("/", methods=["GET"])
def home():
    car_form = CarForm()
    if car_form.validate_on_submit():
        car_age = int(car_form.car_age.data)
        car_fuel = car_form.car_fuel.data
        car_doors = int(car_form.car_doors.data)
        car_cc = int(car_form.car_cc.data)
        car_horsepower = int(car_form.car_horsepower.data)
        car_transmission = car_form.car_transmission.data
        car_odometer = int(car_form.car_odometer.data)
        car_weight = car_form.car_weight.data
        car_color = car_form.car_color.data
    return render_template("car.html", form=car_form)


@app.route("/", methods=["POST"])
def result():
    try:
        form = request.form
        model = joblib.load("mlmodel/car_price_prediction.pkl")
        if int(form['car_fuel']) == 1:
            fuel = 1
        else:
            fuel = 0
        if int(form['car_transmission']) == 1:
            car_transmission = 1
        else:
            car_transmission = 0
        if int(form['car_color']) == 1:
            car_color = 0
        else:
            car_color = 1
        new_car = np.array(
            [int(form['car_odometer']), fuel, int(form['car_doors']), car_transmission,
             int(form['car_horsepower']), car_color, int(form['car_cc']), int(form['car_weight']),
             int(form['car_age'])]).reshape(1, -1)
        predicted_price = model.predict(new_car)
        if predicted_price < 0:
            predicted_price = 0
        return render_template("result.html", price=int(predicted_price))
    except ValueError:
        return render_template("error.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9090, debug=True)
