from flask import Flask, render_template, request 
from datetime import datetime
import model as m
app =Flask(__name__)
@app.route("/", methods = ["GET", "POST"])
def hello():
    models = None
    if request.method == "POST":
        no_of_adults = int(request.form["no_of_adults"])
        no_of_children = int(request.form["no_of_children"])
        no_of_weekend_nights = int(request.form["no_of_weekend_nights"])
        no_of_week_nights = int(request.form["no_of_week_nights"])
        type_of_meal_plan = int(request.form["type_of_meal_plan"])
        required_car_parking_space = int(request.form["required_car_parking_space"])
        room_type_reserved = int(request.form["room_type_reserved"])
        lead_time = int(request.form["lead_time"])
        arrival_date_obj = request.form["arrival_date_obj"]
        market_segment_type = int(request.form["market_segment_type"])
        repeated_guest = int(request.form["repeated_guest"])
        no_of_previous_cancellations = int(request.form["no_of_previous_cancellations"])
        no_of_previous_bookings_not_canceled = int(request.form["no_of_previous_bookings_not_canceled"])
        avg_price_per_room = float(request.form["avg_price_per_room"])
        no_of_special_requests = int(request.form["no_of_special_requests"])

        # extract year, month, and date values from arrival_date
        arrival_date_obj = datetime.strptime(arrival_date_obj, '%Y-%m-%d')
        arrival_year = arrival_date_obj.year
        arrival_month = arrival_date_obj.month
        arrival_date = arrival_date_obj.day

        # pass all the features to the model for prediction
        models = m.model_predict(no_of_adults, no_of_children, no_of_weekend_nights, 
                                no_of_week_nights, type_of_meal_plan, required_car_parking_space,
                                room_type_reserved, lead_time, arrival_year, arrival_month, 
                                arrival_date, market_segment_type, repeated_guest, 
                                no_of_previous_cancellations, no_of_previous_bookings_not_canceled, 
                                avg_price_per_room, no_of_special_requests)
        
    return render_template("index.html", my_marks = models)

if __name__ == "__main__":
    app.run()

