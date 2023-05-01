import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

def model_predict(no_of_adults, no_of_children, no_of_weekend_nights, 
                  no_of_week_nights, type_of_meal_plan, required_car_parking_space,
                 room_type_reserved, lead_time, arrival_year, arrival_month, arrival_date,
                   market_segment_type, repeated_guest, no_of_previous_cancellations,
                     no_of_previous_bookings_not_canceled, avg_price_per_room, no_of_special_requests):
    train = pd.read_csv("train.csv")
    #drop duplicated value
    t = train.drop(['booking_status', 'id'], axis =1)
    t.drop_duplicates(keep=False, inplace = True)
    train = train.loc[t.index]

    X = train.drop(['booking_status','id'], axis=1)
    y = train['booking_status']

    X_train , X_test , y_train , y_test = train_test_split(X, y, random_state = 42 , test_size =0.20, stratify = y)
    #split data
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    data = {
    "no_of_adults": [no_of_adults],
    "no_of_children": [no_of_children],
    "no_of_weekend_nights": [no_of_weekend_nights],
    "no_of_week_nights": [no_of_week_nights],
    "type_of_meal_plan": [type_of_meal_plan],
    "required_car_parking_space": [required_car_parking_space],
    "room_type_reserved": [room_type_reserved],
    "lead_time": [lead_time],
    "arrival_year": [arrival_year],
    "arrival_month": [arrival_month],
    "arrival_date": [arrival_date],
    "market_segment_type": [market_segment_type],
    "repeated_guest": [repeated_guest],
    "no_of_previous_cancellations": [no_of_previous_cancellations],
    "no_of_previous_bookings_not_canceled": [no_of_previous_bookings_not_canceled],
    "avg_price_per_room": [avg_price_per_room],
    "no_of_special_requests": [no_of_special_requests]
}
       

    df = pd.DataFrame(data, columns=data.keys())
    
    return "{:.2%}".format(xgb.predict_proba(df)[:, 1][0])

