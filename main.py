from fastapi import FastAPI, Request, Form, File
from fastapi.templating import Jinja2Templates
import pickle
import datetime
from datetime import datetime
import pandas as pd
from sklearn.preprocessing import RobustScaler

app = FastAPI()
templates = Jinja2Templates(directory="./website")


@app.get("/")
async def read_items(request : Request):
    type_payment = ['Payment', 'Cash_in', 'Cash_out', 'Transfer', 'Debit']
    return templates.TemplateResponse('base.html', {"request" : request, 'type_of_payment' : type_payment})


@app.post("/result")
async def prediction(request : Request,
                    type_payment : str = Form(...),
                    Amount : int = Form(...),
                    balance_old_sender : int = Form(...),
                    balance_new_sender: int = Form(...),
                    balance_old_receiver: int = Form(...),
                    balance_new_receiver: int = Form(...),
                    transaction_date: datetime = Form(...)
                    ):
        # Creating the variables
        type_of_payment = type_payment
        amount_transaction = Amount
        balance_old_sender = balance_old_sender
        balance_new_sender = balance_new_sender
        balance_old_receiver = balance_old_receiver
        balance_new_receiver = balance_new_receiver
        transaction_date_days = (transaction_date.day)*24

        print(transaction_date)
        print(transaction_date_days)




        # Creating binary variables to get the result of the type of payment
        payment = 0
        cash_in = 0
        cash_out = 0 
        transfer = 0 
        debit = 0

        if type_of_payment == "Payment":
            payment = 1
        elif type_of_payment == "Cash_in":
            cash_in = 1
        elif type_of_payment == "Cash_out":
            cash_out = 1
        elif type_of_payment == "transfer":
            transfer = 1
        elif type_of_payment == "debit":
            debit = 1
        
        if balance_old_sender == 0:
            mean_rate_of_change_orig = 0
        else:
            mean_rate_of_change_orig = round((balance_new_sender - balance_old_sender)/balance_old_sender)
        if balance_old_receiver == 0:
            mean_rate_of_change_dest = 0 
        else:
            mean_rate_of_change_dest = round((balance_new_receiver - balance_old_receiver)/balance_old_receiver)


        features = {
            'step' : transaction_date_days,
            'amount' : amount_transaction,
            'oldbalanceOrg'	:balance_old_sender,
            'newbalanceOrig' : balance_new_sender,
            'oldbalanceDest' : 	balance_old_receiver,
            'newbalanceDest' : 	balance_new_receiver,
            'mean_rate_of_change_orig' : mean_rate_of_change_orig,
            'mean_rate_of_change_dest' :mean_rate_of_change_dest,
            'type_payment_CASH_IN' : cash_in,
            'type_payment_CASH_OUT' : cash_out,
            'type_payment_DEBIT' : debit,
            'type_payment_PAYMENT' : payment,
            'type_payment_TRANSFER' : transfer
        }

        # Put the data into the right format
        data = pd.DataFrame(features, index=[0])

        # Scaling the data
        #load scaler
        with open('scaler_model.pkl', 'rb') as f:
            loaded_scaler = pickle.load(f)

        scaled_features = loaded_scaler.transform(data)
        scaled_data = pd.DataFrame(scaled_features, index=data.index, columns=data.columns)

        
        #load model
        with open('credit_fraud_model.pkl', 'rb') as f:
            loaded_model = pickle.load(f)

        prediction = int(loaded_model.predict(scaled_data))
        if prediction == 1:
            result_pred = 'Fraudulent'
        else:
            result_pred = 'Regular'

        return templates.TemplateResponse("result.html", {'request' : request, 'prediction' : result_pred})