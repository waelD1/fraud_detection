from fastapi import FastAPI, Request, Form, File
from fastapi.templating import Jinja2Templates
import pickle

app = FastAPI()
templates = Jinja2Templates(directory="./website")


@app.get("/")
async def read_items(request : Request):
    type_payment = ['Payment', 'Cash_in', 'Cash_out', 'Transfer', 'Debit']
    return templates.TemplateResponse('base.html', {"request" : request, 'type_of_payment' : type_payment})


