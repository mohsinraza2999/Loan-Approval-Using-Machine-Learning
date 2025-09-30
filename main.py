"""
#these lines are for cleaning and training model
#first move data LoanApprovalPrediction.csv to data/raw folder
from scripts.preprocessing import clean_data
from scripts.triaining import train
def runall():
    clean_data()
    train()

if __name__ == "__main__":
    runall()"""

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from scripts.predict import load_model

app = FastAPI()
templates = Jinja2Templates(directory="frontend")

@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/form", response_class=HTMLResponse)
async def form_page(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/submit", response_class=HTMLResponse)
async def submit_form(
    request: Request,
    feature1: str = Form(...),
    feature2: str = Form(...),
    feature3: str = Form(...),
    feature4: str = Form(...),
    feature5: str = Form(...),
    feature6: str = Form(...),
    feature7: str = Form(...),
    feature8: str = Form(...),
    feature9: str = Form(...)
):
   
    get_data = [0,feature1,feature2,feature3,feature4,feature5,feature6,feature7,feature8,feature9]
    message=load_model(get_data)
    
    return templates.TemplateResponse("result.html", {"request": request, "prediction": message})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
