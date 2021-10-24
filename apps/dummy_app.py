from fastapi import FastAPI
from starlette.responses import RedirectResponse

app = FastAPI()


@app.get('/', include_in_schema=False)
async def index():
    return RedirectResponse('/docs')


@app.get('/patient_barcode')
async def patient_barcode(patient_barcode: str):
    return patient_barcode
