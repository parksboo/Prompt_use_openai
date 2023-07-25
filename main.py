import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse
import openai
import io

app = FastAPI()

api_key = ''
openai.api_key = api_key

# address = '../archive/twcs/twcs.csv'
# data = pd.read_csv(address, low_memory=False)

MODEL = "gpt-3.5-turbo"


@app.post("/predict", response_class=PlainTextResponse)
async def predict(file: UploadFile = File(...),  author_id: str = Form(...),  cmd_text: str = Form("please analyse")):
    csv_bytes = await file.read()
    csv_str = io.StringIO(csv_bytes.decode('utf-8'))
    df = pd.read_csv(csv_str)
    str_expr = f"author_id=='{author_id}'"
    data = df.query(str_expr)
    commend = f"text에 중점을 두고 해당 dataframe을 {cmd_text}\ndataframe : '{data}' "
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": commend},
        ],
        temperature=0,
    )
    # 이미지 불러오기 및 전처리
    ans = response['choices'][0]['message']['content']
    return f"commend : {commend}\n\nresponse: {ans}"


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return """
  <html>
  <body>

  <h2>Text data analyzing</h2>

    <form action="/predict" method="post" enctype="multipart/form-data">
      Select the csv file for analyse(limit 20MB) : 
        <input type="file" name="file" id="file">
        <br>
        command (ex. analyse, explain) :
        <input type="text" name="cmd_text" id="cmd_text">
        author_id :
        <input type="text" name="author_id" id="author_id">
        <br>
        <input type="submit" value="Upload CSV" name="submit">
    </form>

  </body>
  </html>
"""


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}
