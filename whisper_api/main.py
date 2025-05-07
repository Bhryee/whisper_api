# main.py
from fastapi import FastAPI
import whisper
import uvicorn

app = FastAPI()
model = whisper.load_model("base")  # tiny de kullanabilirsin

@app.get("/")
def read_root():
    return {"message": "API çalışıyor"}

@app.post("/predict")
async def predict(audio: UploadFile, target: str = Form(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        tmp_path = tmp.name

    result = model.transcribe(tmp_path)
    spoken_text = result["text"].strip().lower()
    print(f"Kullanıcı söyledi: {spoken_text}")

    if target.lower() in spoken_text:
        return "doğru"
    else:
        return "yanlış"

# Render otomatik çalıştırır, aşağısı sadece lokal kullanım içindir
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)



