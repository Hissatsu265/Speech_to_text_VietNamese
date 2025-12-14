

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import numpy as np
from collections import deque
import librosa

app = FastAPI(title="Vietnamese STT Real-time API")

# CORS để frontend có thể kết nối
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LOAD MODEL ---
print("🔄 Đang load model...")
MODEL_PATH = "./whisper-vietnamese-finetuned/final"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained(MODEL_PATH)
model = WhisperForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)

# ✅ CẤU HÌNH CHỐNG LẶP TỪ
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []

print(f"✅ Model loaded on: {device}")

# --- BUFFER AUDIO ---
class AudioBuffer:
    """Buffer để tích lũy audio chunks trước khi transcribe"""
    def __init__(self, min_duration=2.0, sample_rate=16000):
        self.buffer = deque(maxlen=int(30 * sample_rate))  # Max 30s
        self.min_samples = int(min_duration * sample_rate)  # Min 2s
        self.sample_rate = sample_rate
    
    def add(self, audio_chunk):
        """Thêm chunk vào buffer"""
        self.buffer.extend(audio_chunk)
    
    def get_audio(self):
        """Lấy toàn bộ audio trong buffer"""
        if len(self.buffer) >= self.min_samples:
            return np.array(self.buffer, dtype=np.float32)
        return None
    
    def clear(self):
        """Xóa buffer"""
        self.buffer.clear()

@app.get("/")
def read_root():
    return {
        "status": "✅ Server đang chạy",
        "model": MODEL_PATH,
        "device": device
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "device": device}

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("📞 Client connected!")
    
    audio_buffer = AudioBuffer(min_duration=1.5, sample_rate=16000)
    
    try:
        while True:
            # 1. Nhận audio chunk từ client
            data = await websocket.receive_bytes()
            
            # 2. Chuyển bytes → numpy array
            audio_chunk = np.frombuffer(data, dtype=np.float32)
            
            # 3. Thêm vào buffer
            audio_buffer.add(audio_chunk)
            
            # 4. Nếu đủ dữ liệu → transcribe
            audio_array = audio_buffer.get_audio()
            
            if audio_array is not None:
                try:
                    # Chuẩn hóa audio
                    audio_array = librosa.util.normalize(audio_array)
                    
                    # Extract features
                    input_features = processor(
                        audio_array, 
                        sampling_rate=16000, 
                        return_tensors="pt"
                    ).input_features.to(device)
                    
                    # Generate với config chống lặp
                    with torch.no_grad():
                        predicted_ids = model.generate(
                            input_features,
                            max_new_tokens=128,
                            num_beams=5,
                            repetition_penalty=1.2,
                            no_repeat_ngram_size=3,
                            temperature=0.0,
                        )
                    
                    # Decode
                    transcription = processor.batch_decode(
                        predicted_ids, 
                        skip_special_tokens=True
                    )[0].strip()
                    
                    # Gửi kết quả
                    if transcription:
                        print(f"🎤 Transcription: {transcription}")
                        await websocket.send_json({
                            "text": transcription,
                            "status": "success"
                        })
                        
                        # Clear buffer sau khi transcribe
                        audio_buffer.clear()
                
                except Exception as e:
                    print(f"❌ Error during transcription: {e}")
                    await websocket.send_json({
                        "text": "",
                        "status": "error",
                        "message": str(e)
                    })
    
    except WebSocketDisconnect:
        print("📴 Client disconnected")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    # ✅ Ngrok setup
    try:
        from pyngrok import ngrok
        
        API_PORT = 8000
        
        # Tạo public URL với ngrok
        public_url = ngrok.connect(
            API_PORT, 
            "http",
            # ✅ Thay domain của bạn vào đây (hoặc bỏ dòng này để dùng URL random)
            # domain="your-domain.ngrok-free.app"
        )
        
        print("\n" + "="*60)
        print("🌐 NGROK PUBLIC URL")
        print("="*60)
        print(f"🔗 {public_url}")
        print(f"📝 WebSocket: {public_url.replace('http', 'ws')}/ws/transcribe")
        print("="*60 + "\n")
        
    except ImportError:
        print("⚠️  pyngrok not installed. Install: pip install pyngrok")
        print("   Running on localhost only...")
    except Exception as e:
        print(f"⚠️  Ngrok error: {e}")
        print("   Running on localhost only...")
    
    # Chạy server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )