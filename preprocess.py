import os
import re
import pandas as pd
from pydub import AudioSegment
from pydub.utils import mediainfo
from sklearn.model_selection import train_test_split
from tqdm import tqdm

METADATA_FILE ="/home/jupyter-toanlm/multitask/stt/transcriptAll.txt"
AUDIO_FOLDER = "/home/jupyter-toanlm/multitask/stt/mp3"
OUTPUT_FOLDER = "/home/jupyter-toanlm/multitask/stt/wav_16k"
OUTPUT_CSV_TRAIN = "/home/jupyter-toanlm/multitask/stt/train.csv"
OUTPUT_CSV_VAL = "/home/jupyter-toanlm/multitask/stt/validation.csv"
OUTPUT_CSV_TEST = "/home/jupyter-toanlm/multitask/stt/test.csv"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

print("Đọc file metadata...")
df = pd.read_csv(METADATA_FILE, sep="|", header=None, 
                 names=["filename", "sentence", "duration"])

print(f"Tổng số mẫu ban đầu: {len(df)}")

def normalize_text(text):
    """Loại bỏ ký tự đặc biệt, giữ tiếng Việt và số"""
    text = re.sub(r'[^a-zA-ZÀ-ỹ0-9\s]', '', text)
    text = ' '.join(text.split())
    # text = text.lower()  # Uncomment nếu dùng Wav2Vec2
    return text

df['sentence'] = df['sentence'].apply(normalize_text)

# Loại bỏ câu rỗng
df = df[df['sentence'].str.strip() != '']
print(f"Sau khi chuẩn hóa text: {len(df)} mẫu")

# --- 3. CONVERT AUDIO ---
def process_audio(row):
    """Convert MP3 to WAV 16kHz mono"""
    try:
        mp3_path = os.path.join(AUDIO_FOLDER, row['filename'])
        
        if not os.path.exists(mp3_path):
            return None
        
        wav_filename = row['filename'].replace(".mp3", ".wav")
        wav_path = os.path.join(OUTPUT_FOLDER, wav_filename)
        
        if not os.path.exists(wav_path):
            sound = AudioSegment.from_mp3(mp3_path)
            
            if len(sound) == 0:
                return None
                
            sound = sound.set_frame_rate(16000).set_channels(1)
            sound.export(wav_path, format="wav")
            
        return wav_path
    except Exception as e:
        print(f"\nLỗi {row['filename']}: {e}")
        return None

print("\nConvert audio sang WAV 16kHz...")
tqdm.pandas()
df['path'] = df.progress_apply(process_audio, axis=1)

# Loại bỏ file lỗi
df = df.dropna(subset=['path'])
print(f"Sau khi convert: {len(df)} mẫu")

# --- 4. KIỂM TRA ĐỘ DÀI AUDIO ---
def get_duration(audio_path):
    """Lấy độ dài audio (seconds)"""
    try:
        info = mediainfo(audio_path)
        return float(info['duration'])
    except:
        return 0

print("\nKiểm tra độ dài audio...")
df['audio_duration'] = df['path'].progress_apply(get_duration)

# Lọc audio quá ngắn hoặc quá dài
MIN_DURATION = 0.5  # 0.5 giây
MAX_DURATION = 30   # 30 giây
df = df[(df['audio_duration'] >= MIN_DURATION) & (df['audio_duration'] <= MAX_DURATION)]

print(f"Sau khi lọc độ dài ({MIN_DURATION}s - {MAX_DURATION}s): {len(df)} mẫu")

# --- 5. CHUẨN BỊ DATASET CUỐI CÙNG ---
final_df = df[['path', 'sentence']].copy()

# --- 6. CHIA TRAIN/VAL/TEST ---
# 80% train, 10% validation, 10% test
train_df, temp_df = train_test_split(final_df, test_size=0.2, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)

# Lưu ra CSV
train_df.to_csv(OUTPUT_CSV_TRAIN, index=False)
val_df.to_csv(OUTPUT_CSV_VAL, index=False)
test_df.to_csv(OUTPUT_CSV_TEST, index=False)

# --- 7. THỐNG KÊ ---
print("\n" + "="*50)
print("✅ HOÀN TẤT PREPROCESSING!")
print("="*50)
print(f"Tổng số mẫu: {len(final_df)}")
print(f"Train:      {len(train_df)} mẫu ({len(train_df)/len(final_df)*100:.1f}%)")
print(f"Validation: {len(val_df)} mẫu ({len(val_df)/len(final_df)*100:.1f}%)")
print(f"Test:       {len(test_df)} mẫu ({len(test_df)/len(final_df)*100:.1f}%)")
print(f"\nĐộ dài audio trung bình: {df['audio_duration'].mean():.2f}s")
print(f"Độ dài text trung bình: {final_df['sentence'].str.len().mean():.1f} ký tự")
print("\n📁 Files đã lưu:")
print(f"  - {OUTPUT_CSV_TRAIN}")
print(f"  - {OUTPUT_CSV_VAL}")
print(f"  - {OUTPUT_CSV_TEST}")
print("\n🚀 Sẵn sàng cho bước 2: Training!")