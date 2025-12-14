import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
torch.cuda.empty_cache()
import gc
gc.collect()
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset
from transformers import (
    WhisperProcessor, 
    WhisperForConditionalGeneration, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer
)
import evaluate

# --- CẤU HÌNH ---
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "Vietnamese"
TASK = "transcribe"
OUTPUT_DIR = "./whisper-vietnamese-finetuned"
MAX_AUDIO_DURATION = 30.0

# WandB config
USE_WANDB = True
WANDB_PROJECT = "whisper-vietnamese"
WANDB_RUN_NAME = "whisper-small-v1"

if USE_WANDB:
    try:
        import wandb
        wandb.login()
        print("✅ WandB đã kết nối!")
    except Exception as e:
        print(f"⚠️  WandB không khả dụng: {e}")
        USE_WANDB = False

# Kiểm tra GPU
if torch.cuda.is_available():
    device = torch.cuda.current_device()
    print(f"🖥️  Đang dùng GPU: {torch.cuda.get_device_name(device)}")
    print(f"💾 VRAM: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
else:
    print("⚠️  Không tìm thấy GPU, sẽ dùng CPU (rất chậm!)")

# --- 1. LOAD DATASET ---
print("\n📂 Đang load dataset...")
data_files = {
    "train": "train.csv",
    "validation": "validation.csv",
    "test": "test.csv"
}

for split, file in data_files.items():
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Không tìm thấy file: {file}")

data = load_dataset("csv", data_files=data_files)
print(f"✅ Dataset loaded:")
print(f"  - Train: {len(data['train'])} mẫu")
print(f"  - Validation: {len(data['validation'])} mẫu")
print(f"  - Test: {len(data['test'])} mẫu")

# --- 2. LOAD PROCESSOR ---
print(f"\n🔧 Đang load processor từ {MODEL_NAME}...")
processor = WhisperProcessor.from_pretrained(
    MODEL_NAME, 
    language=LANGUAGE, 
    task=TASK
)

# --- 3. PREPROCESSING ---
def prepare_dataset(batch):
    """Chuyển audio → mel-spectrogram và text → token IDs"""
    try:
        audio_path = batch["path"]
        audio_array, sampling_rate = librosa.load(audio_path, sr=16000, mono=True)
        
        duration = len(audio_array) / sampling_rate
        if duration > MAX_AUDIO_DURATION or duration < 0.5:
            return {"input_features": None, "labels": None}
        
        input_features = processor.feature_extractor(
            audio_array, 
            sampling_rate=sampling_rate
        ).input_features[0]
        
        labels = processor.tokenizer(batch["sentence"]).input_ids
        
        return {
            "input_features": input_features,
            "labels": labels
        }
    except Exception as e:
        print(f"❌ Lỗi file {batch['path']}: {e}")
        return {"input_features": None, "labels": None}

print("\n⚙️  Đang preprocess dataset...")

for split in ["train", "validation", "test"]:
    print(f"   Processing {split}...")
    original_len = len(data[split])
    
    data[split] = data[split].map(
        prepare_dataset, 
        remove_columns=data[split].column_names,
        num_proc=1,
        desc=f"Processing {split}"
    )
    
    data[split] = data[split].filter(
        lambda x: x["input_features"] is not None and x["labels"] is not None
    )
    
    removed = original_len - len(data[split])
    if removed > 0:
        print(f"   ⚠️  Removed {removed} invalid samples")

print(f"\n✅ Preprocessing hoàn tất!")
print(f"  - Train: {len(data['train'])} mẫu")
print(f"  - Validation: {len(data['validation'])} mẫu")
print(f"  - Test: {len(data['test'])} mẫu")

# --- 4. DATA COLLATOR ---
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        
        batch["labels"] = labels
        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# --- 5. METRICS ---
metric = evaluate.load("wer")

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}

# --- 6. LOAD MODEL (SỬA LỖI TẠI ĐÂY) ---
print(f"\n🤖 Đang load model {MODEL_NAME}...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)

# ✅ CẤU HÌNH ĐỂ TRÁNH LỖI BACKWARD
model.config.forced_decoder_ids = None
model.config.suppress_tokens = []
model.config.use_cache = False
model.generation_config.use_cache = False

# ✅ Tắt dropout để tương thích với gradient checkpointing
if hasattr(model.config, 'dropout'):
    model.config.dropout = 0.0
if hasattr(model.config, 'attention_dropout'):
    model.config.attention_dropout = 0.0

print(f"✅ Model loaded: {model.num_parameters():,} parameters")

# --- 7. TRAINING ARGUMENTS ---
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    
    # Batch size
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    
    # Learning rate
    learning_rate=1e-5,
    warmup_ratio=0.1,
    
    # Training duration
    num_train_epochs=3,
    
    # Memory optimization
    gradient_checkpointing=True,
    fp16=True,
    
    # ✅ QUAN TRỌNG: Thêm option này để tránh lỗi
    gradient_checkpointing_kwargs={"use_reentrant": False},
    
    # Evaluation & Logging
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=25,
    save_total_limit=2,
    
    # Best model
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    
    # Generation
    predict_with_generate=True,
    generation_max_length=225,
    
    # Logging
    report_to=["wandb"] if USE_WANDB else ["tensorboard"],
    run_name=WANDB_RUN_NAME if USE_WANDB else None,
    
    # Misc
    push_to_hub=False,
    resume_from_checkpoint=True,
    dataloader_num_workers=2,
    remove_unused_columns=False,
)

# --- 8. TRAINER ---
trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=data["train"],
    eval_dataset=data["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

# --- 9. TRAINING ---
print("\n🚀 Bắt đầu training...")
print("="*60)
if USE_WANDB:
    print(f"📊 Xem training tại: https://wandb.ai")
else:
    print(f"📊 Xem TensorBoard: tensorboard --logdir {OUTPUT_DIR}")
print("="*60)

try:
    trainer.train()
    print("\n✅ Training hoàn tất!")
except KeyboardInterrupt:
    print("\n⚠️  Training bị dừng bởi người dùng")
    print("   Đang lưu checkpoint...")
    trainer.save_model(f"{OUTPUT_DIR}/interrupted")
except Exception as e:
    print(f"\n❌ Lỗi training: {e}")
    raise

# --- 10. ĐÁNH GIÁ TRÊN TEST SET ---
print("\n📊 Đánh giá trên test set...")
test_results = trainer.evaluate(data["test"])
print(f"Test WER: {test_results['eval_wer']:.2f}%")

# --- 11. LƯU MODEL ---
print(f"\n💾 Đang lưu model...")
final_model_dir = f"{OUTPUT_DIR}/final"
model.save_pretrained(final_model_dir)
processor.save_pretrained(final_model_dir)

print("\n" + "="*60)
print(f"✅ HOÀN TẤT!")
print(f"📁 Model: {final_model_dir}")
print(f"📈 Test WER: {test_results['eval_wer']:.2f}%")
if USE_WANDB:
    print(f"📊 WandB: https://wandb.ai/{WANDB_PROJECT}")
print("="*60)

print("\n🎯 Cách sử dụng model:")
print(f"from transformers import pipeline")
print(f"pipe = pipeline('automatic-speech-recognition', model='{final_model_dir}')")
print(f"result = pipe('audio.wav')")

if USE_WANDB:
    wandb.finish()