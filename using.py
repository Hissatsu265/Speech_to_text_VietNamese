# from transformers import pipeline

# pipe = pipeline(
#     'automatic-speech-recognition',
#     model='./whisper-vietnamese-finetuned/final',
#     device=0  # Dùng GPU
# )

# result = pipe('./test.wav')
# print(result['text'])
# ===================================================================
# import torch
# from transformers import pipeline
# import pandas as pd
# from tqdm import tqdm
# import jiwer

# # Load model
# print("📥 Loading model...")
# pipe = pipeline(
#     'automatic-speech-recognition',
#     model='./whisper-vietnamese-finetuned/final',
#     device=0 if torch.cuda.is_available() else -1
# )

# # Load test data
# print("📂 Loading test data...")
# test_df = pd.read_csv('test.csv')

# # Transcribe
# print(f"🎤 Transcribing {len(test_df)} files...")
# predictions = []
# references = []

# for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
#     try:
#         result = pipe(row['path'])
#         pred_text = result['text']
        
#         predictions.append(pred_text)
#         references.append(row['sentence'])

#         if idx < 5:
#             print(f"\n--- Example {idx+1} ---")
#             print(f"Ground truth: {row['sentence']}")
#             print(f"Prediction:   {pred_text}")
#     except Exception as e:
#         print(f"❌ Error at {row['path']}: {e}")
#         predictions.append("")
#         references.append(row['sentence'])

# wer = jiwer.wer(references, predictions)
# cer = jiwer.cer(references, predictions)

# print("\n" + "="*60)
# print("📊 KẾT QUẢ ĐÁNH GIÁ")
# print("="*60)
# print(f"WER (Word Error Rate):      {wer*100:.2f}%")
# print(f"CER (Character Error Rate): {cer*100:.2f}%")
# print(f"Accuracy (word-level):      {(1-wer)*100:.2f}%")
# print("="*60)

# results_df = pd.DataFrame({
#     'file': test_df['path'],
#     'ground_truth': references,
#     'prediction': predictions
# })
# results_df.to_csv('predictions.csv', index=False)
# print("\n💾 Saved predictions to: predictions.csv")

# print("\n🔍 PHÂN TÍCH LỖI:")
# errors = []
# for ref, pred in zip(references, predictions):
#     if ref != pred:
#         errors.append({
#             'reference': ref,
#             'prediction': pred,
#             'wer': jiwer.wer(ref, pred)
#         })

# if errors:
#     errors_df = pd.DataFrame(errors).sort_values('wer', ascending=False)
#     print(f"Tổng số lỗi: {len(errors)}/{len(references)} ({len(errors)/len(references)*100:.1f}%)")
#     print("\nTop 5 lỗi nặng nhất:")
#     for idx, row in errors_df.head(5).iterrows():
#         print(f"\n  Ref: {row['reference']}")
#         print(f"  Pred: {row['prediction']}")
#         print(f"  WER: {row['wer']*100:.1f}%")
# ====================comapre and analyze=================================================
import torch
from transformers import pipeline
import pandas as pd
from tqdm import tqdm
import jiwer

def evaluate_model(model_path, model_name, test_df):
    """
    Đánh giá một model trên tập test
    """
    print(f"\n{'='*60}")
    print(f"📥 Đang load model: {model_name}")
    print(f"{'='*60}")
    
    pipe = pipeline(
        'automatic-speech-recognition',
        model=model_path,
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Transcribe
    print(f"🎤 Đang transcribe {len(test_df)} files...")
    predictions = []
    references = []
    
    for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
        try:
            result = pipe(row['path'])
            pred_text = result['text']
            
            predictions.append(pred_text)
            references.append(row['sentence'])
            
            # In 5 ví dụ đầu
            if idx < 5:
                print(f"\n--- Example {idx+1} ---")
                print(f"Ground truth: {row['sentence']}")
                print(f"Prediction:   {pred_text}")
        except Exception as e:
            print(f"❌ Error at {row['path']}: {e}")
            predictions.append("")
            references.append(row['sentence'])
    
    # Tính metrics
    wer = jiwer.wer(references, predictions)
    cer = jiwer.cer(references, predictions)
    
    return {
        'predictions': predictions,
        'references': references,
        'wer': wer,
        'cer': cer,
        'model_name': model_name
    }

def print_results(results):
    """
    In kết quả đánh giá
    """
    print("\n" + "="*60)
    print(f"📊 KẾT QUẢ: {results['model_name']}")
    print("="*60)
    print(f"WER (Word Error Rate):      {results['wer']*100:.2f}%")
    print(f"CER (Character Error Rate): {results['cer']*100:.2f}%")
    print(f"Accuracy (word-level):      {(1-results['wer'])*100:.2f}%")
    print("="*60)

def compare_models(baseline_results, finetuned_results):
    """
    So sánh 2 models
    """
    print("\n" + "="*60)
    print("📈 SO SÁNH BASELINE vs FINE-TUNED")
    print("="*60)
    
    wer_improvement = (baseline_results['wer'] - finetuned_results['wer']) / baseline_results['wer'] * 100
    cer_improvement = (baseline_results['cer'] - finetuned_results['cer']) / baseline_results['cer'] * 100
    
    print(f"\n{'Metric':<25} {'Baseline':<15} {'Fine-tuned':<15} {'Improvement':<15}")
    print("-"*70)
    print(f"{'WER':<25} {baseline_results['wer']*100:>6.2f}%{'':<8} {finetuned_results['wer']*100:>6.2f}%{'':<8} {wer_improvement:>6.2f}%")
    print(f"{'CER':<25} {baseline_results['cer']*100:>6.2f}%{'':<8} {finetuned_results['cer']*100:>6.2f}%{'':<8} {cer_improvement:>6.2f}%")
    print(f"{'Accuracy':<25} {(1-baseline_results['wer'])*100:>6.2f}%{'':<8} {(1-finetuned_results['wer'])*100:>6.2f}%{'':<8}")
    
    if wer_improvement > 0:
        print(f"\n✅ Fine-tuning đã cải thiện WER {wer_improvement:.2f}%")
    else:
        print(f"\n❌ Fine-tuning không cải thiện (WER tệ hơn {abs(wer_improvement):.2f}%)")

# ============= MAIN =============

# Load test data
print("📂 Loading test data...")
test_df = pd.read_csv('test.csv')

# 1. Đánh giá BASELINE MODEL (Whisper Small gốc)
baseline_results = evaluate_model(
    model_path='openai/whisper-small',
    model_name='Whisper Small (Baseline)',
    test_df=test_df
)
print_results(baseline_results)

# Lưu kết quả baseline
baseline_df = pd.DataFrame({
    'file': test_df['path'],
    'ground_truth': baseline_results['references'],
    'prediction': baseline_results['predictions']
})
baseline_df.to_csv('predictions_baseline.csv', index=False)
print("\n💾 Saved baseline predictions to: predictions_baseline.csv")

# 2. Đánh giá FINE-TUNED MODEL
finetuned_results = evaluate_model(
    model_path='./whisper-vietnamese-finetuned-fixed/final',
    model_name='Whisper Small (Fine-tuned)',
    test_df=test_df
)
print_results(finetuned_results)

# Lưu kết quả fine-tuned
finetuned_df = pd.DataFrame({
    'file': test_df['path'],
    'ground_truth': finetuned_results['references'],
    'prediction': finetuned_results['predictions']
})
finetuned_df.to_csv('predictions_finetuned.csv', index=False)
print("\n💾 Saved fine-tuned predictions to: predictions_finetuned.csv")

# 3. SO SÁNH 2 MODELS
compare_models(baseline_results, finetuned_results)

# 4. PHÂN TÍCH LỖI CHI TIẾT
print("\n" + "="*60)
print("🔍 PHÂN TÍCH LỖI CHI TIẾT")
print("="*60)

for model_name, results in [
    ("Baseline", baseline_results),
    ("Fine-tuned", finetuned_results)
]:
    print(f"\n--- {model_name} ---")
    errors = []
    for ref, pred in zip(results['references'], results['predictions']):
        if ref != pred:
            errors.append({
                'reference': ref,
                'prediction': pred,
                'wer': jiwer.wer(ref, pred)
            })
    
    if errors:
        errors_df = pd.DataFrame(errors).sort_values('wer', ascending=False)
        print(f"Tổng số lỗi: {len(errors)}/{len(results['references'])} ({len(errors)/len(results['references'])*100:.1f}%)")
        print(f"\nTop 3 lỗi nặng nhất:")
        for idx, row in errors_df.head(3).iterrows():
            print(f"\n  Ref:  {row['reference']}")
            print(f"  Pred: {row['prediction']}")
            print(f"  WER:  {row['wer']*100:.1f}%")

# 5. Lưu bảng so sánh
comparison_df = pd.DataFrame({
    'file': test_df['path'],
    'ground_truth': baseline_results['references'],
    'baseline_prediction': baseline_results['predictions'],
    'finetuned_prediction': finetuned_results['predictions'],
})
comparison_df.to_csv('comparison.csv', index=False)
print(f"\n💾 Saved comparison to: comparison.csv")

print("\n✅ HOÀN THÀNH!")