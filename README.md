# YouTube Comments Sentiment & Category Classification

Dự án NLP hoàn chỉnh để dự đoán đồng thời:

- `Sentiment`: `Positive`, `Neutral`, `Negative`
- `CategoryID`: bài toán multi-class classification

Dataset đầu vào mặc định là `youtube-comments-sentiment.csv`, có header và các cột chính:

- `CommentText`: văn bản bình luận, feature chính
- `Sentiment`: nhãn sentiment
- `CategoryID`: nhãn category
- Feature phụ nếu có: `VideoTitle`, `Likes`, `Replies`, `CountryCode`, `PublishedAt`

Link nguồn dữ liệu tham khảo: <https://huggingface.co/datasets/vnkat/youtube-comment-sentiment/tree/main>

## Kiến Trúc Dự Án

```text
project/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
│   ├── config/
│   ├── preprocessing/
│   ├── feature_engineering/
│   ├── training/
│   ├── evaluation/
│   ├── inference/
│   └── utils/
├── outputs/
│   ├── figures/
│   ├── metrics/
│   └── models/
├── README.md
├── requirements.txt
└── main.py
```

## Quy Trình NLP

Pipeline gồm các bước độc lập:

1. **Validate**: kiểm tra schema, missing values, duplicate, label bất thường, likes/replies âm, ngày không parse được.
2. **Preprocess**: clean text, xử lý emoji/slang/repeated characters, tạo feature phụ, split train/validation/test.
3. **EDA**: vẽ phân phối lớp, word frequency, word cloud, độ dài comment, quốc gia, thời gian.
4. **Train baseline**: TF-IDF + metadata, huấn luyện riêng model Sentiment và model CategoryID.
5. **Train transformer**: shared encoder transformer với 2 head: sentiment head và category head.
6. **Evaluate**: đánh giá trên test set, sinh metrics, confusion matrix, error examples.
7. **Inference**: dự đoán dữ liệu mới bằng model baseline đã lưu.

## Phương Pháp

### Approach 1: Model Riêng

File: `src/training/baseline.py`

- Text embedding: `TfidfVectorizer`
- Feature phụ: numeric scaling cho `Likes`, `Replies`, length/time features
- Feature categorical: one-hot cho `CountryCode`
- Model: `LogisticRegression`
- Huấn luyện 2 pipeline riêng:
  - `sentiment`
  - `category`

### Approach 2: Multi-task Transformer

File: `src/training/transformer_multitask.py`

- Encoder dùng chung: mặc định `distilbert-base-multilingual-cased`
- Head 1: phân loại `Sentiment`
- Head 2: phân loại `CategoryID`
- Loss: tổng cross-entropy của hai task
- Có thể đổi sang PhoBERT trong `src/config/default.yaml`, ví dụ `vinai/phobert-base`, nếu dữ liệu chủ yếu tiếng Việt.

## Data Preprocessing

Text cleaning trong `src/preprocessing/text_cleaning.py` gồm:

- lowercase
- remove URL
- remove HTML tag
- normalize whitespace
- emoji to text alias nếu có package `emoji`
- repeated character normalization
- social slang replacement
- remove special characters

Feature engineering thêm:

- `clean_comment`
- `clean_title`
- `model_text`
- `comment_length`
- `word_count`
- `title_length`
- `published_hour`
- `published_dayofweek`
- `published_month`

## Cài Đặt

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu chỉ chạy baseline, các thư viện nặng như `torch` và `transformers` chỉ cần thiết khi chạy:

```bash
python main.py --step train_transformer
```

## Cách Chạy

Chạy từng bước:

```bash
python main.py --step validate
python main.py --step preprocess
python main.py --step eda
python main.py --step train_baseline
python main.py --step train_transformer
python main.py --step evaluate
python main.py --step inference
```

Chạy toàn bộ pipeline:

```bash
python main.py --step all
```

Inference với file CSV mới:

```bash
python main.py --step inference --input new_comments.csv --output outputs/metrics/new_predictions.csv
```

Inference bằng transformer sau khi đã chạy `train_transformer`:

```bash
python main.py --step inference --model transformer --input new_comments.csv
```

File inference cần tối thiểu cột `CommentText`. Các cột `VideoTitle`, `Likes`, `Replies`, `CountryCode`, `PublishedAt` là optional.

## Config

Toàn bộ path, feature, model hyperparameter nằm trong:

```text
src/config/default.yaml
```

Các path được resolve theo thư mục project hiện tại. Dataset mặc định đọc từ:

```text
youtube-comments-sentiment.csv
```

Nếu muốn chạy nhanh để học hoặc demo, chỉnh:

```yaml
data:
  sample_size: 50000
```

Transformer cũng có giới hạn riêng:

```yaml
transformer:
  max_train_samples: 5000
  max_eval_samples: 1000
```

## Outputs

Tất cả output được lưu trong `outputs/`.

### Metrics

```text
outputs/metrics/
├── validation_report.json
├── word_frequency.csv
├── baseline_validation_metrics.json
├── baseline_test_metrics.json
├── model_comparison.csv
├── sentiment_error_examples.csv
├── category_error_examples.csv
├── sentiment_tfidf_feature_importance.csv
└── category_tfidf_feature_importance.csv
```

Metrics chính:

- Sentiment: Accuracy, Precision, Recall, F1, Confusion Matrix
- Category: Accuracy, Macro F1, Top-k Accuracy

### Figures

```text
outputs/figures/
├── sentiment_distribution.png
├── category_distribution_top30.png
├── country_distribution_top30.png
├── comment_word_count_distribution.png
├── published_hour_distribution.png
├── word_frequency_top40.png
├── wordcloud.png
├── baseline_sentiment_confusion_matrix.png
├── baseline_category_confusion_matrix.png
├── transformer_training_curve.png
└── transformer_*_confusion_matrix.png
```

### Models

```text
outputs/models/
├── baseline_models.joblib
└── transformer_multitask/
```

## Error Analysis

Sau khi chạy evaluate, các file sau chứa ví dụ dự đoán sai để phân tích:

```text
outputs/metrics/sentiment_error_examples.csv
outputs/metrics/category_error_examples.csv
```

Nên xem các case này để phát hiện:

- comment mỉa mai/sarcasm
- emoji bị mojibake
- title gây nhiễu
- category hiếm
- comment quá ngắn

## Hướng Cải Tiến

- Dùng language detection để tách tiếng Việt/tiếng Anh.
- Thêm normalization cho mojibake emoji như `ðŸ˜‚`.
- Dùng PhoBERT hoặc XLM-R cho dữ liệu đa ngôn ngữ.
- Tối ưu threshold/top-k theo business goal.
- Thêm cross-validation cho baseline.
- Dùng class weighting hoặc focal loss cho category mất cân bằng.
- Deploy inference bằng FastAPI hoặc batch scoring job.
