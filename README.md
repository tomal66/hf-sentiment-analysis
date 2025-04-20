```markdown
# Bi‑LSTM Sentiment Analysis

A TensorFlow/Keras implementation of a multi‑class sentiment analysis model using a bidirectional LSTM on the [“multiclass-sentiment-analysis-dataset”](https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset) from Hugging Face.

---

## 📦 Requirements

- Python 3.7+
- `tensorflow`
- `numpy`
- `pandas`
- `matplotlib`
- `huggingface_hub`
- `fsspec`
- `s3fs`

Install the dependencies:

```bash
pip install tensorflow numpy pandas matplotlib \
            huggingface_hub fsspec s3fs
```

---

## 🔧 Project Structure

```
.
├── notebook.ipynb         # Jupyter notebook with all steps
├── sentiment_bilstm_savedmodel/
│   └── …                  # SavedModel directory
├── sentiment_bilstm_best.h5
├── tokenizer.json
└── README.md              # This file
```

---

## 🚀 Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/tomal66/hf-sentiment-analysis
   cd sentiment-bilstm
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**  
   ```bash
   jupyter notebook notebook.ipynb
   ```

---

## 📝 Notebook Workflow

1. **Install & import**  
   - `huggingface_hub` for `hf://` protocol support  
   - `fsspec` & `s3fs` for remote CSV loading  

2. **Load dataset splits**  
   ```python
   DATASET_ROOT = "hf://datasets/Sp1786/multiclass-sentiment-analysis-dataset/"
   train_df = pd.read_csv(DATASET_ROOT + "train_df.csv")
   val_df   = pd.read_csv(DATASET_ROOT + "val_df.csv")
   test_df  = pd.read_csv(DATASET_ROOT + "test_df.csv")
   ```

3. **Text cleaning**  
   - Strip HTML tags  
   - Remove non‑alphabet characters  
   - Lowercase & trim  

4. **Tokenization & padding**  
   ```python
   tokenizer = Tokenizer(num_words=20000, oov_token="<OOV>")
   tokenizer.fit_on_texts(train_df["text"])
   X_train = pad_sequences(tokenizer.texts_to_sequences(train_df["text"]),
                           maxlen=200, padding="post", truncating="post")
   ```
  
5. **Model architecture**  
   ```python
   model = Sequential([
       Embedding(20000, 128, input_length=200),
       Bidirectional(LSTM(64)),
       Dropout(0.5),
       Dense(32, activation='relu', kernel_regularizer=l2(0.12)),
       BatchNormalization(),
       Dropout(0.6),
       Dense(3, activation='softmax')
   ])
   model.compile(
       optimizer=Adam(learning_rate=0.003),
       loss='sparse_categorical_crossentropy',
       metrics=['accuracy']
   )
   ```

6. **Training**  
   ```python
   history = model.fit(
       X_train, y_train,
       epochs=10, batch_size=128,
       validation_data=(X_val, y_val)
   )
   ```

7. **Visualize performance**  
   ```python
   plt.plot(history.history['accuracy'], label='Train acc')
   plt.plot(history.history['val_accuracy'], label='Val acc')
   plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.show()
   ```

8. **Evaluate & inference**  
   ```python
   test_loss, test_acc = model.evaluate(X_test, y_test)
   print("Test accuracy:", test_acc)
   samples = ["Utterly disappointing; the acting felt wooden.",
              "Let's play a game of chess instead.",
              "Can't wait to see the new movie!"]
   ```

9. **Save & load**  
   - **Model**:  
     ```bash
     model.save("sentiment_bilstm_savedmodel")
     ```
   - **Weights**:  
     ```bash
     model.save_weights("sentiment_bilstm_best.h5")
     ```
   - **Tokenizer**:  
     ```python
     with open("tokenizer.json", "w") as f:
         f.write(tokenizer.to_json())
     ```
   - **Reload**:  
     ```python
     loaded_model = tf.keras.models.load_model("sentiment_bilstm_savedmodel")
     with open("tokenizer.json") as f:
         loaded_tokenizer = tokenizer_from_json(f.read())
     ```

---

## 🗂️ Data Splits

| Split      | Samples |
| ---------- | ------- |
| Train      | ~ (see notebook) |
| Validation | ~ (see notebook) |
| Test       | ~ (see notebook) |

Adjust `DATASET_ROOT` to point to a local path or other dataset if needed.

---

## 📈 Results

After 10 epochs, you should see validation accuracy around **X%**. Tune hyperparameters (dropout, learning rate, L2) for improved performance.

---

## 💡 Extensions

- Replace LSTM with GRU or Transformer blocks.  
- Use pretrained embeddings (GloVe, FastText).  
- Add early stopping & ModelCheckpoint callbacks.  
- Handle class imbalance with class weights.

---

## 📝 License

MIT License. Feel free to adapt and redistribute.

---

## 📚 Citation

If you use this code in your research, please cite:

> Your Name, “Bi‑LSTM Multi‑Class Sentiment Analysis”, GitHub repository, 2025.  
> https://github.com/tomal66/hf-sentiment-analysis
```