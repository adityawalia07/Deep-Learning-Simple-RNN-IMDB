# IMDB Movie Review Sentiment Analysis

> An interactive web application that analyzes the sentiment of movie reviews using a pre-trained RNN model.

## 📋 Overview

This Streamlit application provides a user-friendly interface for analyzing the sentiment of movie reviews. Using a Recurrent Neural Network (RNN) trained on the IMDB movie review dataset, the app can classify reviews as positive, negative, or neutral based on the text content.

## ✨ Features

- **Real-time Sentiment Analysis**: Instantly analyze any movie review with a single click
- **Visual Results**: See sentiment scores displayed through an intuitive gauge chart
- **Sentiment Highlighting**: Automatically highlights positive and negative words in the review
- **Example Reviews**: Pre-loaded examples to demonstrate the application's capabilities
- **Educational Content**: Learn about the sentiment analysis process and model architecture
- **User-friendly Interface**: Clean, intuitive design with helpful tooltips and guidance

## 🛠️ Technologies Used

- **TensorFlow**: For the underlying neural network model
- **Streamlit**: For the web application interface
- **IMDB Dataset**: For training the sentiment analysis model
- **Matplotlib**: For data visualization
- **NumPy**: For numerical operations
- **Regular Expressions**: For text preprocessing


## 📄 Required Files

Make sure you have the following file in your project directory:

- `simple_rnn_imdb.h5`: The trained TensorFlow model


## 🚀 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```

2. Access the application in your web browser at `http://localhost:8501`

3. Options for analyzing sentiment:
   - Type or paste a review in the text area
   - Click one of the example buttons to load a sample review
   - Press the "Analyze Sentiment" button to see results

## 🧠 Model Information

The application uses a Recurrent Neural Network (RNN) model with the following architecture:

- **Embedding Layer**: Converts words to vector representations
- **RNN Layer**: Processes the sequence of word embeddings
- **Dense Layers**: Extract features with ReLU activation
- **Output Layer**: Sigmoid activation for binary classification

The model was trained on 25,000 pre-labeled movie reviews from the IMDB dataset, achieving approximately 87% accuracy on the test set.

## 💡 How It Works

1. **Text Preprocessing**:
   - The input review is converted to lowercase
   - Words are mapped to numbers based on the IMDB vocabulary
   - The sequence is padded to a fixed length (500 tokens)

2. **Sentiment Prediction**:
   - The preprocessed review is fed into the RNN model
   - The model outputs a score between 0 and 1
   - Scores are interpreted as:
     - > 0.7: Positive sentiment
     - < 0.3: Negative sentiment
     - 0.3-0.7: Neutral sentiment

3. **Result Visualization**:
   - Sentiment is displayed with color-coding (green, red, blue)
   - A gauge chart shows the position on the sentiment spectrum
   - Sentiment-bearing words are highlighted in the original text

## 🔧 Customization

You can customize the application by:

1. **Using your own model**: Replace `simple_rnn_imdb.h5` with your trained model
2. **Adding more example reviews**: Modify the example review text in the code
3. **Adjusting sentiment thresholds**: Change the score thresholds for sentiment classification
4. **Extending the word highlighting**: Add more words to the positive/negative word lists

## 📊 Training Your Own Model

To train your own sentiment analysis model:

1. Download the IMDB dataset:
   ```python
   from tensorflow.keras.datasets import imdb
   (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
   ```

2. Build and train an RNN model:
   ```python
   model = tf.keras.Sequential([
       tf.keras.layers.Embedding(10000, 128),
       tf.keras.layers.SimpleRNN(128, activation='relu'),
       tf.keras.layers.Dense(64, activation='relu'),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])
   model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
   model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
   ```

3. Save the trained model:
   ```python
   model.save('simple_rnn_imdb.h5')
   ```

A complete notebook with the training process is available in the `notebooks` directory.

## 📑 Dependencies

The following packages are required:

```
streamlit>=1.24.0
tensorflow>=2.12.0
numpy>=1.24.0
matplotlib>=3.7.0
```

A complete list is available in `requirements.txt`.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- [IMDB Dataset](https://www.tensorflow.org/datasets/catalog/imdb_reviews) - For providing the training data
- [TensorFlow](https://www.tensorflow.org/) - For the machine learning framework
- [Streamlit](https://streamlit.io/) - For the web application framework

