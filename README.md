# 📊 BizInsight AI

BizInsight AI is an AI-powered customer feedback analytics platform that helps businesses understand customer sentiment, identify key issues, track satisfaction trends, and receive intelligent improvement suggestions.

Built as a real-world business intelligence tool using Python, Streamlit, and advanced AI models.

---

## 🚀 Features

- Upload customer feedback CSV files  
- Automatic sentiment analysis  
- Trend tracking over time  
- Top issue detection  
- AI-powered business assistant  
- Persistent data storage  
- Clean dashboard UI  

---

## 🧠 AI Capabilities

- Understands customer pain points  
- Identifies repeating issues  
- Suggests improvement actions  
- Provides business-style insights  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Pandas  
- Matplotlib  
- Scikit-learn  
- TextBlob  
- DeepSeek AI (via OpenRouter)  
- SQLite  

---

## 📂 Project Structure

bizinsight-ai/

│

├── app.py

├── database.py

├── requirements.txt

├── .gitignore

└── README.md



---

## 📥 Installation & Setup

Follow these steps to set up the project locally on your machine.

### 1. Clone the Repository
Open your terminal and run:
```bash
git clone https://github.com/Prateekiiitg56/BizInsight-AI.git
cd BizInsight-AI
```

### 2. Set Up a Virtual Environment
Choose **one** of the options below to isolate your project dependencies.

#### Option A: Using Standard Python (venv)
* **Create the environment:**
  ```bash
  python -m venv venv
  ```
* **Activate the environment:**
  * **Windows:**
    ```bash
    venv\Scripts\activate
    ```
  * **macOS / Linux:**
    ```bash
    source venv/bin/activate
    ```

#### Option B: Using Anaconda (conda)
* **Create and activate the environment:**
  ```bash
  conda create --name BizInsight-AI-env python=3.10 -y
  conda activate BizInsight-AI-env
  ```

### 3. Install Dependencies
Once your virtual environment is active, install the required packages:
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

1. Create a free account at [OpenRouter](https://openrouter.ai/) and get your API key.

2. Copy the example env file:
   - **macOS / Linux:**
```bash
     cp .env.example .env
```
   - **Windows:**
```bash
     copy .env.example .env
```

3. Open the `.env` file and add your API key:
```
   OPENROUTER_API_KEY=your_api_key_here
```

> ⚠️ Never share or commit your `.env` file. It is already listed in `.gitignore`.

---

### 5. Run the Application
Start the Streamlit dashboard:
```bash
streamlit run app.py
```

---

## 📄 CSV Format

Your CSV file must contain a column named `review`.

---

## 📈 Example Use Cases

- E-commerce customer experience analysis  
- Service quality monitoring  
- Product feedback insights  
- Business performance improvement
  
---

## 🏆 Why BizInsight AI?

Manually analyzing customer feedback is time-consuming and error-prone.  
BizInsight AI converts raw reviews into actionable business intelligence using AI.

---

## 📌 Future Enhancements

- Multi-business login system  
- Automated report generation (PDF)  
- Smart complaint clustering  
- Trend alert system  
- Website integration chatbot  

---

## 👨‍💻 Author

Built by **Prateek Singh**  
BTech Student | AI & Software Development Enthusiast

---

⭐ If you like this project, consider giving it a star!
