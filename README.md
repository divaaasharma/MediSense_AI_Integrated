# 🩺 MediSense AI – Integrated Healthcare Assessment System

## 1. Project Overview

MediSense AI is an AI-powered healthcare assessment prototype that combines Machine Learning (ML) and Retrieval-Augmented Generation (RAG) to provide diabetes risk predictions and evidence-based medical information.

The application integrates two major functionalities:

- **Diabetes Risk Prediction:** Uses trained Machine Learning models to analyse patient health indicators and generate diabetes-related predictions.
- **Medical Q&A Assistant:** Uses document retrieval and a Large Language Model (LLM) to answer healthcare-related questions using information from medical documents.

The application is developed using Python and Streamlit.

## 2. Key Features

### A. Diabetes Risk Prediction

- Uses the BRFSS 2015 diabetes health indicators dataset.
- Analyses 21 patient health and demographic features.
- Implements Logistic Regression and Random Forest models.
- Uses StandardScaler for Logistic Regression preprocessing.
- Displays model predictions and estimated class probabilities.

### B. Medical Q&A Assistant

- Retrieves relevant information from medical PDF documents.
- Uses Sentence Transformers for text embeddings.
- Stores document embeddings in a FAISS vector database.
- Retrieves the top 3 relevant passages for a question.
- Uses Groq API to generate context-grounded answers.
- Displays document names, page numbers and similarity scores.

## 3. Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| User Interface | Streamlit |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Text Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| Large Language Model | Groq API |
| PDF Processing | PyPDF |
| Environment Management | Python Virtual Environment |

## 4. Project Architecture

The application follows two integrated workflows.

### Machine Learning Workflow

Patient Health Information  
↓  
Input Data Processing  
↓  
Feature Alignment  
↓  
StandardScaler (Logistic Regression)  
↓  
Random Forest and Logistic Regression  
↓  
Prediction and Estimated Probabilities  
↓  
Streamlit Results Display

### RAG Workflow

User Medical Question  
↓  
Sentence Transformer Embeddings  
↓  
FAISS Similarity Search  
↓  
Top 3 Relevant Medical Passages  
↓  
Groq LLM  
↓  
Generated Answer and Document Sources  
↓  
Streamlit Display

## 5. Dataset Information

**Dataset:** Diabetes Health Indicators Dataset – BRFSS 2015

The dataset contains:

- 253,680 records
- 22 columns
- 21 input features
- 1 target variable (`Diabetes_binary`)
- No missing values in the dataset used for this project.

The dataset is used to train and evaluate the diabetes prediction models.

## 6. Machine Learning Models

Two trained classification models are integrated:

**Logistic Regression**
- Uses standardized input features.
- Produces diabetes class predictions and estimated probabilities.

**Random Forest**
- Uses the original input feature values.
- Produces diabetes class predictions and estimated probabilities.

Both models are loaded from saved `.pkl` files.

## 7. RAG Implementation

The Medical Q&A Assistant uses five main components:

1. Document Processing – Extracts text from medical PDF documents.
2. Text Chunking – Divides extracted content into smaller passages.
3. Embedding Generation – Converts passages into numerical vectors using `all-MiniLM-L6-v2`.
4. Vector Search – Uses FAISS to retrieve relevant passages.
5. Answer Generation – Uses Groq to generate answers based on retrieved medical context.

The vector database contains 514 document vectors with 384 embedding dimensions.
### RAG Indexing Implementation

The document indexing pipeline is implemented using the following modules:

| Component | Source File |
|---|---|
| PDF Text Extraction | `src/document_processor.py` |
| Text Chunking | `src/text_chunker.py` |
| Embedding and Vector Indexing | `src/vector_store.py` |
| Similarity-Based Retrieval | `src/retriever.py` |
| Answer Generation | `src/llm_generator.py` |
| RAG Integration | `src/rag_engine.py` |

The indexing pipeline processes medical PDF documents, converts text chunks into embeddings, and stores the resulting vectors in a FAISS index.

The generated vector database is stored in `data/vector_store/`.

## 8. Project Structure

```text
MediSense_AI_Integrated/
│
├── models/
│   ├── logistic_regression_diabetes.pkl
│   ├── random_forest_diabetes.pkl
│   └── standard_scaler.pkl
│
├── data/
│   ├── medical_documents/
│   ├── vector_store/
│   │   ├── medical_index.faiss
│   │   └── metadata.json
│   ├── document_chunks.json
│   ├── processed_documents.json
│   └── diabetes_binary_health_indicators_BRFSS2015.csv
│
├── src/
│   ├── __init__.py
│   ├── app.py
│   ├── prediction.py
│   ├── rag_engine.py
│   ├── retriever.py
│   ├── llm_generator.py
│   ├── vector_store.py
│   ├── document_processor.py
│   └── text_chunker.py
│
├── screenshots/
│
├── docs/
│   └── MediSense_AI_Project_Screenshots.docx
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## 9. Installation and Setup

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2: Navigate to the Project

```bash
cd MediSense_AI_Integrated
```

### Step 3: Create a Virtual Environment

```bash
python -m venv .venv
```

### Step 4: Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 6: Configure the Groq API Key

Create a `.env` file in the project root directory.

Add:

```env
GROQ_API_KEY=your_groq_api_key
```

Replace the placeholder with your own valid Groq API key.

### Step 7: Run the Application

```bash
python -m streamlit run src/app.py
```

The application will open in your browser at:

```text
http://localhost:8501
```

## 10. Application Screenshots

The project includes screenshots demonstrating:

- Diabetes Risk Prediction Interface
- Random Forest Prediction Results
- Logistic Regression Prediction Results
- Medical Q&A Assistant
- Generated Medical Answers
- Retrieved Medical Document Sources

Refer to the `docs` folder for the screenshot documentation.

## 11. Limitations

- The diabetes prediction models are trained using the BRFSS 2015 dataset.
- Predictions are experimental model outputs and are not clinically validated risk estimates.
- The RAG assistant depends on the availability and relevance of the provided medical documents.
- Answer quality depends on retrieved document passages and the external Groq API.
- The application requires an internet connection for Groq API access and initial embedding-model downloads.

## 12. Future Improvements

- Add additional healthcare prediction models.
- Improve model evaluation and validation.
- Expand the medical document knowledge base.
- Improve retrieval accuracy and source ranking.
- Add patient history and report generation.
- Introduce a more advanced healthcare dashboard.

## 13. Disclaimer

MediSense AI is an educational and demonstration prototype developed for healthcare information retrieval and machine learning experimentation.

It does not provide medical diagnoses, prescribe treatments or replace qualified healthcare professionals.

Users should consult qualified medical professionals for personal healthcare decisions.

---

**Project:** MediSense AI – Integrated Healthcare Assessment System  
**Technologies:** Machine Learning | RAG | FAISS | Groq | Streamlit

---

## Model Training Notebook

The complete diabetes model training and evaluation implementation is available in the following Jupyter Notebook:

[01_Diabetes_Risk_Predictionn.ipynb](notebooks/01_Diabetes_Risk_Predictionn.ipynb)

The notebook covers:

- Data preprocessing and cleaning
- Feature scaling
- Logistic Regression training
- Random Forest training
- Model evaluation
- Saving trained models using Joblib
