# Assignment-5.1


## Links 
Codelabs : https://codelabs-preview.appspot.com/?file_id=1RgLdVbUQtDsAhJ5gTBxKILEPgoRKEFJzlve5-VERqlI#0

Streamlit + FastApi: https://fastapi-app-1057230376331.us-central1.run.app

video link : 
 
## Assignment Overview 
Existing systems for financial research and analysis often rely on static, isolated data sources and lack integration across structured and unstructured data. These systems typically require manual data retrieval and fail to provide real-time, context-aware insights or seamless interaction between different data types. The new system being developed will integrate multiple specialized agents using LangGraph, bringing together real-time web searches, structured financial data from Snowflake, and unstructured quarterly reports through Pinecone-powered retrieval. By combining these agents, the system will automatically generate comprehensive research reports, offering dynamic and context-rich responses. This approach will overcome the limitations of existing systems by enabling metadata filtering, real-time insights, and seamless orchestration between agents. The result will be a more efficient, accurate, and user-friendly solution for financial research.

---

## 🛠️ Technology Used

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/)
[![FastAPI](https://img.shields.io/badge/fastapi-109989?style=for-the-badge&logo=FASTAPI&logoColor=white)](https://fastapi.tiangolo.com/)
[![Amazon AWS](https://img.shields.io/badge/Amazon_AWS-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-%232496ED?style=for-the-badge&logo=Docker&color=blue&logoColor=white)](https://www.docker.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Google Cloud](https://img.shields.io/badge/Google_Cloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)](https://cloud.google.com)
[![Snowflake](https://img.shields.io/badge/Snowflake-00A9E0?style=for-the-badge&logo=snowflake&logoColor=white)](https://www.snowflake.com/)
---

## 🏗️ Architecture Diagram
![AI Application Data Pipeline](https://github.com/Bigdata2025Team5/Assignment-5.1/blob/db337ed8ea89c213c17cf1be8522a0825cdbd409/Architecture%20Diagram.png)

---

## 🔑 How to Use the Application
1. **Launch the Streamlit app**: Open the app at `http://localhost:8501`.
2. **Enter your query**: In the "Research Query" section, input your query about NVIDIA (e.g., "What is NVIDIA's current market cap?").
3. **Select filters**: Choose the year, quarter, and agent(s) to use:
   - **Snowflake Agent**: Retrieves financial metrics.
   - **RAG Agent**: Fetches data from NVIDIA's quarterly reports.
   - **Web Search Agent**: Retrieves real-time web insights.
   - **All Agents**: Uses all agents for the report.
4. **Generate Report**: Click the "Generate Report" button.
5. **View Report and Charts**: The generated report, including charts (e.g., valuation metrics) and web insights, will be displayed.

---

## Project Setup

### Prerequisites
- Python 3.8+
- Snowflake account and credentials
- API keys for OpenAI and Web Search (SerpAPI)
  
### Setup Instructions

1. **Clone the repository**
2. **Install dependencies**:   
   ```
   pip install -r requirements.txt
   ```
3. **Run the FastAPI server**:
   ```
   uvicorn app:app --reload
   ```
4. **Run the Streamlit app**:
   ```
   streamlit run streamlit_app.py
   ```
---

## 📂 Project Structure
```
├── Backend
│   └── Dockerfile
|   ├── main.py
|   ├── opensource.py
|   ├── requirements.txt
├── Diagrams
│   ├── architecture_diagrams.pmg
├── Documentation
├── Frontend
│   ├── Dockerfile
│   ├── app.py
│   ├── requirements.txt
├── POC  
├── AiDisclosure.md
├── README.md

```

---
## References

- LangGraph Documentation
- Apache Airflow Documentation
- Pinecone Documentation 
- FastAPI & Streamlit Documentation
- AWS S3 Best Practices

---

## 👥 Team Information
| Name            | Student ID    | Contribution |
|----------------|--------------|--------------|
| **Pranjal Mahajan** | 002375449  | 33.33% |
| **Srushti Patil**  | 002345025  | 33.33% |
| **Ram Putcha**  | 002304724  | 33.33% |

---
