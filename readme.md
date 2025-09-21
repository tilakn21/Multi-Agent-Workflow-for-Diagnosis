# <img width="122" height="84" alt="Screenshot 2025-09-21 131905" src="https://github.com/user-attachments/assets/51852fea-c620-46f6-a937-7273882a6c1c" />


**MediMind** is an AI-powered multi-agent clinical reasoning assistant that supports doctors in delivering fast, accurate, and guideline-compliant diagnoses. It is **not a replacement for doctors** but a workflow enhancer that reduces cognitive load, references historical cases, literature, and research, and generates structured patient reports and prescriptions.

---

## 🌐 Live Demo & Repository

- **🚀 Web Application:** [Live Demo](https://medbott.vercel.app/)


---

## 🏥 Problem Statement

Misdiagnosis and delayed treatment are significant challenges in healthcare, especially in India. Studies show that **up to 30% of medical errors in rural areas** result from incomplete patient history, scattered information, or lack of reference to medical literature. Doctors often struggle to integrate patient conversations, vitals, medical records, and research into a structured diagnosis efficiently.

**Impact:**  
- Increased medical errors  
- Wasted resources due to repeated tests  
- Delayed or improper treatment  
- Increased burden on healthcare providers

**Our solution:** MediMind – a multi-agent AI system that assists doctors by providing structured, evidence-backed diagnosis within seconds.

---

## 💡 Solution Overview

MediMind uses a **team of specialized agents**:

1. **Conversation Agent:**  
   Captures the patient-doctor conversation, medical history, vitals, doctor's opinion, and optional medical records (like X-rays) into a structured query.

2. **Memory Agent:**  
   - Maintains a **vector database** of **350+ past cases** (queries + diagnoses) embedded using ClinicalBERT.  
   - Provides retrieval and similarity-based recommendations for new queries.  

3. **Literature Agent:**  
   - Implements a **Graph RAG** using **FAISS** with **301163 vectors and 679277 nodes** from medical books and journals.  
   - Fetches structured diagnosis suggestions based on query relevance.  

4. **Scraping Agent:**  
   - Scrapes PubMed and WHO platforms for relevant articles and possible diagnoses.  

5. **Decision Agent:**  
   - Applies **ABHA guidelines** to ensure compliance.  
   - Verifies **patient data privacy** and ranks candidate diagnoses.  

6. **Final Response:**  
   - Doctors receive the best-ranked diagnosis in under **10 seconds** from patient conversation to final output.  
   - Generates a **prescription/report format** for both patient and doctor.  
   - Doctor feedback is stored to improve future recommendations.

---

## ⚙️ Architecture

![MediMind Architecture](https://github.com/user-attachments/assets/b0def1a2-7d26-4355-b850-c1c3962c21f8)

**Flow of Information:**  
1. Patient interaction → Conversation Agent  
2. Query processing → Memory, Literature, Scraping Agents  
3. LLM combines responses → Decision Agent  
4. Final diagnosis → Doctor → Feedback stored in memory  
5. Prescription/report generated → Patient and Doctor

---

## 🖥️ User Interface

**Doctor Dashboard:** View patient info, generated diagnosis, add prescriptions or lab tests.  
**Patient Interface:** Register ABHA ID, input vitals and medical history, view generated reports.

<div align="center">
  
<img src="https://github.com/user-attachments/assets/0b147cdb-6cd1-4bf0-ac94-352dbbda93c7" width="400"/>
<img src="https://github.com/user-attachments/assets/36d68ccd-43f3-43ad-ac92-8d9b5c1011d4" width="400"/>
<img src="https://github.com/user-attachments/assets/a1e143dd-91f8-4895-b805-d82576ed8b27" width="400"/>
<img src="https://github.com/user-attachments/assets/d007e1c7-2548-46bd-a015-cce47f02ce86" width="400"/>


<img src="https://github.com/user-attachments/assets/77bcd212-0433-446f-8e71-e866e8250269" width="400"/>

</div>

<div align="center">
<img src="https://github.com/user-attachments/assets/c5c44ae5-b157-42bf-8c54-cf3f27d2c691" width="400"/>
</div>
---

## 🚀 Features

- **Multi-agent AI workflow** for comprehensive diagnosis support
- **Fast response time** (<10s from query to diagnosis)
- **Historical case referencing** using vector database with 350+ cases
- **Literature-backed diagnosis** with Graph RAG implementation
- **Real-time web scraping** for latest research and guidelines
- **ABHA-compliant** and privacy-safe healthcare data handling
- **Structured reports** and prescription generation
- **User-friendly interfaces** for both patients and doctors
- **Continuous learning** from doctor feedback

---

## 🛠️ Tech Stack

**Backend:**
- FastAPI, Python
- FAISS Vector Database
- ClinicalBERT Embedding Model

**Frontend:**
- Next.js (Web + Mobile)
- Responsive design for cross-platform compatibility

**Data Sources:**
- PubMed medical research
- WHO guidelines and protocols
- Medical books & journals
- Historical case database

**Infrastructure:**
- Supabase (Authentication & Storage)
- RESTful API architecture
- Modular agent communication

---

## 📊 Performance Metrics

- **Response Time:** <10 seconds end-to-end
- **Case Database:** 350+ historical cases
- **Literature Vectors:** 980,440 medical document nodes
- **Accuracy:** Evidence-backed with multiple validation layers
- **Compliance:** ABHA guidelines integrated

---

## 🏗️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/tilakn21/Multi-Agent-Workflow-for-Diagnosis.git
   cd Multi-Agent-Workflow-for-Diagnosis
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment setup:**
   ```bash
   cp .env.example .env
   # Add your Supabase and other API keys to .env
   ```

4. **Run the application:**
   ```bash
   uvicorn main:app --reload
   ```

---

## 📈 Why No Orchestration?

- Agents communicate **modularly and sequentially**, avoiding heavy orchestration frameworks
- This ensures **lightweight deployment, maintainability**, and **faster iteration**
- Suitable for hackathon environments and MVP scenarios
- Simplified debugging and monitoring of individual agent performance

---

## 🔮 Future Enhancements

- Integration with hospital management systems
- Multi-language support for regional healthcare
- Advanced medical imaging analysis
- Real-time vitals monitoring integration
- Telemedicine consultation features

---

## 🤝 Contributing

We welcome contributions! Please read our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit pull requests, report issues, and suggest improvements.

---

## 📚 References

1. [Misdiagnosis in India: Causes and Impact](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6223504/)  
2. [ABHA Guidelines](https://www.abdm.gov.in/)  
3. [FAISS Documentation](https://faiss.ai/)  
4. [ClinicalBERT](https://arxiv.org/abs/1904.03323)
5. [Graph RAG Implementation](https://arxiv.org/abs/2404.16130)

---

## 💬 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Tilak Neema** - [GitHub](https://github.com/tilakn21)
**Shubhang Chakrawarty** - [GitHub](https://github.com/shubhang69)
**Abhinav Gupta** - [GitHub](https://github.com/Abhinav-gupta-123)
**Aadish Sanghvi** - [GitHub](https://github.com/aadish-sanghvi)



## 📌 Important Notes

⚠️ **Disclaimer:** MediMind is a diagnostic assistance tool and should not replace professional medical judgment. Always consult with qualified healthcare professionals for medical decisions.

🔒 **Privacy:** Ensure your `.env` file contains all required API keys securely. Never commit sensitive credentials to version control.

🖼️ **Assets:** Replace placeholder URLs with your actual hosted frontend and demo links before deploying.
