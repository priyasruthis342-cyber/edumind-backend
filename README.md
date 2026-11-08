# EduMind AI — Backend Service

Production Node.js & Express backend API for EduMind AI.

## Features
- **University Exam Q&A Engine:** 2, 3, 8, and 12-mark exam answer generator powered by Google Gemini.
- **AI Chat Assistant:** Real-time conversational study tutor with Mermaid.js diagram generation.
- **Audio Story & Speech Generator:** Natural conversational Tamil & English audio lessons using Edge TTS.
- **Practice Quiz Engine:** Powered by Groq AI (`openai/gpt-oss-120b`).
- **Authentication & Database:** MongoDB Atlas integration for user profiles and progress tracking.
- **Payment Gateway:** Razorpay order generation and verification.

---

## Getting Started

### 1. Install Dependencies
```bash
npm install
```

### 2. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Fill in your credentials:
- `MONGODB_URI`
- `GEMINI_API_KEY`
- `GROQ_API_KEY`
- `RAZORPAY_KEY_ID` & `RAZORPAY_KEY_SECRET`
- `PORT` (defaults to 5050)
- `CORS_ORIGIN` (set to your frontend domain, or `*` for all)

### 3. Run Server
Development:
```bash
npm start
```

Health Check:
```bash
curl http://localhost:5050/api/health
```

---

## Deployment (Render / Railway)
- **Build Command:** `npm install`
- **Start Command:** `npm start`
- Ensure all environment variables from `.env.example` are added in the deployment platform dashboard.
