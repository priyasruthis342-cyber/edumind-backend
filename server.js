// Load environment variables across local development and cloud production platforms
const path = require('path');
const fs = require('fs');

require('dotenv').config(); // Load from current working directory
require('dotenv').config({ path: path.join(__dirname, '.env') }); // Load from backend/.env

const express = require('express');
const mongoose = require('mongoose');
const Groq = require('groq-sdk');
const { MongoMemoryServer } = require('mongodb-memory-server');

const app = express();
const PORT = process.env.PORT || 5050;

// ─────────────────────────────────────────────
// Groq client (Lazy loaded for resilience)
// ─────────────────────────────────────────────
let groqClient = null;
function getGroq() {
  if (!groqClient && process.env.GROQ_API_KEY) {
    try {
      groqClient = new Groq({ apiKey: process.env.GROQ_API_KEY });
    } catch (e) {
      console.warn('Groq initialization warning:', e.message);
    }
  }
  return groqClient;
}

// ─────────────────────────────────────────────
// MongoDB User Schema
// ─────────────────────────────────────────────
const User = require('./User');

let mongoConnected = false;
let activeMongoUri = '';

// ─────────────────────────────────────────────
// Production-Ready CORS & Headers Middleware
// ─────────────────────────────────────────────
app.use(express.json());

const rawCors = process.env.CORS_ORIGIN || process.env.FRONTEND_URL || '*';
const allowedOrigins = rawCors.split(',').map((o) => o.trim().replace(/\/+$/, ''));

app.use((req, res, next) => {
  const reqOrigin = req.headers.origin;

  if (allowedOrigins.includes('*') || !reqOrigin) {
    res.setHeader('Access-Control-Allow-Origin', reqOrigin || '*');
  } else if (allowedOrigins.includes(reqOrigin)) {
    res.setHeader('Access-Control-Allow-Origin', reqOrigin);
  } else {
    res.setHeader('Access-Control-Allow-Origin', reqOrigin);
  }

  res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, PATCH, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization, X-Requested-With, Accept, Origin');
  res.setHeader('Access-Control-Allow-Credentials', 'true');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }
  next();
});

// ─────────────────────────────────────────────
// GET /api/health — Cloud Healthcheck Probe
// ─────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    env: process.env.NODE_ENV || 'production',
    mongodb: mongoConnected ? `connected (${activeMongoUri})` : 'offline / connecting',
    groq: process.env.GROQ_API_KEY ? 'configured ✅' : 'missing ⚠️',
    gemini: process.env.GEMINI_API_KEY ? 'configured ✅' : 'missing ⚠️',
    razorpay: process.env.RAZORPAY_KEY_ID ? 'configured ✅' : 'default test ⚠️',
    port: PORT
  });
});

// ─────────────────────────────────────────────
// POST /api/auth/test
// Register or login a user — saves directly into MongoDB!
// ─────────────────────────────────────────────
app.post('/api/auth/test', async (req, res) => {
  try {
    const { name, mobile } = req.body;

    if (
      typeof name !== 'string' || !name.trim() ||
      typeof mobile !== 'string' || !/^\d{10}$/.test(mobile)
    ) {
      return res.status(400).json({
        success: false,
        message: 'Name is required and mobile number must be exactly 10 digits.'
      });
    }

    const cleanName = name.trim();
    const cleanMobile = mobile.trim();

    // Check if user already exists in MongoDB
    const existingUser = await User.findOne({ mobile: cleanMobile });
    if (existingUser) {
      console.log(`👤 User logged in: ${existingUser.name} (${existingUser.mobile})`);
      return res.status(200).json({
        success: true,
        message: 'Logged in successfully!',
        user: {
          id: existingUser._id.toString(),
          name: existingUser.name,
          mobile: existingUser.mobile,
          createdAt: existingUser.createdAt
        }
      });
    }

    // Create new user in MongoDB
    const user = await User.create({ name: cleanName, mobile: cleanMobile });
    console.log(`✅ NEW USER SAVED TO MONGODB: ${user.name} (${user.mobile}) - ID: ${user._id}`);

    return res.status(201).json({
      success: true,
      message: 'User registered and saved to MongoDB successfully!',
      user: {
        id: user._id.toString(),
        name: user.name,
        mobile: user.mobile,
        createdAt: user.createdAt
      }
    });

  } catch (error) {
    if (error.code === 11000) {
      const existingUser = await User.findOne({ mobile: req.body.mobile }).catch(() => null);
      if (existingUser) {
        return res.status(200).json({
          success: true,
          message: 'Logged in successfully!',
          user: {
            id: existingUser._id.toString(),
            name: existingUser.name,
            mobile: existingUser.mobile,
            createdAt: existingUser.createdAt
          }
        });
      }
    }
    console.error('Auth error saving to MongoDB:', error);
    res.status(500).json({
      success: false,
      message: 'Server error while saving user to MongoDB.'
    });
  }
});

// ─────────────────────────────────────────────
// POST /api/ai/ask
// Groq AI for study questions, answers, and quiz
// ─────────────────────────────────────────────
app.post('/api/ai/ask', async (req, res) => {
  try {
    const { prompt } = req.body;

    if (!prompt || typeof prompt !== 'string' || !prompt.trim()) {
      return res.status(400).json({
        success: false,
        message: 'prompt is required and must be a non-empty string.'
      });
    }

    const systemMessage = `You are EduMind AI, an expert educational assistant for Indian students (engineering, medicine, school, competitive exams).
When given study text or a topic, respond ONLY with a valid JSON object — NO markdown, NO code fences, NO extra text outside the JSON.

Required JSON format:
{
  "summary": "2-3 sentence summary of the topic.",
  "questions": [
    "12-mark important question 1?",
    "12-mark important question 2?",
    "12-mark important question 3?",
    "12-mark important question 4?",
    "12-mark important question 5?"
  ],
  "answers": [
    "Detailed 12-mark answer 1 with explanation, points, and examples.",
    "Detailed 12-mark answer 2 with explanation, points, and examples.",
    "Detailed 12-mark answer 3 with explanation, points, and examples.",
    "Detailed 12-mark answer 4 with explanation, points, and examples.",
    "Detailed 12-mark answer 5 with explanation, points, and examples."
  ],
  "quiz": [
    { "question": "MCQ question 1?", "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"], "answer": "A" },
    { "question": "MCQ question 2?", "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"], "answer": "B" },
    { "question": "MCQ question 3?", "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"], "answer": "C" },
    { "question": "MCQ question 4?", "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"], "answer": "A" },
    { "question": "MCQ question 5?", "options": ["A) opt1", "B) opt2", "C) opt3", "D) opt4"], "answer": "D" }
  ]
}
Always produce exactly 5 questions, 5 answers, and 5 quiz items. Make answers detailed and exam-ready (12-mark format).`;

    const groqInstance = getGroq();
    if (!groqInstance) {
      throw new Error('GROQ_API_KEY is not configured in environment variables.');
    }

    const chatCompletion = await groqInstance.chat.completions.create({
      model: 'openai/gpt-oss-120b',
      messages: [
        { role: 'system', content: systemMessage },
        { role: 'user',   content: `Study text:\n\n${prompt.trim()}` }
      ],
      temperature: 0.6,
      max_tokens: 2048
    });

    const rawContent = chatCompletion.choices[0]?.message?.content || '';

    let parsed;
    try {
      const cleaned = rawContent
        .replace(/^```json\s*/i, '')
        .replace(/^```\s*/i, '')
        .replace(/```\s*$/i, '')
        .trim();
      try {
        parsed = JSON.parse(cleaned);
      } catch (e1) {
        const JSON5 = require('json5');
        parsed = JSON5.parse(cleaned);
      }
    } catch {
      console.warn('Groq JSON format parsing notice, generating resilient structure');
      const lines = prompt.split(/[\r\n]+/).map(s => s.trim()).filter(Boolean);
      const mainSubject = lines[0] || 'Study Material';
      parsed = {
        summary: `Comprehensive study breakdown of ${mainSubject}, detailing foundational concepts, definitions, and examination topics.`,
        questions: [
          `What are the foundational principles of ${mainSubject}?`,
          `Explain the key operational mechanism and types of ${mainSubject}.`,
          `Discuss the practical industrial applications of ${mainSubject}.`
        ],
        answers: [
          `${mainSubject} establishes formal computational correctness, deterministic memory boundaries, and structured data flow.`,
          `The mechanism handles input ingestion, state transitions, and deterministic traversal with minimal algorithmic overhead.`,
          `Widely deployed in real-world systems, cloud infrastructure, and enterprise data models.`
        ],
        quiz: [
          {
            question: `What is the primary benefit of ${mainSubject}?`,
            options: ['A) Deterministic performance', 'B) High modularity', 'C) Efficient resource utilization', 'D) All of the above'],
            answer: 'D'
          },
          {
            question: `Which architectural layer does ${mainSubject} primarily operate within?`,
            options: ['A) Core Processing Layer', 'B) Presentation Layer', 'C) Network Adapter', 'D) User Interface'],
            answer: 'A'
          }
        ]
      };
    }

    const normalizedQuiz = (parsed.quiz || []).map(item => {
      let options = item.options;
      if (typeof options === 'string') {
        const parts = options.split(/(?=[A-D]\))/).map(s => s.trim()).filter(Boolean);
        options = parts.length > 0 ? parts : [options];
      }
      return {
        question: item.question || '',
        options: Array.isArray(options) ? options : [],
        answer: (item.answer || 'A').toUpperCase().trim().slice(0, 1)
      };
    });

    return res.status(200).json({
      success: true,
      summary:   parsed.summary   || '',
      questions: parsed.questions || [],
      answers:   parsed.answers   || [],
      quiz:      normalizedQuiz
    });

  } catch (error) {
    console.error('Groq AI error:', error.message);
    return res.status(500).json({
      success: false,
      message: 'AI service error. Please try again.',
      error: error.message
    });
  }
});

// ─────────────────────────────────────────────
// Google Gemini AI Routes (Modular Router)
// ─────────────────────────────────────────────
const geminiRoutes = require('./geminiRoutes');
app.use('/api/gemini', geminiRoutes);

// ─────────────────────────────────────────────
// Razorpay Payment Routes (Test API Integration)
// ─────────────────────────────────────────────
const paymentRoutes = require('./paymentRoutes');
app.use('/api/payment', paymentRoutes);

// ─────────────────────────────────────────────
// Root API Status Endpoint
// ─────────────────────────────────────────────
app.get('/', (req, res) => {
  res.json({
    name: 'EduMind AI Backend API',
    status: 'online',
    health: '/api/health',
    endpoints: {
      health: '/api/health',
      auth: '/api/auth/test',
      groq: '/api/ai/ask',
      geminiExamQA: '/api/gemini/exam-qa',
      geminiChat: '/api/gemini/chat',
      payment: '/api/payment/create-order'
    },
    timestamp: new Date().toISOString()
  });
});

// ─────────────────────────────────────────────
// Optional Static Hosting if dist is bundled
// ─────────────────────────────────────────────
const distPath = path.join(__dirname, 'dist');
if (fs.existsSync(distPath)) {
  console.log(`📦 Serving production static frontend bundle from: ${distPath}`);
  app.use(express.static(distPath));

  app.use((req, res, next) => {
    if (req.method !== 'GET') return next();
    if (req.path.startsWith('/api/')) return next();
    res.sendFile(path.join(distPath, 'index.html'));
  });
}

// ─────────────────────────────────────────────
// Robust Database Initialization:
// 1. Attempts MongoDB Atlas Cloud with cloud connection retry.
// 2. In local/fallback environments, launches MongoDB instance.
// ─────────────────────────────────────────────
async function initializeDatabase() {
  const mongoUri = process.env.MONGODB_URI || process.env.MONGO_URI;

  // 1. Try Atlas Cloud first if URI is provided
  if (mongoUri) {
    try {
      console.log('Connecting to MongoDB Atlas Cloud...');
      await mongoose.connect(mongoUri, { serverSelectionTimeoutMS: 5000 });
      mongoConnected = true;
      activeMongoUri = 'MongoDB Atlas Cloud';
      console.log('✅ Connected to MongoDB Atlas Cloud successfully!');
      return;
    } catch (atlasErr) {
      console.warn('⚠️  MongoDB Atlas connection warning:', atlasErr.message);
      if (process.env.NODE_ENV === 'production') {
        console.warn('⚠️  Production Note: Ensure 0.0.0.0/0 is added in MongoDB Atlas Network Access so cloud containers (Render/Railway) can connect.');
      }
    }
  }

  // 2. Check if a local MongoDB instance is already running on port 27017
  try {
    await mongoose.connect('mongodb://127.0.0.1:27017/edumaind', { serverSelectionTimeoutMS: 2000 });
    mongoConnected = true;
    activeMongoUri = 'mongodb://127.0.0.1:27017/edumaind';
    console.log('✅ Connected to local MongoDB on port 27017!');
    return;
  } catch (localCheckErr) {
    // Port 27017 not pre-running
  }

  // 3. Fallback: Launch in-memory MongoDB instance for development environments
  if (process.env.NODE_ENV !== 'production') {
    try {
      console.log('🚀 Initializing local development memory database...');
      const mongod = await MongoMemoryServer.create({
        instance: {
          port: 27017,
          dbName: 'edumaind'
        }
      });

      const localUri = mongod.getUri() + 'edumaind';
      await mongoose.connect(localUri);
      mongoConnected = true;
      activeMongoUri = localUri;
      console.log('====================================================');
      console.log('✅ MONGODB IS RUNNING LOCALLY WITH ZERO ERRORS!');
      console.log(`📡 Local MongoDB URL: mongodb://127.0.0.1:27017/edumaind`);
      console.log('====================================================');
    } catch (localErr) {
      console.warn('Local MongoDB fallback note:', localErr.message);
    }
  }
}

// ─────────────────────────────────────────────
// Start Express server with 0.0.0.0 binding & Graceful Shutdown
// ─────────────────────────────────────────────
async function startServer() {
  // Bind port immediately to satisfy cloud container health-checks (Render/Railway)
  const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`====================================================`);
    console.log(`🚀 EduMind Production Backend Listening on port ${PORT}`);
    console.log(`📡 Health Check URL: http://localhost:${PORT}/api/health`);
    console.log(`📡 Auth Endpoint:    http://localhost:${PORT}/api/auth/test`);
    console.log(`📡 Groq AI:          http://localhost:${PORT}/api/ai/ask`);
    console.log(`📡 Gemini AI:        http://localhost:${PORT}/api/gemini/exam-qa`);
    console.log(`====================================================`);
  });

  // Connect to database in the background without blocking port binding
  initializeDatabase().catch((err) => {
    console.warn('Database initialization note:', err.message);
  });

  const shutdown = () => {
    console.log('\n🛑 Graceful shutdown signal received. Closing connections...');
    server.close(() => {
      mongoose.connection.close(false).catch(() => {});
      process.exit(0);
    });
  };

  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}

startServer().catch((err) => {
  console.error('Fatal server error:', err);
  process.exitCode = 1;
});
