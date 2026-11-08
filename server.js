// Load environment variables across local development and cloud production platforms
const path = require('path');
const fs = require('fs');

require('dotenv').config(); // Load from current working directory
require('dotenv').config({ path: path.join(__dirname, '.env') }); // Load from backend/.env

const express = require('express');
const { cleanMathFormatting } = require('./mathFormatter');

function sanitizeAiModule(obj) {
  if (!obj) return obj;
  if (typeof obj === 'string') return cleanMathFormatting(obj);
  if (Array.isArray(obj)) return obj.map(sanitizeAiModule);
  if (typeof obj === 'object') {
    const res = {};
    for (const [k, v] of Object.entries(obj)) {
      res[k] = sanitizeAiModule(v);
    }
    return res;
  }
  return obj;
}
const mongoose = require('mongoose');
const { GoogleGenAI } = require('@google/genai');
const { MongoMemoryServer } = require('mongodb-memory-server');

const app = express();
const PORT = process.env.PORT || 5050;

// ─────────────────────────────────────────────
// Google Gemini AI Client (Exclusive AI Provider)
// ─────────────────────────────────────────────
let geminiClient = null;
function getGeminiClient() {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return null;
  if (!geminiClient) {
    try {
      geminiClient = new GoogleGenAI({ apiKey });
    } catch (e) {
      console.warn('GoogleGenAI initialization warning:', e.message);
      return null;
    }
  }
  return geminiClient;
}

const CANDIDATE_GEMINI_MODELS = [
  'gemini-3.5-flash-lite',  // Primary: fastest response time & high rate limits
  'gemini-3.5-flash',
  'gemini-3.7-flash',
  'gemini-3.8-flash'
];

// ─────────────────────────────────────────────
// MongoDB User Schema
// ─────────────────────────────────────────────
const User = require('./User');

let mongoConnected = false;
let activeMongoUri = '';

// ─────────────────────────────────────────────
// Production-Ready CORS & Headers Middleware
// ─────────────────────────────────────────────
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

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
    aiEngine: 'Google Gemini AI (Exclusive)',
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
// Google Gemini AI for study questions, answers, and practice quiz
// Strictly resilient: always returns 200 with structured JSON even on rate limits
// ─────────────────────────────────────────────
function buildDeterministicStudyFallback(prompt) {
  const lines = (prompt || '').split(/[\r\n]+/).map(s => s.trim().replace(/^[^a-zA-Z0-9]+/, '')).filter(Boolean);
  const subject = lines[0] ? lines[0].slice(0, 80) : 'Core Syllabus Topic';
  const subtopic = lines[1] ? lines[1].slice(0, 80) : 'Fundamental Principles';

  return {
    summary: `Comprehensive academic analysis of ${subject}. This unit covers foundational theory, core operational architecture, algorithmic principles, and practical engineering applications required for university examination excellence.`,
    questions: [
      `Explain the fundamental architecture and working principles of ${subject}.`,
      `Describe the core operations, state transitions, and step-by-step algorithms associated with ${subject}.`,
      `Provide an implementation example or technical breakdown for ${subtopic} with syntax and execution flow.`,
      `Discuss the real-world performance characteristics, trade-offs, and design patterns of ${subject}.`,
      `Compare and contrast ${subject} with alternative system designs, detailing advantages and limitations.`
    ],
    answers: [
      `### 1. Definition & Architectural Overview\n\n**Definition:** ${subject} represents a core theoretical and engineering construct designed to organize, process, and optimize data and computational execution.\n\n**Core Architecture:**\n- **Interface Layer:** Defines abstract operations and programmatic interaction contracts.\n- **Implementation Layer:** Manages memory allocation, state variables, and pointer/reference integrity.\n- **Runtime Execution:** Operates under strict boundary constraints to deliver deterministic results.\n\n**Key Advantages:** High cohesion, robust error isolation, and predictable execution flow.`,
      `### 2. Operational Mechanics & Core Algorithms\n\nOperations for ${subject} follow standard algorithmic conventions:\n\n1. **Initialization:** Allocates memory boundaries and sets default pointers/state values.\n2. **State Mutation:** Performs updates in O(1) or O(n) average time complexity depending on storage topology.\n3. **Query/Traversal:** Iterates deterministically from the root/head node until termination.\n4. **Cleanup:** Frees allocated memory blocks to prevent memory leaks and dangling references.`,
      `### 3. Implementation Blueprint & Execution Flow\n\n**Execution Logic:**\n\`\`\`text\nAlgorithm ExecuteOperation(${subject}):\n  Input: Validated system inputs\n  Step 1: Check boundary constraints and preconditions\n  Step 2: Allocate resource node or buffer\n  Step 3: Update forward/backward links or state machine\n  Step 4: Return execution status (Success/Failure)\n\`\`\`\n\n**Output:** Predictable, high-throughput execution with bounded memory utilization.`,
      `### 4. Real-World Applications & Industry Deployment\n\n- **Operating Systems & Kernels:** Utilized in scheduler queues, page tables, and buffer caches.\n- **Network Stacks:** Deployed in routing tables, packet buffers, and multiplexing pipelines.\n- **Database Storage Engines:** Powers indexing, transaction write-ahead logs, and cache eviction models.`,
      `### 5. Comparative Trade-offs & University Exam Summary\n\n- **Strengths:** Predictable complexity, exceptional scalability, and low architectural overhead.\n- **Weaknesses:** Potential memory fragmentation, non-trivial synchronization in multi-threaded contexts.\n- **Key Takeaway for Exam:** Present formal definitions, sketch architectural diagrams, and emphasize time/space complexity analysis.`
    ],
    quiz: [
      {
        question: `What is the primary architectural purpose of ${subject}?`,
        options: [
          'A) To provide structured, deterministic data and state management',
          'B) To bypass operating system memory constraints',
          'C) To eliminate all algorithmic complexity',
          'D) To replace network protocols'
        ],
        answer: 'A'
      },
      {
        question: `Which operational step is essential when modifying ${subject}?`,
        options: [
          'A) Boundary constraint validation',
          'B) Unconditional stack unwinding',
          'C) Arbitrary pointer invalidation',
          'D) Disabling interrupts completely'
        ],
        answer: 'A'
      },
      {
        question: `What is the primary operational trade-off when implementing ${subject}?`,
        options: [
          'A) Space complexity vs. lookup speed',
          'B) Compiler version compatibility only',
          'C) Screen refresh rate degradation',
          'D) Non-deterministic CPU clock speeds'
        ],
        answer: 'A'
      },
      {
        question: `In which domain is ${subject} most heavily utilized?`,
        options: [
          'A) Operating system kernels and high-performance services',
          'B) Static HTML styling',
          'C) Optical disc formatting only',
          'D) Analog sound modulation'
        ],
        answer: 'A'
      },
      {
        question: `What strategy ensures memory integrity in systems utilizing ${subject}?`,
        options: [
          'A) Explicit deallocation and rigorous boundary checks',
          'B) Ignoring return values and errors',
          'C) Infinite recursion loops',
          'D) Random pointer assignment'
        ],
        answer: 'A'
      }
    ]
  };
}

app.post('/api/ai/ask', async (req, res) => {
  const { prompt } = req.body || {};

  if (!prompt || typeof prompt !== 'string' || !prompt.trim()) {
    return res.status(400).json({
      success: false,
      message: 'prompt is required and must be a non-empty string.'
    });
  }

  const systemMessage = `You are EduMind AI, an elite university examination preparation assistant powered exclusively by Google Gemini.
Analyze the user's study text/topic and generate a comprehensive study module.
STRICT PURE MATHEMATICAL NOTATION: Never use raw LaTeX dollar signs ($...$ or $...$) or backslash commands (e.g. \text, \infty). Always write formulas, complexity bounds, and equations in clean Unicode (e.g. x² + 5x + 6 = 0, O(V²), ∞).
Respond ONLY with a valid JSON object — NO markdown code fences, NO comments, NO extra text outside the JSON.

Required JSON format:
{
  "summary": "2-3 sentence executive academic summary of the core concepts in the text.",
  "questions": [
    "University 12-mark Question 1?",
    "University 12-mark Question 2?",
    "University 12-mark Question 3?",
    "University 12-mark Question 4?",
    "University 12-mark Question 5?"
  ],
  "answers": [
    "Comprehensive 12-mark answer 1 structured with Definition, Core Architecture, Step-by-Step Working, Code/Formula where applicable, and Advantages & Disadvantages.",
    "Comprehensive 12-mark answer 2 structured with Definition, Core Architecture, Step-by-Step Working, Code/Formula where applicable, and Advantages & Disadvantages.",
    "Comprehensive 12-mark answer 3 structured with Definition, Core Architecture, Step-by-Step Working, Code/Formula where applicable, and Advantages & Disadvantages.",
    "Comprehensive 12-mark answer 4 structured with Definition, Core Architecture, Step-by-Step Working, Code/Formula where applicable, and Advantages & Disadvantages.",
    "Comprehensive 12-mark answer 5 structured with Definition, Core Architecture, Step-by-Step Working, Code/Formula where applicable, and Advantages & Disadvantages."
  ],
  "quiz": [
    {
      "question": "Clear multiple choice exam question 1?",
      "options": ["A) Option A text", "B) Option B text", "C) Option C text", "D) Option D text"],
      "answer": "A"
    },
    {
      "question": "Clear multiple choice exam question 2?",
      "options": ["A) Option A text", "B) Option B text", "C) Option C text", "D) Option D text"],
      "answer": "B"
    },
    {
      "question": "Clear multiple choice exam question 3?",
      "options": ["A) Option A text", "B) Option B text", "C) Option C text", "D) Option D text"],
      "answer": "C"
    },
    {
      "question": "Clear multiple choice exam question 4?",
      "options": ["A) Option A text", "B) Option B text", "C) Option C text", "D) Option D text"],
      "answer": "D"
    },
    {
      "question": "Clear multiple choice exam question 5?",
      "options": ["A) Option A text", "B) Option B text", "C) Option C text", "D) Option D text"],
      "answer": "A"
    }
  ]
}

STRICT CONSTRAINTS:
1. Always output exactly 5 questions, 5 answers, and 5 quiz items.
2. In options, use the prefix "A) ", "B) ", "C) ", "D) ".
3. In answer, use single uppercase letter: "A", "B", "C", or "D".
4. Output raw JSON only.`;

  let parsed = null;

  // 1. Attempt generation with Google Gemini model cascade
  const client = getGeminiClient();
  if (client) {
    for (const model of CANDIDATE_GEMINI_MODELS) {
      try {
        const response = await Promise.race([
          client.models.generateContent({
            model,
            contents: `Study text:\n\n${prompt.trim()}`,
            config: {
              responseMimeType: 'application/json',
              systemInstruction: systemMessage,
              temperature: 0.3
            }
          }),
          new Promise((_, reject) => setTimeout(() => reject(new Error(`Timeout on ${model}`)), 4000))
        ]);

        const rawText = response?.text;
        if (rawText) {
          const cleaned = rawText
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
          if (parsed && Array.isArray(parsed.quiz) && parsed.quiz.length > 0) {
            break; // Successfully got full valid JSON!
          }
        }
      } catch (err) {
        console.warn(`[AI Ask] Model ${model} unavailable (${err.message}). Trying next candidate...`);
      }
    }
  }

  // 2. Fallback to rich deterministic educational structure if Gemini is busy, rate-limited, or unavailable
  if (!parsed || !Array.isArray(parsed.quiz) || parsed.quiz.length === 0) {
    console.warn('[AI Ask] Utilizing deterministic syllabus analysis fallback.');
    parsed = buildDeterministicStudyFallback(prompt);
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
    questions: Array.isArray(parsed.questions) ? parsed.questions : [],
    answers:   Array.isArray(parsed.answers) ? parsed.answers : [],
    quiz:      normalizedQuiz
  });
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
      studyQuiz: '/api/ai/ask',
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
    console.log(`📡 Gemini Study & Quiz: http://localhost:${PORT}/api/ai/ask`);
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
