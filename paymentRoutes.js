const express = require('express');
const router = express.Router();
const Razorpay = require('razorpay');
const crypto = require('crypto');

// Live Razorpay API credentials
const key_id = process.env.RAZORPAY_KEY_ID || 'rzp_live_U0SAD2W1CHDQG0';
const key_secret = process.env.RAZORPAY_KEY_SECRET || 'zbLVt5TrT2RQ6zDjf1MS9xlD';

const razorpay = new Razorpay({
  key_id,
  key_secret,
});

/**
 * Calculates the exact timestamp of next 12:00 AM (Midnight)
 * Enforces strict 12 AM to 12 AM daily validity window.
 */
function getNextMidnightTimestamp() {
  const now = new Date();
  const nextMidnight = new Date(now.getFullYear(), now.getMonth(), now.getDate() + 1, 0, 0, 0, 0);
  return nextMidnight.getTime();
}

function getTodayDateString() {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, '0')}-${String(now.getDate()).padStart(2, '0')}`;
}

/**
 * GET /api/payment/config
 * Returns live public key ID & pricing info
 */
router.get('/config', (req, res) => {
  return res.json({
    success: true,
    key_id,
    currency: 'INR',
    amount: 100, // 100 paise = ₹1.00
    displayAmount: '₹1.00',
    mode: 'live',
    expiresAt: getNextMidnightTimestamp(),
    todayDate: getTodayDateString()
  });
});

/**
 * POST /api/payment/create-order
 * Creates an official Live Razorpay order for ₹1.00 (100 paise)
 */
router.post('/create-order', async (req, res) => {
  try {
    const { receipt, notes = {} } = req.body;

    const options = {
      amount: 100, // Strictly ₹1.00 (100 paise)
      currency: 'INR',
      receipt: receipt || `rcpt_${Date.now().toString(36)}`,
      payment_capture: 1, // Auto-capture payment
      notes: {
        app: 'EduMind AI',
        plan: 'Daily Pass (12 AM to 12 AM)',
        ...notes
      }
    };

    const order = await razorpay.orders.create(options);
    console.log(`💳 Live Razorpay Order Created: ${order.id} for amount ₹${order.amount / 100}`);

    return res.status(200).json({
      success: true,
      order_id: order.id,
      amount: order.amount,
      currency: order.currency,
      key_id,
      mode: 'live'
    });
  } catch (error) {
    console.error('Razorpay Create Order Error:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to create Razorpay live order.',
      error: error.message
    });
  }
});

/**
 * POST /api/payment/verify
 * Verifies live Razorpay payment signature & returns midnight expiration
 */
router.post('/verify', (req, res) => {
  try {
    const { razorpay_order_id, razorpay_payment_id, razorpay_signature } = req.body;

    if (!razorpay_order_id || !razorpay_payment_id || !razorpay_signature) {
      return res.status(400).json({
        success: false,
        message: 'Order ID, Payment ID, and Signature are required.'
      });
    }

    const generatedSignature = crypto
      .createHmac('sha256', key_secret)
      .update(`${razorpay_order_id}|${razorpay_payment_id}`)
      .digest('hex');

    if (generatedSignature === razorpay_signature) {
      const expiresAt = getNextMidnightTimestamp();
      const paidDate = getTodayDateString();

      console.log(`✅ Live Razorpay Payment Verified: ${razorpay_payment_id} (Expires at midnight: ${new Date(expiresAt).toLocaleString()})`);
      return res.status(200).json({
        success: true,
        verified: true,
        payment_id: razorpay_payment_id,
        order_id: razorpay_order_id,
        paidDate,
        expiresAt,
        message: 'Payment verified successfully! Access granted until 12:00 AM Midnight.'
      });
    } else {
      console.warn(`❌ Razorpay Signature Mismatch for payment ${razorpay_payment_id}`);
      return res.status(400).json({
        success: false,
        verified: false,
        message: 'Invalid payment signature.'
      });
    }
  } catch (error) {
    console.error('Razorpay Verify Error:', error);
    return res.status(500).json({
      success: false,
      message: 'Payment verification failed.',
      error: error.message
    });
  }
});

module.exports = router;
