const express = require('express');
const router = express.Router();
const Razorpay = require('razorpay');
const crypto = require('crypto');

const key_id = process.env.RAZORPAY_KEY_ID || 'rzp_test_TwxrpdkuvcZNdH';
const key_secret = process.env.RAZORPAY_KEY_SECRET || '996rZ0hCsnE0B5T6eP7XQb42';

const razorpay = new Razorpay({
  key_id,
  key_secret,
});

/**
 * GET /api/payment/config
 * Returns public test key ID & pricing info
 */
router.get('/config', (req, res) => {
  return res.json({
    success: true,
    key_id,
    currency: 'INR',
    amount: 100, // 100 paise = ₹1.00
    displayAmount: '₹1.00'
  });
});

/**
 * POST /api/payment/create-order
 * Creates a real Razorpay test order for ₹1.00 (100 paise)
 */
router.post('/create-order', async (req, res) => {
  try {
    const { amount = 100, receipt, notes = {} } = req.body;

    let parsedAmount = parseInt(amount, 10) || 100;
    // Razorpay requires amounts in paise (100 paise = ₹1.00). Ensure minimum 100 paise.
    if (parsedAmount < 100) {
      parsedAmount = parsedAmount * 100;
    }
    if (parsedAmount < 100) parsedAmount = 100;

    const options = {
      amount: parsedAmount, // amount in paise (minimum 100 = ₹1)
      currency: 'INR',
      receipt: receipt || `rcpt_${Date.now().toString(36)}`,
      notes: {
        app: 'EduMind AI',
        plan: 'Daily Pass',
        ...notes
      }
    };

    const order = await razorpay.orders.create(options);
    console.log(`💳 Razorpay Test Order Created: ${order.id} for amount ₹${order.amount / 100}`);

    return res.status(200).json({
      success: true,
      order_id: order.id,
      amount: order.amount,
      currency: order.currency,
      key_id
    });
  } catch (error) {
    console.error('Razorpay Create Order Error:', error);
    return res.status(500).json({
      success: false,
      message: 'Failed to create Razorpay test order.',
      error: error.message
    });
  }
});

/**
 * POST /api/payment/verify
 * Verifies Razorpay payment signature
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
      console.log(`✅ Razorpay Payment Verified: ${razorpay_payment_id} for order ${razorpay_order_id}`);
      return res.status(200).json({
        success: true,
        verified: true,
        payment_id: razorpay_payment_id,
        message: 'Payment verified successfully!'
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

