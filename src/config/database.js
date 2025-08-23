const mongoose = require('mongoose');

const connectDB = async () => {
  try {
    if (!process.env.MONGODB_URI) {
      console.log('Database disabled - no MONGODB_URI');
      return false;
    }
    await mongoose.connect(process.env.MONGODB_URI);
    console.log('MongoDB connected');
    return true;
  } catch (error) {
    console.error('MongoDB error:', error.message);
    return false;
  }
};

module.exports = { connectDB };
