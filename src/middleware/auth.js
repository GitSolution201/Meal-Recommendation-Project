const jwt = require('jsonwebtoken');
const User = require('../models/User');

const createContext = async ({ req }) => {
  try {
    const token = req.headers.authorization?.split(' ')[1];
    if (!token) return { user: null, isAuthenticated: false };
    
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    const user = await User.findById(decoded.userId).select('-password');
    
    return { user, isAuthenticated: !!user };
  } catch (error) {
    return { user: null, isAuthenticated: false };
  }
};

const requireAuth = (resolver) => {
  return (parent, args, context, info) => {
    if (!context.isAuthenticated) throw new Error('Authentication required');
    return resolver(parent, args, context, info);
  };
};

module.exports = { createContext, requireAuth };