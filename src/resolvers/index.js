
const User = require('../models/User');
const { requireAuth } = require('../middleware/auth');
const { getKNNRecommendations, sampleMeals } = require('../services/mlService');

const resolvers = {
  Query: {
    me: requireAuth((_, __, context) => context.user),
    getMeals: () => sampleMeals,
    getRecommendations: (_, { weightGoal }) => {
      return getKNNRecommendations({ weightGoal }, 3);
    }
  },
  
  Mutation: {
    signup: async (_, { input }) => {
      try {
        const existingUser = await User.findOne({ email: input.email });
        if (existingUser) {
          return { success: false, message: 'User exists', user: null, token: null };
        }
        
        const user = new User(input);
        await user.save();
        
        const jwt = require('jsonwebtoken');
        const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET);
        
        return { success: true, message: 'User created', user, token };
      } catch (error) {
        return { success: false, message: 'Signup failed', user: null, token: null };
      }
    },
    
    login: async (_, { input }) => {
      try {
        const user = await User.findOne({ email: input.email });
        if (!user || !(await user.comparePassword(input.password))) {
          return { success: false, message: 'Invalid credentials', user: null, token: null };
        }
        
        const jwt = require('jsonwebtoken');
        const token = jwt.sign({ userId: user._id }, process.env.JWT_SECRET);
        
        return { success: true, message: 'Login successful', user, token };
      } catch (error) {
        return { success: false, message: 'Login failed', user: null, token: null };
      }
    }
  }
};

module.exports = { resolvers };
