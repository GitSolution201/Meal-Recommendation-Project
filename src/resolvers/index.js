
// In-memory storage (no MongoDB needed)
let users = [];
let nextUserId = 1;

const resolvers = {
  Query: {
    me: (_, __, context) => {
      if (!context.user) throw new Error('Authentication required');
      return context.user;
    },
    getMeals: () => [
      { id: '1', name: 'Chicken Breast', calories: 165, protein: 31, fat: 3.6 },
      { id: '2', name: 'Salmon', calories: 208, protein: 25, fat: 12 },
      { id: '3', name: 'Quinoa', calories: 120, protein: 4.4, fat: 1.9 }
    ],
    getRecommendations: (_, { weightGoal }) => {
      const meals = [
        { id: '1', name: 'Chicken Breast', calories: 165, protein: 31, fat: 3.6 },
        { id: '2', name: 'Salmon', calories: 208, protein: 25, fat: 12 },
        { id: '3', name: 'Quinoa', calories: 120, protein: 4.4, fat: 1.9 }
      ];
      
      if (weightGoal === 'aggressive') {
        return meals.filter(meal => meal.calories < 200);
      } else if (weightGoal === 'moderate') {
        return meals.filter(meal => meal.protein > 20);
      }
      return meals;
    }
  },
  
  Mutation: {
    signup: async (_, { input }) => {
      try {
        // Check if user exists (in memory)
        const existingUser = users.find(u => u.email === input.email);
        if (existingUser) {
          return { success: false, message: 'User exists', user: null, token: null };
        }
        
        // Create user in memory
        const user = { 
          id: nextUserId++, 
          ...input, 
          createdAt: new Date(),
          updatedAt: new Date()
        };
        users.push(user);
        
        // Generate simple token
        const token = `token_${user.id}_${Date.now()}`;
        
        console.log('User created:', user.email);
        console.log('Total users:', users.length);
        
        return { success: true, message: 'User created', user, token };
      } catch (error) {
        console.error('Signup error:', error);
        return { success: false, message: 'Signup failed', user: null, token: null };
      }
    },
    
    login: async (_, { input }) => {
      try {
        // Find user in memory
        const user = users.find(u => u.email === input.email);
        if (!user || user.password !== input.password) {
          return { success: false, message: 'Invalid credentials', user: null, token: null };
        }
        
        // Generate simple token
        const token = `token_${user.id}_${Date.now()}`;
        
        console.log('User logged in:', user.email);
        
        return { success: true, message: 'Login successful', user, token };
      } catch (error) {
        console.error('Login error:', error);
        return { success: false, message: 'Login failed', user: null, token: null };
      }
    }
  }
};

module.exports = { resolvers };
