
const User = require('../models/User');
const { requireAuth } = require('../middleware/auth');
const fs = require('fs');
const path = require('path');

// Function to read and parse CSV data
const readMealData = () => {
  try {
    const csvPath = path.join(__dirname, '../../df_combinedUser_data_sample.csv');
    console.log('Reading CSV from:', csvPath);
    
    if (!fs.existsSync(csvPath)) {
      console.error('CSV file not found at:', csvPath);
      return [];
    }
    
    const csvData = fs.readFileSync(csvPath, 'utf8');
    console.log('CSV file size:', csvData.length, 'characters');
    
    // Split by lines and filter empty lines
    const lines = csvData.split('\n').filter(line => line.trim());
    console.log('Total lines in CSV:', lines.length);
    
    if (lines.length === 0) {
      console.error('No lines found in CSV');
      return [];
    }
    
    // Parse headers (first line)
    const headers = lines[0].split(',').map(h => h.trim());
    console.log('CSV headers:', headers);
    console.log('Number of columns:', headers.length);
    
    const meals = [];
    
    // Process each data line (skip header)
    for (let i = 1; i < lines.length; i++) {
      try {
        const line = lines[i];
        if (!line.trim()) continue;
        
        // More robust CSV parsing - handle quoted values
        const values = [];
        let currentValue = '';
        let inQuotes = false;
        
        for (let j = 0; j < line.length; j++) {
          const char = line[j];
          
          if (char === '"') {
            inQuotes = !inQuotes;
          } else if (char === ',' && !inQuotes) {
            values.push(currentValue.trim());
            currentValue = '';
          } else {
            currentValue += char;
          }
        }
        values.push(currentValue.trim()); // Add last value
        
        // Create meal object
        const meal = {};
        headers.forEach((header, index) => {
          let value = values[index] || '';
          
          // Remove quotes if present
          if (value.startsWith('"') && value.endsWith('"')) {
            value = value.slice(1, -1);
          }
          
          // Convert numeric values
          if (['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 
               'SodiumContent', 'CarbohydrateContent', 'FiberContent', 'SugarContent', 
               'ProteinContent', 'WeightLossScore', 'IsGoodMeal', 'UserAge', 'UserWeight', 
               'UserBMI', 'UserBMR', 'RecipeServings'].includes(header)) {
            value = parseFloat(value) || 0;
          }
          
          meal[header] = value;
        });
        
        meals.push(meal);
        
      } catch (lineError) {
        console.error(`Error processing line ${i}:`, lineError);
        continue; // Skip problematic lines
      }
    }
    
    console.log('Successfully parsed meals:', meals.length);
    
    // Filter out meals with all zero nutritional values
    const filteredMeals = meals.filter(meal => {
      const hasCalories = meal.Calories > 0;
      const hasProtein = meal.ProteinContent > 0;
      const hasFat = meal.FatContent > 0;
      const hasCarbs = meal.CarbohydrateContent > 0;
      
      // Keep meal if it has at least 2 nutritional values
      const nutritionalValues = [hasCalories, hasProtein, hasFat, hasCarbs];
      return nutritionalValues.filter(Boolean).length >= 2;
    });
    
    console.log('Meals after filtering zero values:', filteredMeals.length);
    
    // Show sample meal structure
    if (filteredMeals.length > 0) {
      console.log('Sample meal keys:', Object.keys(filteredMeals[0]));
      console.log('Sample meal with images:', {
        name: filteredMeals[0].Name,
        images: filteredMeals[0].Images,
        calories: filteredMeals[0].Calories,
        protein: filteredMeals[0].ProteinContent
      });
    }
    
    return filteredMeals;
    
  } catch (error) {
    console.error('Error reading CSV file:', error);
    // Fallback to hardcoded data if CSV reading fails
    return [
      { id: '1', name: 'Chicken Breast', calories: 165, protein: 31, fat: 3.6 },
      { id: '2', name: 'Salmon', calories: 208, protein: 25, fat: 12 },
      { id: '3', name: 'Quinoa', calories: 120, protein: 4.4, fat: 1.9 }
    ];
  }
};

const resolvers = {
  Query: {
    me: requireAuth((_, __, context) => context.user),
    
    getMeals: () => {
      const meals = readMealData();
      console.log("Total meals found after filtering:", meals.length);
      
      // Debug: Check first few meals for images
      if (meals.length > 0) {
        console.log("=== DEBUG: First 3 meals image data ===");
        for (let i = 0; i < Math.min(3, meals.length); i++) {
          const meal = meals[i];
          console.log(`Meal ${i + 1}:`, {
            name: meal.Name,
            rawImages: meal.Images,
            extractedImage: extractFirstImage(meal.Images || ''),
            hasImages: !!meal.Images
          });
        }
        console.log("=== END DEBUG ===");
      }
      
      return meals.map((meal, index) => {
        const extractedImage = extractFirstImage(meal.Images || '');
        
        // Debug: Log each meal's image extraction
        if (index < 5) {
          console.log(`Processing meal ${index + 1}:`, {
            name: meal.Name,
            rawImages: meal.Images,
            extractedImage: extractedImage
          });
        }
        
        return {
          id: meal.RecipeId || meal.id || (index + 1).toString(),
          name: meal.Name || meal.name || 'Unknown Meal',
          calories: meal.Calories || 0,
          protein: meal.ProteinContent || 0,
          fat: meal.FatContent || 0,
          carbs: meal.CarbohydrateContent || 0,
          fiber: meal.FiberContent || 0,
          sugar: meal.SugarContent || 0,
          sodium: meal.SodiumContent || 0,
          cholesterol: meal.CholesterolContent || 0,
          saturatedFat: meal.SaturatedFatContent || 0,
          weightLossScore: meal.WeightLossScore || 0,
          isGoodMeal: meal.IsGoodMeal || 0,
          category: meal.RecipeCategory || meal.category || 'General',
          prepTime: meal.PrepTime || meal.prepTime || 'Unknown',
          totalTime: meal.TotalTime || meal.totalTime || 'Unknown',
          servings: meal.RecipeServings || meal.servings || 1,
          // Add picture/image field - extract first image URL if multiple
          images: extractedImage,
          keywords: meal.Keywords || meal.keywords || '',
          ingredients: meal.RecipeIngredientParts || meal.ingredients || ''
        };
      });
    },
    
    getRecommendations: (_, { weightGoal, maxCalories, minProtein }) => {
      const meals = readMealData();
      console.log("Getting recommendations from", meals.length, "filtered meals");
      
      let filteredMeals = meals;
      
      // Filter by weight goal
      if (weightGoal === 'aggressive') {
        filteredMeals = meals.filter(meal => meal.Calories < 300 && meal.ProteinContent > 20);
      } else if (weightGoal === 'moderate') {
        filteredMeals = meals.filter(meal => meal.Calories < 500 && meal.ProteinContent > 15);
      } else if (weightGoal === 'maintenance') {
        filteredMeals = meals.filter(meal => meal.Calories < 800);
      }
      
      // Filter by max calories if provided
      if (maxCalories) {
        filteredMeals = filteredMeals.filter(meal => meal.Calories <= maxCalories);
      }
      
      // Filter by min protein if provided
      if (minProtein) {
        filteredMeals = filteredMeals.filter(meal => meal.ProteinContent >= minProtein);
      }
      
      // Sort by weight loss score (best meals first)
      filteredMeals.sort((a, b) => (b.WeightLossScore || 0) - (a.WeightLossScore || 0));
      
      console.log("Filtered to", filteredMeals.length, "recommendations");
      
      // Return top 20 recommendations
      return filteredMeals.slice(0, 20).map((meal, index) => ({
        id: meal.RecipeId || meal.id || (index + 1).toString(),
        name: meal.Name || meal.name || 'Unknown Meal',
        calories: meal.Calories || 0,
        protein: meal.ProteinContent || 0,
        fat: meal.FatContent || 0,
        carbs: meal.CarbohydrateContent || 0,
        fiber: meal.FiberContent || 0,
        sugar: meal.SugarContent || 0,
        sodium: meal.SodiumContent || 0,
        cholesterol: meal.CholesterolContent || 0,
        saturatedFat: meal.SaturatedFatContent || 0,
        weightLossScore: meal.WeightLossScore || 0,
        isGoodMeal: meal.IsGoodMeal || 0,
        category: meal.RecipeCategory || meal.category || 'General',
        prepTime: meal.PrepTime || meal.prepTime || 'Unknown',
        totalTime: meal.TotalTime || meal.totalTime || 'Unknown',
        servings: meal.RecipeServings || meal.servings || 1,
        images: extractFirstImage(meal.Images || meal.images || ''),
        keywords: meal.Keywords || meal.keywords || '',
        ingredients: meal.RecipeIngredientParts || meal.ingredients || ''
      }));
    },
    
    getMealById: (_, { id }) => {
      const meals = readMealData();
      const meal = meals.find(m => m.RecipeId === id || m.Name === id || m.id === id);
      
      if (!meal) return null;
      
      return {
        id: meal.RecipeId || meal.id || id,
        name: meal.Name || meal.name || 'Unknown Meal',
        calories: meal.Calories || 0,
        protein: meal.ProteinContent || 0,
        fat: meal.FatContent || 0,
        carbs: meal.CarbohydrateContent || 0,
        fiber: meal.FiberContent || 0,
        sugar: meal.SugarContent || 0,
        sodium: meal.SodiumContent || 0,
        cholesterol: meal.CholesterolContent || 0,
        saturatedFat: meal.SaturatedFatContent || 0,
        weightLossScore: meal.WeightLossScore || 0,
        isGoodMeal: meal.IsGoodMeal || 0,
        category: meal.RecipeCategory || meal.category || 'General',
        prepTime: meal.PrepTime || meal.prepTime || 'Unknown',
        totalTime: meal.TotalTime || meal.totalTime || 'Unknown',
        servings: meal.RecipeServings || meal.servings || 1,
        ingredients: meal.RecipeIngredientParts || meal.ingredients || '',
        instructions: meal.RecipeInstructions || meal.instructions || '',
        keywords: meal.Keywords || meal.keywords || '',
        images: extractFirstImage(meal.Images || meal.images || '')
      };
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
        console.error('Signup error:', error);
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
        console.error('Login error:', error);
        return { success: false, message: 'Login failed', user: null, token: null };
      }
    }
  }
};

// Helper function to extract first image URL from the Images field
function extractFirstImage(imagesField) {
  console.log('=== extractFirstImage called with:', imagesField);
  
  if (!imagesField) {
    console.log('No images field provided');
    return '';
  }
  
  try {
    // Handle R-style vector format: c("url1", "url2", "url3")
    if (imagesField.includes('c(') && imagesField.includes(')')) {
      console.log('Detected R-style format');
      const urls = imagesField.match(/"([^"]+)"/g);
      console.log('Extracted URLs:', urls);
      if (urls && urls.length > 0) {
        // Remove quotes and return first URL
        const firstUrl = urls[0].replace(/"/g, '');
        console.log('Returning first URL:', firstUrl);
        return firstUrl;
      }
    }
    
    // Handle comma-separated URLs
    if (imagesField.includes(',')) {
      console.log('Detected comma-separated format');
      const urls = imagesField.split(',').map(url => url.trim().replace(/"/g, ''));
      console.log('Split URLs:', urls);
      const firstUrl = urls[0] || '';
      console.log('Returning first URL:', firstUrl);
      return firstUrl;
    }
    
    // Single URL
    console.log('Detected single URL format');
    const cleanUrl = imagesField.replace(/"/g, '');
    console.log('Returning cleaned URL:', cleanUrl);
    return cleanUrl;
    
  } catch (error) {
    console.error('Error extracting image:', error);
    return imagesField || '';
  }
}

module.exports = { resolvers };
