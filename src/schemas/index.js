const { gql } = require('apollo-server-express');

const typeDefs = gql`
  type User {
    id: ID!
    email: String!
    firstName: String!
    lastName: String!
    age: Int!
    gender: String!
    weightKg: Float!
    heightCm: Float!
    activityLevel: String!
    weightGoal: String!
  }

  type Meal {
    id: ID!
    name: String!
    calories: Float!
    protein: Float!
    fat: Float!
    carbs: Float!
    fiber: Float!
    sugar: Float!
    sodium: Float!
    cholesterol: Float!
    saturatedFat: Float!
    weightLossScore: Float!
    isGoodMeal: Int!
    category: String!
    prepTime: String!
    totalTime: String!
    servings: Int!
    images: String!
    keywords: String!
    ingredients: String!
  }

  type MealDetail {
    id: ID!
    name: String!
    calories: Float!
    protein: Float!
    fat: Float!
    carbs: Float!
    fiber: Float!
    sugar: Float!
    sodium: Float!
    cholesterol: Float!
    saturatedFat: Float!
    weightLossScore: Float!
    isGoodMeal: Int!
    category: String!
    prepTime: String!
    totalTime: String!
    servings: Int!
    ingredients: String!
    instructions: String!
    keywords: String!
    images: String!
  }

  type AuthResponse {
    success: Boolean!
    message: String!
    user: User
    token: String
  }

  input UserInput {
    email: String!
    password: String!
    firstName: String!
    lastName: String!
    age: Int!
    gender: String!
    weightKg: Float!
    heightCm: Float!
    activityLevel: String!
    weightGoal: String!
  }

  input LoginInput {
    email: String!
    password: String!
  }

  type Query {
    me: User
    getMeals: [Meal!]!
    getRecommendations(
      weightGoal: String
      maxCalories: Float
      minProtein: Float
    ): [Meal!]!
    getMealById(id: String!): MealDetail
  }

  type Mutation {
    signup(input: UserInput!): AuthResponse!
    login(input: LoginInput!): AuthResponse!
  }
`;

module.exports = { typeDefs };
