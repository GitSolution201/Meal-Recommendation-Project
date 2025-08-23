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
    weightKg: String!
    heightCm: String!
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
    getRecommendations(weightGoal: String!): [Meal!]!
  }
  
  type Mutation {
    signup(input: UserInput!): AuthResponse!
    login(input: LoginInput!): AuthResponse!
  }
`;

module.exports = { typeDefs };
