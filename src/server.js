require('dotenv').config();
const express = require('express');
const { ApolloServer } = require('apollo-server-express');
const cors = require('cors');
const { createContext } = require('./middleware/auth');
const { connectDB } = require('./config/database');
const { typeDefs } = require('./schemas');
const { resolvers } = require('./resolvers');

const app = express();
app.use(cors());
app.use(express.json());

const apolloServer = new ApolloServer({
  typeDefs,
  resolvers,
  context: createContext
});

async function startServer() {
  await apolloServer.start();
  apolloServer.applyMiddleware({ app });
  
  await connectDB();
  
  const PORT = process.env.PORT || 4000;
  app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`GraphQL: http://localhost:${PORT}/graphql`);
  });
}

startServer();
