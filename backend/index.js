const express = require('express');
const cors = require('cors');
const app = express();
const port = process.env.PORT || 5000;

app.use(cors()); // Enable CORS for local frontend requests
app.use(express.json()); // Parse JSON bodies

// Placeholder for /token endpoint (e.g., mock authentication)
app.get('/token', (req, res) => {
  // In production, implement real auth logic (e.g., JWT generation)
  res.json({ access_token: 'mock-token-for-local-dev' });
});

// Placeholder for /insights endpoint
app.post('/insights', (req, res) => {
  const { coin } = req.body;
  if (!coin) {
    return res.status(400).json({ error: 'Coin is required' });
  }
  // Replace with actual logic (e.g., fetch data from an API like CoinGecko)
  const mockInsight = `Local dev insight for ${coin}: Buy low, sell high!`; // Placeholder
  res.json({ insight: mockInsight });
});

// Placeholder for /portfolio endpoint
app.post('/portfolio', (req, res) => {
  const { portfolio } = req.body;
  if (!portfolio || !Array.isArray(portfolio)) {
    return res.status(400).json({ error: 'Portfolio array is required' });
  }
  // Replace with actual logic (e.g., calculate total value based on current prices)
  const mockTotalValue = portfolio.reduce((sum, item) => sum + (parseFloat(item.amount) || 0), 0) * 100; // Mock calculation
  const mockSuggestion = 'Diversify your holdings!'; // Placeholder
  res.json({ total_value: mockTotalValue, suggestion: mockSuggestion });
});

// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});