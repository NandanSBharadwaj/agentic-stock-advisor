#  Agentic Stock Advisor

AI-powered portfolio optimization system that transforms natural language investment goals into mathematically optimal stock allocations.

##  Features

-  **Natural Language Input**: Describe your investment goals in plain English
-  **AI Stock Discovery**: Claude AI identifies relevant stocks based on your criteria
-  **Real Market Data**: Fetches live historical data from Yahoo Finance
-  **Mathematical Optimization**: Gurobi solver finds provably optimal allocations
-  **Export Reports**: Generate professional PDF and CSV reports
-  **Risk-Aware**: Automatically manages portfolio risk and diversification

##  Tech Stack

- **LLM**: Anthropic Claude (Sonnet 4)
- **Optimization**: Gurobi Optimizer
- **Data**: yfinance (Yahoo Finance API)
- **Language**: Python 3.10+
- **Libraries**: pandas, reportlab

##  Installation

### Prerequisites
- Python 3.9+
- Gurobi academic license (free for students)
- Anthropic API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/YOUR-USERNAME/agentic-stock-advisor.git
cd agentic-stock-advisor
```

2. Create virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install anthropic yfinance gurobipy pandas reportlab
```

4. Set up environment variables:

Create a `.env` file:
```
ANTHROPIC_API_KEY=your-claude-api-key-here
```

##  Usage

Run the advisor:
```bash
python3 ultimate_stock_advisor.py
```

### Example Prompts
```
💭 Investment goal: Aggressive AI stocks
💭 Investment goal: Conservative healthcare dividend stocks
💭 Investment goal: Clean energy with moderate risk
💭 Investment goal: High growth tech stocks
```

### Sample Output
```
================================================================================
 RECOMMENDED PORTFOLIO
================================================================================

NVDA: 20.00%
  NVIDIA Corporation
  Leading AI chip manufacturer...

TSLA: 20.00%
  Tesla Inc
  Electric vehicle pioneer...

 Expected Annual Return: 54.54%
 Portfolio Beta: 1.88
================================================================================
```

##  Architecture
```
Natural Language Goal
    ↓
AI Stock Discovery (Claude) → Identifies relevant stocks
    ↓
Market Data Fetcher (yfinance) → Gets real returns & risk metrics
    ↓
Portfolio Optimizer (Gurobi) → Finds mathematically optimal allocation
    ↓
Report Generator → Exports to PDF/CSV
```

##  Features in Detail

### 1. Natural Language Understanding
Uses Claude AI to parse investment goals and extract:
- Risk tolerance (conservative/moderate/aggressive)
- Industry preferences
- Investment themes
- Target returns

### 2. Dynamic Stock Discovery
No pre-defined stock lists. The AI discovers relevant stocks based on:
- Sector alignment
- Risk profile matching
- Market capitalization
- Liquidity requirements

### 3. Mathematical Optimization
Gurobi solver maximizes expected returns subject to:
- Budget constraint (weights sum to 100%)
- Risk limits (portfolio beta constraints)
- Diversification (maximum allocation per stock)
- Non-negativity (no short positions)

### 4. Real-Time Data
Fetches 1-year historical data to calculate:
- Expected annual returns
- Beta (systematic risk vs S&P 500)
- Volatility (standard deviation)

##  Educational Value

This project demonstrates:
- **LLM Integration**: Practical use of Claude API for structured extraction
- **Financial Engineering**: Portfolio optimization theory in practice
- **Multi-Agent Systems**: Orchestrating AI and mathematical solvers
- **Real-World APIs**: Working with financial data sources
- **Software Engineering**: Clean architecture, error handling, export functionality

##  Security Notes

- Never commit `.env` file (contains API keys)
- `.gitignore` is configured to exclude sensitive files
- API keys should be rotated periodically

##  License

This project is for educational purposes. Gurobi academic license is for non-commercial use only.

##  Acknowledgments

- Anthropic Claude for natural language understanding
- Gurobi for optimization engine
- Yahoo Finance for market data

## 📧 Contact

Nandan Shankara Bharadwaj - nandanbharadwaj2@gmail.com

---

** Disclaimer**: This tool is for educational purposes only. Not financial advice. Always consult a licensed financial advisor before making investment decisions.
