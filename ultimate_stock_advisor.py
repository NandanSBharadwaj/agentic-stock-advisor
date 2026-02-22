"""
Ultimate Stock Advisor: Goal + Industry → Auto Stock Discovery → Portfolio
"""

import os
from dotenv import load_dotenv
import anthropic
import yfinance as yf
import gurobipy as gp
from gurobipy import GRB
import json
import pandas as pd

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def discover_stocks_for_goal(user_goal: str) -> dict:
    """Use Claude to identify relevant stocks based on user's goal"""
    
    prompt = f"""You are a stock market expert. Based on this investment goal, identify relevant stocks.

USER GOAL: "{user_goal}"

Return ONLY valid JSON:
{{
  "goal_analysis": {{
    "industries": ["list of industries"],
    "risk_tolerance": "conservative|moderate|aggressive",
    "investment_theme": "brief description"
  }},
  "recommended_stocks": [
    {{
      "ticker": "NVDA",
      "company": "NVIDIA",
      "reason": "AI chip leader",
      "risk_level": "high",
      "sector": "AI"
    }}
  ],
  "portfolio_constraints": {{
    "max_single_stock": 0.35,
    "target_return": 0.15,
    "max_beta": 1.5
  }}
}}

List 8-12 US-traded stocks. Return ONLY JSON, no markdown."""

    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = message.content[0].text.strip()
        
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        return json.loads(response_text)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


def validate_and_fetch_stock_data(stocks, period="1y"):
    """Validate tickers and fetch real market data"""
    
    tickers = [s['ticker'] for s in stocks]
    print(f"\n📈 Fetching data for {len(tickers)} stocks: {', '.join(tickers)}")
    
    stock_data = {}
    
    try:
        # Download market data first
        print("   Downloading S&P 500 data...")
        market_raw = yf.download("^GSPC", period=period, progress=False)
        
        # Check structure and extract
        if isinstance(market_raw, pd.DataFrame):
            if 'Adj Close' in market_raw.columns:
                market_prices = market_raw['Adj Close']
            elif len(market_raw.columns) > 0:
                market_prices = market_raw.iloc[:, 0]  # Take first column
            else:
                print("❌ Cannot extract market data")
                return None
        else:
            market_prices = market_raw
        
        market_returns = market_prices.pct_change().dropna()
        market_var = market_returns.var()
        
        print("   Downloading stock data...")
        # Download stocks one by one for reliability
        for stock_info in stocks:
            ticker = stock_info['ticker']
            
            try:
                stock_raw = yf.download(ticker, period=period, progress=False)
                
                # Extract price data
                if isinstance(stock_raw, pd.DataFrame):
                    if 'Adj Close' in stock_raw.columns:
                        stock_prices = stock_raw['Adj Close']
                    elif len(stock_raw.columns) > 0:
                        stock_prices = stock_raw.iloc[:, 0]
                    else:
                        print(f"   ⚠️  Skipping {ticker} - no price data")
                        continue
                else:
                    stock_prices = stock_raw
                
                # Calculate returns
                stock_returns = stock_prices.pct_change().dropna()
                
                # Align with market
                common_dates = stock_returns.index.intersection(market_returns.index)
                
                if len(common_dates) < 50:
                    print(f"   ⚠️  Skipping {ticker} - insufficient data ({len(common_dates)} days)")
                    continue
                
                stock_returns_aligned = stock_returns.loc[common_dates]
                market_returns_aligned = market_returns.loc[common_dates]
                
                # Calculate metrics
                expected_return = float(stock_returns_aligned.mean() * 252)
                volatility = float(stock_returns_aligned.std() * (252 ** 0.5))
                beta = float(stock_returns_aligned.cov(market_returns_aligned) / market_var)
                
                stock_data[ticker] = {
                    "ticker": ticker,
                    "company": stock_info['company'],
                    "expected_return": expected_return,
                    "volatility": volatility,
                    "beta": beta,
                    "reason": stock_info['reason'],
                    "sector": stock_info.get('sector', 'Unknown'),
                    "risk_level": stock_info.get('risk_level', 'moderate')
                }
                
                print(f"   ✓ {ticker}: Return {expected_return*100:.1f}%, Beta {beta:.2f}")
                
            except Exception as e:
                print(f"   ⚠️  Skipping {ticker} - error: {str(e)[:50]}")
                continue
        
        print(f"\n✓ Successfully fetched data for {len(stock_data)} stocks")
        return stock_data
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return None
    
def optimize_portfolio(stock_data, constraints):
    """Build optimal portfolio using Gurobi"""
    
    tickers = list(stock_data.keys())
    
    if len(tickers) < 2:
        print("❌ Need at least 2 stocks")
        return None
    
    print(f"\n🔧 Optimizing {len(tickers)} stocks...")
    
    m = gp.Model("Portfolio")
    m.setParam('OutputFlag', 0)
    
    max_single = constraints.get('max_single_stock', 0.4)
    weights = m.addVars(tickers, lb=0.0, ub=max_single, name="weight")
    
    m.addConstr(gp.quicksum(weights[t] for t in tickers) == 1.0, "budget")
    
    max_beta = constraints.get('max_beta', 2.0)
    portfolio_beta = gp.quicksum(weights[t] * stock_data[t]['beta'] for t in tickers)
    m.addConstr(portfolio_beta <= max_beta, "beta_limit")
    
    portfolio_return = gp.quicksum(weights[t] * stock_data[t]['expected_return'] for t in tickers)
    m.setObjective(portfolio_return, GRB.MAXIMIZE)
    

    
    m.optimize()
    
    if m.Status == GRB.OPTIMAL:
        allocation = {t: weights[t].X for t in tickers if weights[t].X > 0.01}
        
        total_return = sum(allocation[t] * stock_data[t]['expected_return'] for t in allocation)
        total_beta = sum(allocation[t] * stock_data[t]['beta'] for t in allocation)
        
        return {
            "allocation": allocation,
            "expected_return": total_return,
            "portfolio_beta": total_beta,
            "num_stocks": len(allocation)
        }
    else:
        print(f"❌ Optimization failed")
        return None
    
def export_to_csv(user_goal, discovery, stock_data, result, filename="portfolio.csv"):
    """Export portfolio to CSV file"""
    import csv
    from datetime import datetime
    
    try:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Header info
            writer.writerow(['Portfolio Analysis Report'])
            writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Investment Goal:', user_goal])
            writer.writerow([])
            
            # Goal Analysis
            writer.writerow(['GOAL ANALYSIS'])
            writer.writerow(['Industries:', ', '.join(discovery['goal_analysis']['industries'])])
            writer.writerow(['Risk Tolerance:', discovery['goal_analysis']['risk_tolerance']])
            writer.writerow(['Theme:', discovery['goal_analysis']['investment_theme']])
            writer.writerow([])
            
            # Portfolio Metrics
            writer.writerow(['PORTFOLIO METRICS'])
            writer.writerow(['Expected Annual Return:', f"{result['expected_return']*100:.2f}%"])
            writer.writerow(['Portfolio Beta:', f"{result['portfolio_beta']:.2f}"])
            writer.writerow(['Number of Stocks:', result['num_stocks']])
            writer.writerow([])
            
            # Allocation
            writer.writerow(['PORTFOLIO ALLOCATION'])
            writer.writerow(['Ticker', 'Company', 'Allocation %', 'Expected Return', 'Beta', 'Volatility', 'Reason'])
            
            for ticker, weight in sorted(result['allocation'].items(), key=lambda x: -x[1]):
                data = stock_data[ticker]
                writer.writerow([
                    ticker,
                    data['company'],
                    f"{weight*100:.2f}%",
                    f"{data['expected_return']*100:.2f}%",
                    f"{data['beta']:.2f}",
                    f"{data['volatility']*100:.2f}%",
                    data['reason']
                ])
            
            writer.writerow([])
            
            # All Discovered Stocks
            writer.writerow(['ALL DISCOVERED STOCKS'])
            writer.writerow(['Ticker', 'Company', 'Expected Return', 'Beta', 'Volatility', 'Risk Level', 'Sector'])
            
            for ticker, data in sorted(stock_data.items(), key=lambda x: -x[1]['expected_return']):
                writer.writerow([
                    ticker,
                    data['company'],
                    f"{data['expected_return']*100:.2f}%",
                    f"{data['beta']:.2f}",
                    f"{data['volatility']*100:.2f}%",
                    data['risk_level'],
                    data['sector']
                ])
        
        print(f"\n✓ Portfolio exported to {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error exporting CSV: {e}")
        return None


def export_to_pdf(user_goal, discovery, stock_data, result, filename="portfolio.pdf"):
    """Export portfolio to PDF file"""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from datetime import datetime
    
    try:
        doc = SimpleDocTemplate(filename, pagesize=letter)
        elements = []
        styles = getSampleStyleSheet()
        
        # Title
        title = Paragraph("<b>Portfolio Analysis Report</b>", styles['Title'])
        elements.append(title)
        elements.append(Spacer(1, 12))
        
        # Header info
        header_data = [
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Investment Goal:', user_goal],
        ]
        header_table = Table(header_data, colWidths=[120, 400])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 20))
        
        # Goal Analysis Section
        elements.append(Paragraph("<b>Goal Analysis</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        goal_data = [
            ['Industries:', ', '.join(discovery['goal_analysis']['industries'][:3]) + '...'],
            ['Risk Tolerance:', discovery['goal_analysis']['risk_tolerance'].upper()],
            ['Theme:', discovery['goal_analysis']['investment_theme']],
        ]
        goal_table = Table(goal_data, colWidths=[120, 400])
        goal_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
        ]))
        elements.append(goal_table)
        elements.append(Spacer(1, 20))
        
        # Portfolio Metrics Section
        elements.append(Paragraph("<b>Portfolio Metrics</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        metrics_data = [
            ['Expected Annual Return:', f"{result['expected_return']*100:.2f}%"],
            ['Portfolio Beta:', f"{result['portfolio_beta']:.2f}"],
            ['Number of Stocks:', str(result['num_stocks'])],
        ]
        metrics_table = Table(metrics_data, colWidths=[150, 100])
        metrics_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (1, 0), (1, 0), colors.green),
            ('BACKGROUND', (0, 0), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ]))
        elements.append(metrics_table)
        elements.append(Spacer(1, 20))
        
        # Allocation Table
        elements.append(Paragraph("<b>Recommended Allocation</b>", styles['Heading2']))
        elements.append(Spacer(1, 6))
        
        allocation_data = [['Ticker', 'Company', 'Allocation', 'Return', 'Beta']]
        
        for ticker, weight in sorted(result['allocation'].items(), key=lambda x: -x[1]):
            data = stock_data[ticker]
            allocation_data.append([
                ticker,
                data['company'][:25],
                f"{weight*100:.1f}%",
                f"{data['expected_return']*100:.1f}%",
                f"{data['beta']:.2f}"
            ])
        
        allocation_table = Table(allocation_data, colWidths=[60, 200, 70, 70, 50])
        allocation_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        elements.append(allocation_table)
        
        # Build PDF
        doc.build(elements)
        
        print(f"\n✓ Portfolio exported to {filename}")
        return filename
        
    except Exception as e:
        print(f"❌ Error exporting PDF: {e}")
        import traceback
        traceback.print_exc()
        return None

def run_advisor(user_goal: str):
    """Complete pipeline"""
    
    print("=" * 80)
    print("🚀 ULTIMATE STOCK ADVISOR")
    print("=" * 80)
    print(f"\n💭 YOUR GOAL: \"{user_goal}\"")
    
    print("\n🤖 Discovering stocks...")
    discovery = discover_stocks_for_goal(user_goal)
    
    if not discovery:
        print("Failed to discover stocks")
        return
    
    print(f"✓ Industries: {', '.join(discovery['goal_analysis']['industries'])}")
    print(f"✓ Risk: {discovery['goal_analysis']['risk_tolerance']}")
    print(f"✓ Found {len(discovery['recommended_stocks'])} stocks")
    
    stock_data = validate_and_fetch_stock_data(discovery['recommended_stocks'])
    
    if not stock_data or len(stock_data) < 2:
        print("Not enough valid stocks")
        return
    
    print("\n📊 STOCKS DISCOVERED:")
    for ticker, data in sorted(stock_data.items(), key=lambda x: -x[1]['expected_return']):
        risk_emoji = "🔴" if data['risk_level'] == 'high' else "🟡" if data['risk_level'] == 'moderate' else "🟢"
        print(f"{risk_emoji} {ticker}: Return {data['expected_return']*100:.1f}% | Beta {data['beta']:.2f}")
    
    result = optimize_portfolio(stock_data, discovery['portfolio_constraints'])
    
    if result:
        print("\n" + "=" * 80)
        print("🎯 RECOMMENDED PORTFOLIO")
        print("=" * 80)
        
        for ticker, weight in sorted(result['allocation'].items(), key=lambda x: -x[1]):
            print(f"\n{ticker}: {weight*100:.2f}%")
            print(f"  {stock_data[ticker]['company']}")
            print(f"  {stock_data[ticker]['reason'][:60]}...")
        
        print(f"\n💰 Expected Return: {result['expected_return']*100:.2f}%")
        print(f"📈 Portfolio Beta: {result['portfolio_beta']:.2f}")
        print("=" * 80)
        
        # ADD THIS NEW SECTION HERE:
        print("\n📄 Export Options:")
        export_choice = input("Export to CSV, PDF, both, or skip? (csv/pdf/both/skip): ").lower()
        
        if export_choice in ['csv', 'both']:
            csv_file = export_to_csv(user_goal, discovery, stock_data, result)
            if csv_file:
                print(f"✅ CSV saved: {csv_file}")
        
        if export_choice in ['pdf', 'both']:
            pdf_file = export_to_pdf(user_goal, discovery, stock_data, result)
            if pdf_file:
                print(f"✅ PDF saved: {pdf_file}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Welcome to Ultimate Stock Advisor!")
    print("=" * 80)
    
    while True:
        user_goal = input("\n💭 Investment goal (or 'quit'): ")
        
        if user_goal.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if user_goal.strip():
            run_advisor(user_goal)
        else:
            print("Please enter a goal")