import discord
from discord.ext import commands, tasks
import yfinance as yf
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Tuple, Optional
import logging
from dotenv import load_dotenv

# Load environment variables (works locally and on cloud)
if os.path.exists('.env'):
    load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Bot configuration
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# Trading pairs configuration with Yahoo Finance symbols
TRADING_PAIRS = {
    # Major Pairs - 5 Min
    'major_pairs': {
        'timeframe': '5m',
        'pairs': {
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X', 
            'USDJPY': 'USDJPY=X',
            'USDCHF': 'USDCHF=X',
            'AUDUSD': 'AUDUSD=X',
            'USDCAD': 'USDCAD=X',
            'NZDUSD': 'NZDUSD=X'
        }
    },
    # Minor & Cross Pairs - 15 Min
    'minor_cross_pairs': {
        'timeframe': '15m',
        'pairs': {
            'EURGBP': 'EURGBP=X',
            'EURJPY': 'EURJPY=X',
            'EURCHF': 'EURCHF=X',
            'EURAUD': 'EURAUD=X',
            'EURCAD': 'EURCAD=X',
            'EURNZD': 'EURNZD=X',
            'GBPJPY': 'GBPJPY=X',
            'GBPCHF': 'GBPCHF=X',
            'GBPAUD': 'GBPAUD=X',
            'GBPCAD': 'GBPCAD=X',
            'GBPNZD': 'GBPNZD=X'
        }
    },
    # Indices - 5 Min
    'indices': {
        'timeframe': '5m',
        'pairs': {
            'US500': '^GSPC',  # S&P 500
            'US30': '^DJI',    # Dow Jones
            'US100': '^IXIC',  # NASDAQ
            'UK100': '^FTSE',  # FTSE 100
            'DE40': '^GDAXI',  # DAX
            'JP225': '^N225'   # Nikkei
        }
    },
    # Commodities - 15 Min  
    'commodities': {
        'timeframe': '15m',
        'pairs': {
            'GOLD': 'GC=F',    # Gold futures
            'SILVER': 'SI=F',  # Silver futures
            'OIL': 'CL=F',     # Oil futures
            'NATGAS': 'NG=F'   # Natural Gas
        }
    }
}

class RSICalculator:
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> List[float]:
        """Calculate RSI for given price series"""
        if len(prices) < period + 1:
            return [50] * len(prices)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # Calculate initial averages
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        
        rsi_values = []
        
        # Calculate RSI for each point
        for i in range(period, len(deltas) + 1):
            if i == period:
                current_avg_gain = avg_gain
                current_avg_loss = avg_loss
            else:
                current_avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                current_avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
                avg_gain = current_avg_gain
                avg_loss = current_avg_loss
            
            if current_avg_loss == 0:
                rsi = 100
            else:
                rs = current_avg_gain / current_avg_loss
                rsi = 100 - (100 / (1 + rs))
            
            rsi_values.append(rsi)
        
        return rsi_values

class DivergenceDetector:
    @staticmethod
    def find_peaks_troughs(data: List[float], window: int = 3) -> Tuple[List[int], List[int]]:
        """Find peaks and troughs in data"""
        peaks = []
        troughs = []
        
        for i in range(window, len(data) - window):
            # Check for peak
            if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
               all(data[i] >= data[i+j] for j in range(1, window+1)):
                peaks.append(i)
            
            # Check for trough  
            if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
               all(data[i] <= data[i+j] for j in range(1, window+1)):
                troughs.append(i)
        
        return peaks, troughs
    
    @staticmethod
    def detect_divergence(prices: List[float], rsi_values: List[float]) -> Dict:
        """Detect bullish and bearish divergences"""
        if len(prices) < 20 or len(rsi_values) < 20:
            return {"bullish": [], "bearish": []}
        
        price_peaks, price_troughs = DivergenceDetector.find_peaks_troughs(prices)
        rsi_peaks, rsi_troughs = DivergenceDetector.find_peaks_troughs(rsi_values)
        
        bullish_divergences = []
        bearish_divergences = []
        
        # Bullish divergence: price makes lower lows, RSI makes higher lows
        if len(price_troughs) >= 2 and len(rsi_troughs) >= 2:
            for i in range(1, min(len(price_troughs), len(rsi_troughs))):
                price_idx1, price_idx2 = price_troughs[i-1], price_troughs[i]
                
                # Find closest RSI trough
                rsi_idx1 = min(rsi_troughs, key=lambda x: abs(x - price_idx1))
                rsi_idx2 = min(rsi_troughs, key=lambda x: abs(x - price_idx2))
                
                if (prices[price_idx2] < prices[price_idx1] and 
                    rsi_values[rsi_idx2] > rsi_values[rsi_idx1]):
                    bullish_divergences.append({
                        'type': 'bullish',
                        'strength': abs(rsi_values[rsi_idx2] - rsi_values[rsi_idx1])
                    })
        
        # Bearish divergence: price makes higher highs, RSI makes lower highs
        if len(price_peaks) >= 2 and len(rsi_peaks) >= 2:
            for i in range(1, min(len(price_peaks), len(rsi_peaks))):
                price_idx1, price_idx2 = price_peaks[i-1], price_peaks[i]
                
                # Find closest RSI peak
                rsi_idx1 = min(rsi_peaks, key=lambda x: abs(x - price_idx1))
                rsi_idx2 = min(rsi_peaks, key=lambda x: abs(x - price_idx2))
                
                if (prices[price_idx2] > prices[price_idx1] and 
                    rsi_values[rsi_idx2] < rsi_values[rsi_idx1]):
                    bearish_divergences.append({
                        'type': 'bearish',
                        'strength': abs(rsi_values[rsi_idx1] - rsi_values[rsi_idx2])
                    })
        
        return {
            "bullish": bullish_divergences,
            "bearish": bearish_divergences
        }

class YahooFinanceProvider:
    def __init__(self):
        """No API key needed for Yahoo Finance!"""
        pass
    
    async def get_intraday_data(self, symbol: str, yahoo_symbol: str, interval: str) -> Optional[Dict]:
        """Fetch intraday data from Yahoo Finance"""
        try:
            logger.info(f"Fetching data for {symbol} ({yahoo_symbol}) - {interval}")
            
            # Get data from Yahoo Finance
            ticker = yf.Ticker(yahoo_symbol)
            
            # Get last 2 days of data to ensure we have enough
            data = ticker.history(period='2d', interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for {yahoo_symbol}")
                return None
            
            # Convert to our expected format
            converted_data = {}
            for timestamp, row in data.iterrows():
                converted_data[timestamp.strftime('%Y-%m-%d %H:%M:%S')] = {
                    '4. close': str(row['Close'])
                }
            
            logger.info(f"Retrieved {len(converted_data)} data points for {symbol}")
            return converted_data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return None

class RSIDivergenceBot:
    def __init__(self, market_data_provider: YahooFinanceProvider, channel_id: int):
        self.market_data = market_data_provider
        self.channel_id = channel_id
        self.last_alerts = {}
        self.rsi_calculator = RSICalculator()
        self.divergence_detector = DivergenceDetector()
    
    async def analyze_pair(self, symbol: str, yahoo_symbol: str, timeframe: str) -> Optional[Dict]:
        """Analyze a trading pair for RSI divergence"""
        data = await self.market_data.get_intraday_data(symbol, yahoo_symbol, timeframe)
        
        if not data:
            return None
        
        # Convert to lists
        timestamps = []
        prices = []
        
        for timestamp, ohlcv in sorted(data.items()):
            timestamps.append(timestamp)
            prices.append(float(ohlcv['4. close']))
        
        if len(prices) < 50:
            logger.warning(f"Not enough data for {symbol}: {len(prices)} points")
            return None
        
        # Calculate RSI
        rsi_values = self.rsi_calculator.calculate_rsi(prices)
        
        # Detect divergences
        divergences = self.divergence_detector.detect_divergence(prices, rsi_values)
        
        if divergences['bullish'] or divergences['bearish']:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'current_price': prices[-1],
                'current_rsi': rsi_values[-1] if rsi_values else 50,
                'divergences': divergences,
                'timestamp': timestamps[-1]
            }
        
        return None
    
    def create_alert_embed(self, analysis: Dict) -> discord.Embed:
        """Create Discord embed for divergence alert"""
        symbol = analysis['symbol']
        divergences = analysis['divergences']
        
        if divergences['bullish']:
            title = f"🟢 BULLISH RSI DIVERGENCE - {symbol}"
            color = 0x00ff00
            div_type = "BULLISH"
        else:
            title = f"🔴 BEARISH RSI DIVERGENCE - {symbol}"
            color = 0xff0000
            div_type = "BEARISH"
        
        embed = discord.Embed(
            title=title,
            color=color,
            timestamp=datetime.utcnow()
        )
        
        embed.add_field(
            name="📊 Current Data",
            value=f"**Price:** {analysis['current_price']:.5f}\n"
                  f"**RSI:** {analysis['current_rsi']:.2f}\n"
                  f"**Timeframe:** {analysis['timeframe']}",
            inline=True
        )
        
        embed.add_field(
            name="🎯 Signal Type", 
            value=f"**{div_type} DIVERGENCE**\n"
                  f"Strong momentum shift detected",
            inline=True
        )
        
        embed.add_field(
            name="⏰ Time",
            value=f"{analysis['timestamp']}\n"
                  f"Alert: {datetime.now().strftime('%H:%M:%S')}",
            inline=True
        )
        
        total_divs = len(divergences['bullish']) + len(divergences['bearish'])
        embed.add_field(
            name="📈 Divergence Strength",
            value=f"**{total_divs}** divergence(s) detected\n"
                  f"Data source: Yahoo Finance",
            inline=False
        )
        
        embed.set_footer(text="CAMBIST Trading Bot | Yahoo Finance Data")
        
        return embed
    
    async def scan_and_alert(self):
        """Scan all pairs and send alerts"""
        channel = bot.get_channel(self.channel_id)
        if not channel:
            logger.error(f"Channel {self.channel_id} not found")
            return
        
        total_pairs = 0
        alerts_sent = 0
        successful_fetches = 0
        failed_fetches = 0
        
        # Send scan start message
        start_embed = discord.Embed(
            title="🔍 RSI Divergence Scan Started",
            description="Scanning all pairs with Yahoo Finance...",
            color=0x0099ff,
            timestamp=datetime.utcnow()
        )
        scan_message = await channel.send(embed=start_embed)
        
        for category, config in TRADING_PAIRS.items():
            timeframe = config['timeframe']
            
            for symbol, yahoo_symbol in config['pairs'].items():
                total_pairs += 1
                try:
                    # Check cooldown
                    last_alert_time = self.last_alerts.get(symbol, datetime.min)
                    if datetime.now() - last_alert_time < timedelta(minutes=30):
                        logger.info(f"Skipping {symbol} - still in cooldown")
                        continue
                    
                    logger.info(f"Analyzing {symbol} ({yahoo_symbol})")
                    analysis = await self.analyze_pair(symbol, yahoo_symbol, timeframe)
                    
                    if analysis:
                        embed = self.create_alert_embed(analysis)
                        await channel.send(embed=embed)
                        
                        self.last_alerts[symbol] = datetime.now()
                        alerts_sent += 1
                        logger.info(f"🚨 Alert sent for {symbol}")
                        successful_fetches += 1
                    else:
                        logger.info(f"✅ {symbol} - No divergence detected")
                        successful_fetches += 1
                    
                    # Small delay to be respectful
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing {symbol}: {e}")
                    failed_fetches += 1
                    continue
        
        # Send scan complete message with details
        complete_embed = discord.Embed(
            title="✅ RSI Divergence Scan Complete",
            color=0x00ff00,
            timestamp=datetime.utcnow()
        )
        
        complete_embed.add_field(
            name="📊 Scan Results",
            value=f"**Total Pairs:** {total_pairs}\n"
                  f"**Successful:** {successful_fetches}\n" 
                  f"**Failed:** {failed_fetches}\n"
                  f"**Alerts Sent:** {alerts_sent}",
            inline=True
        )
        
        complete_embed.add_field(
            name="📈 Data Quality",
            value=f"**Success Rate:** {(successful_fetches/total_pairs*100):.1f}%\n"
                  f"**API Source:** Yahoo Finance\n"
                  f"**Next Scan:** 10 minutes",
            inline=True
        )
        
        if alerts_sent == 0:
            complete_embed.add_field(
                name="🎯 No Divergences Found",
                value="This is normal! RSI divergences are rare patterns.\n"
                      "The bot will alert you when significant divergences occur.",
                inline=False
            )
        
        # Update the original scan message
        await scan_message.edit(embed=complete_embed)
        
        logger.info(f"Scan complete: {total_pairs} pairs checked, {successful_fetches} successful, {alerts_sent} alerts sent")

@bot.command(name='test')
async def test_alert(ctx):
    """Test alert functionality"""
    # Create a fake divergence for testing
    test_analysis = {
        'symbol': 'EURUSD',
        'timeframe': '5m',
        'current_price': 1.0850,
        'current_rsi': 32.5,
        'divergences': {
            'bullish': [{'type': 'bullish', 'strength': 15.2}],
            'bearish': []
        },
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    embed = scanner.create_alert_embed(test_analysis)
    await ctx.send("🧪 **TEST ALERT** - This is what a real divergence looks like:")
    await ctx.send(embed=embed)

@bot.command(name='datacheck')
async def data_quality_check(ctx):
    """Check if we're getting good data from Yahoo Finance"""
    await ctx.send("📊 Checking data quality from Yahoo Finance...")
    
    # Test a few major pairs
    test_pairs = [
        ('EURUSD', 'EURUSD=X'),
        ('GBPUSD', 'GBPUSD=X'), 
        ('USDJPY', 'USDJPY=X')
    ]
    
    results = []
    
    for symbol, yahoo_symbol in test_pairs:
        try:
            data = await market_provider.get_intraday_data(symbol, yahoo_symbol, '5m')
            if data:
                data_points = len(data)
                latest_time = max(data.keys()) if data else "No data"
                results.append(f"✅ **{symbol}**: {data_points} points, latest: {latest_time}")
            else:
                results.append(f"❌ **{symbol}**: No data received")
        except Exception as e:
            results.append(f"❌ **{symbol}**: Error - {str(e)}")
    
    embed = discord.Embed(
        title="📊 Data Quality Check Results",
        description="\n".join(results),
        color=0x0099ff,
        timestamp=datetime.utcnow()
    )
    
    await ctx.send(embed=embed)


# Initialize components
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
CHANNEL_ID = int(os.getenv('CHANNEL_ID', '0'))

market_provider = YahooFinanceProvider()
scanner = RSIDivergenceBot(market_provider, CHANNEL_ID)

@bot.event
async def on_ready():
    logger.info(f'{bot.user} has connected to Discord!')
    logger.info('Using Yahoo Finance for market data (unlimited free API)')
    if not market_scanner.is_running():
        market_scanner.start()

@tasks.loop(minutes=10)  # Scan every 10 minutes (can be more frequent with Yahoo)
async def market_scanner():
    """Main scanning loop"""
    try:
        await scanner.scan_and_alert()
    except Exception as e:
        logger.error(f"Error in market scanner: {e}")

@bot.command(name='scan')
async def manual_scan(ctx):
    """Manual scan command"""
    await ctx.send("🔍 Starting manual scan with Yahoo Finance...")
    await scanner.scan_and_alert()
    await ctx.send("✅ Scan complete!")

@bot.command(name='status')
async def bot_status(ctx):
    """Check bot status"""
    embed = discord.Embed(
        title="🤖 CAMBIST RSI Bot Status",
        color=0x0099ff,
        timestamp=datetime.utcnow()
    )
    
    total_pairs = sum(len(config['pairs']) for config in TRADING_PAIRS.values())
    
    embed.add_field(
        name="📊 Monitoring",
        value=f"**{total_pairs}** trading pairs\n"
              f"**Yahoo Finance** API (FREE!)",
        inline=True
    )
    
    embed.add_field(
        name="⏰ Scan Frequency", 
        value="Every 10 minutes\nAutomatic scanning active",
        inline=True
    )
    
    embed.add_field(
        name="🎯 Last Alerts",
        value=f"{len(scanner.last_alerts)} pairs alerted recently",
        inline=True
    )
    
    await ctx.send(embed=embed)

@bot.command(name='pairs')
async def list_pairs(ctx):
    """List all monitored pairs"""
    embed = discord.Embed(
        title="📋 Monitored Trading Pairs",
        description="All pairs monitored with Yahoo Finance data",
        color=0x9932cc,
        timestamp=datetime.utcnow()
    )
    
    for category, config in TRADING_PAIRS.items():
        pairs_text = ', '.join(config['pairs'].keys())
        embed.add_field(
            name=f"{category.replace('_', ' ').title()} ({config['timeframe']})",
            value=pairs_text,
            inline=False
        )
    
    await ctx.send(embed=embed)

if __name__ == "__main__":
    if not DISCORD_TOKEN:
        print("❌ DISCORD_TOKEN environment variable not set!")
        exit(1)
    if not CHANNEL_ID:
        print("❌ CHANNEL_ID environment variable not set!")
        exit(1)
    
    print("🚀 Starting CAMBIST RSI Divergence Bot with Yahoo Finance...")
    print("📊 Free unlimited market data API")
    bot.run(DISCORD_TOKEN)