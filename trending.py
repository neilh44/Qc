
import pennylane as qml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import requests
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class EnhancedQuantumCryptoPredictor:
    def __init__(self, n_qubits: int = 3):
        self.n_qubits = n_qubits
        self.feature_scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
        self.base_url = "https://api.binance.com/api/v3"
        
        self.feature_dev = qml.device("default.qubit", wires=self.n_qubits)
        self.prediction_dev = qml.device("default.qubit", wires=self.n_qubits)
        
        self.feature_circuit = qml.QNode(self.quantum_feature_extraction, self.feature_dev)
        self.prediction_circuit = qml.QNode(self.quantum_prediction_circuit, self.prediction_dev)

    def quantum_feature_extraction(self, features):
        # Input encoding layer
        for i in range(self.n_qubits):
            qml.RY(features[i], wires=i)
            qml.RZ(np.pi * features[i]**2, wires=i)
        
        # Entanglement layer
        for i in range(self.n_qubits - 1):
            qml.CRZ(np.pi/2, wires=[i, i+1])
            qml.CNOT(wires=[i, i+1])
        
        # Non-linear transformation
        for i in range(self.n_qubits):
            qml.Rot(features[i], features[i]**2, features[i]**3, wires=i)
        
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliY(2))
        ]

    def quantum_prediction_circuit(self, features, weights):
        # Input encoding
        for i in range(self.n_qubits):
            qml.RX(features[i], wires=i)
            qml.RY(features[i]**2, wires=i)
            qml.RZ(features[i]**3, wires=i)
        
        # Entanglement layers
        for _ in range(2):
            for i in range(self.n_qubits - 1):
                qml.CNOT(wires=[i, i + 1])
            for i in range(self.n_qubits):
                qml.RY(weights[i % len(weights)], wires=i)
        
        return [
            qml.expval(qml.PauliZ(0)),
            qml.expval(qml.PauliX(1)),
            qml.expval(qml.PauliY(2))
        ]

    def get_current_price(self, symbol: str) -> float:
        try:
            url = f"{self.base_url}/ticker/price"
            response = requests.get(url, params={'symbol': symbol}, verify=False)
            data = response.json()
            return float(data['price'])
        except Exception as e:
            print(f"Error fetching current price: {e}")
            return None

    def get_historical_data(self, symbol: str, interval: str, lookback_days: int) -> pd.DataFrame:
        try:
            start_time = int((datetime.now() - timedelta(days=lookback_days + 30)).timestamp() * 1000)
            
            url = f"{self.base_url}/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'startTime': start_time,
                'limit': 1000
            }
            
            response = requests.get(url, params=params, verify=False)
            klines = response.json()
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades_count',
                'taker_buy_volume', 'taker_buy_quote_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df.dropna(subset=['close'])
            
        except Exception as e:
            print(f"Error fetching historical data: {e}")
            return pd.DataFrame()

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        normalized_features = self.feature_scaler.fit_transform(features.reshape(-1, 1)).flatten()
        return normalized_features

    def prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        try:
            # Calculate traditional features
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std()
            df['momentum'] = df['returns'].rolling(window=10).mean()
            
            df_clean = df.dropna()
            if df_clean.empty:
                return None
            
            # Extract features
            features = np.column_stack([
                df_clean['close'].values,
                df_clean['volatility'].values,
                df_clean['momentum'].values
            ])
            
            # Normalize features
            normalized_features = np.array([
                self.normalize_features(feat) 
                for feat in features
            ])
            
            return normalized_features
            
        except Exception as e:
            print(f"Error in feature preparation: {e}")
            return None

    def analyze_market_condition(self, df: pd.DataFrame) -> Dict:
        try:
            # Calculate necessary indicators
            df['SMA20'] = df['close'].rolling(window=20).mean()
            df['SMA50'] = df['close'].rolling(window=50).mean()
            df['STD20'] = df['close'].rolling(window=20).std()
            df['Returns'] = df['close'].pct_change()
            
            # Get recent data
            recent_data = df.tail(50).copy()
            
            # Calculate trend strength metrics
            price_trend = (recent_data['SMA20'].iloc[-1] - recent_data['SMA20'].iloc[-20]) / recent_data['SMA20'].iloc[-20]
            volatility = recent_data['STD20'].iloc[-1] / recent_data['close'].iloc[-1]
            avg_movement = abs(recent_data['Returns']).mean()
            
            # Define thresholds
            TREND_THRESHOLD = 0.02
            VOLATILITY_THRESHOLD = 0.015
            
            # Analyze market conditions
            if abs(price_trend) > TREND_THRESHOLD and volatility < VOLATILITY_THRESHOLD:
                condition = "TRENDING"
                direction = "UPWARD" if price_trend > 0 else "DOWNWARD"
                strength = abs(price_trend) / TREND_THRESHOLD
            elif volatility > VOLATILITY_THRESHOLD * 1.5:
                condition = "CHOPPY"
                direction = "VOLATILE"
                strength = volatility / VOLATILITY_THRESHOLD
            else:
                condition = "DIRECTIONLESS"
                direction = "SIDEWAYS"
                strength = avg_movement / VOLATILITY_THRESHOLD
            
            return {
                "condition": condition,
                "direction": direction,
                "strength": float(strength),
                "metrics": {
                    "trend_strength": float(abs(price_trend)),
                    "volatility": float(volatility),
                    "average_movement": float(avg_movement)
                }
            }
            
        except Exception as e:
            print(f"Error in market condition analysis: {e}")
            return None

    def predict_prices(self, features: np.ndarray, current_price: float, 
                    steps: int) -> Tuple[List[float], List[float]]:
        weights = np.random.uniform(-np.pi, np.pi, self.n_qubits * 3)
        predictions = []
        confidences = []
        current_features = features[-1]
        
        for _ in range(steps):
            # Get quantum predictions
            measurements = self.prediction_circuit(current_features, weights)
            
            # Calculate prediction and confidence
            pred_change = np.mean(measurements)
            confidence = 0.5 + abs(np.std(measurements)) * 0.5
            
            predicted_price = current_price * (1 + pred_change * 0.01)
            predictions.append(predicted_price)
            confidences.append(min(0.95, max(0.51, confidence)))
            
            # Update for next prediction
            current_price = predicted_price
            current_features = self.normalize_features(
                np.array([predicted_price, features[-1][1], features[-1][2]])
            )
        
        return predictions, confidences

    def predict_price(self, symbol: str, interval: str = '1d', 
                     lookback_days: int = 30, prediction_steps: int = 4) -> Dict:
        try:
            current_price = self.get_current_price(symbol)
            if current_price is None:
                return None
            
            df = self.get_historical_data(symbol, interval, lookback_days)
            if df.empty:
                return None
            
            df.iloc[-1, df.columns.get_loc('close')] = current_price
            
            # Analyze market condition
            market_condition = self.analyze_market_condition(df)
            
            # Prepare features and make predictions
            features = self.prepare_features(df)
            if features is None:
                return None
            
            predictions, confidences = self.predict_prices(
                features, current_price, prediction_steps
            )
            
            return {
                'symbol': symbol,
                'current_price': float(current_price),
                'market_condition': market_condition,
                'predictions': [{
                    'timestamp': (datetime.now() + timedelta(minutes=i+1)).strftime('%Y-%m-%d %H:%M:%S'),
                    'predicted_price': float(price),
                    'confidence': float(conf)
                } for i, (price, conf) in enumerate(zip(predictions, confidences))]
            }
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None

def main():
    # Suppress SSL warnings
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    predictor = EnhancedQuantumCryptoPredictor(n_qubits=3)
    symbols = ['XVGUSDT']
    
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        predictions = predictor.predict_price(
            symbol=symbol,
            interval='1d',
            lookback_days=30,
            prediction_steps=3
        )
        
        if predictions:
            print(f"Current {symbol} price: ${predictions['current_price']:,.8f}")
            print("\nMarket Condition:")
            market_condition = predictions['market_condition']
            print(f"State: {market_condition['condition']}")
            print(f"Direction: {market_condition['direction']}")
            print(f"Strength: {market_condition['strength']:.2f}")
            print(f"Metrics:")
            print(f"- Trend Strength: {market_condition['metrics']['trend_strength']:.4f}")
            print(f"- Volatility: {market_condition['metrics']['volatility']:.4f}")
            print(f"- Average Movement: {market_condition['metrics']['average_movement']:.4f}")
            
            print("\nPredictions:")
            for pred in predictions['predictions']:
                print(f"Time: {pred['timestamp']}")
                print(f"Predicted Price: ${pred['predicted_price']:,.8f}")
                print(f"Confidence: {pred['confidence']:.2%}\n")
        else:
            print(f"Failed to get predictions for {symbol}")

if __name__ == "__main__":
    main()