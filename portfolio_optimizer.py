from scipy.optimize import minimize
import numpy as np
import pandas as pd

class PortfolioOptimizer:
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.optimization_methods = {
            'sharpe': self.optimize_sharpe_ratio,
            'minimum_volatility': self.optimize_minimum_volatility,
            'maximum_diversification': self.optimize_maximum_diversification
        }

    async def optimize_portfolio(self, symbols, method='sharpe'):
        """Optimize portfolio weights using selected method"""
        try:
            # Get historical data for all symbols
            historical_data = await self.get_historical_data(symbols)
            
            # Calculate returns and covariance
            returns = self.calculate_returns(historical_data)
            cov_matrix = self.calculate_covariance(returns)
            
            # Run selected optimization method
            optimizer = self.optimization_methods.get(method, self.optimize_sharpe_ratio)
            optimal_weights = await optimizer(returns, cov_matrix)
            
            # Calculate portfolio metrics
            metrics = self.calculate_portfolio_metrics(optimal_weights, returns, cov_matrix)
            
            return {
                'weights': dict(zip(symbols, optimal_weights)),
                'metrics': metrics
            }
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            return None

    async def get_historical_data(self, symbols, lookback_days=365):
        """Fetch historical price data for all symbols"""
        try:
            data = {}
            for symbol in symbols:
                prices = await get_historical_prices(symbol, lookback_days)
                data[symbol] = prices
            return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Historical data fetch error: {str(e)}")
            return None

    def calculate_returns(self, prices):
        """Calculate daily returns"""
        return prices.pct_change().dropna()

    def calculate_covariance(self, returns):
        """Calculate covariance matrix"""
        return returns.cov()

    async def optimize_sharpe_ratio(self, returns, cov_matrix):
        """Optimize portfolio for maximum Sharpe ratio"""
        try:
            n = len(returns.columns)
            
            def objective(weights):
                portfolio_return = np.sum(returns.mean() * weights) * 252
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
                return -sharpe_ratio  # Minimize negative Sharpe ratio
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}  # Weights sum to 1
            ]
            bounds = tuple((0, 1) for _ in range(n))  # Weights between 0 and 1
            
            result = minimize(
                objective,
                x0=np.array([1/n] * n),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x
            
        except Exception as e:
            logger.error(f"Sharpe optimization error: {str(e)}")
            return None

    async def optimize_minimum_volatility(self, returns, cov_matrix):
        """Optimize portfolio for minimum volatility"""
        try:
            n = len(returns.columns)
            
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
                
            constraints = [
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            ]
            bounds = tuple((0, 1) for _ in range(n))
            
            result = minimize(
                objective,
                x0=np.array([1/n] * n),
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            return result.x
            
        except Exception as e:
            logger.error(f"Minimum volatility optimization error: {str(e)}")
            return None

    def calculate_portfolio_metrics(self, weights, returns, cov_matrix):
        """Calculate portfolio performance metrics"""
        try:
            portfolio_return = np.sum(returns.mean() * weights) * 252
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_std
            
            # Calculate diversification ratio
            weighted_std = np.sum(np.sqrt(np.diag(cov_matrix)) * weights)
            diversification_ratio = portfolio_std / weighted_std
            
            return {
                'expected_return': portfolio_return,
                'volatility': portfolio_std,
                'sharpe_ratio': sharpe_ratio,
                'diversification_ratio': diversification_ratio
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {str(e)}")
            return None
