"""
Classical Optimizer - Benchmark local sem APIs externas.
Implementa métodos clássicos para comparação com solução quântica.
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List
import time


class LocalClassicalOptimizer:
    """Otimizador clássico local para benchmark"""
    
    def __init__(self):
        pass
        
    def solve_portfolio_classical(self, returns: np.ndarray,
                                covariance: np.ndarray,
                                risk_aversion: float = 0.5) -> Dict:
        """
        Resolve portfólio usando métodos clássicos locais.
        
        Args:
            returns: Retornos esperados
            covariance: Matriz de covariância
            risk_aversion: Aversão ao risco
            
        Returns:
            Dict: Resultado da otimização clássica
        """
        n_assets = len(returns)
        
        if n_assets <= 20:
            return self._exhaustive_search(returns, covariance, risk_aversion)
        elif n_assets <= 100:
            return self._greedy_optimization(returns, covariance, risk_aversion)
        else:
            return self._heuristic_large_scale(returns, covariance, risk_aversion)
    
    def _exhaustive_search(self, returns: np.ndarray,
                          covariance: np.ndarray,
                          risk_aversion: float) -> Dict:
        """Busca exaustiva para problemas pequenos"""
        n = len(returns)
        k = max(1, n // 3)  # Selecionar k ativos
        
        best_objective = float('inf')
        best_solution = None
        
        # Testar todas combinações
        from itertools import combinations
        
        for asset_combo in combinations(range(n), k):
            solution = np.zeros(n)
            solution[list(asset_combo)] = 1.0 / k
            
            # Avaliar objetivo
            portfolio_return = np.dot(solution, returns)
            portfolio_risk = np.sqrt(np.dot(solution, np.dot(covariance, solution)))
            objective = risk_aversion * portfolio_risk - portfolio_return
            
            if objective < best_objective:
                best_objective = objective
                best_solution = solution
        
        return {
            'weights': best_solution,
            'objective': best_objective,
            'method': 'exhaustive_search',
            'num_holdings': np.sum(best_solution > 0)
        }
    
    def _greedy_optimization(self, returns: np.ndarray,
                           covariance: np.ndarray,
                           risk_aversion: float) -> Dict:
        """Otimização greedy para problemas médios"""
        n = len(returns)
        
        # Calcular Sharpe ratio para cada ativo
        volatilities = np.sqrt(np.diag(covariance))
        sharpe_ratios = returns / volatilities
        
        # Selecionar ativos com melhor Sharpe
        k = max(5, n // 10)
        top_assets = np.argsort(sharpe_ratios)[-k:]
        
        # Otimizar pesos dentro dos selecionados
        selected_returns = returns[top_assets]
        selected_cov = covariance[np.ix_(top_assets, top_assets)]
        
        weights_selected = self._optimize_weights_analytical(
            selected_returns, selected_cov, risk_aversion
        )
        
        # Mapear para vetor completo
        full_weights = np.zeros(n)
        full_weights[top_assets] = weights_selected
        
        return {
            'weights': full_weights,
            'objective': self._evaluate_objective(full_weights, returns, covariance, risk_aversion),
            'method': 'greedy_selection',
            'num_holdings': len(top_assets)
        }
    
    def _heuristic_large_scale(self, returns: np.ndarray,
                             covariance: np.ndarray,
                             risk_aversion: float) -> Dict:
        """Heurística para problemas grandes"""
        n = len(returns)
        
        # Estratégia: market cap weighted nos top performers
        # Simular market caps
        market_caps = np.random.lognormal(15, 2, n)
        
        # Score combinado (Sharpe + size)
        volatilities = np.sqrt(np.diag(covariance))
        sharpe_ratios = returns / volatilities
        
        # Normalizar scores
        norm_sharpe = (sharpe_ratios - np.mean(sharpe_ratios)) / np.std(sharpe_ratios)
        norm_caps = (market_caps - np.mean(market_caps)) / np.std(market_caps)
        
        combined_score = 0.7 * norm_sharpe + 0.3 * norm_caps
        
        # Selecionar top 5%
        k = max(10, n // 20)
        selected_indices = np.argsort(combined_score)[-k:]
        
        # Pesos proporcionais ao score
        selected_scores = combined_score[selected_indices]
        selected_scores = np.maximum(selected_scores, 0)  # Remove negativos
        
        if np.sum(selected_scores) > 0:
            weights_selected = selected_scores / np.sum(selected_scores)
        else:
            weights_selected = np.ones(k) / k
        
        # Mapear para vetor completo
        full_weights = np.zeros(n)
        full_weights[selected_indices] = weights_selected
        
        return {
            'weights': full_weights,
            'objective': self._evaluate_objective(full_weights, returns, covariance, risk_aversion),
            'method': 'large_scale_heuristic',
            'num_holdings': k
        }
    
    def _optimize_weights_analytical(self, returns: np.ndarray,
                                   covariance: np.ndarray,
                                   risk_aversion: float) -> np.ndarray:
        """Otimização analítica de pesos (Markowitz)"""
        try:
            # Solução Markowitz
            inv_cov = np.linalg.inv(covariance + 1e-6 * np.eye(len(covariance)))
            ones = np.ones(len(returns))
            
            # Pesos ótimos
            numerator = inv_cov @ (returns / risk_aversion + ones)
            denominator = ones.T @ inv_cov @ ones
            
            weights = numerator / denominator
            weights = np.maximum(weights, 0)  # Sem short selling
            
            if np.sum(weights) > 0:
                weights = weights / np.sum(weights)
            else:
                weights = np.ones(len(returns)) / len(returns)
                
            return weights
            
        except:
            # Fallback: equal weight
            return np.ones(len(returns)) / len(returns)
    
    def _evaluate_objective(self, weights: np.ndarray,
                          returns: np.ndarray,
                          covariance: np.ndarray,
                          risk_aversion: float) -> float:
        """Avalia função objetivo"""
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        return risk_aversion * portfolio_risk - portfolio_return
    
    def benchmark_performance(self, returns: np.ndarray,
                            covariance: np.ndarray) -> Dict:
        """Benchmark de performance clássica"""
        start_time = time.time()
        
        result = self.solve_portfolio_classical(returns, covariance)
        
        execution_time = time.time() - start_time
        
        # Métricas adicionais
        weights = result['weights']
        portfolio_return = np.dot(weights, returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(covariance, weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        
        return {
            'execution_time': execution_time,
            'portfolio_return': portfolio_return,
            'portfolio_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'num_holdings': result['num_holdings'],
            'method': result['method']
        }