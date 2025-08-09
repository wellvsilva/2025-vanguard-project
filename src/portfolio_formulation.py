"""
Portfolio Formulation - Metodologia JPMorgan/Amazon local.
Implementa decomposição com Random Matrix Theory.
"""

import numpy as np
from typing import Tuple, List, Dict
from scipy.linalg import eigh


class IndustrialPortfolioFormulation:
    """Formulação industrial de portfólio"""
    
    def __init__(self, decomposition_threshold: int = 50):
        self.decomposition_threshold = decomposition_threshold
        
    def formulate_problem(self, returns: np.ndarray, 
                         covariance: np.ndarray,
                         risk_aversion: float = 0.5) -> Dict:
        """
        Formulação industrial seguindo JPMorgan methodology.
        
        Args:
            returns: Retornos esperados
            covariance: Matriz de covariância  
            risk_aversion: Aversão ao risco
            
        Returns:
            Dict: Problema formulado com decomposição
        """
        n_assets = len(returns)
        
        if n_assets <= self.decomposition_threshold:
            # Problema pequeno - formulação direta
            Q = self._create_qubo_matrix(returns, covariance, risk_aversion)
            return {
                'type': 'direct',
                'Q': Q,
                'n_assets': n_assets,
                'clusters': [list(range(n_assets))]
            }
        else:
            # Problema grande - decomposição
            return self._decomposed_formulation(returns, covariance, risk_aversion)
    
    def _decomposed_formulation(self, returns: np.ndarray,
                              covariance: np.ndarray,
                              risk_aversion: float) -> Dict:
        """Formulação com decomposição"""
        # 1. Random Matrix Theory filtering
        filtered_corr = self._random_matrix_filter(covariance)
        
        # 2. Clustering baseado em correlação
        clusters = self._correlation_clustering(filtered_corr)
        
        # 3. Criar QUBO para cada cluster
        cluster_problems = []
        for cluster in clusters:
            cluster_returns = returns[cluster]
            cluster_cov = covariance[np.ix_(cluster, cluster)]
            
            Q_cluster = self._create_qubo_matrix(
                cluster_returns, cluster_cov, risk_aversion
            )
            
            cluster_problems.append({
                'assets': cluster,
                'Q': Q_cluster,
                'returns': cluster_returns
            })
        
        return {
            'type': 'decomposed',
            'clusters': clusters,
            'cluster_problems': cluster_problems,
            'n_assets': len(returns),
            'filtered_correlation': filtered_corr
        }
    
    def _random_matrix_filter(self, covariance: np.ndarray) -> np.ndarray:
        """Random Matrix Theory filtering"""
        # Converter para correlação
        std_devs = np.sqrt(np.diag(covariance))
        correlation = covariance / np.outer(std_devs, std_devs)
        
        # Eigenvalue decomposition
        eigenvals, eigenvecs = eigh(correlation)
        eigenvals = eigenvals[::-1]
        eigenvecs = eigenvecs[:, ::-1]
        
        # Marchenko-Pastur threshold
        n = correlation.shape[0]
        q = 1.0  # Simplified
        lambda_max = (1 + np.sqrt(q))**2
        
        # Filter noise
        signal_mask = eigenvals > lambda_max
        if np.any(signal_mask):
            filtered_eigenvals = eigenvals[signal_mask]
            filtered_eigenvecs = eigenvecs[:, signal_mask]
            
            # Reconstruct
            filtered_corr = filtered_eigenvecs @ np.diag(filtered_eigenvals) @ filtered_eigenvecs.T
            np.fill_diagonal(filtered_corr, 1.0)
        else:
            filtered_corr = np.eye(n)
            
        return filtered_corr
    
    def _correlation_clustering(self, correlation: np.ndarray) -> List[List[int]]:
        """Clustering baseado em correlação"""
        n = correlation.shape[0]
        
        # Threshold para correlação significativa
        threshold = 0.3
        
        # Greedy clustering
        clusters = []
        unassigned = set(range(n))
        
        while unassigned:
            # Iniciar novo cluster
            seed = unassigned.pop()
            cluster = [seed]
            
            # Adicionar ativos correlacionados
            to_add = []
            for asset in unassigned:
                avg_corr = np.mean([abs(correlation[asset, c]) for c in cluster])
                if avg_corr > threshold and len(cluster) < self.decomposition_threshold:
                    to_add.append(asset)
            
            for asset in to_add:
                cluster.append(asset)
                unassigned.remove(asset)
            
            clusters.append(cluster)
        
        return clusters
    
    def _create_qubo_matrix(self, returns: np.ndarray,
                           covariance: np.ndarray,
                           risk_aversion: float) -> np.ndarray:
        """Cria matriz QUBO para otimização"""
        n = len(returns)
        Q = np.zeros((n, n))
        
        # Termo de risco (quadrático)
        Q += risk_aversion * covariance
        
        # Termo de retorno (linear na diagonal)
        np.fill_diagonal(Q, np.diag(Q) - returns)
        
        # Constraint de budget (selecionar k ativos)
        k = max(1, n // 3)
        penalty = 1000
        
        for i in range(n):
            Q[i, i] += penalty * (2*k - 1)
            for j in range(i + 1, n):
                Q[i, j] += 2 * penalty
                Q[j, i] += 2 * penalty
        
        return Q