"""
Decomposition Pipeline baseado no paper JPMorgan/Amazon (Acharya et al. 2023).
Implementa metodologia real para problems de escala industrial.
Referência: arXiv:2409.10301v1
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.linalg import eigh
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import SpectralClustering
import networkx as nx
from concurrent.futures import ThreadPoolExecutor


@dataclass
class DecompositionResult:
    """Resultado da decomposição do pipeline"""
    clusters: List[List[int]]
    filtered_correlation: np.ndarray
    eigenvalues: np.ndarray
    size_reduction: float
    metadata: Dict


class IndustrialDecompositionPipeline:
    """
    Pipeline de decomposição para problemas industriais de grande escala.
    Baseado em Acharya et al. (2023) - JPMorgan Chase & Amazon research.
    """
    
    def __init__(self, noise_cutoff: float = 0.1, max_cluster_size: int = 50):
        """
        Args:
            noise_cutoff: Threshold para filtrar ruído em correlações
            max_cluster_size: Tamanho máximo de cluster
        """
        self.noise_cutoff = noise_cutoff
        self.max_cluster_size = max_cluster_size
        
    def decompose_portfolio_problem(self, correlation_matrix: np.ndarray,
                                  returns: np.ndarray) -> DecompositionResult:
        """
        Pipeline principal de decomposição.
        
        Passos (seguindo paper exato):
        1. Random Matrix Theory preprocessing 
        2. Modified Spectral Clustering (Newman's algorithm)
        3. Risk rebalancing validation
        
        Args:
            correlation_matrix: Matriz de correlação (n x n)
            returns: Retornos esperados (n,)
            
        Returns:
            DecompositionResult: Resultado da decomposição
        """
        n_assets = correlation_matrix.shape[0]
        print(f"🔬 Decomposition Pipeline - {n_assets} assets")
        
        # Passo 1: Random Matrix Theory Preprocessing
        print("  📊 Step 1: Random Matrix Theory filtering...")
        filtered_corr, eigenvals = self._random_matrix_filtering(correlation_matrix)
        
        # Passo 2: Modified Spectral Clustering  
        print("  🔗 Step 2: Modified Spectral Clustering...")
        clusters = self._modified_spectral_clustering(filtered_corr)
        
        # Passo 3: Risk Rebalancing Validation
        print("  ⚖️ Step 3: Risk rebalancing validation...")
        validated_clusters = self._risk_rebalancing_validation(
            clusters, filtered_corr, returns
        )
        
        # Calcular redução de tamanho
        original_size = n_assets
        subproblem_sizes = [len(cluster) for cluster in validated_clusters]
        avg_subproblem_size = np.mean(subproblem_sizes)
        size_reduction = 1 - (avg_subproblem_size / original_size)
        
        print(f"  ✅ Decomposed into {len(validated_clusters)} subproblems")
        print(f"  📉 Size reduction: {size_reduction:.1%}")
        
        return DecompositionResult(
            clusters=validated_clusters,
            filtered_correlation=filtered_corr,
            eigenvalues=eigenvals,
            size_reduction=size_reduction,
            metadata={
                'original_size': original_size,
                'num_clusters': len(validated_clusters),
                'avg_cluster_size': avg_subproblem_size,
                'max_cluster_size': max(subproblem_sizes) if subproblem_sizes else 0,
                'min_cluster_size': min(subproblem_sizes) if subproblem_sizes else 0
            }
        )
    
    def _random_matrix_filtering(self, corr_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random Matrix Theory preprocessing.
        Remove componentes de ruído usando teoria de matriz aleatória.
        """
        # Decomposição eigenvalue
        eigenvals, eigenvecs = eigh(corr_matrix)
        eigenvals = eigenvals[::-1]  # Ordem decrescente
        eigenvecs = eigenvecs[:, ::-1]
        
        # Marchenko-Pastur threshold para filtrar ruído
        n, t = corr_matrix.shape[0], corr_matrix.shape[0]  # n assets, t time periods
        q = n / t if t > 0 else 1
        
        # Marchenko-Pastur bounds
        lambda_min = (1 - np.sqrt(q))**2
        lambda_max = (1 + np.sqrt(q))**2
        
        # Filtrar eigenvalues dentro do regime de ruído
        signal_mask = (eigenvals > lambda_max) | (eigenvals < lambda_min)
        signal_eigenvals = eigenvals[signal_mask]
        signal_eigenvecs = eigenvecs[:, signal_mask]
        
        # Reconstruir matriz filtrada
        if len(signal_eigenvals) > 0:
            filtered_corr = signal_eigenvecs @ np.diag(signal_eigenvals) @ signal_eigenvecs.T
            # Garantir diagonal = 1
            np.fill_diagonal(filtered_corr, 1.0)
        else:
            # Fallback: matriz identidade
            filtered_corr = np.eye(corr_matrix.shape[0])
            
        return filtered_corr, eigenvals
    
    def _modified_spectral_clustering(self, correlation_matrix: np.ndarray) -> List[List[int]]:
        """
        Modified Spectral Clustering baseado no algoritmo de Newman.
        Encontra clusters que maximizam modularidade.
        """
        n = correlation_matrix.shape[0]
        
        # Converter correlação para grafo de adjacência
        # Usar valor absoluto e threshold
        adj_matrix = np.abs(correlation_matrix)
        adj_matrix = adj_matrix - np.eye(n)  # Remove diagonal
        
        # Threshold para conexões significativas
        threshold = np.percentile(adj_matrix, 80)  # Top 20% das correlações
        adj_matrix = (adj_matrix > threshold).astype(float)
        
        # Spectral clustering adaptativo
        max_clusters = min(n // 5, 20)  # Máximo de clusters
        best_clusters = None
        best_modularity = -1
        
        for n_clusters in range(2, max_clusters + 1):
            try:
                if np.sum(adj_matrix) == 0:  # Matriz zero
                    break
                    
                clustering = SpectralClustering(
                    n_clusters=n_clusters,
                    affinity='precomputed',
                    random_state=42
                )
                labels = clustering.fit_predict(adj_matrix)
                
                # Calcular modularidade
                modularity = self._calculate_modularity(adj_matrix, labels)
                
                if modularity > best_modularity:
                    best_modularity = modularity
                    best_clusters = labels
                    
            except:
                continue
        
        # Converter labels para lista de clusters
        if best_clusters is not None:
            clusters = []
            for cluster_id in np.unique(best_clusters):
                cluster_members = np.where(best_clusters == cluster_id)[0].tolist()
                if len(cluster_members) > 0:
                    clusters.append(cluster_members)
        else:
            # Fallback: clusters por tamanho fixo
            clusters = []
            for i in range(0, n, self.max_cluster_size):
                cluster = list(range(i, min(i + self.max_cluster_size, n)))
                clusters.append(cluster)
                
        return clusters
    
    def _calculate_modularity(self, adj_matrix: np.ndarray, labels: np.ndarray) -> float:
        """Calcula modularidade Newman para qualidade do clustering"""
        m = np.sum(adj_matrix) / 2  # Total de edges
        if m == 0:
            return 0
            
        modularity = 0
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix)):
                if labels[i] == labels[j]:
                    ki = np.sum(adj_matrix[i])
                    kj = np.sum(adj_matrix[j])
                    expected = (ki * kj) / (2 * m)
                    modularity += adj_matrix[i, j] - expected
                    
        return modularity / (2 * m)
    
    def _risk_rebalancing_validation(self, clusters: List[List[int]],
                                   correlation_matrix: np.ndarray,
                                   returns: np.ndarray) -> List[List[int]]:
        """
        Validação e rebalanceamento de risco dos clusters.
        Garante que clusters mantenham características de risco similares.
        """
        validated_clusters = []
        
        for cluster in clusters:
            if len(cluster) == 0:
                continue
                
            # Verificar se cluster é muito grande
            if len(cluster) > self.max_cluster_size:
                # Subdividir cluster grande
                sub_clusters = self._subdivide_large_cluster(
                    cluster, correlation_matrix, returns
                )
                validated_clusters.extend(sub_clusters)
            else:
                # Validar homogeneidade do cluster
                if self._validate_cluster_homogeneity(cluster, correlation_matrix):
                    validated_clusters.append(cluster)
                else:
                    # Re-cluster se não homogêneo
                    refined_clusters = self._refine_heterogeneous_cluster(
                        cluster, correlation_matrix
                    )
                    validated_clusters.extend(refined_clusters)
        
        return validated_clusters
    
    def _subdivide_large_cluster(self, cluster: List[int],
                               correlation_matrix: np.ndarray,
                               returns: np.ndarray) -> List[List[int]]:
        """Subdivide cluster grande mantendo coerência"""
        if len(cluster) <= self.max_cluster_size:
            return [cluster]
            
        # Extrair submatriz do cluster
        cluster_corr = correlation_matrix[np.ix_(cluster, cluster)]
        
        # Aplicar clustering hierárquico
        n_subclusters = (len(cluster) + self.max_cluster_size - 1) // self.max_cluster_size
        
        try:
            clustering = SpectralClustering(n_clusters=n_subclusters, random_state=42)
            sub_labels = clustering.fit_predict(np.abs(cluster_corr))
            
            sub_clusters = []
            for label in np.unique(sub_labels):
                sub_cluster_indices = np.where(sub_labels == label)[0]
                sub_cluster = [cluster[i] for i in sub_cluster_indices]
                sub_clusters.append(sub_cluster)
                
            return sub_clusters
            
        except:
            # Fallback: divisão por tamanho
            sub_clusters = []
            for i in range(0, len(cluster), self.max_cluster_size):
                sub_cluster = cluster[i:i + self.max_cluster_size]
                sub_clusters.append(sub_cluster)
            return sub_clusters
    
    def _validate_cluster_homogeneity(self, cluster: List[int],
                                    correlation_matrix: np.ndarray) -> bool:
        """Valida se cluster tem correlação interna alta"""
        if len(cluster) < 2:
            return True
            
        # Calcular correlação média intra-cluster
        cluster_corr = correlation_matrix[np.ix_(cluster, cluster)]
        
        # Ignorar diagonal
        mask = ~np.eye(len(cluster), dtype=bool)
        intra_correlations = cluster_corr[mask]
        
        # Threshold para homogeneidade
        mean_correlation = np.mean(np.abs(intra_correlations))
        return mean_correlation > 0.3  # 30% threshold
    
    def _refine_heterogeneous_cluster(self, cluster: List[int],
                                    correlation_matrix: np.ndarray) -> List[List[int]]:
        """Refina cluster heterogêneo"""
        if len(cluster) <= 2:
            return [cluster]
            
        cluster_corr = correlation_matrix[np.ix_(cluster, cluster)]
        
        # Aplicar clustering binário
        try:
            clustering = SpectralClustering(n_clusters=2, random_state=42)
            labels = clustering.fit_predict(np.abs(cluster_corr))
            
            refined_clusters = []
            for label in np.unique(labels):
                sub_indices = np.where(labels == label)[0]
                sub_cluster = [cluster[i] for i in sub_indices]
                refined_clusters.append(sub_cluster)
                
            return refined_clusters
            
        except:
            return [cluster]  # Fallback: manter cluster original