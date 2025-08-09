"""
ETF Creation Engine - Local sem APIs externas.
Criação de ETFs usando decomposição JPMorgan/Amazon.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class BondType(Enum):
    TREASURY = "treasury"
    CORPORATE = "corporate"
    MUNICIPAL = "municipal"


@dataclass
class LocalBondSecurity:
    """Bond security local"""
    id: str
    bond_type: BondType
    duration: float
    yield_to_maturity: float
    credit_rating: str
    liquidity_score: float
    market_value: float
    sector: str


@dataclass
class LocalETFObjective:
    """ETF objective local"""
    target_duration: float
    duration_tolerance: float
    target_yield: float
    min_liquidity_score: float
    min_holdings: int
    max_holdings: int


class LocalETFCreator:
    """Criador de ETF 100% local"""
    
    def __init__(self):
        pass
        
    def create_etf_local(self, bond_universe: List[LocalBondSecurity],
                        objective: LocalETFObjective) -> Dict:
        """
        Cria ETF localmente usando decomposição.
        
        Args:
            bond_universe: Universo de bonds
            objective: Objetivos do ETF
            
        Returns:
            Dict: ETF criado
        """
        # 1. Filtrar universo
        eligible_bonds = self._filter_bonds_local(bond_universe, objective)
        
        # 2. Clustering por duration/setor
        clusters = self._cluster_bonds_local(eligible_bonds)
        
        # 3. Seleção otimizada
        selected_bonds = self._select_bonds_local(clusters, objective)
        
        # 4. Otimizar pesos
        final_weights = self._optimize_weights_local(selected_bonds, objective)
        
        return self._build_etf_result(selected_bonds, final_weights, objective)
    
    def _filter_bonds_local(self, bonds: List[LocalBondSecurity],
                           objective: LocalETFObjective) -> List[LocalBondSecurity]:
        """Filtra bonds por critérios"""
        filtered = []
        
        for bond in bonds:
            # Liquidez mínima
            if bond.liquidity_score < objective.min_liquidity_score:
                continue
                
            # Duration range
            duration_diff = abs(bond.duration - objective.target_duration)
            if duration_diff > objective.duration_tolerance * 2:
                continue
                
            filtered.append(bond)
        
        return filtered
    
    def _cluster_bonds_local(self, bonds: List[LocalBondSecurity]) -> List[List[int]]:
        """Agrupa bonds por características"""
        if len(bonds) <= 20:
            return [list(range(len(bonds)))]
        
        # Cluster por setor
        sector_groups = {}
        for i, bond in enumerate(bonds):
            if bond.sector not in sector_groups:
                sector_groups[bond.sector] = []
            sector_groups[bond.sector].append(i)
        
        # Subdivir clusters grandes
        clusters = []
        for sector_indices in sector_groups.values():
            if len(sector_indices) <= 15:
                clusters.append(sector_indices)
            else:
                # Subdivir por duration
                sector_bonds = [bonds[i] for i in sector_indices]
                durations = [bond.duration for bond in sector_bonds]
                median_duration = np.median(durations)
                
                low_duration = [sector_indices[i] for i, d in enumerate(durations) if d <= median_duration]
                high_duration = [sector_indices[i] for i, d in enumerate(durations) if d > median_duration]
                
                if low_duration:
                    clusters.append(low_duration)
                if high_duration:
                    clusters.append(high_duration)
        
        return clusters
    
    def _select_bonds_local(self, clusters: List[List[int]],
                           objective: LocalETFObjective) -> List[int]:
        """Seleciona bonds representativos"""
        selected = []
        
        for cluster in clusters:
            # Selecionar melhor bond do cluster
            if len(cluster) == 1:
                selected.extend(cluster)
            else:
                # Critério: yield ajustado por duration
                best_idx = cluster[0]  # Fallback
                selected.append(best_idx)
        
        # Garantir limites
        if len(selected) < objective.min_holdings:
            # Adicionar mais bonds
            all_bonds = set(range(len(selected)))
            remaining = list(all_bonds - set(selected))
            additional = remaining[:objective.min_holdings - len(selected)]
            selected.extend(additional)
        
        if len(selected) > objective.max_holdings:
            selected = selected[:objective.max_holdings]
        
        return selected
    
    def _optimize_weights_local(self, selected_indices: List[int],
                               objective: LocalETFObjective) -> np.ndarray:
        """Otimiza pesos localmente"""
        n = len(selected_indices)
        if n == 0:
            return np.array([])
        
        # Equal weight como baseline
        weights = np.ones(n) / n
        
        return weights
    
    def _build_etf_result(self, selected_bonds: List[LocalBondSecurity],
                         weights: np.ndarray,
                         objective: LocalETFObjective) -> Dict:
        """Constrói resultado final do ETF"""
        if len(weights) == 0:
            return {
                'num_holdings': 0,
                'duration': 0,
                'yield': 0,
                'tracking_error': 1.0,
                'holdings': []
            }
        
        # Calcular métricas
        portfolio_duration = np.average([bond.duration for bond in selected_bonds], weights=weights)
        portfolio_yield = np.average([bond.yield_to_maturity for bond in selected_bonds], weights=weights)
        
        # Duration error
        duration_error = abs(portfolio_duration - objective.target_duration)
        
        # Holdings
        holdings = []
        for i, (bond, weight) in enumerate(zip(selected_bonds, weights)):
            if weight > 0:
                holdings.append({
                    'bond_id': bond.id,
                    'weight': weight,
                    'duration': bond.duration,
                    'yield': bond.yield_to_maturity,
                    'sector': bond.sector
                })
        
        return {
            'num_holdings': len(holdings),
            'duration': portfolio_duration,
            'yield': portfolio_yield,
            'tracking_error': duration_error / objective.target_duration,
            'holdings': holdings,
            'weights': weights
        }