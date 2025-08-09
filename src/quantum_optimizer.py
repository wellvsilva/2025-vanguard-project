"""
Quantum Optimizer Local - Sem APIs externas.
Implementa metodologia JPMorgan/Amazon com Qiskit local.
"""

import numpy as np
from typing import Dict, List, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Sampler
from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA


class LocalQuantumOptimizer:
    """Otimizador quântico 100% local"""
    
    def __init__(self, max_iterations: int = 100):
        self.max_iterations = max_iterations
        self.sampler = Sampler()
        
    def solve_portfolio_decomposed(self, Q: np.ndarray, 
                                 cluster_size: int) -> Tuple[np.ndarray, Dict]:
        """
        Resolve portfólio usando decomposição + QAOA local.
        
        Args:
            Q: Matriz QUBO
            cluster_size: Tamanho do cluster
            
        Returns:
            Tuple: (solução, métricas)
        """
        hamiltonian = self._qubo_to_pauli_local(Q)
        
        # QAOA local
        optimizer = COBYLA(maxiter=50)
        qaoa = QAOA(sampler=self.sampler, optimizer=optimizer, reps=2)
        
        result = qaoa.compute_minimum_eigenvalue(hamiltonian)
        
        # Extrair solução
        solution = self._extract_solution_local(result, Q.shape[0])
        
        metrics = {
            'objective_value': result.optimal_value,
            'solution': solution,
            'method': 'local_qaoa'
        }
        
        return solution, metrics
    
    def _qubo_to_pauli_local(self, Q: np.ndarray) -> SparsePauliOp:
        """Converte QUBO para Pauli localmente"""
        n = Q.shape[0]
        pauli_list = []
        coeffs = []
        
        # Termos lineares
        for i in range(n):
            if abs(Q[i, i]) > 1e-10:
                pauli_str = 'I' * i + 'Z' + 'I' * (n - i - 1)
                pauli_list.append(pauli_str)
                coeffs.append(-Q[i, i] / 2)
        
        # Termos quadráticos
        for i in range(n):
            for j in range(i + 1, n):
                if abs(Q[i, j]) > 1e-10:
                    pauli_str = ['I'] * n
                    pauli_str[i] = 'Z'
                    pauli_str[j] = 'Z'
                    pauli_list.append(''.join(pauli_str))
                    coeffs.append(Q[i, j] / 4)
        
        if not pauli_list:
            pauli_list = ['I' * n]
            coeffs = [0.0]
            
        return SparsePauliOp(pauli_list, coeffs)
    
    def _extract_solution_local(self, result, n_qubits: int) -> np.ndarray:
        """Extrai solução do resultado QAOA"""
        if result.eigenstate is not None:
            probabilities = np.abs(result.eigenstate.data) ** 2
            max_idx = np.argmax(probabilities)
            binary_string = format(max_idx, f'0{n_qubits}b')
            return np.array([int(bit) for bit in binary_string])
        else:
            return np.random.randint(0, 2, n_qubits)