#!/usr/bin/env python3
#Por favor NÃO REMOVER O SHEBANG!!!
#Para testes rápidos em caso de aplicação em larga escala onde os demais testes vão consumir muito tempo

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model import Model

def main():
    print("=== Teste Simples dos Pesos Dinâmicos ===")
    
    parameters = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8000, 8000, 8000, 8000, 8000, 5.0, 2.0, 2.0, 2.0, 3.0, 3, 3, 3, 3, 3, 4, 4]
    
    model = Model(parameters)
    
    print("Pesos iniciais:", model.get_dynamic_weights())
    
    test_cases = [(0x1000, 1), (0x1004, 0), (0x1008, 1)]
    
    for i, (pc, outcome) in enumerate(test_cases):
        print(f"\nTeste {i+1}: PC=0x{pc:x}, Outcome={outcome}")
        is_correct = model.predict_and_train(pc, outcome)
        print(f"Resultado: {'Correto' if is_correct else 'Incorreto'}")
        print(f"Pesos atuais: {model.get_dynamic_weights()}")
    
    print("\n=== Teste concluído ===")

if __name__ == "__main__":
    main()

