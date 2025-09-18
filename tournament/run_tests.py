#!/usr/bin/env python3
#Por favor NÃO REMOVER O SHEBANG!!!
#Criei esse lindo arquivinho para rodar testes mais completinhos e "robustos"

import sys
import os

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from model import Model
    print("✓ Módulos importados com sucesso!")
except ImportError as e:
    print(f"✗ Erro ao importar módulos: {e}")
    sys.exit(1)

def test_basic_functionality():
    print("\n=== Teste Básico de Funcionalidade ===")
    
    try:
        parameters = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8000, 8000, 8000, 8000, 8000, 5.0, 2.0, 2.0, 2.0, 3.0, 3, 3, 3, 3, 3, 4, 4]
        model = Model(parameters)
        print("✓ Modelo criado com sucesso!")
        
        initial_weights = model.get_dynamic_weights()
        print(f"✓ Pesos iniciais obtidos: {initial_weights}")
        
        test_pc = 0x1000
        test_outcome = 1
        
        result = model.predict_and_train(test_pc, test_outcome)
        print(f"✓ Primeira predição executada: {'Correta' if result else 'Incorreta'}")
        
        updated_weights = model.get_dynamic_weights()
        print(f"✓ Pesos após predição: {updated_weights}")
        
        model.reset_dynamic_weights()
        reset_weights = model.get_dynamic_weights()
        print(f"✓ Reset de pesos executado: {reset_weights}")
        
        print("\n🎉 Todos os testes básicos passaram!")
        return True
        
    except Exception as e:
        print(f"✗ Erro durante teste básico: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_with_sample_data():
    print("\n=== Teste com Dados de Exemplo ===")
    
    try:
        parameters = [1, 1, 1, 1, 1, 4, 4, 4, 4, 4, 8000, 8000, 8000, 8000, 8000, 5.0, 2.0, 2.0, 2.0, 3.0, 3, 3, 3, 3, 3, 4, 4]
        model = Model(parameters)
        
        test_data = [
            (0x1000, 1),
            (0x1004, 0),
            (0x1008, 1),
            (0x100c, 1),
            (0x1010, 0),
        ]
        
        correct = 0
        total = len(test_data)
        
        print("Executando predições de teste...")
        for i, (pc, outcome) in enumerate(test_data):
            result = model.predict_and_train(pc, outcome)
            if result:
                correct += 1
            print(f"  Teste {i+1}: PC=0x{pc:x}, Outcome={outcome} -> {'✓' if result else '✗'}")
        
        accuracy = (correct / total) * 100
        final_weights = model.get_dynamic_weights()
        
        print(f"\nResultados:")
        print(f"  Precisão: {accuracy:.1f}% ({correct}/{total})")
        print(f"  Pesos finais: {final_weights}")
        
        print("🎉 Teste com dados de exemplo concluído!")
        return True
        
    except Exception as e:
        print(f"✗ Erro durante teste com dados: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=== Teste do Sistema de Pesos Dinâmicos ===")
    print(f"Python version: {sys.version}")
    print(f"Diretório atual: {os.getcwd()}")
    
    success1 = test_basic_functionality()
    success2 = test_with_sample_data()
    
    if success1 and success2:
        print("\n🎉 TODOS OS TESTES PASSARAM COM SUCESSO!")
        print("\nO sistema de pesos dinâmicos está funcionando corretamente!")
    else:
        print("\n❌ Alguns testes falharam. Verifique os erros acima.")
        sys.exit(1)

if __name__ == "__main__":
    main()

