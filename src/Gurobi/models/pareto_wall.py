import matplotlib.pyplot as plt
import numpy as np


def plot_pareto_front_2d(pareto_front, obj1='FMAX', obj2='TFT', 
                        save_path=None, title=None):
    """
    Plota a fronteira de Pareto em 2D para dois objetivos.
    
    Args:
        pareto_front (ParetoFront): Fronteira de Pareto
        obj1 (str): Primeiro objetivo para o eixo X
        obj2 (str): Segundo objetivo para o eixo Y  
        save_path (str): Caminho para salvar o gráfico
        title (str): Título do gráfico
    """
    if pareto_front.size() == 0:
        print("Fronteira de Pareto vazia!")
        return
    
    # Extrai valores dos objetivos
    x_values = [sol.get_objective_value(obj1) for sol in pareto_front.solutions]
    y_values = [sol.get_objective_value(obj2) for sol in pareto_front.solutions]
    
    # Remove pontos com valores zero (soluções inválidas)
    valid_points = [(x, y) for x, y in zip(x_values, y_values) if x > 0 and y > 0]
    if not valid_points:
        print("Nenhum ponto válido para plotar!")
        return
    
    x_values, y_values = zip(*valid_points)
    
    # Cria o gráfico no estilo da imagem
    plt.figure(figsize=(8, 6))
    
    # Plota pontos
    plt.scatter(x_values, y_values, c='steelblue', s=80, alpha=0.8, 
                edgecolors='darkblue', linewidth=1, zorder=5)
    
    # Conecta os pontos da fronteira de Pareto
    if len(x_values) > 1:
        # Ordena pontos para conectar adequadamente (por X crescente)
        points = list(zip(x_values, y_values))
        points.sort()
        sorted_x, sorted_y = zip(*points)
        plt.plot(sorted_x, sorted_y, color='steelblue', linewidth=2, alpha=0.7, zorder=3)
    
    # Labels dos eixos
    x_label = "Tool switches" if obj1 == "TS" else ("Makespan" if obj1 == "FMAX" else "Flow time")
    y_label = "Makespan" if obj2 == "FMAX" else ("Flow time" if obj2 == "TFT" else "Tool switches")
    
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel(y_label, fontsize=12)
    plt.title(title or f'Fronteira de Pareto: {y_label} vs. {x_label}', fontsize=14, pad=20)
    
    # Grid estilo da imagem
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    
    # Ajusta layout
    plt.tight_layout()
    
    # Salva o gráfico
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Gráfico 2D salvo em: {save_path}")
    
    plt.close()  # Fecha a figura para economizar memória


