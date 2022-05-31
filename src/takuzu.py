# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from sys import stdin
from typing import Tuple
from search import (
    Problem,
    Node,
    astar_search,
    breadth_first_tree_search,
    depth_first_tree_search,
    greedy_search,
    recursive_best_first_search,
)


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    # TODO: outros metodos da classe


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    def __init__(self, board, size):
        self.board = board
        self.size = size
    
    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> Tuple[int, int]:
        """Devolve os valores imediatamente abaixo e acima,
        respectivamente."""

        # Check if it is the last row
        if row == self.size - 1:
            return (None, self.board[row-1][col])
        
        # Check if it is the first row
        if row == 0:
            return (self.board[row+1][col], None)
        
        return (self.board[row+1][col], self.board[row-1][col])
        

    def adjacent_horizontal_numbers(self, row: int, col: int) -> Tuple[int, int]:
        """Devolve os valores imediatamente à esquerda e à direita,
        respectivamente."""
        
        # Check if it is the last column
        if col == self.size - 1:
            return (self.board[row][col-1], None)

        # Check if it is the first column
        if col == 0:
            return (None, self.board[row][col+1])

        return (self.board[row][col-1], self.board[row][col+1])

    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """

        board = []
        size = int(stdin.readline())
        
        for i in range(size):
            board.append([])
            row = stdin.readline().split(sep='\t')
            for j in range(size):
                board[i].append(int(row[j]))

        return Board(board, size)

    def __str__(self):
        """Converte o tabuleiro para string."""

        string = ""
        for i in range(self.size):
            for j in range(self.size):
                string += str(self.board[i][j]) + " "
            string += "\n"

        return string

    # TODO: outros metodos da classe


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        # TODO
        pass

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        # TODO
        pass

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        # TODO
        pass

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        # TODO
        pass
        
    def h(self, node: Node):
        """Função heuristica utilizada para a procura A*."""
        # TODO
        pass

    # TODO: outros metodos da classe


if __name__ == "__main__":
    # TODO:
    # Ler o ficheiro de input de sys.argv[1],
    # Usar uma técnica de procura para resolver a instância,
    # Retirar a solução a partir do nó resultante,
    # Imprimir para o standard output no formato indicado.

    board = Board.parse_instance_from_stdin()