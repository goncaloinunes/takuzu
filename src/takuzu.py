# takuzu.py: Template para implementação do projeto de Inteligência Artificial 2021/2022.
# Devem alterar as classes e funções neste ficheiro de acordo com as instruções do enunciado.
# Além das funções e classes já definidas, podem acrescentar outras que considerem pertinentes.

# Grupo 00:
# 00000 Nome1
# 00000 Nome2

from sys import stdin
from more_itertools import adjacent
import numpy as np
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


class Board:
    """Representação interna de um tabuleiro de Takuzu."""

    EMPTY = 2

    def __init__(self, board: np.ndarray, size: int):
        self.board = board
        self.size = size
    
    def get_number(self, row: int, col: int) -> int:
        """Devolve o valor na respetiva posição do tabuleiro."""
        
        return self.board[row][col]

    def adjacent_vertical_numbers(self, row: int, col: int) -> Tuple[int, int]:
        """Devolve os valores imediatamente acima e abaixo,
        respectivamente."""

        # Check if it is the last row
        if row == self.size - 1:
            return (self.board[row-1][col], None)
        
        # Check if it is the first row
        if row == 0:
            return (None, self.board[row+1][col])
        
        return (self.board[row-1][col], self.board[row+1][col])
        
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

    def change_number(self, row: int, col: int, number: int):
        """Altera o valor da posição especificada."""

        self.board[row][col] = number

    def copy(self):
        """Cria uma cópia do tabuleiro."""

        return Board(self.board.copy(), self.size)

    def get_col(self, col: int) -> np.ndarray:
        """Devolve a coluna especificada."""

        return self.board[:, col]

    def get_row(self, row: int) -> np.ndarray:
        """Devolve a linha especificada."""

        return self.board[row]

    @staticmethod
    def equal_arrays(array1: np.ndarray, array2: np.ndarray) -> bool:
        """Verifica se duas arrays são iguais e não contêm posições vazias."""

        for i in range(array1.size):
            if array1[i] != array2[i] or array1[i] == Board.EMPTY:
                return False

        return True

    def all_diff_rows_and_cols(self) -> bool:
        """Verifica se todas as linhas e colunas são diferentes."""

        for i in range(self.size):
            for j in range(self.size):
                if(i != j and (self.equal_arrays(self.get_row(i), self.get_row(j)) or self.equal_arrays(self.get_col(i), self.get_col(j)))):
                    return False

        return True

    def equal_zeros_and_ones(self) -> bool:
        """
        Verifica se cada linha e coluna do tabuleiro contem igual numero de 0's e 1's
        Caution, this function is not 100% accurate, it can return false positives 
        if the board is not completly filled with 0's and 1's.
        """

        for i in range(self.size):
            if(sum(self.get_row(i)) - self.size / 2 not in (0, 0.5) or sum(self.get_col(i)) - self.size / 2 not in (0, 0.5)):
                return False
            
        return True

    def less_than_two_equal_adjacents(self) -> bool:
        """Verifica se existem mais de dois valores adjacentes iguais."""
        
        for i in range(self.size):
            for j in range(self.size):

                if(self.board[i][j] == Board.EMPTY):
                    continue

                vertical = self.adjacent_vertical_numbers(i, j)
                horizontal = self.adjacent_horizontal_numbers(i, j)
                if (self.board[i][j] == vertical[0] and self.board[i][j] == vertical[1]) or \
                (self.board[i][j] == horizontal[0] and self.board[i][j] == horizontal[1]):
                    return False

        return True


    def all_filled(self) -> bool:
        """Verifica se todos os valores do tabuleiro estão preenchidos."""

        for i in range(self.size):
            for j in range(self.size):
                if self.board[i][j] == Board.EMPTY:
                    return False
        
        return True

                
    @staticmethod
    def parse_instance_from_stdin():
        """Lê o test do standard input (stdin) que é passado como argumento
        e retorna uma instância da classe Board.

        Por exemplo:
            $ python3 takuzu.py < input_T01

            > from sys import stdin
            > stdin.readline()
        """

        size = int(stdin.readline())
        board = np.array([[int(x) for x in stdin.readline().split()] for _ in range(size)])

        return Board(board, size)


    def __str__(self):
        """Converte o tabuleiro para string."""

        string = ""
        for i in range(self.size):
            for j in range(self.size):
                string += str(self.board[i][j])
                string += "\t" if j < self.size - 1 else ""
            string += "\n"

        return string


class TakuzuState:
    state_id = 0

    def __init__(self, board):
        self.board = board
        self.id = TakuzuState.state_id
        TakuzuState.state_id += 1

    def __lt__(self, other):
        return self.id < other.id

    @staticmethod
    def count_occurrences(array: np.ndarray, value: int):
        count = 0
        for i in range(array.size):
            if array[i] == value:
                count += 1

        return count

    @staticmethod
    def find_empty_position(array: np.ndarray) -> int:
        for i in range(array.size):
            if array[i] == Board.EMPTY:
                return i

        return -1

    def find_action_due_to_adjacents(self) -> list:

        for i in range(self.board.size):
            for j in range(self.board.size):
                if self.board.get_number(i, j) == Board.EMPTY:
                    horizontal = self.board.adjacent_horizontal_numbers(i, j)
                    if horizontal[0] == horizontal[1] and horizontal[0] != Board.EMPTY:
                        return [(i, j, 0 if horizontal[0] == 1 else 1)]
                    
                    vertical = self.board.adjacent_vertical_numbers(i, j)
                    if vertical[0] == vertical[1] and vertical[0] != Board.EMPTY:
                        return [(i, j, 0 if vertical[0] == 1 else 1)]
        
        return []

    def find_action_due_to_zeros_and_ones(self) -> list:

        for i in range(self.board.size):
            row = self.board.get_row(i)
            j = TakuzuState.find_empty_position(row)
            if j == -1:
                continue

            if self.board.size % 2 == 0:
                if TakuzuState.count_occurrences(row, 1) >= self.board.size / 2:
                    return [(i, j, 0)]
            else:
                if TakuzuState.count_occurrences(row, 1) > self.board.size / 2 :
                    return [(i, j, 0)]

            if TakuzuState.count_occurrences(row, 0) >= self.board.size / 2:
                return [(i, j, 1)]

        del j
        del i

        for j in range(self.board.size):
            col = self.board.get_col(j)
            i = TakuzuState.find_empty_position(col)
            if i == -1:
                continue


            if self.board.size % 2 == 0:
                if TakuzuState.count_occurrences(col, 1) >= self.board.size / 2:
                    return [(i, j, 0)]
            else:
                if TakuzuState.count_occurrences(col, 1) > self.board.size / 2 :
                    return [(i, j, 0)]

            if TakuzuState.count_occurrences(col, 0) >= self.board.size / 2:
                return [(i, j, 1)]
      
        return []
                    
    def play_action(self, action : Tuple[int, int, int]) -> "TakuzuState":
        new_board = self.board.copy()
        new_board.change_number(action[0], action[1], action[2])

        return TakuzuState(new_board)

    def is_valid(self) -> bool:
        return self.board.less_than_two_equal_adjacents() and self.board.equal_zeros_and_ones() and self.board.all_diff_rows_and_cols()
    
    def is_goal(self) -> bool:
        return self.is_valid()


class Takuzu(Problem):
    def __init__(self, board: Board):
        """O construtor especifica o estado inicial."""
        super().__init__(TakuzuState(board))
        self.initial = TakuzuState(board)

    def actions(self, state: TakuzuState):
        """Retorna uma lista de ações que podem ser executadas a
        partir do estado passado como argumento."""
        
        adjacent = state.find_action_due_to_adjacents()
        if adjacent != []:
            return adjacent

        zeros_and_ones = state.find_action_due_to_zeros_and_ones()
        if zeros_and_ones != []:
            return zeros_and_ones

        if not state.board.less_than_two_equal_adjacents() or not state.board.all_diff_rows_and_cols():
            return []


        actions = []
        for i in range(state.board.size):
            for j in range(state.board.size):
                if state.board.get_number(i, j) == 2:
                    actions.append((i, j, 0))
                    actions.append((i, j, 1))

        return actions

    def result(self, state: TakuzuState, action):
        """Retorna o estado resultante de executar a 'action' sobre
        'state' passado como argumento. A ação a executar deve ser uma
        das presentes na lista obtida pela execução de
        self.actions(state)."""
        
        return state.play_action(action)

    def goal_test(self, state: TakuzuState):
        """Retorna True se e só se o estado passado como argumento é
        um estado objetivo. Deve verificar se todas as posições do tabuleiro
        estão preenchidas com uma sequência de números adjacentes."""
        
        return state.is_goal()

    
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
   
    problem = Takuzu(board)
    
    goal_node = depth_first_tree_search(problem)
   
    print(goal_node.state.board, end='')

    

    
