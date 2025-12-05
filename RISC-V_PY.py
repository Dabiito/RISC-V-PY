#!/usr/bin/env python3
"""
Simulador RISC-V RV32I - Versão Simplificada
Implementa apenas as instruções básicas da tabela fornecida
"""

import sys
# Importa o módulo sys, mas não é usado diretamente no corpo principal do simulador.
from enum import Enum
# Importa a classe Enum para definir tipos enumerados (Instrução, Opcode, etc.).
from typing import List, Tuple
# Importa tipos para anotação de tipo (melhora a clareza do código).

# --- Enumerações (Constantes RISC-V) ---

class InstructionType(Enum):
    # Enumeração para os formatos de instrução do RISC-V.
    R = "R" # Tipo Registrador (aritméticas/lógicas)
    I = "I" # Tipo Imediato (carga, ADDI)
    S = "S" # Tipo Store (armazenamento)
    B = "B" # Tipo Branch (desvios)
    U = "U" # Tipo Upper Imediato (LUI, AUIPC)
    J = "J" # Tipo Jump (JAL)

class Opcode(Enum):
    # Enumeração para os Opcodes principais (7 bits, b[6:0]).
    LOAD = 0b0000011     # Instruções de Load (Tipo I)
    OP_IMM = 0b0010011   # Instruções Aritméticas/Lógicas com Imediato (Tipo I)
    AUIPC = 0b0010111    # Add Upper Immediate to PC (Tipo U)
    STORE = 0b0100011    # Instruções de Store (Tipo S)
    OP = 0b0110011       # Instruções Aritméticas/Lógicas de Registrador (Tipo R)
    LUI = 0b0110111      # Load Upper Immediate (Tipo U)
    BRANCH = 0b1100011   # Instruções de Desvio Condicional (Tipo B)
    JALR = 0b1100111     # Jump and Link Register (Tipo I)
    JAL = 0b1101111      # Jump and Link (Tipo J)
    SYSTEM = 0b1110011   # Instruções de Sistema (ECALL, EBREAK, CSR) (Tipo I)

class Funct3(Enum):
    # Enumeração para o campo funct3 (3 bits, b[14:12]), usado para sub-classificar Opcode.
    
    # Branch (Opcode 0b1100011)
    BEQ = 0b000
    BNE = 0b001
    BLT = 0b100
    BGE = 0b101
    BLTU = 0b110
    BGEU = 0b111
    
    # Load/Store (Opcode 0b0000011 e 0b0100011)
    LB = 0b000
    LH = 0b001
    LW = 0b010
    LBU = 0b100
    LHU = 0b101
    SB = 0b000
    SH = 0b001
    SW = 0b010
    
    # OP-IMM (Opcode 0b0010011)
    ADDI = 0b000
    SLTI = 0b010
    SLTIU = 0b011
    XORI = 0b100
    ORI = 0b110
    ANDI = 0b111
    SLLI = 0b001 # O funct7 diferencia SLLI
    SRLI = 0b101 # O funct7 diferencia SRLI de SRAI
    SRAI = 0b101 # O funct7 diferencia SRLI de SRAI
    
    # OP (Opcode 0b0110011)
    ADD = 0b000 # O funct7 diferencia ADD de SUB
    SUB = 0b000 # O funct7 diferencia ADD de SUB
    SLL = 0b001
    SLT = 0b010
    SLTU = 0b011
    XOR = 0b100
    SRL = 0b101 # O funct7 diferencia SRL de SRA
    SRA = 0b101 # O funct7 diferencia SRL de SRA
    OR = 0b110
    AND = 0b111

class Register(Enum):
    # Enumeração para os registradores e seus índices (convenções de chamada/uso).
    ZERO = 0   # x0: Always zero
    RA = 1     # x1: Return address
    SP = 2     # x2: Stack pointer
    GP = 3     # x3: Global pointer
    TP = 4     # x4: Thread pointer
    T0 = 5     # x5: Temp/Alternate Link register
    T1 = 6     # x6
    T2 = 7     # x7
    S0 = 8     # x8: Saved register / Frame pointer
    S1 = 9     # x9: Saved register
    A0 = 10    # x10: Argument / Return value
    A1 = 11    # x11: Argument / Return value
    A2 = 12    # x12: Argument
    A3 = 13    # x13: Argument
    A4 = 14    # x14: Argument
    A5 = 15    # x15: Argument
    A6 = 16    # x16: Argument
    A7 = 17    # x17: Argument (System call number)
    S2 = 18    # x18: Saved register
    S3 = 19    # x19
    S4 = 20    # x20
    S5 = 21    # x21
    S6 = 22    # x22
    S7 = 23    # x23
    S8 = 24    # x24
    S9 = 25    # x25
    S10 = 26   # x26
    S11 = 27   # x27
    T3 = 28    # x28: Temporary register
    T4 = 29    # x29
    T5 = 30    # x30
    T6 = 31    # x31

# --- Classe CPU ---

class CPU:
    def __init__(self, memory, bus):
        # Inicializa 32 registradores (x0 a x31) com zero.
        self.registers = [0] * 32
        # Program Counter (PC), começa em 0.
        self.pc = 0
        # Referência à memória principal (não usada diretamente, mas passada para o Bus).
        self.memory = memory
        # Referência ao barramento de comunicação (Bus) para acesso à memória/I/O.
        self.bus = bus
        
        # Inicializar registradores importantes conforme convenção
        # Configura o Stack Pointer para um endereço alto (0x80000).
        self.registers[Register.SP.value] = 0x80000
        # Configura o Global Pointer para um endereço comum (0x1000).
        self.registers[Register.GP.value] = 0x1000
        
        # Statistics
        self.instructions_executed = 0
        self.vram_display_interval = 50  # Exibir VRAM a cada 50 instruções
        self.last_vram_display = 0

    def fetch(self) -> int:
        """Busca instrução da memória"""
        # Verifica se o PC está alinhado em 4 bytes (instruções são de 4 bytes).
        if self.pc % 4 != 0:
            raise Exception(f"PC não alinhado: 0x{self.pc:08x}")
        
        # Lê a palavra (32 bits) da memória usando o Bus.
        instruction = self.bus.read_word(self.pc)
        # Incrementa o PC em 4 (tamanho da instrução).
        self.pc += 4
        # Retorna a instrução lida.
        return instruction

    def decode(self, instruction: int) -> Tuple[Opcode, int, int, int, int, int]:
        """Decodifica instrução"""
        # Extrai o Opcode (bits 6:0).
        opcode = instruction & 0x7F
        # Extrai o Registrador de Destino (rd) (bits 11:7).
        rd = (instruction >> 7) & 0x1F
        # Extrai o Funct3 (bits 14:12).
        funct3 = (instruction >> 12) & 0x7
        # Extrai o Registrador Fonte 1 (rs1) (bits 19:15).
        rs1 = (instruction >> 15) & 0x1F
        # Extrai o Registrador Fonte 2 (rs2) (bits 24:20).
        rs2 = (instruction >> 20) & 0x1F
        # Extrai o Funct7 (bits 31:25).
        funct7 = (instruction >> 25) & 0x7F
        
        # Retorna os campos da instrução.
        return Opcode(opcode), rd, funct3, rs1, rs2, funct7

    def execute(self, instruction: int):
        """Executa instrução"""
        # Decodifica a instrução para obter seus campos.
        opcode, rd, funct3, rs1, rs2, funct7 = self.decode(instruction)
        
        try:
            # Seleciona o método de execução baseado no Opcode.
            if opcode == Opcode.OP_IMM:
                self.execute_op_imm(instruction, rd, funct3, rs1)
            elif opcode == Opcode.OP:
                self.execute_op(instruction, rd, funct3, rs1, rs2, funct7)
            elif opcode == Opcode.LUI:
                self.execute_lui(instruction, rd)
            elif opcode == Opcode.AUIPC:
                self.execute_auipc(instruction, rd)
            elif opcode == Opcode.JAL:
                self.execute_jal(instruction, rd)
            elif opcode == Opcode.JALR:
                self.execute_jalr(instruction, rd, funct3, rs1)
            elif opcode == Opcode.BRANCH:
                self.execute_branch(instruction, funct3, rs1, rs2)
            elif opcode == Opcode.LOAD:
                self.execute_load(instruction, rd, funct3, rs1)
            elif opcode == Opcode.STORE:
                self.execute_store(instruction, funct3, rs1, rs2)
            elif opcode == Opcode.SYSTEM:
                self.execute_system(instruction, rd, funct3, rs1)
            else:
                # Lança exceção para opcodes não implementados.
                raise Exception(f"Opcode não implementado: {opcode}")
                
        except Exception as e:
            # Captura e imprime erro com o endereço do PC e a instrução.
            print(f"Erro executando instrução 0x{instruction:08x} em PC 0x{self.pc-4:08x}: {e}")
            raise # Re-lança a exceção.
            
        # Incrementa o contador de instruções executadas.
        self.instructions_executed += 1

        if (self.instructions_executed - self.last_vram_display >= 
            self.vram_display_interval):
            self.display_vram()
            self.last_vram_display = self.instructions_executed

    def sign_extend(self, value: int, bits: int) -> int:
        """Estende sinal de um valor com determinado número de bits"""
        # Verifica se o bit de sinal (o bit mais à esquerda) está ativo.
        if value & (1 << (bits - 1)):
            # Se ativo, realiza a extensão preenchendo os bits superiores com 1.
            return value | (~((1 << bits) - 1) & 0xFFFFFFFF)
        # Se não ativo, retorna o valor original (preenche com 0, o que já ocorre por padrão).
        return value

    # --- Funções de Extração de Imediato (Immediate) ---
    
    def get_immediate_i(self, instruction: int) -> int:
        """Extrai immediate do formato I (12 bits)"""
        # Extrai os bits [31:20] e estende o sinal (12 bits).
        return self.sign_extend((instruction >> 20) & 0xFFF, 12)

    def get_immediate_s(self, instruction: int) -> int:
        """Extrai immediate do formato S (12 bits)"""
        # Constrói o imediato: bits [11:5] de [31:25].
        imm_11_5 = (instruction >> 25) & 0x7F
        # Constrói o imediato: bits [4:0] de [11:7].
        imm_4_0 = (instruction >> 7) & 0x1F
        # Combina e estende o sinal.
        return self.sign_extend((imm_11_5 << 5) | imm_4_0, 12)

    def get_immediate_b(self, instruction: int) -> int:
        """Extrai immediate do formato B (13 bits, sempre par)"""
        # Extrai o bit [12] de [31].
        imm_12 = (instruction >> 31) & 0x1
        # Extrai o bit [11] de [7].
        imm_11 = (instruction >> 7) & 0x1
        # Extrai os bits [10:5] de [30:25].
        imm_10_5 = (instruction >> 25) & 0x3F
        # Extrai os bits [4:1] de [11:8].
        imm_4_1 = (instruction >> 8) & 0xF
        # Combina os bits. O bit 0 é implicitamente zero (<< 1).
        imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
        # Estende o sinal (13 bits).
        return self.sign_extend(imm, 13)

    def get_immediate_u(self, instruction: int) -> int:
        """Extrai immediate do formato U (20 bits superiores)"""
        # Extrai os 20 bits superiores [31:12] e adiciona 12 zeros à direita (<< 12).
        return (instruction & 0xFFFFF000)

    def get_immediate_j(self, instruction: int) -> int:
        """Extrai immediate do formato J (21 bits, sempre par)"""
        # Extrai o bit [20] de [31].
        imm_20 = (instruction >> 31) & 0x1
        # Extrai os bits [19:12] de [19:12].
        imm_19_12 = (instruction >> 12) & 0xFF
        # Extrai o bit [11] de [20].
        imm_11 = (instruction >> 20) & 0x1
        # Extrai os bits [10:1] de [30:21].
        imm_10_1 = (instruction >> 21) & 0x3FF
        # Combina os bits. O bit 0 é implicitamente zero (<< 1).
        imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
        # Estende o sinal (21 bits).
        return self.sign_extend(imm, 21)

    # --- Funções de Execução ---

    def execute_op_imm(self, instruction: int, rd: int, funct3: int, rs1: int):
        """Executa instruções OP-IMM: ADDI, SLTI, SLTIU, XORI, ORI, ANDI, SLLI, SRLI, SRAI"""
        # Obtém o imediato.
        imm = self.get_immediate_i(instruction)
        # Obtém o valor do registrador fonte.
        rs1_val = self.registers[rs1]
        
        # O resultado deve ser inicializado, mas pode ser calculado dentro do if/elif.
        
        if funct3 == Funct3.ADDI.value:
            # ADDI: Soma.
            result = rs1_val + imm
        elif funct3 == Funct3.SLTI.value:
            # SLTI (Set Less Than Immediate): Assinada.
            result = 1 if rs1_val < imm else 0
        elif funct3 == Funct3.SLTIU.value:
            # SLTIU (Set Less Than Immediate Unsigned): Não assinada (usando 0xFFFFFFFF para simular).
            result = 1 if (rs1_val & 0xFFFFFFFF) < (imm & 0xFFFFFFFF) else 0
        elif funct3 == Funct3.XORI.value:
            # XORI: XOR bit a bit.
            result = rs1_val ^ imm
        elif funct3 == Funct3.ORI.value:
            # ORI: OR bit a bit.
            result = rs1_val | imm
        elif funct3 == Funct3.ANDI.value:
            # ANDI: AND bit a bit.
            result = rs1_val & imm
        elif funct3 == Funct3.SLLI.value:
            # SLLI (Shift Left Logical Immediate): O imediato é o shamt (shift amount).
            shamt = imm & 0x1F # Apenas os 5 bits inferiores do imediato (0-31).
            result = rs1_val << shamt
        elif funct3 == Funct3.SRLI.value:
            # SRLI (Shift Right Logical Immediate): Deslocamento lógico (preenche com zeros).
            shamt = imm & 0x1F
            # O & 0xFFFFFFFF garante que rs1_val seja tratado como valor de 32 bits não assinado antes do shift.
            result = (rs1_val & 0xFFFFFFFF) >> shamt
        elif funct3 == Funct3.SRAI.value:
            # SRAI (Shift Right Arithmetic Immediate): Deslocamento aritmético (preserva o bit de sinal).
            shamt = imm & 0x1F
            # Verifica se o bit de sinal está ativo (bit 31).
            if rs1_val & 0x80000000:
                # Se for negativo, preenche os bits superiores com 1.
                result = (rs1_val >> shamt) | (~0 << (32 - shamt))
            else:
                # Se for positivo, descola normalmente (preenche com 0).
                result = rs1_val >> shamt
        else:
            # Lança exceção se Funct3 não for reconhecido.
            raise Exception(f"Funct3 OP-IMM não implementado: {funct3}")
        
        # Garante que o resultado seja um valor de 32 bits antes de armazenar (truncagem).
        self.registers[rd] = result & 0xFFFFFFFF

    def execute_op(self, instruction: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int):
        """Executa instruções OP: ADD, SUB, SLL, SLT, SLTU, XOR, SRL, SRA, OR, AND"""
        # Obtém os valores dos registradores fonte.
        rs1_val = self.registers[rs1]
        rs2_val = self.registers[rs2]
        
        if funct3 == Funct3.ADD.value:
            # Funct3=0b000: ADD ou SUB (diferenciados por Funct7).
            if funct7 == 0:
                # ADD: Soma.
                result = rs1_val + rs2_val
            elif funct7 == 0x20:
                # SUB: Subtração.
                result = rs1_val - rs2_val
            else:
                raise Exception(f"Funct7 inválido para ADD/SUB: {funct7}")
        elif funct3 == Funct3.SLL.value:
            # SLL (Shift Left Logical): Deslocamento lógico.
            shamt = rs2_val & 0x1F
            result = rs1_val << shamt
        elif funct3 == Funct3.SLT.value:
            # SLT (Set Less Than): Assinada.
            result = 1 if rs1_val < rs2_val else 0
        elif funct3 == Funct3.SLTU.value:
            # SLTU (Set Less Than Unsigned): Não assinada.
            result = 1 if (rs1_val & 0xFFFFFFFF) < (rs2_val & 0xFFFFFFFF) else 0
        elif funct3 == Funct3.XOR.value:
            # XOR: XOR bit a bit.
            result = rs1_val ^ rs2_val
        elif funct3 == Funct3.SRL.value:
            # Funct3=0b101: SRL ou SRA (diferenciados por Funct7).
            if funct7 == 0:
                # SRL (Shift Right Logical): Deslocamento lógico (preenche com zeros).
                shamt = rs2_val & 0x1F
                result = (rs1_val & 0xFFFFFFFF) >> shamt
            elif funct7 == 0x20:
                # SRA (Shift Right Arithmetic): Deslocamento aritmético.
                shamt = rs2_val & 0x1F
                # Implementação do shift aritmético (preserva o bit de sinal).
                if rs1_val & 0x80000000:
                    result = (rs1_val >> shamt) | (~0 << (32 - shamt))
                else:
                    result = rs1_val >> shamt
            else:
                raise Exception(f"Funct7 inválido para SRL/SRA: {funct7}")
        elif funct3 == Funct3.OR.value:
            # OR: OR bit a bit.
            result = rs1_val | rs2_val
        elif funct3 == Funct3.AND.value:
            # AND: AND bit a bit.
            result = rs1_val & rs2_val
        else:
            raise Exception(f"Funct3 OP não implementado: {funct3}")
        
        # Garante o resultado de 32 bits.
        self.registers[rd] = result & 0xFFFFFFFF

    def execute_lui(self, instruction: int, rd: int):
        """Executa LUI - Load Upper Immediate"""
        # Obtém o imediato de 20 bits (já shiftado para os bits 31:12).
        imm = self.get_immediate_u(instruction)
        # Armazena o imediato em rd (os 12 bits inferiores são 0).
        self.registers[rd] = imm & 0xFFFFFFFF

    def execute_auipc(self, instruction: int, rd: int):
        """Executa AUIPC - Add Upper Immediate to PC"""
        # Obtém o imediato de 20 bits.
        imm = self.get_immediate_u(instruction)
        # Calcula o endereço: (PC da instrução atual) + imediato.
        # O PC atual é 'self.pc - 4' (pois self.pc já foi incrementado no fetch).
        self.registers[rd] = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_jal(self, instruction: int, rd: int):
        """Executa JAL - Jump and Link"""
        # Obtém o imediato de desvio (offset).
        imm = self.get_immediate_j(instruction)
        # Salva o endereço de retorno (PC + 4) no registrador de destino (rd).
        self.registers[rd] = self.pc
        # Atualiza o PC: PC atual + offset.
        self.pc = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_jalr(self, instruction: int, rd: int, funct3: int, rs1: int):
        """Executa JALR - Jump and Link Register"""
        # JALR requer Funct3=0b000.
        if funct3 != 0:
            raise Exception(f"Funct3 JALR não zero: {funct3}")
        
        # Obtém o imediato (offset).
        imm = self.get_immediate_i(instruction)
        # Obtém o valor do registrador base.
        rs1_val = self.registers[rs1]
        # Calcula o endereço de destino: rs1 + imediato.
        # E zera o bit menos significativo (Bit 0) para garantir alinhamento par (especificação RV32I).
        target = (rs1_val + imm) & 0xFFFFFFFE
        
        # Salva o endereço de retorno (PC + 4) no registrador de destino (rd).
        self.registers[rd] = self.pc
        # Atualiza o PC para o endereço de destino.
        self.pc = target

    def execute_branch(self, instruction: int, funct3: int, rs1: int, rs2: int):
        """Executa instruções de branch: BEQ, BNE, BLT, BGE, BLTU, BGEU"""
        # Obtém os valores dos registradores.
        rs1_val = self.registers[rs1]
        rs2_val = self.registers[rs2]
        # Obtém o imediato de desvio (offset).
        imm = self.get_immediate_b(instruction)
        
        branch_taken = False
        
        # Comparações (assinadas vs. não assinadas).
        if funct3 == Funct3.BEQ.value:
            branch_taken = (rs1_val == rs2_val)
        elif funct3 == Funct3.BNE.value:
            branch_taken = (rs1_val != rs2_val)
        elif funct3 == Funct3.BLT.value:
            branch_taken = (rs1_val < rs2_val) # Assinado
        elif funct3 == Funct3.BGE.value:
            branch_taken = (rs1_val >= rs2_val) # Assinado
        elif funct3 == Funct3.BLTU.value:
            # Não Assinado (requer máscara para tratamento correto em Python).
            branch_taken = ((rs1_val & 0xFFFFFFFF) < (rs2_val & 0xFFFFFFFF))
        elif funct3 == Funct3.BGEU.value:
            # Não Assinado.
            branch_taken = ((rs1_val & 0xFFFFFFFF) >= (rs2_val & 0xFFFFFFFF))
        else:
            raise Exception(f"Funct3 BRANCH não implementado: {funct3}")
        
        # Se a condição de desvio for verdadeira.
        if branch_taken:
            # Atualiza o PC: PC atual + offset.
            self.pc = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_load(self, instruction: int, rd: int, funct3: int, rs1: int):
        """Executa instruções de load: LB, LH, LW, LBU, LHU"""
        # Obtém o imediato (offset).
        imm = self.get_immediate_i(instruction)
        # Obtém o valor do registrador base.
        rs1_val = self.registers[rs1]
        # Calcula o endereço de memória.
        address = (rs1_val + imm) & 0xFFFFFFFF
        
        # Inicializa 'value' (o valor lido da memória).
        value = 0
        
        if funct3 == Funct3.LB.value:
            # LB (Load Byte): Lê 1 byte.
            value = self.bus.read_byte(address)
            # Estende o sinal do byte lido para 32 bits.
            value = self.sign_extend(value, 8)
        elif funct3 == Funct3.LH.value:
            # LH (Load Halfword): Lê 2 bytes.
            value = self.bus.read_halfword(address)
            # Estende o sinal da meia palavra lida para 32 bits.
            value = self.sign_extend(value, 16)
        elif funct3 == Funct3.LW.value:
            # LW (Load Word): Lê 4 bytes.
            value = self.bus.read_word(address)
            # Não precisa de extensão de sinal, pois já é 32 bits.
        elif funct3 == Funct3.LBU.value:
            # LBU (Load Byte Unsigned): Lê 1 byte e preenche com zeros.
            value = self.bus.read_byte(address) & 0xFF
        elif funct3 == Funct3.LHU.value:
            # LHU (Load Halfword Unsigned): Lê 2 bytes e preenche com zeros.
            value = self.bus.read_halfword(address) & 0xFFFF
        else:
            raise Exception(f"Funct3 LOAD não implementado: {funct3}")
        
        # Armazena o valor lido (truncado para 32 bits) no registrador de destino.
        self.registers[rd] = value & 0xFFFFFFFF

    def execute_store(self, instruction: int, funct3: int, rs1: int, rs2: int):
        """Executa instruções de store: SB, SH, SW"""
        # Obtém o imediato (offset).
        imm = self.get_immediate_s(instruction)
        # Obtém o valor do registrador base.
        rs1_val = self.registers[rs1]
        # Obtém o valor a ser armazenado.
        rs2_val = self.registers[rs2]
        # Calcula o endereço de memória.
        address = (rs1_val + imm) & 0xFFFFFFFF
        
        # Armazena o valor, truncando-o para o tamanho correto.
        if funct3 == Funct3.SB.value:
            # SB (Store Byte): Escreve 1 byte (bits 7:0 de rs2).
            self.bus.write_byte(address, rs2_val & 0xFF)
        elif funct3 == Funct3.SH.value:
            # SH (Store Halfword): Escreve 2 bytes (bits 15:0 de rs2).
            self.bus.write_halfword(address, rs2_val & 0xFFFF)
        elif funct3 == Funct3.SW.value:
            # SW (Store Word): Escreve 4 bytes (bits 31:0 de rs2).
            self.bus.write_word(address, rs2_val & 0xFFFFFFFF)
        else:
            raise Exception(f"Funct3 STORE não implementado: {funct3}")

    def execute_system(self, instruction: int, rd: int, funct3: int, rs1: int):
        """Executa instruções SYSTEM: ECALL, EBREAK"""
        # Obtém o imediato (usado para diferenciar ECALL/EBREAK).
        imm = self.get_immediate_i(instruction)
        
        if funct3 == 0:  # ECALL/EBREAK
            if imm == 0:
                # ECALL - Chamada de sistema.
                return self.handle_ecall()
            elif imm == 1:
                # EBREAK - Breakpoint.
                return self.handle_ebreak()
        else:
            raise Exception(f"Funct3 SYSTEM não implementado: {funct3}")
        
        return False # Retorna False se a instrução não parar a execução.

    def handle_ecall(self):
        """Trata chamada de sistema ECALL"""
        # O número da chamada de sistema é lido do registrador A7 (x17).
        syscall_num = self.registers[Register.A7.value]
        
        if syscall_num == 10:  # exit
            # Finaliza a simulação.
            print("\nPrograma finalizado com exit (ECALL)")
            return True
        elif syscall_num == 11:  # print character
            # O caractere a ser impresso está em A0 (x10).
            char = self.registers[Register.A0.value] & 0xFF
            print(chr(char), end='')
        elif syscall_num == 1:  # print integer
            # O inteiro a ser impresso está em A0.
            num = self.registers[Register.A0.value]
            print(num, end='')
        else:
            print(f"Chamada de sistema {syscall_num} não implementada.")
            
        return False # Continua a execução.

    def handle_ebreak(self):
        """Trata breakpoint EBREAK"""
        # Sinaliza um breakpoint e sugere parada.
        print(f"\nEBREAK encontrado em PC 0x{self.pc-4:08x}")
        return True # Indica parada.

    def run(self, max_instructions=1000):
        """Executa CPU até encontrar ECALL/EBREAK ou máximo de instruções"""
        instructions = 0
        
        while instructions < max_instructions:
            try:
                # Busca a próxima instrução.
                instruction = self.fetch()
                
                # Verificar se é instrução de parada (ECALL é 0x00000073).
                if instruction == 0x00000073:
                    # Se for ECALL, executa (para obter o tratamento) e sai.
                    self.execute(instruction)
                    break
                
                # Executa a instrução.
                stop = self.execute(instruction)
                # Verifica se a instrução executada sinalizou uma parada (ECALL/EBREAK).
                if stop:
                    break
                    
                instructions += 1
                
            except Exception as e:
                # Trata erros inesperados.
                print(f"Erro executando instrução: {e}")
                break
        
        # Imprime o número de instruções processadas.
        print(f"Executadas {instructions} instruções")
        return instructions
    
def display_vram(self):
    """Exibe conteúdo da VRAM no terminal"""
    print(f"\n=== VRAM após {self.instructions_executed} instruções ===")
    
    # Definir região da VRAM (0x80000 - 0x8FFFF conforme especificação)
    vram_start = 0x80000
    vram_end = 0x8FFFF
    
    # Exibir apenas uma parte (ex: 256 bytes) para não poluir o terminal
    display_size = 256
    
    for addr in range(vram_start, min(vram_start + display_size, vram_end), 16):
        line_hex = ""
        line_ascii = ""
        
        for i in range(16):
            try:
                byte_val = self.bus.read_byte(addr + i)
                line_hex += f"{byte_val:02x} "
                
                # Converter para ASCII se for caractere imprimível
                if 32 <= byte_val <= 126:  # Caracteres imprimíveis
                    line_ascii += chr(byte_val)
                else:
                    line_ascii += "."
            except:
                line_hex += "?? "
                line_ascii += "?"
        
        print(f"0x{addr:08x}: {line_hex} | {line_ascii}")
    
    print("=" * 60)

# --- Classe Memory (Simulação de Memória Principal) ---

class Memory:
    def __init__(self, size=0x100000):  # 1MB de memória (1024KB)
        self.size = size
        # Array de bytes para armazenar os dados da memória.
        self.memory = bytearray(size)
        
        # Mapeamento de memória conforme especificação
        # Define as regiões de memória (RAM, VRAM, I/O).
        self.RAM_START = 0x00000
        self.RAM_END = 0x7FFFF
        self.VRAM_START = 0x80000
        self.VRAM_END = 0x8FFFF
        self.RESERVED_START = 0x90000
        self.RESERVED_END = 0x9FBFF
        self.IO_START = 0x9FC00
        self.IO_END = 0x9FFFF

    # --- Métodos de Leitura/Escrita de Byte ---
    
    def read_byte(self, address: int) -> int:
        """Lê byte da memória"""
        if 0 <= address < self.size:
            # Retorna o valor do byte.
            return self.memory[address]
        else:
            # Endereço fora dos limites.
            raise Exception(f"Endereço de memória inválido: 0x{address:08x}")

    def write_byte(self, address: int, value: int):
        """Escreve byte na memória"""
        if 0 <= address < self.size:
            # Armazena o valor, garantindo que seja um byte (0-255).
            self.memory[address] = value & 0xFF
        else:
            raise Exception(f"Endereço de memória inválido: 0x{address:08x}")

    # --- Métodos de Leitura/Escrita de Meia Palavra (Halfword) ---

    def read_halfword(self, address: int) -> int:
        """Lê meia palavra (16 bits) da memória"""
        # Verifica alinhamento de 2 bytes (meia palavra).
        if address % 2 != 0:
            raise Exception(f"Endereço não alinhado para meia palavra: 0x{address:08x}")
        
        # Combina os dois bytes lidos (Little-Endian: byte[0] + byte[1]<<8).
        return (self.read_byte(address) |
               (self.read_byte(address + 1) << 8))

    def write_halfword(self, address: int, value: int):
        """Escreve meia palavra (16 bits) na memória"""
        if address % 2 != 0:
            raise Exception(f"Endereço não alinhado para meia palavra: 0x{address:08x}")
        
        # Escreve o byte menos significativo.
        self.write_byte(address, value & 0xFF)
        # Escreve o byte mais significativo.
        self.write_byte(address + 1, (value >> 8) & 0xFF)

    # --- Métodos de Leitura/Escrita de Palavra (Word) ---
        
    def read_word(self, address: int) -> int:
        """Lê palavra (32 bits) da memória"""
        # Verifica alinhamento de 4 bytes (palavra).
        if address % 4 != 0:
            raise Exception(f"Endereço não alinhado para palavra: 0x{address:08x}")
        
        # Combina os quatro bytes lidos (Little-Endian).
        return (self.read_byte(address) |
               (self.read_byte(address + 1) << 8) |
               (self.read_byte(address + 2) << 16) |
               (self.read_byte(address + 3) << 24))

    def write_word(self, address: int, value: int):
        """Escreve palavra (32 bits) na memória"""
        if address % 4 != 0:
            raise Exception(f"Endereço não alinhado para palavra: 0x{address:08x}")
        
        # Escreve os quatro bytes em ordem Little-Endian.
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
        self.write_byte(address + 2, (value >> 16) & 0xFF)
        self.write_byte(address + 3, (value >> 24) & 0xFF)

    # --- Métodos de Utilitários de Memória ---
        
    def load_program(self, program: List[int], start_address: int = 0):
        """Carrega programa na memória"""
        # Escreve cada instrução (palavra de 4 bytes) na memória a partir do endereço inicial.
        for i, instruction in enumerate(program):
            address = start_address + i * 4
            self.write_word(address, instruction)

    def display_vram(self):
        """Exibe conteúdo da VRAM como caracteres ASCII"""
        # Função de depuração para mostrar o que foi escrito na VRAM.
        print("\n--- VRAM Content ---")
        # Itera sobre um bloco da VRAM (0x80000 a 0x80100).
        for addr in range(self.VRAM_START, min(self.VRAM_START + 256, self.VRAM_END), 16):
            line = ""
            for i in range(16):
                byte_val = self.read_byte(addr + i)
                # Converte para ASCII se for um caractere imprimível.
                if 32 <= byte_val <= 126:  # Caracteres ASCII imprimíveis
                    line += chr(byte_val)
                else:
                    line += "." # Usa ponto para não imprimíveis.
            print(f"0x{addr:08x}: {line}")
        print("-------------------\n")

# --- Classe Bus (Barramento de Comunicação) ---

class Bus:
    def __init__(self, memory: Memory):
        # O Bus age como um intermediário que roteia as requisições para o dispositivo correto (aqui, apenas a Memória).
        self.memory = memory

    # Métodos de leitura/escrita do Bus que delegam para a Memória.
    def read_byte(self, address: int) -> int:
        """Lê byte do barramento"""
        return self.memory.read_byte(address)

    def write_byte(self, address: int, value: int):
        """Escreve byte no barramento"""
        self.memory.write_byte(address, value)

    def read_halfword(self, address: int) -> int:
        """Lê meia palavra do barramento"""
        return self.memory.read_halfword(address)

    def write_halfword(self, address: int, value: int):
        """Escreve meia palavra no barramento"""
        self.memory.write_halfword(address, value)

    def read_word(self, address: int) -> int:
        """Lê palavra do barramento"""
        return self.memory.read_word(address)

    def write_word(self, address: int, value: int):
        """Escreve palavra no barramento"""
        self.memory.write_word(address, value)

# --- Classe Simulator (Coordenação) ---

class Simulator:
    def __init__(self):
        # Cria a Memória.
        self.memory = Memory()
        # Cria o Bus, conectado à Memória.
        self.bus = Bus(self.memory)
        # Cria a CPU, conectada à Memória e ao Bus.
        self.cpu = CPU(self.memory, self.bus)

    def load_test_program(self):
        """Carrega programa de teste que demonstra todas as instruções RV32I"""
        
        # Programa de teste (valores hexadecimais das instruções).
        test_program = [
            # === ARITMÉTICA (Tipo I e R) === 
            # Endereços 0x00 a 0x0C
            0x00a00093,  # 0x00: addi x1, x0, 10  (x1 = 10)
            0x01400113,  # 0x04: addi x2, x0, 20  (x2 = 20)
            0x002081b3,  # 0x08: add x3, x1, x2   (x3 = 30)
            0x40110233,  # 0x0C: sub x4, x2, x1   (x4 = 10)
            
            # === LÓGICA (Tipo I e R) === 
            # Endereços 0x10 a 0x24
            0x00f0e293,  # 0x10: ori x5, x1, 15   (x5 = 10 | 15 = 15)
            0x00a17313,  # 0x14: andi x6, x2, 10  (x6 = 20 & 10 = 0)
            0x0050c393,  # 0x18: xori x7, x1, 5   (x7 = 10 ^ 5 = 15)
            0x0020e433,  # 0x1C: or x8, x1, x2    (x8 = 10 | 20 = 30)
            0x0020f4b3,  # 0x20: and x9, x1, x2   (x9 = 10 & 20 = 0)
            0x0020c533,  # 0x24: xor x10, x1, x2  (x10 = 10 ^ 20 = 30)
            
            # === SHIFTS (Tipo I e R) === 
            # Endereços 0x28 a 0x3C
            0x00209593,  # 0x28: slli x11, x1, 2  (x11 = 10 << 2 = 40)
            0x00115593,  # 0x2C: srli x11, x2, 1  (x11 = 20 >> 1 = 10)
            0xff600613,  # 0x30: addi x12, x0, -10 (x12 = -10)
            0x40165693,  # 0x34: srai x13, x12, 1 (x13 = -10 >>_a 1 = -5)
            0x00209733,  # 0x38: sll x14, x1, x2  (x14 = 10 << 20)
            0x001157b3,  # 0x3C: srl x15, x2, x1  (x15 = 20 >> 10 = 0)
            
            # === COMPARAÇÃO (Tipo I e R) === 
            # Endereços 0x40 a 0x4C
            0x00f0a813,  # 0x40: slti x16, x1, 15 (x16 = 1)
            0x00f0b893,  # 0x44: sltiu x17, x1, 15 (x17 = 1)
            0x0020a913,  # 0x48: slt x18, x1, x2  (x18 = 1)
            0x0020b993,  # 0x4C: sltu x19, x1, x2  (x19 = 1)
            
            # === LOAD/STORE (Tipo I e S) ===
            # Endereços 0x50 a 0x54
            0x00102223,  # 0x50: sw x1, 4(x0)     (Armazena x1=10 no endereço RAM 0x4)
            0x00402a03,  # 0x54: lw x20, 4(x0)    (Carrega 10 do endereço 0x4 para x20)
            
            # === NOPs (Substituindo Branch/Jump para execução sequencial) ===
            # Endereços 0x58 a 0x60
            0x00000013,  # 0x58: NOP (No Operation: addi x0, x0, 0)
            0x00000013,  # 0x5C: NOP
            0x00000013,  # 0x60: NOP
            
            # === LUI/AUIPC (Tipo U) ===
            # Endereços 0x64 a 0x68
            0x12345ab7,  # 0x64: lui x21, 0x12345 (x21 = 0x12345000)
            0x00000b17,  # 0x68: auipc x22, 0     (x22 = PC da instrução (0x68) + 0 = 0x68)
            
            # === FIM (Manipulação da VRAM e I/O) ===
            # Endereços 0x6C a 0x84
            0x000804b7,  # 0x6C: lui x9, 0x80000  (x9 = 0x80000000, início da VRAM)
            0x04600513,  # 0x70: addi x10, x0, 70 ('F' ASCII)
            0x00a48023,  # 0x74: sb x10, 0(x9)    (Store Byte 'F' em 0x80000000)
            0x04900513,  # 0x78: addi x10, x0, 73 ('I' ASCII)
            0x00a480a3,  # 0x7C: sb x10, 1(x9)    (Store Byte 'I' em 0x80000001)
            0x04d00513,  # 0x80: addi x10, x0, 77 ('M' ASCII)
            0x00a48123,  # 0x84: sb x10, 2(x9)    (Store Byte 'M' em 0x80000002)
            
            # === EXIT (SYSTEM) ===
            # Endereços 0x88 a 0x8C
            0x00a00893,  # 0x88: addi x17, x0, 10 (a7 = 10, código para a chamada de sistema 'exit')
            0x00000073,  # 0x8C: ecall            (Executa a chamada de sistema, parando a simulação)
        ]
        
        self.memory.load_program(test_program)

    def run_simulation(self):
        """Executa simulação completa"""
        print("=== Simulador RISC-V RV32I ===")
        print("Implementando instruções básicas da tabela")
        print("Iniciando execução...")
        
        # Carregar programa de teste.
        self.load_test_program()
        
        # Executar CPU (limite de 1000 instruções).
        instructions = self.cpu.run(1000)
        
        # Mostrar estado final.
        print("\n=== Estado Final ===")
        self.display_cpu_state()
        # Exibe o conteúdo da VRAM para verificar o store.
        self.memory.display_vram()
        
        print(f"Total de instruções executadas: {instructions}")
        print("Simulação concluída!")

    def display_cpu_state(self):
        """Exibe estado atual da CPU"""
        # Exibe os registradores formatados com seus nomes de convenção.
        print("\n--- Registradores ---")
        reg_names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", 
                     "s0/fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                     "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                     "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
        
        # Imprime 4 registradores por linha.
        for i in range(0, 32, 4):
            line = ""
            for j in range(4):
                reg_idx = i + j
                if reg_idx < 32:
                    # Formata a saída: x<índice>(<nome>)=<valor em hexa>
                    line += f"x{reg_idx:2}({reg_names[reg_idx]:5})=0x{self.cpu.registers[reg_idx]:08x} "
            print(line)
        
        # Exibe o valor final do PC.
        print(f"PC: 0x{self.cpu.pc:08x}")

def main():
    """Função principal"""
    # Cria a instância do Simulador.
    simulator = Simulator()
    
    try:
        # Inicia a simulação.
        simulator.run_simulation()
    except KeyboardInterrupt:
        # Tratamento para interrupção do usuário (Ctrl+C).
        print("\nSimulação interrompida pelo usuário")
    except Exception as e:
        # Tratamento de erros gerais.
        print(f"Erro durante simulação: {e}")
        import traceback
        # Imprime o stack trace para depuração.
        traceback.print_exc()

if __name__ == "__main__":
    # Garante que a função principal seja chamada quando o script for executado.
    main()