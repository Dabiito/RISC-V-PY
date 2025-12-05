import sys
from enum import Enum
from typing import List, Tuple

# --- Enumerações (Constantes RISC-V) ---
class InstructionType(Enum):
    R = "R"
    I = "I"
    S = "S"
    B = "B"
    U = "U"
    J = "J"

class Opcode(Enum):
    LOAD = 0b0000011
    OP_IMM = 0b0010011
    AUIPC = 0b0010111
    STORE = 0b0100011
    OP = 0b0110011
    LUI = 0b0110111
    BRANCH = 0b1100011
    JALR = 0b1100111
    JAL = 0b1101111
    SYSTEM = 0b1110011

class Funct3(Enum):
    BEQ = 0b000
    BNE = 0b001
    BLT = 0b100
    BGE = 0b101
    BLTU = 0b110
    BGEU = 0b111
    
    LB = 0b000
    LH = 0b001
    LW = 0b010
    LBU = 0b100
    LHU = 0b101
    SB = 0b000
    SH = 0b001
    SW = 0b010
    
    ADDI = 0b000
    SLTI = 0b010
    SLTIU = 0b011
    XORI = 0b100
    ORI = 0b110
    ANDI = 0b111
    SLLI = 0b001
    SRLI = 0b101
    SRAI = 0b101
    
    ADD = 0b000
    SUB = 0b000
    SLL = 0b001
    SLT = 0b010
    SLTU = 0b011
    XOR = 0b100
    SRL = 0b101
    SRA = 0b101
    OR = 0b110
    AND = 0b111

class Register(Enum):
    ZERO = 0
    RA = 1
    SP = 2
    GP = 3
    TP = 4
    T0 = 5
    T1 = 6
    T2 = 7
    S0 = 8
    S1 = 9
    A0 = 10
    A1 = 11
    A2 = 12
    A3 = 13
    A4 = 14
    A5 = 15
    A6 = 16
    A7 = 17
    S2 = 18
    S3 = 19
    S4 = 20
    S5 = 21
    S6 = 22
    S7 = 23
    S8 = 24
    S9 = 25
    S10 = 26
    S11 = 27
    T3 = 28
    T4 = 29
    T5 = 30
    T6 = 31

# --- Classe CPU ---
class CPU:
    def __init__(self, memory, bus):
        self.registers = [0] * 32
        self.pc = 0
        self.memory = memory
        self.bus = bus
        
        self.registers[Register.SP.value] = 0x80000
        self.registers[Register.GP.value] = 0x1000
        
        self.instructions_executed = 0
        self.vram_display_interval = 50
        self.last_vram_display = 0

    def fetch(self) -> int:
        if self.pc % 4 != 0:
            raise Exception(f"PC não alinhado: 0x{self.pc:08x}")
        
        instruction = self.bus.read_word(self.pc)
        self.pc += 4
        return instruction

    def decode(self, instruction: int) -> Tuple[Opcode, int, int, int, int, int]:
        opcode = instruction & 0x7F
        rd = (instruction >> 7) & 0x1F
        funct3 = (instruction >> 12) & 0x7
        rs1 = (instruction >> 15) & 0x1F
        rs2 = (instruction >> 20) & 0x1F
        funct7 = (instruction >> 25) & 0x7F
        
        return Opcode(opcode), rd, funct3, rs1, rs2, funct7

    def execute(self, instruction: int):
        opcode, rd, funct3, rs1, rs2, funct7 = self.decode(instruction)
        
        try:
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
                raise Exception(f"Opcode não implementado: {opcode}")
                
        except Exception as e:
            print(f"Erro executando instrução 0x{instruction:08x} em PC 0x{self.pc-4:08x}: {e}")
            raise
            
        self.instructions_executed += 1

        if (self.instructions_executed - self.last_vram_display >= 
            self.vram_display_interval):
            self.display_vram()
            self.last_vram_display = self.instructions_executed

    def sign_extend(self, value: int, bits: int) -> int:
        if value & (1 << (bits - 1)):
            return value | (~((1 << bits) - 1) & 0xFFFFFFFF)
        return value
    
    def get_immediate_i(self, instruction: int) -> int:
        return self.sign_extend((instruction >> 20) & 0xFFF, 12)

    def get_immediate_s(self, instruction: int) -> int:
        imm_11_5 = (instruction >> 25) & 0x7F
        imm_4_0 = (instruction >> 7) & 0x1F
        return self.sign_extend((imm_11_5 << 5) | imm_4_0, 12)

    def get_immediate_b(self, instruction: int) -> int:
        imm_12 = (instruction >> 31) & 0x1
        imm_11 = (instruction >> 7) & 0x1
        imm_10_5 = (instruction >> 25) & 0x3F
        imm_4_1 = (instruction >> 8) & 0xF
        imm = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
        return self.sign_extend(imm, 13)

    def get_immediate_u(self, instruction: int) -> int:
        return (instruction & 0xFFFFF000)

    def get_immediate_j(self, instruction: int) -> int:
        imm_20 = (instruction >> 31) & 0x1
        imm_19_12 = (instruction >> 12) & 0xFF
        imm_11 = (instruction >> 20) & 0x1
        imm_10_1 = (instruction >> 21) & 0x3FF
        imm = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
        return self.sign_extend(imm, 21)

    def execute_op_imm(self, instruction: int, rd: int, funct3: int, rs1: int):
        imm = self.get_immediate_i(instruction)
        rs1_val = self.registers[rs1]
        
        if funct3 == Funct3.ADDI.value:
            result = rs1_val + imm
        elif funct3 == Funct3.SLTI.value:
            result = 1 if rs1_val < imm else 0
        elif funct3 == Funct3.SLTIU.value:
            result = 1 if (rs1_val & 0xFFFFFFFF) < (imm & 0xFFFFFFFF) else 0
        elif funct3 == Funct3.XORI.value:
            result = rs1_val ^ imm
        elif funct3 == Funct3.ORI.value:
            result = rs1_val | imm
        elif funct3 == Funct3.ANDI.value:
            result = rs1_val & imm
        elif funct3 == Funct3.SLLI.value:
            shamt = imm & 0x1F
            result = rs1_val << shamt
        elif funct3 == Funct3.SRLI.value:
            shamt = imm & 0x1F
            result = (rs1_val & 0xFFFFFFFF) >> shamt
        elif funct3 == Funct3.SRAI.value:
            shamt = imm & 0x1F
            if rs1_val & 0x80000000:
                result = (rs1_val >> shamt) | (~0 << (32 - shamt))
            else:
                result = rs1_val >> shamt
        else:
            raise Exception(f"Funct3 OP-IMM não implementado: {funct3}")
        
        self.registers[rd] = result & 0xFFFFFFFF

    def execute_op(self, instruction: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int):
        rs1_val = self.registers[rs1]
        rs2_val = self.registers[rs2]
        
        if funct3 == Funct3.ADD.value:
            if funct7 == 0:
                result = rs1_val + rs2_val
            elif funct7 == 0x20:
                result = rs1_val - rs2_val
            else:
                raise Exception(f"Funct7 inválido para ADD/SUB: {funct7}")
        elif funct3 == Funct3.SLL.value:
            shamt = rs2_val & 0x1F
            result = rs1_val << shamt
        elif funct3 == Funct3.SLT.value:
            result = 1 if rs1_val < rs2_val else 0
        elif funct3 == Funct3.SLTU.value:
            result = 1 if (rs1_val & 0xFFFFFFFF) < (rs2_val & 0xFFFFFFFF) else 0
        elif funct3 == Funct3.XOR.value:
            result = rs1_val ^ rs2_val
        elif funct3 == Funct3.SRL.value:
            if funct7 == 0:
                shamt = rs2_val & 0x1F
                result = (rs1_val & 0xFFFFFFFF) >> shamt
            elif funct7 == 0x20:
                shamt = rs2_val & 0x1F
                if rs1_val & 0x80000000:
                    result = (rs1_val >> shamt) | (~0 << (32 - shamt))
                else:
                    result = rs1_val >> shamt
            else:
                raise Exception(f"Funct7 inválido para SRL/SRA: {funct7}")
        elif funct3 == Funct3.OR.value:
            result = rs1_val | rs2_val
        elif funct3 == Funct3.AND.value:
            result = rs1_val & rs2_val
        else:
            raise Exception(f"Funct3 OP não implementado: {funct3}")
        
        self.registers[rd] = result & 0xFFFFFFFF

    def execute_lui(self, instruction: int, rd: int):
        imm = self.get_immediate_u(instruction)
        self.registers[rd] = imm & 0xFFFFFFFF

    def execute_auipc(self, instruction: int, rd: int):
        imm = self.get_immediate_u(instruction)
        self.registers[rd] = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_jal(self, instruction: int, rd: int):
        imm = self.get_immediate_j(instruction)
        self.registers[rd] = self.pc
        self.pc = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_jalr(self, instruction: int, rd: int, funct3: int, rs1: int):
        if funct3 != 0:
            raise Exception(f"Funct3 JALR não zero: {funct3}")
        
        imm = self.get_immediate_i(instruction)
        rs1_val = self.registers[rs1]
        target = (rs1_val + imm) & 0xFFFFFFFE
        
        self.registers[rd] = self.pc
        self.pc = target

    def execute_branch(self, instruction: int, funct3: int, rs1: int, rs2: int):
        rs1_val = self.registers[rs1]
        rs2_val = self.registers[rs2]
        imm = self.get_immediate_b(instruction)
        
        branch_taken = False
        
        if funct3 == Funct3.BEQ.value:
            branch_taken = (rs1_val == rs2_val)
        elif funct3 == Funct3.BNE.value:
            branch_taken = (rs1_val != rs2_val)
        elif funct3 == Funct3.BLT.value:
            branch_taken = (rs1_val < rs2_val)
        elif funct3 == Funct3.BGE.value:
            branch_taken = (rs1_val >= rs2_val)
        elif funct3 == Funct3.BLTU.value:
            branch_taken = ((rs1_val & 0xFFFFFFFF) < (rs2_val & 0xFFFFFFFF))
        elif funct3 == Funct3.BGEU.value:
            branch_taken = ((rs1_val & 0xFFFFFFFF) >= (rs2_val & 0xFFFFFFFF))
        else:
            raise Exception(f"Funct3 BRANCH não implementado: {funct3}")
        
        if branch_taken:
            self.pc = (self.pc - 4 + imm) & 0xFFFFFFFF

    def execute_load(self, instruction: int, rd: int, funct3: int, rs1: int):
        imm = self.get_immediate_i(instruction)
        rs1_val = self.registers[rs1]
        address = (rs1_val + imm) & 0xFFFFFFFF
        
        value = 0
        
        if funct3 == Funct3.LB.value:
            value = self.bus.read_byte(address)
            value = self.sign_extend(value, 8)
        elif funct3 == Funct3.LH.value:
            value = self.bus.read_halfword(address)
            value = self.sign_extend(value, 16)
        elif funct3 == Funct3.LW.value:
            value = self.bus.read_word(address)
        elif funct3 == Funct3.LBU.value:
            value = self.bus.read_byte(address) & 0xFF
        elif funct3 == Funct3.LHU.value:
            value = self.bus.read_halfword(address) & 0xFFFF
        else:
            raise Exception(f"Funct3 LOAD não implementado: {funct3}")
        
        self.registers[rd] = value & 0xFFFFFFFF

    def execute_store(self, instruction: int, funct3: int, rs1: int, rs2: int):
        imm = self.get_immediate_s(instruction)
        rs1_val = self.registers[rs1]
        rs2_val = self.registers[rs2]
        address = (rs1_val + imm) & 0xFFFFFFFF
        
        if funct3 == Funct3.SB.value:
            self.bus.write_byte(address, rs2_val & 0xFF)
        elif funct3 == Funct3.SH.value:
            self.bus.write_halfword(address, rs2_val & 0xFFFF)
        elif funct3 == Funct3.SW.value:
            self.bus.write_word(address, rs2_val & 0xFFFFFFFF)
        else:
            raise Exception(f"Funct3 STORE não implementado: {funct3}")

    def execute_system(self, instruction: int, rd: int, funct3: int, rs1: int):
        imm = self.get_immediate_i(instruction)
        
        if funct3 == 0:
            if imm == 0:
                return self.handle_ecall()
            elif imm == 1:
                return self.handle_ebreak()
        else:
            raise Exception(f"Funct3 SYSTEM não implementado: {funct3}")
        
        return False

    def handle_ecall(self):
        syscall_num = self.registers[Register.A7.value]
        
        if syscall_num == 10:
            print("\nPrograma finalizado com exit (ECALL)")
            return True
        elif syscall_num == 11:
            char = self.registers[Register.A0.value] & 0xFF
            print(chr(char), end='')
        elif syscall_num == 1:
            num = self.registers[Register.A0.value]
            print(num, end='')
        else:
            print(f"Chamada de sistema {syscall_num} não implementada.")
            
        return False

    def handle_ebreak(self):
        print(f"\nEBREAK encontrado em PC 0x{self.pc-4:08x}")
        return True

    def run(self, max_instructions=1000):
        instructions = 0
        
        while instructions < max_instructions:
            try:
                instruction = self.fetch()
                
                if instruction == 0x00000073:
                    self.execute(instruction)
                    break
                
                stop = self.execute(instruction)
                if stop:
                    break
                    
                instructions += 1
                
            except Exception as e:
                print(f"Erro executando instrução: {e}")
                break
        
        print(f"Executadas {instructions} instruções")
        return instructions
    
    def display_vram(self):
        print(f"\n=== VRAM após {self.instructions_executed} instruções ===")
        
        vram_start = 0x80000
        vram_end = 0x8FFFF
        display_size = 256
        
        for addr in range(vram_start, min(vram_start + display_size, vram_end), 16):
            line_hex = ""
            line_ascii = ""
            
            for i in range(16):
                try:
                    byte_val = self.bus.read_byte(addr + i)
                    line_hex += f"{byte_val:02x} "
                    
                    if 32 <= byte_val <= 126:
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
    def __init__(self, size=0x100000):
        self.size = size
        self.memory = bytearray(size)
        
        self.RAM_START = 0x00000
        self.RAM_END = 0x7FFFF
        self.VRAM_START = 0x80000
        self.VRAM_END = 0x8FFFF
        self.RESERVED_START = 0x90000
        self.RESERVED_END = 0x9FBFF
        self.IO_START = 0x9FC00
        self.IO_END = 0x9FFFF

    def read_byte(self, address: int) -> int:
        if 0 <= address < self.size:
            return self.memory[address]
        else:
            raise Exception(f"Endereço de memória inválido: 0x{address:08x}")

    def write_byte(self, address: int, value: int):
        if 0 <= address < self.size:
            self.memory[address] = value & 0xFF
        else:
            raise Exception(f"Endereço de memória inválido: 0x{address:08x}")

    def read_halfword(self, address: int) -> int:
        if address % 2 != 0:
            raise Exception(f"Endereço não alinhado para meia palavra: 0x{address:08x}")
        
        return (self.read_byte(address) |
               (self.read_byte(address + 1) << 8))

    def write_halfword(self, address: int, value: int):
        if address % 2 != 0:
            raise Exception(f"Endereço não alinhado para meia palavra: 0x{address:08x}")
        
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)

    def read_word(self, address: int) -> int:
        if address % 4 != 0:
            raise Exception(f"Endereço não alinhado para palavra: 0x{address:08x}")
        
        return (self.read_byte(address) |
               (self.read_byte(address + 1) << 8) |
               (self.read_byte(address + 2) << 16) |
               (self.read_byte(address + 3) << 24))

    def write_word(self, address: int, value: int):
        if address % 4 != 0:
            raise Exception(f"Endereço não alinhado para palavra: 0x{address:08x}")
        
        self.write_byte(address, value & 0xFF)
        self.write_byte(address + 1, (value >> 8) & 0xFF)
        self.write_byte(address + 2, (value >> 16) & 0xFF)
        self.write_byte(address + 3, (value >> 24) & 0xFF)

    def load_program(self, program: List[int], start_address: int = 0):
        for i, instruction in enumerate(program):
            address = start_address + i * 4
            self.write_word(address, instruction)

    def display_vram(self):
        print("\n--- VRAM Content ---")
        for addr in range(self.VRAM_START, min(self.VRAM_START + 256, self.VRAM_END), 16):
            line = ""
            for i in range(16):
                byte_val = self.read_byte(addr + i)
                if 32 <= byte_val <= 126:
                    line += chr(byte_val)
                else:
                    line += "."
            print(f"0x{addr:08x}: {line}")
        print("-------------------\n")

# --- Classe Bus (Barramento de Comunicação) ---
class Bus:
    def __init__(self, memory: Memory):
        self.memory = memory

    def read_byte(self, address: int) -> int:
        return self.memory.read_byte(address)

    def write_byte(self, address: int, value: int):
        self.memory.write_byte(address, value)

    def read_halfword(self, address: int) -> int:
        return self.memory.read_halfword(address)

    def write_halfword(self, address: int, value: int):
        self.memory.write_halfword(address, value)

    def read_word(self, address: int) -> int:
        return self.memory.read_word(address)

    def write_word(self, address: int, value: int):
        self.memory.write_word(address, value)

# --- Classe Simulator (Coordenação) ---
class Simulator:
    def __init__(self):
        self.memory = Memory()
        self.bus = Bus(self.memory)
        self.cpu = CPU(self.memory, self.bus)

    def load_test_program(self):
        test_program = [
            0x00a00093,
            0x01400113,
            0x002081b3,
            0x40110233,
            
            0x00f0e293,
            0x00a17313,
            0x0050c393,
            0x0020e433,
            0x0020f4b3,
            0x0020c533,
            
            0x00209593,
            0x00115593,
            0xff600613,
            0x40165693,
            0x00209733,
            0x001157b3,
            
            0x00f0a813,
            0x00f0b893,
            0x0020a913,
            0x0020b993,
            
            0x00102223,
            0x00402a03,
            
            0x00000013,
            0x00000013,
            0x00000013,
            
            0x12345ab7,
            0x00000b17,
            
            0x000804b7,
            0x04600513,
            0x00a48023,
            0x04900513,
            0x00a480a3,
            0x04d00513,
            0x00a48123,
            
            0x00a00893,
            0x00000073,
        ]
        
        self.memory.load_program(test_program)

    def run_simulation(self):
        print("=== Simulador RISC-V RV32I ===")
        print("Iniciando execução...")
        
        self.load_test_program()
        
        instructions = self.cpu.run(1000)
        
        print("\n=== Estado Final ===")
        self.display_cpu_state()
        self.memory.display_vram()
        
        print(f"Total de instruções executadas: {instructions}")
        print("Simulação concluída!")

    def display_cpu_state(self):
        print("\n--- Registradores ---")
        reg_names = ["zero", "ra", "sp", "gp", "tp", "t0", "t1", "t2", 
                     "s0/fp", "s1", "a0", "a1", "a2", "a3", "a4", "a5",
                     "a6", "a7", "s2", "s3", "s4", "s5", "s6", "s7",
                     "s8", "s9", "s10", "s11", "t3", "t4", "t5", "t6"]
        
        for i in range(0, 32, 4):
            line = ""
            for j in range(4):
                reg_idx = i + j
                if reg_idx < 32:
                    line += f"x{reg_idx:2}({reg_names[reg_idx]:5})=0x{self.cpu.registers[reg_idx]:08x} "
            print(line)
        
        print(f"PC: 0x{self.cpu.pc:08x}")

def main():
    simulator = Simulator()
    
    try:
        simulator.run_simulation()
    except KeyboardInterrupt:
        print("\nSimulação interrompida pelo usuário")
    except Exception as e:
        print(f"Erro durante simulação: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
