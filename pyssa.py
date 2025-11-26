# Copyright (c) 2025 Jifeng Wu
# Licensed under the Apache-2.0 License. See LICENSE file in the project root for full license information.
import inspect
from collections import OrderedDict, deque
from dis import Instruction, get_instructions
from enum import Enum
from itertools import chain
from types import CodeType
from typing import Iterable, List, Mapping, Dict, Optional, Sequence, Tuple

import networkx as nx
from put_back_iterator import PutBackIterator

NoneType = type(None)
EllipsisType = type(Ellipsis)


# =============================
# IR (Intermediate Representation) classes
# =============================


class IRInstruction(object): pass


class IRInPlaceBinaryOp(IRInstruction):
    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs


class IRStore(IRInstruction):
    def __init__(self, name, value, force_global=False):
        self.name = name
        self.value = value
        self.force_global = force_global


class IRStoreSubscr(IRInstruction):
    def __init__(self, key, container, value):
        self.key = key
        self.container = container
        self.value = value


class IRDeleteSubscr(IRInstruction):
    def __init__(self, key, container):
        self.key = key
        self.container = container


class IRStoreAttr(IRInstruction):
    def __init__(self, obj, attribute, value):
        self.obj = obj
        self.attribute = attribute
        self.value = value


class IRDelete(IRInstruction):
    def __init__(self, name):
        self.name = name


class IRSetException(IRInstruction):
    def __init__(self, exception):
        self.exception = exception


class IRBranch(IRInstruction):
    def __init__(self, condition, if_true):
        self.condition = condition
        self.if_true = if_true


class IRJump(IRInstruction):
    def __init__(self, target):
        self.target = target


class IRReturn(IRInstruction):
    def __init__(self, value):
        self.value = value


class IRRaise(IRInstruction):
    def __init__(self, exception_instance_or_type):
        self.exception_instance_or_type = exception_instance_or_type


class IRImportStar(IRInstruction):
    def __init__(self, module):
        self.module = module


class IRDeleteAttr(IRInstruction):
    def __init__(self, obj, attribute):
        self.obj = obj
        self.attribute = attribute


class IRSetupAnnotations(IRInstruction): pass


# Used for unique IR objects such as basic blocks, constants, temporaries, etc.
class IRValue(object): pass


class IRRegion(IRValue):
    def __init__(self, name, posonlyargs, args, varargs, kwonlyargs, varkeywords):
        self.name = name
        self.basic_blocks = []
        self.posonlyargs = posonlyargs
        self.args = args
        self.varargs = varargs
        self.kwonlyargs = kwonlyargs
        self.varkeywords = varkeywords


class IRBasicBlock(IRValue):
    def __init__(self):
        self.instructions = []


class IRConstant(IRInstruction, IRValue):
    def __init__(self, literal_value):
        self.literal_value = literal_value


class IRLoad(IRInstruction, IRValue):
    def __init__(self, name):
        self.name = name


class IRLoadRegion(IRInstruction, IRValue):
    def __init__(self, name):
        self.name = name


class IRImport(IRInstruction, IRValue):
    def __init__(self, name, level):
        self.name = name
        self.level = level


class IRImportFrom(IRInstruction, IRValue):
    def __init__(self, module, name):
        self.module = module
        self.name = name


class IRLoadBuiltIn(IRInstruction, IRValue):
    def __init__(self, builtin):
        self.builtin = builtin


class IRMakeFunction(IRInstruction, IRValue):
    def __init__(
            self,
            loaded_region,
            parameter_default_values,
            keyword_only_parameter_default_values,
            free_variable_cells,
            annotations,
    ):
        self.loaded_region = loaded_region
        self.parameter_default_values = parameter_default_values
        self.keyword_only_parameter_default_values = keyword_only_parameter_default_values
        self.free_variable_cells = free_variable_cells
        self.annotations = annotations


class IRLoadAttr(IRInstruction, IRValue):
    def __init__(self, value, attribute):
        self.value = value
        self.attribute = attribute


class IRLoadSuperAttr(IRInstruction, IRValue):
    def __init__(self, cls_value, self_value, attribute):
        self.cls_value = cls_value
        self.self_value = self_value
        self.attribute = attribute


class IRUnaryOperator(Enum):
    INVERT = '~'
    NOT = 'not'
    UNARY_ADD = '+'
    UNARY_SUB = '-'


class IRUnaryOp(IRInstruction, IRValue):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand


class IRBinaryOperator(Enum):
    AND = 'and'
    OR = 'or'
    ADD = '+'
    SUB = '-'
    MULT = '*'
    MAT_MULT = '@'
    DIV = '/'
    MOD = '%'
    POW = '**'
    LEFT_SHIFT = '<<'
    RIGHT_SHIFT = '>>'
    BITWISE_OR = '|'
    BITWISE_XOR = '^'
    BITWISE_AND = '&'
    FLOOR_DIV = '//'
    EQ = '=='
    NOT_EQ = '!='
    LT = '<'
    LE = '<='
    GT = '>'
    GE = '>='
    IS = 'is'
    IS_NOT = 'is not'
    IN = 'in'
    NOT_IN = 'not in'


# Mapping bytecode argument to IR operator.
ARGVAL_TO_IR_BINARY_OPERATORS = {
    0: IRBinaryOperator.ADD,
    1: IRBinaryOperator.BITWISE_AND,
    2: IRBinaryOperator.FLOOR_DIV,
    4: IRBinaryOperator.MAT_MULT,
    5: IRBinaryOperator.MULT,
    6: IRBinaryOperator.MOD,
    7: IRBinaryOperator.BITWISE_OR,
    8: IRBinaryOperator.POW,
    10: IRBinaryOperator.SUB,
    11: IRBinaryOperator.DIV,
    12: IRBinaryOperator.BITWISE_XOR,
    '==': IRBinaryOperator.EQ,
    '!=': IRBinaryOperator.NOT_EQ,
    '<': IRBinaryOperator.LT,
    '>': IRBinaryOperator.GT,
    '>=': IRBinaryOperator.GE,
    '<=': IRBinaryOperator.LE,
}

ARGVAL_TO_IR_INPLACE_BINARY_OPERATORS = {
    13: IRBinaryOperator.ADD,
    15: IRBinaryOperator.FLOOR_DIV,
    18: IRBinaryOperator.MULT,
    23: IRBinaryOperator.SUB,
    24: IRBinaryOperator.DIV,
}


class IRBinaryOp(IRInstruction, IRValue):
    def __init__(self, lhs, op, rhs):
        self.lhs = lhs
        self.op = op
        self.rhs = rhs


class IRBinarySubscr(IRInstruction, IRValue):
    def __init__(self, key, container):
        self.key = key
        self.container = container


class IRUnpackSequence(IRInstruction, IRValue):
    def __init__(self, sequence, size):
        self.sequence = sequence
        self.size = size


class IRUnpackEx(IRInstruction, IRValue):
    def __init__(self, sequence, leading, trailing):
        self.sequence = sequence
        self.leading = leading
        self.trailing = trailing


class IRBuildList(IRInstruction, IRValue):
    def __init__(self, values):
        self.values = values


class IRBuildTuple(IRInstruction, IRValue):
    def __init__(self, values):
        self.values = values


class IRBuildSet(IRInstruction, IRValue):
    def __init__(self, values, frozen=False):
        self.values = values
        self.frozen = frozen


class IRCall(IRInstruction, IRValue):
    def __init__(self, callee, arguments, keywords):
        self.callee = callee
        self.arguments = arguments
        self.keywords = keywords


class IRCallFunctionEx(IRInstruction, IRValue):
    def __init__(self, callee, args, kwargs):
        self.callee = callee
        self.args = args
        self.kwargs = kwargs


class IRGetIter(IRInstruction, IRValue):
    def __init__(self, value):
        self.value = value


class IRForIter(IRInstruction, IRValue):
    def __init__(self, iterator, target):
        self.iterator = iterator
        self.target = target


class IRBuildSlice(IRInstruction, IRValue):
    def __init__(self, start, stop, step):
        self.start = start
        self.stop = stop
        self.step = step


class IRFormatString(IRInstruction, IRValue):
    def __init__(self, value, fmt_spec):
        self.value = value
        self.fmt_spec = fmt_spec


class IRConcatenateStrings(IRInstruction, IRValue):
    def __init__(self, strings):
        self.strings = strings


class IRYield(IRInstruction, IRValue):
    def __init__(self, value):
        self.value = value


class IRBuildMap(IRInstruction, IRValue):
    def __init__(self, keys, values):
        self.keys = keys
        self.values = values


class IRGetException(IRInstruction, IRValue): pass


class ImplementationError(Exception): pass


class UnexpectedError(Exception): pass


class IRRegionBuilder(object):
    def __init__(self):
        self.region_names_to_regions = OrderedDict()  # type: Dict[str, IRRegion]
        self.is_generator = OrderedDict()  # type: Dict[str, bool]

    def build_region(
            self,
            region_name_components,  # type: Sequence[str]
            region_code,  # type: CodeType
    ):
        region_name = '.'.join(region_name_components)

        posonlyargcount = region_code.co_posonlyargcount
        argcount = region_code.co_argcount

        if region_code.co_flags & inspect.CO_VARARGS:
            varargscount = 1
        else:
            varargscount = 0

        kwonlyargcount = region_code.co_kwonlyargcount

        if region_code.co_flags & inspect.CO_VARKEYWORDS:
            varkeywordscount = 1
        else:
            varkeywordscount = 0

        count = posonlyargcount + argcount + varargscount + kwonlyargcount + varkeywordscount
        all_arguments = deque(region_code.co_varnames[:count])

        posonlyargs = tuple(all_arguments.popleft() for _ in range(posonlyargcount))
        args = tuple(all_arguments.popleft() for _ in range(argcount))

        if varargscount:
            varargs = all_arguments.popleft()
        else:
            varargs = None

        kwonlyargs = tuple(all_arguments.popleft() for _ in range(kwonlyargcount))

        if varkeywordscount:
            varkeywords = all_arguments.popleft()
        else:
            varkeywords = None

        self.region_names_to_regions[region_name] = IRRegion(
            name=region_name,
            posonlyargs=posonlyargs,
            args=args,
            varargs=varargs,
            kwonlyargs=kwonlyargs,
            varkeywords=varkeywords
        )
        self.is_generator[region_name] = False

        region_bytecode_instructions = get_instructions(region_code)
        bytecode_basic_blocks, is_generator, cfg = partition_bytecode_basic_blocks(region_bytecode_instructions)

        # Translate bytecode instructions in each bytecode basic block
        ir_basic_block_builder = IRBasicBlockBuilder(
            parent_ir_region_builder=self,
            region_name_components=region_name_components,
            bytecode_basic_blocks=bytecode_basic_blocks,
        )

        stack = []  # type: List[Optional[IRValue]]
        for offset in bytecode_basic_blocks:
            stack = ir_basic_block_builder.build_basic_block(
                offset=offset,
                initial_stack=stack,
            )

        if stack:
            raise ImplementationError(
                'stack not empty after translating bytecode instructions in each bytecode basic block')

        self.region_names_to_regions[region_name].basic_blocks.extend(ir_basic_block_builder.ir_basic_blocks.values())
        self.is_generator[region_name] = is_generator


OPNAMES_WITH_KNOWN_CONTROL_FLOW_SEMANTICS = {
    'NOP',
    'COPY_FREE_VARS',
    'RESUME',
    'MAKE_CELL',
    'KW_NAMES',
    'EXTENDED_ARG',
    'POP_TOP',
    'END_FOR',
    'COPY',
    'SWAP',
    'PUSH_NULL',
    'UNARY_NEGATIVE',
    'UNARY_NOT',
    'UNARY_INVERT',
    'GET_ITER',
    'BINARY_OP',
    'BINARY_SUBSCR',
    'STORE_SUBSCR',
    'DELETE_SUBSCR',
    'BINARY_SLICE',
    'BUILD_SLICE',
    'BUILD_STRING',
    'BUILD_TUPLE',
    'BUILD_LIST',
    'BUILD_SET',
    'LIST_APPEND',
    'LIST_EXTEND',
    'SET_ADD',
    'SET_UPDATE',
    'YIELD_VALUE',
    'RETURN_VALUE',
    'RETURN_CONST',
    'LOAD_ASSERTION_ERROR',
    'LOAD_BUILD_CLASS',
    'LOAD_NAME',
    'LOAD_DEREF',
    'LOAD_FAST',
    'LOAD_FAST_AND_CLEAR',
    'LOAD_GLOBAL',
    'LOAD_CLOSURE',
    'LOAD_ATTR',
    'LOAD_CONST',
    'LOAD_SUPER_ATTR',
    'IMPORT_NAME',
    'IMPORT_FROM',
    'COMPARE_OP',
    'CALL',
    'CALL_FUNCTION_EX',
    'CALL_INTRINSIC_1',
    'MAKE_FUNCTION',
    'UNPACK_SEQUENCE',
    'FOR_ITER',
    'STORE_NAME',
    'STORE_FAST',
    'STORE_ATTR',
    'STORE_DEREF',
    'STORE_GLOBAL',
    'RAISE_VARARGS',
    'RERAISE',
    'JUMP_FORWARD',
    'JUMP_BACKWARD',
    'POP_JUMP_IF_TRUE',
    'POP_JUMP_IF_FALSE',
    'POP_JUMP_IF_NONE',
    'POP_JUMP_IF_NOT_NONE',
    'FORMAT_VALUE',
    'RETURN_GENERATOR',
    'IS_OP',
    'CONTAINS_OP',
    'BUILD_MAP',
    'BUILD_CONST_KEY_MAP',
    'BEFORE_WITH',
    'PUSH_EXC_INFO',
    'POP_EXCEPT',
    'WITH_EXCEPT_START',
    'CHECK_EXC_MATCH',
    'DELETE_NAME',
    'DELETE_FAST',
    'LOAD_FAST_CHECK',
    'DICT_UPDATE',
    'DICT_MERGE',
    'DELETE_ATTR',
    'MAP_ADD',
    'STORE_SLICE',
    'UNPACK_EX',
    'SETUP_ANNOTATIONS',
}

BRANCHING_OPNAMES = {
    'FOR_ITER',
    'POP_JUMP_IF_TRUE',
    'POP_JUMP_IF_FALSE',
    'POP_JUMP_IF_NONE',
    'POP_JUMP_IF_NOT_NONE',
}

SUSPENDING_OPNAMES = {
    'YIELD_VALUE',
}

JUMPING_OPNAMES = {
    'JUMP_FORWARD',
    'JUMP_BACKWARD',
}

RETURNING_OPNAMES = {
    'RETURN_VALUE',
    'RETURN_CONST',
}

RAISING_OPNAMES = {
    'RAISE_VARARGS',
    'RERAISE',
}

TERMINATING_OPNAMES = set().union(
    BRANCHING_OPNAMES,
    SUSPENDING_OPNAMES,
    JUMPING_OPNAMES,
    RETURNING_OPNAMES,
    RAISING_OPNAMES,
)


def partition_bytecode_basic_blocks(bytecode_instructions):
    # type: (Iterable[Instruction]) -> Tuple[Mapping[int, Sequence[Instruction]], bool, nx.DiGraph]
    """
    Partition bytecode instructions into bytecode basic blocks, keyed by their offsets.
    Each basic block is a list of bytecode instructions.
    """
    labels_to_bytecode_basic_blocks = OrderedDict()
    is_generator = False
    cfg = nx.DiGraph()

    bytecode_instruction_iterator = PutBackIterator(bytecode_instructions)

    # Initialize current_basic_block and associate with the offset of the peeked first instruction
    current_basic_block = []
    if bytecode_instruction_iterator.has_next():
        first_stack_instruction = next(bytecode_instruction_iterator)
        first_label = first_stack_instruction.offset
        labels_to_bytecode_basic_blocks[first_label] = current_basic_block
        bytecode_instruction_iterator.put_back(first_stack_instruction)

        # Iterate through all stack instructions
        for stack_instruction in bytecode_instruction_iterator:
            opname = stack_instruction.opname
            argval = stack_instruction.argval
            is_jump_target = stack_instruction.is_jump_target
            offset = stack_instruction.offset

            # Do we know the control flow semantics of the opname?
            if opname not in OPNAMES_WITH_KNOWN_CONTROL_FLOW_SEMANTICS:
                raise NotImplementedError('unknown control flow semantics: %s' % (stack_instruction,))

            # Are we at the start of a new basic block?
            # In that case, reset current_basic_block
            if is_jump_target:
                if offset not in labels_to_bytecode_basic_blocks:
                    labels_to_bytecode_basic_blocks[offset] = []

                current_basic_block = labels_to_bytecode_basic_blocks[offset]

            # Add the current instruction to current_basic_block
            # The next instruction should be the start of a new basic block
            # Peek the next instruction and check
            # Reset current_basic_block if not
            if opname in TERMINATING_OPNAMES:
                current_basic_block.append(stack_instruction)

                if bytecode_instruction_iterator.has_next():
                    next_instruction = next(bytecode_instruction_iterator)
                    if not next_instruction.is_jump_target:
                        labels_to_bytecode_basic_blocks[next_instruction.offset] = current_basic_block = []
                    bytecode_instruction_iterator.put_back(next_instruction)
            # RETURN_GENERATOR, POP_TOP are the first two instructions of a generator
            # In this case, skip them
            elif opname == 'RETURN_GENERATOR':
                is_generator = True

                if bytecode_instruction_iterator.has_next() and next(bytecode_instruction_iterator).opname == 'POP_TOP':
                    continue

                raise UnexpectedError('RETURN_GENERATOR not followed by POP_TOP')
            # consecutive CALL_INTRINSIC_1(3), RERAISE in a generator
            # In this case, skip them
            elif is_generator and opname == 'CALL_INTRINSIC_1' and argval == 3:
                if bytecode_instruction_iterator.has_next() and next(bytecode_instruction_iterator).opname == 'RERAISE':
                    continue

                raise UnexpectedError('CALL_INTRINSIC_1(3) not followed by RERAISE in a generator')
            else:
                current_basic_block.append(stack_instruction)

        # Analyze CFG
        label_iterator = PutBackIterator(labels_to_bytecode_basic_blocks)
        for label in label_iterator:
            cfg.add_node(label)

            basic_block = labels_to_bytecode_basic_blocks[label]
            if basic_block:
                last_bytecode_instruction_index = len(basic_block) - 1

                for i in range(last_bytecode_instruction_index):
                    if basic_block[i].opname in TERMINATING_OPNAMES:
                        raise UnexpectedError(
                            'basic block %d contains terminating opname in the middle' % (label,)
                        )

                last_bytecode_instruction = basic_block[last_bytecode_instruction_index]
                last_bytecode_instruction_opname = last_bytecode_instruction.opname
                last_bytecode_instruction_argval = last_bytecode_instruction.argval

                if last_bytecode_instruction_opname not in TERMINATING_OPNAMES:
                    if not label_iterator.has_next():
                        raise UnexpectedError(
                            'last basic block %d cannot terminate with a non-terminating opname' % (label,)
                        )
                    else:
                        next_label = next(label_iterator)
                        cfg.add_edge(label, next_label)
                        label_iterator.put_back(next_label)
                elif last_bytecode_instruction_opname in BRANCHING_OPNAMES:
                    if not label_iterator.has_next():
                        raise UnexpectedError(
                            'last basic block %d cannot terminate with a branching opname' % (label,)
                        )
                    else:
                        cfg.add_edge(label, last_bytecode_instruction_argval)
                        next_label = next(label_iterator)
                        cfg.add_edge(label, next_label)
                        label_iterator.put_back(next_label)
                elif last_bytecode_instruction_opname in SUSPENDING_OPNAMES:
                    if not label_iterator.has_next():
                        raise UnexpectedError(
                            'last basic block %d cannot terminate with a suspending opname' % (label,)
                        )
                    else:
                        next_label = next(label_iterator)
                        cfg.add_edge(label, next_label)
                        label_iterator.put_back(next_label)
                elif last_bytecode_instruction_opname in JUMPING_OPNAMES:
                    cfg.add_edge(label, last_bytecode_instruction_argval)
            else:
                if label_iterator.has_next():
                    next_label = next(label_iterator)
                    cfg.add_edge(label, next_label)
                    label_iterator.put_back(next_label)

        # Trim basic blocks based on CFG
        reachable_labels = nx.descendants(cfg, first_label) | {first_label}
        non_reachable_labels = set(labels_to_bytecode_basic_blocks) - reachable_labels
        for non_reachable_label in non_reachable_labels:
            del labels_to_bytecode_basic_blocks[non_reachable_label]
            cfg.remove_node(non_reachable_label)

    return labels_to_bytecode_basic_blocks, is_generator, cfg


class IRBasicBlockBuilder(object):
    def __init__(
            self,
            parent_ir_region_builder,  # type: IRRegionBuilder
            region_name_components,  # type: Sequence[str]
            bytecode_basic_blocks,  # type: Mapping[int, Sequence[Instruction]]
    ):
        self.parent_ir_region_builder = parent_ir_region_builder  # type: IRRegionBuilder
        self.region_name_components = region_name_components
        self.bytecode_basic_blocks = bytecode_basic_blocks  # type: Mapping[int, Sequence[Instruction]]

        self.ir_basic_blocks = OrderedDict()  # type: Dict[int, IRBasicBlock]
        # Initialize IR basic blocks for each bytecode basic block (to set up jump targets)
        for offset in bytecode_basic_blocks:
            self.ir_basic_blocks[offset] = IRBasicBlock()
        self.final_stacks = OrderedDict()  # type: Dict[int, List[Optional[IRValue]]]

    def build_basic_block(
            self,
            offset,  # type: int
            initial_stack,  # type: Sequence[Optional[IRValue]]
    ):
        # type: (...) -> List[Optional[IRValue]]
        """
        Converts bytecode instructions in a basic block to IR; recursively follows control-flow targets.
        Returns the final stack after this basic block.
        Updates self.ir_basic_blocks.
        """
        if offset in self.final_stacks:
            return self.final_stacks[offset]
        else:
            # A copy of initial_stack
            stack = list(initial_stack)

            # A reference
            ir_basic_block_instruction_list = self.ir_basic_blocks[offset].instructions

            # Used before CALL
            kw_names = None

            put_back_iterator = PutBackIterator(self.bytecode_basic_blocks[offset])

            for stack_instruction in put_back_iterator:
                opname = stack_instruction.opname
                argval = stack_instruction.argval
                arg = stack_instruction.arg

                # These instructions are NOPs in our analysis
                if opname in ('NOP', 'COPY_FREE_VARS', 'RESUME', 'MAKE_CELL', 'EXTENDED_ARG'):
                    pass
                elif opname == 'POP_TOP':
                    stack.pop()
                elif opname == 'END_FOR':
                    stack.pop()
                    stack.pop()
                elif opname == 'COPY':
                    stack.append(stack[-argval])
                elif opname == 'SWAP':
                    stack[-argval], stack[-1] = stack[-1], stack[-argval]
                elif opname == 'PUSH_NULL':
                    stack.append(IRConstant(literal_value=None))
                elif opname == 'KW_NAMES':
                    kw_names = argval
                elif opname == 'LOAD_CONST':
                    if isinstance(argval, (int, float, complex, str, bytes, NoneType, EllipsisType)):
                        instruction = IRConstant(argval)
                    elif isinstance(argval, (tuple, frozenset)):
                        def recursively_handle_tuple_or_frozenset_constant(tuple_or_frozenset_constant):
                            element_constant_list = []
                            for element in tuple_or_frozenset_constant:
                                if isinstance(element, (int, float, complex, str, bytes, NoneType, EllipsisType)):
                                    element_constant = IRConstant(element)
                                elif isinstance(element, (tuple, frozenset)):
                                    element_constant = recursively_handle_tuple_or_frozenset_constant(element)
                                else:
                                    raise UnexpectedError('unknown constant: %r' % (tuple_or_frozenset_constant,))

                                ir_basic_block_instruction_list.append(element_constant)
                                element_constant_list.append(element_constant)

                            if isinstance(tuple_or_frozenset_constant, tuple):
                                return IRBuildTuple(values=tuple(element_constant_list))
                            else:
                                return IRBuildSet(values=tuple(element_constant_list), frozen=True)

                        instruction = recursively_handle_tuple_or_frozenset_constant(argval)
                    elif isinstance(argval, CodeType):
                        # Compile the bytecode into a separate region
                        name_component = argval.co_name
                        self.parent_ir_region_builder.build_region(
                            region_name_components=self.region_name_components + [name_component],
                            region_code=argval
                        )
                        instruction = IRLoadRegion('.'.join(self.region_name_components + [name_component]))
                    else:
                        raise UnexpectedError('unknown constant: %r' % (argval,))

                    ir_basic_block_instruction_list.append(instruction)
                    stack.append(instruction)
                elif opname == 'LOAD_BUILD_CLASS':
                    instruction = IRLoadBuiltIn(builtin=__build_class__)
                    ir_basic_block_instruction_list.append(instruction)
                    stack.append(instruction)
                elif opname in (
                        'LOAD_NAME',
                        'LOAD_DEREF',
                        'LOAD_FAST',
                        'LOAD_FAST_CHECK',
                        'LOAD_FAST_AND_CLEAR',
                        'LOAD_GLOBAL',
                        'LOAD_CLOSURE'
                ):
                    instruction = IRLoad(argval)
                    ir_basic_block_instruction_list.append(instruction)

                    if opname == 'LOAD_GLOBAL' and arg & 0b1:
                        """Changed in version 3.11: If the low bit of namei is set, then a NULL is pushed to the stack before the global variable."""
                        stack.append(IRConstant(literal_value=None))
                    stack.append(instruction)
                elif opname == 'LOAD_SUPER_ATTR':
                    """
                    LOAD_SUPER_ATTR(namei)
                    This opcode implements super(), both in its zero-argument and two-argument forms (e.g. super().method(), super().attr and super(cls, self).method(), super(cls, self).attr).

                    It pops three values from the stack (from top of stack down): - self: the first argument to the current method - cls: the class within which the current method was defined - the global super

                    With respect to its argument, it works similarly to LOAD_ATTR, except that namei is shifted left by 2 bits instead of 1.

                    The low bit of namei signals to attempt a method load, as with LOAD_ATTR, which results in pushing NULL and the loaded method. When it is unset a single value is pushed to the stack.

                    The second-low bit of namei, if set, means that this was a two-argument call to super() (unset means zero-argument).
                    """
                    self_value = stack.pop()
                    cls_vale = stack.pop()
                    global_super_value = stack.pop()

                    instruction = IRLoadSuperAttr(cls_vale, self_value, argval)
                    ir_basic_block_instruction_list.append(instruction)

                    if arg & 0b1:
                        stack.append(IRConstant(literal_value=None))
                    stack.append(instruction)
                elif opname == 'IMPORT_NAME':
                    """
                    Imports the module co_names[namei].
                    STACK[-1] and STACK[-2] are popped and provide the fromlist and level arguments of __import__().
                    The module object is pushed onto the stack.

                    When the name variable is of the form package.module, normally, the top-level package (the name up till the first dot) is returned, not the module named by name.
                    However, when a non-empty fromlist argument is given, the module named by name is returned.
                    """
                    fromlist = stack.pop()
                    level = stack.pop()

                    if (
                            (not isinstance(level, IRConstant))
                            or (not isinstance(level.literal_value, int))
                    ):
                        raise UnexpectedError('cannot handle this IMPORT_NAME: %s' % (stack_instruction,))

                    if isinstance(fromlist, IRBuildTuple) and len(fromlist.values):
                        name = argval
                    elif isinstance(fromlist, IRConstant) and isinstance(fromlist.literal_value, NoneType):
                        name = argval.split('.')[0]
                    else:
                        raise UnexpectedError('cannot handle this IMPORT_NAME: %s' % (stack_instruction,))

                    instruction = IRImport(name, level.literal_value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'IMPORT_FROM':
                    module = stack[-1]
                    instruction = IRImportFrom(module, argval)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname in ('BINARY_OP', 'COMPARE_OP'):
                    """
                    Binary operations remove the top two items from the stack (STACK[-1] and STACK[-2]). They perform the operation, then put the result back on the stack.

                    In-place operations are like binary operations, but the operation is done in-place when STACK[-2] supports it, and the resulting STACK[-1] may be (but does not have to be) the original STACK[-2].

                    BINARY_OP(op)
                    Implements the binary and in-place operators (depending on the value of op):

                    rhs = STACK.pop()
                    lhs = STACK.pop()
                    """
                    rhs = stack.pop()
                    lhs = stack.pop()
                    if argval in ARGVAL_TO_IR_BINARY_OPERATORS:
                        op = ARGVAL_TO_IR_BINARY_OPERATORS[argval]
                        instruction = IRBinaryOp(lhs, op, rhs)
                        ir_basic_block_instruction_list.append(instruction)

                        stack.append(instruction)
                    elif argval in ARGVAL_TO_IR_INPLACE_BINARY_OPERATORS:
                        op = ARGVAL_TO_IR_INPLACE_BINARY_OPERATORS[argval]
                        instruction = IRInPlaceBinaryOp(lhs, op, rhs)
                        ir_basic_block_instruction_list.append(instruction)

                        stack.append(lhs)
                    else:
                        raise UnexpectedError('unknown BINARY_OP: %s' % (stack_instruction,))
                elif opname == 'BINARY_SUBSCR':
                    key = stack.pop()
                    container = stack.pop()

                    instruction = IRBinarySubscr(key, container)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'UNPACK_SEQUENCE':
                    sequence = stack.pop()

                    unpack_sequence_instruction = IRUnpackSequence(sequence=sequence, size=argval)
                    ir_basic_block_instruction_list.append(unpack_sequence_instruction)

                    for index in range(argval - 1, -1, -1):
                        index_constant = IRConstant(literal_value=index)
                        ir_basic_block_instruction_list.append(index_constant)

                        binary_subscr_instruction = IRBinarySubscr(
                            key=index_constant,
                            container=unpack_sequence_instruction
                        )
                        ir_basic_block_instruction_list.append(binary_subscr_instruction)

                        stack.append(binary_subscr_instruction)
                elif opname == 'UNPACK_EX':
                    sequence = stack.pop()

                    leading = argval & 0b11111111
                    trailing = (argval >> 8)

                    unpack_ex_instruction = IRUnpackEx(sequence=sequence, leading=leading, trailing=trailing)
                    ir_basic_block_instruction_list.append(unpack_ex_instruction)

                    for index in range(leading + trailing, -1, -1):
                        index_constant = IRConstant(literal_value=index)
                        ir_basic_block_instruction_list.append(index_constant)

                        binary_subscr_instruction = IRBinarySubscr(
                            key=index_constant,
                            container=unpack_ex_instruction
                        )
                        ir_basic_block_instruction_list.append(binary_subscr_instruction)

                        stack.append(binary_subscr_instruction)
                elif opname == 'LOAD_ATTR':
                    """
                    LOAD_ATTR(namei)

                    STACK[-1] is popped.

                    If the low bit of namei is not set, this pushes getattr(STACK[-1], co_names[namei>>1]).

                    If the low bit of namei is set, this will attempt to load a method named co_names[namei>>1] from the STACK[-1] object. This bytecode distinguishes two cases:

                    If STACK[-1] has a method with the correct name, the bytecode pushes the (unbound) method and STACK[-1]. STACK[-1] will be used as the first argument (self) by CALL when calling the unbound method.

                    unbound_meth | self | arg1 | ... | argN

                    Otherwise, NULL and the object returned by the attribute lookup are pushed.

                    NULL | meth | arg1 | ... | argN

                    This requires runtime inspection. Thus, we model with usual getattr (which returns a bound method object on an object with that method) and push NULL and the result of getattr onto the stack.
                    """
                    value = stack.pop()

                    if not (arg & 0b1):
                        pass
                    else:
                        stack.append(IRConstant(literal_value=None))

                    instruction = IRLoadAttr(value, argval)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname in ('BUILD_LIST', 'BUILD_TUPLE', 'BUILD_SET'):
                    if argval == 0:
                        values = tuple()
                    else:
                        value_deque = deque()
                        for _ in range(argval):
                            value_deque.appendleft(stack.pop())
                        values = tuple(value_deque)
                    if opname == 'BUILD_LIST':
                        instruction = IRBuildList(values)
                    elif opname == 'BUILD_TUPLE':
                        instruction = IRBuildTuple(values)
                    else:
                        instruction = IRBuildSet(values)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'MAKE_FUNCTION':
                    """
                    MAKE_FUNCTION(flags)
                    Pushes a new function object on the stack. From bottom to top, the consumed stack must consist of values if the argument carries a specified flag value

                    0x01 a tuple of default values for positional-only and positional-or-keyword parameters in positional order
                    0x02 a dictionary of keyword-only parameters’ default values
                    0x04 a tuple of strings containing parameters’ annotations
                    0x08 a tuple containing cells for free variables, making a closure
                    the code associated with the function (at STACK[-1])
                    """
                    code_associated_with_function = stack.pop()
                    if not isinstance(code_associated_with_function, IRLoadRegion):
                        raise UnexpectedError(
                            "code_associated_with_function is not IRLoadRegion: %s" % (
                                code_associated_with_function,
                            )
                        )

                    if arg & 0x08:
                        free_variable_cells = stack.pop()
                    else:
                        free_variable_cells = IRConstant(literal_value=None)
                        ir_basic_block_instruction_list.append(free_variable_cells)

                    annotations = OrderedDict()  # type: Dict[str, IRValue]
                    if arg & 0x04:
                        annotation_tuple = stack.pop()
                        if (
                                (not isinstance(annotation_tuple, IRBuildTuple))
                                or (len(annotation_tuple.values) % 2)
                        ):
                            raise UnexpectedError("unknown annotation tuple layout")

                        len_parameter_annotations = len(annotation_tuple.values) // 2
                        for i in range(len_parameter_annotations):
                            parameter = annotation_tuple.values[2 * i]
                            annotation = annotation_tuple.values[2 * i + 1]

                            if (
                                    (not isinstance(parameter, IRConstant))
                                    or (not isinstance(parameter.literal_value, str))
                                    or (not isinstance(annotation, IRValue))
                            ):
                                raise UnexpectedError("unknown annotation tuple layout")

                            annotations[parameter.literal_value] = annotation

                    if arg & 0x02:
                        keyword_only_parameter_default_values = stack.pop()
                    else:
                        keyword_only_parameter_default_values = IRConstant(literal_value=None)
                        ir_basic_block_instruction_list.append(keyword_only_parameter_default_values)

                    if arg & 0x01:
                        parameter_default_values = stack.pop()
                    else:
                        parameter_default_values = IRConstant(literal_value=None)
                        ir_basic_block_instruction_list.append(parameter_default_values)

                    instruction = IRMakeFunction(
                        loaded_region=code_associated_with_function,
                        parameter_default_values=parameter_default_values,
                        keyword_only_parameter_default_values=keyword_only_parameter_default_values,
                        annotations=annotations,
                        free_variable_cells=free_variable_cells,
                    )
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'CALL':
                    """
                    CALL(argc)
                    Calls a callable object with the number of arguments specified by argc, including the named arguments specified by the preceding KW_NAMES, if any. On the stack are (in ascending order), either:

                    NULL
                    The callable
                    The positional arguments
                    The named arguments

                    or:

                    The callable
                    self
                    The remaining positional arguments
                    The named arguments
                    argc is the total of the positional and named arguments, excluding self when a NULL is not present.

                    CALL pops all arguments and the callable object off the stack, calls the callable object with those arguments, and pushes the return value returned by the callable object.
                    """
                    null_or_callable = stack[-(argval + 2)]
                    callable_or_self = stack[-(argval + 1)]
                    if argval:
                        arguments_and_keywords = tuple(stack[-argval:])
                    else:
                        arguments_and_keywords = tuple()

                    for _ in range(argval + 2):
                        stack.pop()

                    if kw_names:
                        number_of_keywords = len(kw_names)
                        arguments = arguments_and_keywords[:-number_of_keywords]
                        keywords = dict(zip(kw_names, arguments_and_keywords[-number_of_keywords:]))

                        # Clear kw_names
                        kw_names = None
                    else:
                        arguments = arguments_and_keywords
                        keywords = {}

                    if isinstance(null_or_callable, IRConstant) and null_or_callable.literal_value is None:
                        callable = callable_or_self
                    else:
                        callable = null_or_callable
                        arguments = (callable_or_self,) + arguments

                    instruction = IRCall(callable, arguments, keywords)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'GET_ITER':
                    value = stack.pop()
                    instruction = IRGetIter(value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'FOR_ITER':
                    # before: [iter]; after: [iter, iter()] *or* [] (and jump over END_FOR.)
                    iterator = stack[-1]
                    target = self.ir_basic_blocks[argval]
                    instruction = IRForIter(iterator, target)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'POP_JUMP_IF_TRUE':
                    condition = stack.pop()
                    if_true = self.ir_basic_blocks[argval]

                    instruction = IRBranch(condition, if_true)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'POP_JUMP_IF_FALSE':
                    condition = stack.pop()
                    if_false = self.ir_basic_blocks[argval]

                    not_condition = IRUnaryOp(IRUnaryOperator.NOT, condition)
                    ir_basic_block_instruction_list.append(not_condition)

                    instruction = IRBranch(not_condition, if_false)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'POP_JUMP_IF_NONE':
                    value = stack.pop()
                    if_not_none = self.ir_basic_blocks[argval]

                    none = IRConstant(literal_value=None)
                    ir_basic_block_instruction_list.append(none)

                    value_is_none = IRBinaryOp(lhs=value, op=IRBinaryOperator.IS, rhs=none)
                    ir_basic_block_instruction_list.append(value_is_none)

                    instruction = IRBranch(value_is_none, if_not_none)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'POP_JUMP_IF_NOT_NONE':
                    value = stack.pop()
                    if_not_none = self.ir_basic_blocks[argval]

                    none = IRConstant(literal_value=None)
                    ir_basic_block_instruction_list.append(none)

                    value_is_not_none = IRBinaryOp(lhs=value, op=IRBinaryOperator.IS_NOT, rhs=none)
                    ir_basic_block_instruction_list.append(value_is_not_none)

                    instruction = IRBranch(value_is_not_none, if_not_none)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'JUMP_FORWARD':
                    target = self.ir_basic_blocks[argval]
                    instruction = IRJump(target)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname == 'JUMP_BACKWARD':
                    target = self.ir_basic_blocks[argval]
                    instruction = IRJump(target)
                    ir_basic_block_instruction_list.append(instruction)

                    # Branch out and translate stack instructions in target
                    self.build_basic_block(
                        offset=argval,
                        initial_stack=stack,
                    )
                elif opname in ('STORE_NAME', 'STORE_FAST', 'STORE_DEREF', 'STORE_GLOBAL'):
                    value = stack.pop()
                    instruction = IRStore(name=argval, value=value, force_global=opname == 'STORE_GLOBAL')
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'STORE_SUBSCR':
                    key = stack.pop()
                    container = stack.pop()
                    value = stack.pop()

                    instruction = IRStoreSubscr(key, container, value)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'DELETE_SUBSCR':
                    key = stack.pop()
                    container = stack.pop()

                    instruction = IRDeleteSubscr(key=key, container=container)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'STORE_ATTR':
                    obj = stack.pop()
                    value = stack.pop()

                    instruction = IRStoreAttr(obj, argval, value)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'RETURN_CONST':
                    constant_instruction = IRConstant(argval)
                    ir_basic_block_instruction_list.append(constant_instruction)
                    return_instruction = IRReturn(constant_instruction)
                    ir_basic_block_instruction_list.append(return_instruction)

                    break
                elif opname == 'RETURN_VALUE':
                    value = stack.pop()
                    instruction = IRReturn(value)
                    ir_basic_block_instruction_list.append(instruction)

                    break
                elif opname == 'RAISE_VARARGS':
                    if arg == 1:
                        exception_instance_or_type = stack.pop()
                        instruction = IRRaise(exception_instance_or_type)
                        ir_basic_block_instruction_list.append(instruction)

                        break
                    else:
                        raise NotImplementedError(stack_instruction)
                elif opname == 'BINARY_SLICE':
                    end = stack.pop()
                    start = stack.pop()
                    container = stack.pop()

                    step = IRConstant(literal_value=None)
                    ir_basic_block_instruction_list.append(step)

                    build_slice_instruction = IRBuildSlice(start=start, stop=end, step=step)
                    ir_basic_block_instruction_list.append(build_slice_instruction)

                    binary_subscr_instruction = IRBinarySubscr(key=build_slice_instruction, container=container)
                    ir_basic_block_instruction_list.append(binary_subscr_instruction)
                    stack.append(binary_subscr_instruction)
                elif opname == 'LIST_EXTEND':
                    seq = stack.pop()
                    list_to_be_extended = stack[-argval]

                    extend_method = IRLoadAttr(value=list_to_be_extended, attribute='extend')
                    ir_basic_block_instruction_list.append(extend_method)

                    call_instruction = IRCall(callee=extend_method, arguments=(seq,), keywords={})
                    ir_basic_block_instruction_list.append(call_instruction)
                elif opname == 'LIST_APPEND':
                    item = stack.pop()
                    list_to_be_appended = stack[-argval]

                    append_method = IRLoadAttr(value=list_to_be_appended, attribute='append')
                    ir_basic_block_instruction_list.append(append_method)

                    call_instruction = IRCall(callee=append_method, arguments=(item,), keywords={})
                    ir_basic_block_instruction_list.append(call_instruction)
                elif opname == 'SET_ADD':
                    item = stack.pop()
                    set_to_be_updated = stack[-argval]

                    add_method = IRLoadAttr(value=set_to_be_updated, attribute='add')
                    ir_basic_block_instruction_list.append(add_method)

                    call_instruction = IRCall(callee=add_method, arguments=(item,), keywords={})
                    ir_basic_block_instruction_list.append(call_instruction)
                elif opname == 'SET_UPDATE':
                    seq = stack.pop()
                    set_to_be_updated = stack[-argval]

                    update_method = IRLoadAttr(value=set_to_be_updated, attribute='update')
                    ir_basic_block_instruction_list.append(update_method)

                    call_instruction = IRCall(callee=update_method, arguments=(seq,), keywords={})
                    ir_basic_block_instruction_list.append(call_instruction)
                elif opname == 'CALL_INTRINSIC_1':
                    # Passes STACK[-1] as the argument and sets STACK[-1] to the result.

                    # INTRINSIC_IMPORT_STAR
                    if argval == 2:
                        module = stack[-1]
                        instruction = IRImportStar(module=module)
                        ir_basic_block_instruction_list.append(instruction)
                    # INTRINSIC_LIST_TO_TUPLE
                    elif argval == 6:
                        argument = stack.pop()

                        load_tuple = IRLoadBuiltIn(builtin=tuple)
                        ir_basic_block_instruction_list.append(load_tuple)

                        result = IRCall(callee=load_tuple, arguments=(argument,), keywords={})
                        ir_basic_block_instruction_list.append(result)

                        stack.append(result)
                    else:
                        raise NotImplementedError(stack_instruction)
                elif opname == 'CALL_FUNCTION_EX':
                    """
                    Calls a callable object with variable set of positional and keyword arguments.
                    If the lowest bit of flags is set, the top of the stack contains a mapping object containing additional keyword arguments.
                    Before the callable is called, the mapping object and iterable object are each unpacked and their contents passed in as keyword and positional arguments respectively.
                    CALL_FUNCTION_EX pops all arguments and the callable object off the stack, calls the callable object with those arguments, and pushes the return value returned by the callable object.
                    """
                    if argval & 0b1:
                        kwargs = stack.pop()
                    else:
                        kwargs = IRConstant(None)
                        ir_basic_block_instruction_list.append(kwargs)

                    args = stack.pop()
                    callable = stack.pop()

                    # In the C source code:
                    # STACK_SHRINK(((oparg & 1) ? 1 : 0));
                    # STACK_SHRINK(2);
                    # stack_pointer[-1] = result;
                    # Overwriting top-of-stack with result = popping top-of-stack and pushing result
                    placeholder = stack.pop()
                    if (
                            not isinstance(placeholder, IRConstant)
                            or placeholder.literal_value is not None
                    ):
                        raise UnexpectedError('unknown placeholder used for CALL_FUNCTION_EX stack instruction')

                    result = IRCallFunctionEx(callee=callable, args=args, kwargs=kwargs)
                    ir_basic_block_instruction_list.append(result)

                    stack.append(result)
                elif opname == 'FORMAT_VALUE':
                    if (arg & 0x04) == 0x04:
                        fmt_spec = stack.pop()
                    else:
                        fmt_spec = IRConstant(None)
                        ir_basic_block_instruction_list.append(fmt_spec)

                    value = stack.pop()

                    if (arg & 0x03) == 0x01:
                        # call str() on value before formatting it
                        loaded_str = IRLoadBuiltIn(builtin=str)
                        ir_basic_block_instruction_list.append(loaded_str)

                        modified_value = IRCall(callee=loaded_str, arguments=(value,), keywords={})
                        ir_basic_block_instruction_list.append(modified_value)
                    elif (arg & 0x03) == 0x02:
                        # call repr() on value before formatting it
                        loaded_repr = IRLoadBuiltIn(builtin=repr)
                        ir_basic_block_instruction_list.append(loaded_repr)

                        modified_value = IRCall(callee=loaded_repr, arguments=(value,), keywords={})
                        ir_basic_block_instruction_list.append(modified_value)
                    elif (arg & 0x03) == 0x03:
                        # call ascii() on value before formatting it
                        loaded_ascii = IRLoadBuiltIn(builtin=ascii)
                        ir_basic_block_instruction_list.append(loaded_ascii)

                        modified_value = IRCall(callee=loaded_ascii, arguments=(value,), keywords={})
                        ir_basic_block_instruction_list.append(modified_value)
                    else:
                        modified_value = value

                    instruction = IRFormatString(value=modified_value, fmt_spec=fmt_spec)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'BUILD_STRING':
                    strings = tuple(stack[-argval:])
                    for _ in range(argval):
                        stack.pop()

                    instruction = IRConcatenateStrings(strings=strings)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'UNARY_NEGATIVE':
                    value = stack.pop()
                    instruction = IRUnaryOp(op=IRUnaryOperator.UNARY_SUB, operand=value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'UNARY_NOT':
                    value = stack.pop()
                    instruction = IRUnaryOp(op=IRUnaryOperator.NOT, operand=value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'UNARY_INVERT':
                    value = stack.pop()
                    instruction = IRUnaryOp(op=IRUnaryOperator.INVERT, operand=value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'BUILD_SLICE':
                    if argval == 3:
                        step = stack.pop()
                    else:
                        step = IRConstant(literal_value=None)
                        ir_basic_block_instruction_list.append(step)
                    end = stack.pop()
                    start = stack.pop()

                    instruction = IRBuildSlice(start=start, stop=end, step=step)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'LOAD_ASSERTION_ERROR':
                    instruction = IRLoadBuiltIn(builtin=AssertionError)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'YIELD_VALUE':
                    value = stack.pop()
                    instruction = IRYield(value=value)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'IS_OP':
                    if argval:
                        op = IRBinaryOperator.IS_NOT
                    else:
                        op = IRBinaryOperator.IS
                    rhs = stack.pop()
                    lhs = stack.pop()
                    instruction = IRBinaryOp(lhs=lhs, op=op, rhs=rhs)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'CONTAINS_OP':
                    if argval:
                        op = IRBinaryOperator.NOT_IN
                    else:
                        op = IRBinaryOperator.IN
                    rhs = stack.pop()
                    lhs = stack.pop()
                    instruction = IRBinaryOp(lhs=lhs, op=op, rhs=rhs)
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'BUILD_MAP':
                    # Pops 2*argval items: key1, value1, key2, value2, ..., keyN, valueN.
                    # Dictionary holds count entries:  {..., STACK[-4]: STACK[-3], STACK[-2]: STACK[-1]}
                    key_deque = deque()
                    value_deque = deque()

                    for _ in range(argval):
                        value = stack.pop()
                        key = stack.pop()

                        key_deque.appendleft(key)
                        value_deque.appendleft(value)

                    instruction = IRBuildMap(keys=tuple(key_deque), values=tuple(value_deque))
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'BUILD_CONST_KEY_MAP':
                    key_tuple = stack.pop()
                    value_deque = deque()

                    if not isinstance(key_tuple, IRBuildTuple):
                        raise UnexpectedError('key tuple is not a tuple literal in BUILD_CONST_KEY_MAP')

                    for _ in range(argval):
                        value = stack.pop()
                        value_deque.appendleft(value)

                    instruction = IRBuildMap(keys=key_tuple.values, values=tuple(value_deque))
                    ir_basic_block_instruction_list.append(instruction)

                    stack.append(instruction)
                elif opname == 'BEFORE_WITH':
                    manager = stack.pop()

                    __exit__ = IRLoadAttr(value=manager, attribute='__exit__')
                    ir_basic_block_instruction_list.append(__exit__)
                    stack.append(__exit__)

                    __enter__ = IRLoadAttr(value=manager, attribute='__exit__')
                    ir_basic_block_instruction_list.append(__enter__)

                    result = IRCall(callee=__enter__, arguments=(), keywords={})
                    ir_basic_block_instruction_list.append(result)

                    stack.append(result)
                elif opname == 'PUSH_EXC_INFO':
                    new_exc = stack.pop()

                    prev_exec = IRGetException()
                    ir_basic_block_instruction_list.append(prev_exec)

                    stack.append(prev_exec)
                    stack.append(new_exc)
                elif opname in ('DICT_UPDATE', 'DICT_MERGE'):
                    mapping = stack.pop()

                    dictionary = stack[-argval]

                    update_method = IRLoadAttr(value=dictionary, attribute='update')
                    ir_basic_block_instruction_list.append(update_method)

                    call_instruction = IRCall(callee=update_method, arguments=(mapping,), keywords={})
                    ir_basic_block_instruction_list.append(call_instruction)
                elif opname in ('DELETE_NAME', 'DELETE_FAST'):
                    instruction = IRDelete(name=argval)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'DELETE_ATTR':
                    obj = stack.pop()
                    instruction = IRDeleteAttr(obj=obj, attribute=argval)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'MAP_ADD':
                    value = stack.pop()
                    key = stack.pop()
                    dictionary = stack[-argval]

                    instruction = IRStoreSubscr(key=key, container=dictionary, value=value)
                    ir_basic_block_instruction_list.append(instruction)
                elif opname == 'STORE_SLICE':
                    end = stack.pop()
                    start = stack.pop()
                    container = stack.pop()
                    values = stack.pop()

                    step = IRConstant(literal_value=None)
                    ir_basic_block_instruction_list.append(step)

                    build_slice_instruction = IRBuildSlice(start=start, stop=end, step=step)
                    ir_basic_block_instruction_list.append(build_slice_instruction)

                    store_subscr_instruction = IRStoreSubscr(key=build_slice_instruction, container=container,
                                                             value=values)
                    ir_basic_block_instruction_list.append(store_subscr_instruction)
                elif opname == 'SETUP_ANNOTATIONS':
                    instruction = IRSetupAnnotations()
                    ir_basic_block_instruction_list.append(instruction)
                else:
                    raise NotImplementedError(stack_instruction)

            self.final_stacks[offset] = stack
            return stack


def dump(
        region_names_to_regions,  # type: Dict[str, IRRegion]
):
    for region_name, region in region_names_to_regions.items():
        print(
            'Region name=%r posonlyargs=%r args=%r varargs=%r kwonlyargs=%r varkeywords=%r:' % (
                region_name,
                region.posonlyargs,
                region.args,
                region.varargs,
                region.kwonlyargs,
                region.varkeywords
            )
        )

        values_to_indices = {}

        def get_index(value):
            if value not in values_to_indices:
                values_to_indices[value] = len(values_to_indices)
            return values_to_indices[value]

        for basic_block in region.basic_blocks:
            print('Basic block $%d:' % (get_index(basic_block),))

            for instruction in basic_block.instructions:
                if isinstance(instruction, IRValue):
                    index = get_index(instruction)

                    if isinstance(instruction, IRConstant):
                        print('$%d = constant %r' % (index, instruction.literal_value))
                    elif isinstance(instruction, IRLoad):
                        print('$%d = load %r' % (index, instruction.name))
                    elif isinstance(instruction, IRLoadRegion):
                        print('$%d = load_region %r' % (index, instruction.name))
                    elif isinstance(instruction, IRMakeFunction):
                        print(
                            '$%d = make_function $%d $%d $%d $%d %s' % (
                                index,
                                get_index(instruction.loaded_region),
                                get_index(instruction.parameter_default_values),
                                get_index(instruction.keyword_only_parameter_default_values),
                                get_index(instruction.free_variable_cells),
                                ' '.join(
                                    '%s=$%d' % (parameter, get_index(annotation))
                                    for parameter, annotation in instruction.annotations.items()
                                ),

                            )
                        )
                    elif isinstance(instruction, IRImport):
                        print('$%d = import %r %d' % (index, instruction.name, instruction.level))
                    elif isinstance(instruction, IRImportFrom):
                        print('$%d = import_from $%d %r' % (index, get_index(instruction.module), instruction.name))
                    elif isinstance(instruction, IRLoadAttr):
                        print('$%d = load_attr $%d %r' % (index, get_index(instruction.value), instruction.attribute))
                    elif isinstance(instruction, IRLoadSuperAttr):
                        print(
                            '$%d = load_super_attr $%d $%d %r' % (
                                index,
                                get_index(instruction.cls_value),
                                get_index(instruction.self_value),
                                instruction.attribute
                            )
                        )
                    elif isinstance(instruction, IRLoadBuiltIn):
                        print('$%d = load_builtin %r' % (index, instruction.builtin.__name__))
                    elif isinstance(instruction, IRUnaryOp):
                        print(
                            '$%d = %s $%d' % (
                                index,
                                instruction.op.value,
                                get_index(instruction.operand)
                            )
                        )
                    elif isinstance(instruction, IRBinaryOp):
                        print(
                            '$%d = $%d %s $%d' % (
                                index,
                                get_index(instruction.lhs),
                                instruction.op.value,
                                get_index(instruction.rhs)
                            )
                        )
                    elif isinstance(instruction, IRBinarySubscr):
                        print('$%d = $%d[$%d]' % (index, get_index(instruction.container), get_index(instruction.key)))
                    elif isinstance(instruction, IRUnpackSequence):
                        print(
                            '$%d = unpack_sequence $%d %d' % (
                                index,
                                get_index(instruction.sequence),
                                instruction.size
                            )
                        )
                    elif isinstance(instruction, IRUnpackEx):
                        print(
                            '$%d = unpack_ex $%d %d %d' % (
                                index,
                                get_index(instruction.sequence),
                                instruction.leading,
                                instruction.trailing
                            )
                        )
                    elif isinstance(instruction, IRBuildList):
                        print(
                            '$%d = build_list %s' % (
                                index,
                                ' '.join(
                                    '$%d' % (get_index(value),)
                                    for value in instruction.values
                                )
                            )
                        )
                    elif isinstance(instruction, IRBuildTuple):
                        print(
                            '$%d = build_tuple %s' % (
                                index,
                                ' '.join(
                                    '$%d' % (get_index(value),)
                                    for value in instruction.values
                                )
                            )
                        )
                    elif isinstance(instruction, IRBuildSet):
                        print(
                            '$%d = build_set %s frozen=%r' % (
                                index,
                                ' '.join(
                                    '$%d' % (get_index(value),) for value in instruction.values
                                ),
                                instruction.frozen
                            )
                        )
                    elif isinstance(instruction, IRBuildSlice):
                        print(
                            '$%d = build_slice $%d $%d $%d' % (
                                index,
                                get_index(instruction.start),
                                get_index(instruction.stop),
                                get_index(instruction.step)
                            )
                        )
                    elif isinstance(instruction, IRCall):
                        print(
                            '$%d = call $%d %s' % (
                                index,
                                get_index(instruction.callee),
                                ' '.join(
                                    chain(
                                        ('$%d' % (get_index(value),) for value in instruction.arguments),
                                        (
                                            '%s=$%d' % (keyword_name, get_index(keyword))
                                            for keyword_name, keyword in instruction.keywords.items()
                                        ),
                                    )
                                )
                            )
                        )
                    elif isinstance(instruction, IRCallFunctionEx):
                        print(
                            '$%d = call_function_ex $%d $%d $%d' % (
                                index,
                                get_index(instruction.callee),
                                get_index(instruction.args),
                                get_index(instruction.kwargs),
                            )
                        )
                    elif isinstance(instruction, IRGetIter):
                        print('$%d = get_iter $%d' % (index, get_index(instruction.value)))
                    elif isinstance(instruction, IRForIter):
                        print(
                            '$%d = for_iter $%d $%d' % (
                                index,
                                get_index(instruction.iterator),
                                get_index(instruction.target)
                            )
                        )
                    elif isinstance(instruction, IRFormatString):
                        print(
                            '$%d = format_string $%d $%d' % (
                                index,
                                get_index(instruction.value),
                                get_index(instruction.fmt_spec)
                            )
                        )
                    elif isinstance(instruction, IRConcatenateStrings):
                        print(
                            '$%d = concatenate_strings %s' % (
                                index,
                                ' '.join(
                                    '$%d' % (get_index(string),)
                                    for string in instruction.strings
                                )
                            )
                        )
                    elif isinstance(instruction, IRYield):
                        print('$%d = yield $%d' % (index, get_index(instruction.value)))
                    elif isinstance(instruction, IRBuildMap):
                        print(
                            '$%d = build_map %s' % (
                                index,
                                ' '.join(
                                    '$%d:$%d' % (get_index(key), get_index(value))
                                    for key, value in zip(instruction.keys, instruction.values)
                                )
                            )
                        )
                    else:
                        raise NotImplementedError(instruction)
                else:
                    if isinstance(instruction, IRInPlaceBinaryOp):
                        print(
                            '$%d %s= $%d' % (
                                get_index(instruction.lhs),
                                instruction.op.value,
                                get_index(instruction.rhs)
                            )
                        )
                    elif isinstance(instruction, IRStore):
                        print(
                            'store $%d %r force_global=%r' % (
                                get_index(instruction.value),
                                instruction.name,
                                instruction.force_global
                            )
                        )
                    elif isinstance(instruction, IRStoreSubscr):
                        print(
                            '$%d[$%d] = $%d' % (
                                get_index(instruction.container),
                                get_index(instruction.key),
                                get_index(instruction.value)
                            )
                        )
                    elif isinstance(instruction, IRDeleteSubscr):
                        print('del $%d[$%d]' % (get_index(instruction.container), get_index(instruction.key)))
                    elif isinstance(instruction, IRStoreAttr):
                        print(
                            'store_attr $%d %r $%d' % (
                                get_index(instruction.obj),
                                instruction.attribute,
                                get_index(instruction.value)
                            )
                        )
                    elif isinstance(instruction, IRDelete):
                        print('del %r' % (instruction.name,))
                    elif isinstance(instruction, IRBranch):
                        print('branch $%d $%d' % (get_index(instruction.condition), get_index(instruction.if_true)))
                    elif isinstance(instruction, IRJump):
                        print('jump $%d' % (get_index(instruction.target),))
                    elif isinstance(instruction, IRReturn):
                        print('return $%d' % (get_index(instruction.value),))
                    elif isinstance(instruction, IRRaise):
                        print('raise $%d' % (get_index(instruction.exception_instance_or_type),))
                    elif isinstance(instruction, IRImportStar):
                        print('import_star $%d' % (get_index(instruction.module),))
                    elif isinstance(instruction, IRDeleteAttr):
                        print('delete_attr $%d %r' % (get_index(instruction.obj), instruction.attribute))
                    elif isinstance(instruction, IRSetupAnnotations):
                        print('setup_annotations')
                    else:
                        raise NotImplementedError(instruction)

            print()

        print()
