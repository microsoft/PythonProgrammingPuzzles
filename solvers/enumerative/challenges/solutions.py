from typing import List, Set, Dict, Callable, Tuple
import logging
from challenges import extract_constants

from tython import Program, TastNode, _RULES_BY_KIND, RULES, Rule, str2name
from tython.rules import DEF_RULE

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generatable_answer(q: Program, a: Program):
    def get_src(node):
        return Program(node).src(safe=False, simplify=False)

    consts = extract_constants(q)
    const_srcs = {k: [get_src(c) for c in consts[k]] for k in consts}

    def helper(anode):
        if anode.rule.name == "COPY":
            return get_src(anode.children[0]) in const_srcs[anode.rule.nt]
        return anode.rule.name != "literal" and all(helper(n) for n in anode.children)

    return helper(a.tree)


def verify_solutions(challenges):
    '''
    Verify all provided solutions to the given challenges and store the parsed solution
    program in the challenge object.
    '''
    successes = 0
    all_correct = True
    for ch in challenges:
        verified_sols = []
        f = ch.prog.run(max_ticks=ch.max_ticks)['sat']
        args_node = ch.prog.tree.children[0].children[1].children[-1]
        for sol_p in ch.gold_solutions:
            s = sol_p.string
            if s == '':
                continue
            # Verify solution is correct.
            if True:
                assert s.startswith('def ')
                try:
                    sol_prog = Program(s)
                except Exception as e:
                    logger.error(f"Exception parsing solution for {ch.name} '{s}': {e}")
                    continue

                # Inject the value assignments for the variables to the call of the sol func.
                p_body = sol_prog.tree.children[0].children[2]
                sol_prog.tree.children[0].children[1].children = args_node.children
                sol_prog = Program(TastNode(DEF_RULE, [str2name("sol"), args_node, p_body]))

                a_safe = sol_prog.src(simplify=False)
                x = sol_prog.run(max_ticks=ch.max_ticks)["sol"]()
                ch.prog.reset_clock()

                v = f(x)
                assert isinstance(v, bool)

                if not generatable_answer(ch.prog, sol_prog):
                    logger.error(f'Challenge "{ch.name}" cannot be used to automatically generate solution "{s}"')

                # TODO
                # if type(y) != ch.type:
                #    print(f'Challenge "{ch.name}" has wrong solution type: "{type(y)}"')
                #    all_correct = False
                if v is not True:  # checks both False and None
                    logger.error(f'Challenge "{ch.name}" not satisfied by solution "{s}"')
                else:
                    sol_p.prog = sol_prog
                    verified_sols.append(sol_p)
                    successes += 1

        ch.gold_solutions = verified_sols

    logger.info(
        f"Tython confirmed {successes:,} solutions to {len(challenges)} challenges."
    )
    return
