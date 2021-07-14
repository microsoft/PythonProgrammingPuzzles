from math import cos, sin, pi, log, exp, log2, inf

import ast
import itertools
import logging

import utils

from . import program
from . import rules
from . import nonterminals as nt


logger = logging.getLogger(__name__)


class ParseFailure(Exception):
    pass


def cast(node, t):
    if node.nt == t:
        return node
    return make(f'cast:{t}', node)


def to_digits(n):
    digits = None
    for i in str(n)[::-1]:
        n2 = make(i, ())
        digits = make("Digits", (n2, digits) if digits else (n2,))
    return digits


def make(rule_name, *children):  # returns rule number, (casted) children
    children = tuple(utils.flatten(children))

    kids = tuple(k.nt for k in children)
    idx = (rule_name, *kids)
    if idx in rules._rule_lookup:
        rule = rules._rule_lookup[idx]  # easy case, no casting
    else:
        try:
            # print()
            # print(rule_name, [Program(c).src(simplify=False) for c in children])
            # print("kids", kids)
            # print(list(itertools.product(*(rules._cast_costs[k] for k in kids))))
            search_space_size = 1
            for k in kids:
                search_space_size *= len(rules._cast_costs[k])
            if search_space_size > 10**6:
                raise ParseFailure("Too big a product search space for casting")
            best_pairs = min(
                (pairs for pairs in itertools.product(*(rules._cast_costs[k] for k in kids))
                 if (rule_name, *(t for _cost, t in pairs)) in rules._rule_lookup),
                key=lambda pairs: sum(cost for cost, _nt in pairs)
            )
            rule = rules._rule_lookup[(rule_name, *(t for _cost, t in best_pairs))]
            children = [cast(child, t) for (child, (_cost, t)) in zip(children, best_pairs)]

        except ValueError:
            raise ParseFailure(f"No rules with name='{rule_name}' kids={idx[1:]}")

    return program.TastNode(rules.rules[rule.index], children)


def str2name(s):
    ans = None
    for c in s:
        if c.lower() in rules._alpha_chars:
            n = make(f"'{c.lower()}'", [])
            if c != c.lower():
                n = make(f"upper_alpha_char", n)
            ans = make('name', [ans, n] if ans else [n])
        else:
            assert c in rules._digits, f"Illegal character in NAME `{s}`"
            assert ans, "NAME cannot start with digit"
            ans = make('name', ans, make(c))
    return ans


def from_ast(source: ast.AST, var_nts):  # Returns TastNode, not optimized for speed

    var_nts_lists = {v: [k] for v, k in
                     (var_nts or {}).items()}  # (imperfect) queue of stock by scope for variables

    def get_return_nts(node):
        if node.rule.name == "def":
            return []  # don't get confused with the return stock inside a function
        if node.rule.name == "return":
            return [node.children[0].nt]
        return [k for n in node.children for k in get_return_nts(n)]

    def make_flex(node):
        return node if node.nt == nt.Z else make('cast:Z', [node])

    # def make_statement(node):
    #     return node if node.nt == STMT else make("statement", [make_flex(node)])

    def body_helper(body: list):
        children = [helper(body[0])]
        if len(body) > 1:
            children.append(body_helper(body[1:]))
        return make('body', children)

    def str2chars(s):
        if not s:
            return make("Chars", [])
        c = s[0]
        if c.lower() in rules._alpha_chars:
            n = make(f"'{c.lower()}'")
            if c != c.lower():
                n = make(f"upper_alpha_char", n)
        elif c in rules._digits or c in rules._other_chars:
            n = make(c)
        else:
            n = make("other_char", to_digits(ord(c)))
        return make("Chars", n, str2chars(s[1:]))

    def resolve_names(node, t):  # x = 12   or (x, y) = [12, 'cat'] or [_ for x, y in zip(range(3), 'cat')]
        if isinstance(node, ast.Name):
            name = node.id
            var_nts_lists.setdefault(name, []).append(t)
            if (f"var:{t}", nt.NAME) not in rules._rule_lookup:
                t = nt.Z
            return [name], make(f"var:{t}", str2name(name))  # easy case like: x = 12
        if (isinstance(node, (ast.Tuple, ast.List)) and
                all(isinstance(e, ast.Name) for e in node.elts) and
                len(node.elts) > 1 and
                (len(t.kids) == len(node.elts) or t.isa(nt.LIST))):
            names = [e.id for e in node.elts]
            kids = list(t.kids) if t.isa(nt.TUPLE) else [t.kid] * len(node.elts)
            children = []
            for n, k in zip(names, kids):
                var_nts_lists.setdefault(n, []).append(k)
                children.append(str2name(n))

            return names, make(f'var:TUPLE{kids}', children)
        return [], cast(cast(helper(node), nt.Z), nt.Z)  # Default just make it a Z

    def helper(node):
        if node is None:
            return make("none", ())

        if isinstance(node, ast.arguments):
            make_args = ['*args']
            if node.vararg:
                make_args.append(helper(node.vararg))
            if node.kwarg:
                if not node.vararg:
                    make_args.append(make("empty"))
                make_args.append(helper(node.kwarg))
            ans = make(*make_args)

            defaults = [None] * (len(node.args) - len(node.defaults)) + node.defaults
            for default, arg in list(zip(defaults, node.args))[::-1]:
                make_args = [
                    "default_arg" if default else "arg",
                    str2name(arg.arg)
                ]
                arg_nt = None
                if arg.annotation:
                    arg_nt = nt.annotation2nt(arg.annotation)
                    make_args.append(make(f'type:{arg_nt}'))
                if default:
                    d = helper(default)
                    make_args.append(d)
                    arg_nt = arg_nt or d.nt
                if arg_nt:
                    var_nts_lists.setdefault(arg.arg, []).append(arg_nt)
                make_args.append(ans)
                ans = make(*make_args)

            return ans

        if isinstance(node, ast.Assert):
            return make("assert", [helper(node.test)] + ([] if node.msg is None else [helper(node.msg)]))

        if isinstance(node, ast.Assign):  # doesn't check for l-values (e.g., currently allows 1 = 7)
            assert len(node.targets) == 1, "Cannot do x = y = ... for now"
            value = helper(node.value)
            var_nt = value.nt if value.nt in rules._var_nts else nt.Z

            _var_names, var_node = resolve_names(node.targets[0], value.nt)
            return make("=", [var_node, value])

            # tar = node.targets[0]
            # if isinstance(tar, ast.Name):
            #     n = str2name(tar.id)
            #     if var_nts_lists.get(tar.id):
            #         k2 = var_nts_lists[tar.id][-1]
            #         assert k2 == var_nt, f"variable `{tar.id}` used in different stock {k2} != {var_nt}"
            #     else:
            #         var_nts_lists[tar.id] = [var_nt]
            # else:
            #     n = helper(tar)
            #     if n.nt != var_nt:
            #         var_nt = nt.Z
            #         n = make_flex(n)
            #         value = make_flex(value)
            #
            # return make("=", [make(f"var:{var_nt}", n), value])

        if isinstance(node, ast.AugAssign):  # doesn't check for l-values (e.g., currently allows 1 = 7)
            if isinstance(node.target, ast.Name):
                name = node.target.id
                l = var_nts_lists.get(name)
                t = l[-1] if l else nt.Z
                target = make(f"var:{t}", str2name(name))
            else:
                target = helper(node.target)

            op = _ast2binop[type(node.op)]

            value = helper(node.value)

            return make(f"{op}=", target, value)


        if isinstance(node, ast.BinOp):
            kids = helper(node.left), helper(node.right)
            op = _ast2binop[type(node.op)]
            return make(op, kids)

        if isinstance(node, ast.BoolOp):
            op = {ast.And: "and", ast.Or: "or"}[type(node.op)]
            kids = [helper(v) for v in node.values]
            assert all(k.nt == nt.BOOL for k in kids), "todo: implement and/or for non-boolean types"
            ans = kids[-1]
            for k in kids[-2::-1]:
                ans = make(op, (k, ans))
            return ans

        # "Module(body=[Expr(value=Call(func=Attribute(value=Str(s='cat'), attr='startswith'), args=[Str(s='c')], keywords=[]))])"
        if isinstance(node, ast.Call):

            if isinstance(node.func, ast.Attribute):
                if node.func.attr not in rules._fixed_dot_funcs:
                    raise ParseFailure(f"Unknown dot member function '.{node.func.attr}'")
                kids = [helper(node.func.value), *(helper(v) for v in node.args)]
                return make(node.func.attr, kids)

            kids = [helper(v) for v in node.args]
            if isinstance(node.func, ast.Name) and node.func.id in rules._builtin_funcs:
                name = node.func.id
                if node.keywords:
                    if name == "sorted" and len(node.keywords) == 1 and node.keywords[0].arg == 'reverse':
                        assert node.keywords[0].value.value is True
                        return make("revsorted", kids)
                    else:
                        raise ParseFailure("Do not handle keyword arguments yet (TODO soon)")
                return make(name, kids)
            assert not node.keywords
            f = helper(node.func)
            if not f.nt.isa(nt.FUNCTION):
                raise ParseFailure(f"Unknown function `{f}`")
            return make("call", (f, *kids))

        if isinstance(node, ast.Compare):
            if len({type(n) for n in node.ops}) == 1 and 1 <= len(node.comparators) <= 2:
                kids = [helper(n) for n in [node.left] + node.comparators]
                op = {ast.Eq: "==", ast.Gt: ">", ast.GtE: ">=", ast.Lt: "<", ast.LtE: "<=", ast.NotEq: "!=",
                      ast.In: "in", ast.NotIn: "not in", ast.Is: "is", ast.IsNot: "is not"}[type(node.ops[0])]
                return make(op, kids)
            raise ParseFailure("Cannot handle a>b<c!=d==e yet")

        # ast.Dict --> handled together with ast.Set

        if isinstance(node, ast.DictComp):
            raise ParseFailure("ast.DictComp e.g. `{i: i*2 for i in range(5)}` not supported yet")

        if isinstance(node, ast.Expr):
            return helper(node.value)

        if isinstance(node, ast.For):
            it = helper(node.iter)
            i_nt = nt.iterand(it.nt)  # Like List[int] => 'int'

            _new_names, target_node = resolve_names(node.target, i_nt)

            return make("for", target_node, it, body_helper(node.body))

        if isinstance(node, ast.FormattedValue):
            if node.conversion != -1:
                raise NotImplementedError
            children = [helper(node.value)]
            if node.format_spec:
                children.append(helper(node.format_spec).children[0])
            return make("formatted_value", children)

        if isinstance(node, ast.FunctionDef):
            old = {k: v.copy() for k, v in var_nts_lists.items()}
            ans = make('def', [
                str2name(node.name),
                helper(node.args),
                body_helper(node.body)
            ])
            var_nts_lists.clear()
            var_nts_lists.update(old) # reset environment
            return ans

        # ast.GeneratorExp -> handled together with ast.SetComp

        if isinstance(node, ast.If):
            children = [make_flex(helper(node.test)), body_helper(node.body)]
            if node.orelse:
                children.append(body_helper(node.orelse))
            return make("if", children)

        if isinstance(node, ast.IfExp):
            test, a, b = [make_flex(helper(node.test)), helper(node.body), helper(node.orelse)]
            try:
                return make("ifExp", [test, a, b])
            except ParseFailure:
                return make("ifExp", [test, make_flex(a), make_flex(b)])

        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ParseFailure("Imports not supported yet")
            # logger.warning("imports not supported yet")
            # return from_ast(ast.parse('assert False, "Cannot handle imports yet"'), {})

        if isinstance(node, ast.JoinedStr):
            n = make("f_string_inside", [])
            for v in node.values:
                c = helper(v)
                if c.nt == nt.STR:
                    c = c.children[0]
                n = make("f_string_inside", [n, c])
            return make("f_string", [n])

        # ast.List and ast.ListComp --> handled together with ast.Set and ast.SetComp, resp.

        if isinstance(node, ast.Lambda):
            raise NotImplementedError("TODO: implement lambdas, for now use `def problem` instead.")
            # arg_names = [x.arg for x in node.args.args]
            # assert len(set(arg_names)) == len(arg_names), "SyntaxError: duplicate argument name in lambda definition"
            # for name in arg_names:
            #     assert name in _var_nts, f"Can only use predefined variable names in lambda: `lambda {name}: ...`"
            #
            # body = helper(node.body)
            #
            # arg_nts = [_var_nts[name] for name in arg_names]
            #
            # return make("lambda", (*[make("def:" + a, ()) for a in arg_names], body),
            #                 t=nt.FUNCTION(*arg_nts, body.nt))

        if isinstance(node, ast.Module):
            return body_helper(node.body)
            assert len(node.body) == 1, "from_python: too many lines of python"
            inside = node.body[0]
            if isinstance(inside, ast.FunctionDef):
                assert inside.name in ["sat",
                                       "sol"], f"(def) name `{inside.name}` not in [`sat`, `sol`]"
                args = inside.args.args

                if inside.name == "sat":
                    assert len(args) == 1, "Expecting exactly 1 argument to a problem"
                    a = args[0]
                    arg_name = a.arg
                    arg_nt = nt.annotation2nt(a.annotation)
                    var_nts_lists.setdefault(arg_name, []).append(arg_nt)
                    body = body_helper(inside.body)  # called after the above assignment -- important
                    var_nts_lists[arg_name].pop()
                    assert set(get_return_nts(body)) == {nt.BOOL}

                    return make(f'def_problem(: {arg_nt})', [str2name(arg_name), body])

                    # t = PROBLEM[arg_nt]
                    # return make(f"def_problem({arg_nt})", [body], t)

                body = body_helper(inside.body)
                assert len(args) == 0, "Expecting 0 arguments to a solution"
                assert NotImplementedError
                return_nts = get_return_nts(body)
                assert len(set(return_nts)) == 1
                return_nt = return_nts[0]
                return make(f"def_solution->{return_nt}", [body])
                # t = ("solution", return_nt)
                # return make(f"def_solution->{return_nt}", [body], t)

            return helper(node.body[0])

        if isinstance(node, ast.Name):
            name = node.id
            if name in rules._builtin_constants:
                raise NotImplementedError  # have to add this zzz
                return make("const", make("literal", name, "LIT"), t=type(eval(name)))
            if var_nts_lists.get(name):
                t = var_nts_lists[name][-1]
            else:
                t = nt.Z
                var_nts_lists[name] = [t]

            return make(
                f"access_var:{t}" if (f"access_var:{t}", nt.NAME) in rules._rule_lookup else f"access_var:Z",
                [str2name(name)]
            )

        if isinstance(node, ast.NameConstant):
            v = node.value if isinstance(node, ast.NameConstant) else node.s if isinstance(node, ast.Str) else node.n
            return make(utils.stringify(v), ())

        if isinstance(node, ast.Num):
            if isinstance(node.n, int):
                return make("int-const", to_digits(node.n))
            assert isinstance(node.n, float)
            if node.n == inf:
                return make("inf", ())
            s = str(node.n)
            s = (s if "." in s else s.replace("e", ".0e")).split(".")
            assert len(s) == 2
            a = to_digits("0" if len(s) == 1 or s[0] == "" else s[0])
            s = s[-1].split("e")
            b = to_digits(s[0])
            if len(s) == 1:
                return make("float-const", (a, b))
            if s[-1][0] == "-":
                return make("float-const-tiny", (a, b, to_digits(s[-1][1:])))
            return make("float-const-large", (a, b, to_digits(s[-1].lstrip("+"))))

        if isinstance(node, ast.Pass):
            return make("pass")

        if isinstance(node, ast.Return):
            return make("return", [helper(node.value)])

        if isinstance(node, (ast.Set, ast.Tuple, ast.List, ast.Dict)):
            (container, t) = {
                ast.Set: ("{set}", nt.SET),
                ast.Tuple: ("(tuple)", nt.TUPLE),
                ast.List: ("[list]", nt.LIST),
                ast.Dict: ("{dict}", nt.DICT)
            }[type(node)]

            if container == "{dict}":
                children = [(helper(k), helper(v)) for k, v in zip(node.keys, node.values)]
                kids = [(k1.nt, k2.nt) for k1, k2 in children]
                inside_nt = (nt.Z, nt.Z)
            else:
                children = [helper(n) for n in node.elts]
                if container == "(tuple)":
                    return make("(tuple)", children)
                kids = [k.nt for k in children]
                inside_nt = nt.Z

            if len(set(kids)) == 1 and nt.ELTS[kids[0]] in nt.stock:
                inside_nt = kids[0]

            elts = make(f"Elts0:{inside_nt}", [])

            for k in children[::-1]:
                elts = make("Elts", (k, elts) if container != "{dict}" else (*k, elts))

            return make(container, elts)

        if isinstance(node, (ast.SetComp, ast.ListComp, ast.GeneratorExp)):
            for_in_ifs = []
            target_names = []
            for g in node.generators:

                assert isinstance(g, ast.comprehension)

                it = helper(g.iter)
                i_nt = nt.iterand(it.nt)  # Like List[int] => 'int'

                new_names, target_node = resolve_names(g.target, i_nt)

                target_names += new_names

                children = [target_node, it]

                if g.ifs:
                    assert len(g.ifs) == 1, "Can't handle double-if in [_ for _ in _ if _ if _]"
                    children.append(helper(g.ifs[0]))
                else:
                    children.append(make('True', []))  # add True if there is no if

                for_in_ifs.append(make('for_in_if', children))

            elt = helper(node.elt)

            for var_names in target_names:
                var_nts_lists[var_names].pop()

            rule_name = {ast.ListComp: "[ListComp]",
                         ast.SetComp: "{SetComp}",
                         ast.GeneratorExp: "(GeneratorComp)"}[type(node)]

            return make(rule_name, make("Comp", [elt] + for_in_ifs))

        if isinstance(node, ast.Slice):
            return make(":slice", (helper(node.lower) if node.lower else make('empty', ()),
                                   helper(node.upper) if node.upper else make('empty', ()),
                                   helper(node.step) if node.step else make('empty', ())))

        if isinstance(node, ast.Str):
            return make("str-const", str2chars(node.s))

        if isinstance(node, ast.Subscript):
            a = helper(node.value)
            i = helper(node.slice.value if isinstance(node.slice, ast.Index) else node.slice)
            k = a.nt
            if k.isa(nt.TUPLE):

                def extract_num(n):
                    if n is None:
                        return None
                    if isinstance(n, ast.Num):
                        return n.n
                    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub) and isinstance(n.operand, ast.Num):
                        return -n.operand.n
                    assert False, "Tuple subscripts must be constants for typing"

                if isinstance(node.slice, ast.Slice):
                    assert False, "TODO: add tuple slices"
                    # subscript = Slice(extract_num(node.slice.lower),
                    #                   extract_num(node.slice.upper),
                    #                   extract_num(node.slice.step))
                assert isinstance(node.slice, ast.Index)
                subscript = extract_num(node.slice.value)
                return make(f"[{subscript}]", (a,))
            return make("[i]", a, i)

        if isinstance(node, ast.UnaryOp):
            op = {ast.UAdd: "+unary",
                  ast.USub: "-unary",
                  ast.Not: "not"
                  }[type(node.op)]
            return make(op, helper(node.operand))

        # ast.Tuple --> handled together with ast.Set

        assert False, f"from_python: Unknown type in ast '{type(node)}'"

    # try: # zzz
    tree = helper(source)
    # except ParseFailure as e:
    #     raise ParseFailure(f"Failed to parse '{source}': {''.join(e.args)}")

    return tree


_ast2binop = {ast.Mult: "*",
              ast.Add: "+",
              ast.Sub: "-",
              ast.Mod: "%",
              ast.BitAnd: "&",
              ast.BitOr: "|",
              ast.BitXor: "^",
              ast.Div: "/",
              ast.FloorDiv: "//",
              ast.LShift: "<<",
              ast.RShift: ">>",
              ast.Pow: "**"}
