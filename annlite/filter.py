from typing import Dict

LOGICAL_OPERATORS = {'$and': 'AND', '$or': 'OR'}

COMPARISON_OPERATORS = {
    '$lt': '<',
    '$gt': '>',
    '$lte': '<=',
    '$gte': '>=',
    '$eq': '=',
    '$neq': '!=',
}

MEMBERSHIP_OPERATORS = {'$in': 'IN', '$nin': 'NOT IN'}


def _sql_parsing(data, default_logic: str = 'AND'):
    """
    :param data: JSON Object (dict).
    :param parameters: dict.
    :return: where clause (str) built from data
    """
    where_clause = ''
    parameters = []

    if isinstance(data, dict):
        for i, (key, value) in enumerate(data.items()):
            if key in LOGICAL_OPERATORS:
                clause, params = _sql_parsing(
                    value, default_logic=LOGICAL_OPERATORS[key]
                )
                if i == 0:
                    where_clause += clause
                else:
                    where_clause += f' {LOGICAL_OPERATORS[key]} {clause}'
                parameters.extend(params)
            elif key.startswith('$'):
                raise ValueError(
                    f'The operator {key} is not supported yet, please double check the given filters!'
                )
            else:
                if i > 0:
                    where_clause += f' {default_logic} '

                items = list(value.items())

                if len(items) == 0:
                    raise ValueError(f'The query express is illegal: {data}')
                elif len(items) > 1:
                    clause_list, params_list = [], []

                    for op, val in items:
                        _clause, _params = _sql_parsing({key: {op: val}})
                        clause_list.append(_clause)
                        params_list.extend(_params)

                    where_clause += f' AND '.join(clause_list)
                    parameters.extend(params_list)
                else:
                    op, val = items[0]
                    if op in LOGICAL_OPERATORS:
                        clause, params = _sql_parsing(
                            val, default_logic=LOGICAL_OPERATORS[op]
                        )
                        where_clause += clause
                        parameters.extend(params)
                    elif op in COMPARISON_OPERATORS:
                        parameters.append(val)
                        where_clause += f'({key} {COMPARISON_OPERATORS[op]} ?)'
                    elif op in MEMBERSHIP_OPERATORS:
                        parameters.extend(val)
                        where_clause += f'({key} {MEMBERSHIP_OPERATORS[op]}({", ".join(["?"]*len(val))}))'
                    else:
                        raise ValueError(
                            f'The operator {op} is not supported yet, please double check the given filters!'
                        )
    elif isinstance(data, list):
        clause_list, params_list = [], []
        for d in data:
            _clause, _params = _sql_parsing(d)
            clause_list.append(_clause)
            params_list.extend(_params)
        where_clause += '(' + f' {default_logic} '.join(clause_list) + ')'
        parameters.extend(params_list)

    elif isinstance(data, str):
        return data, parameters
    else:
        raise ValueError(f'The query express is illegal: {data}')
    return where_clause, tuple(parameters)


class Filter(object):
    """A class to parse query language to SQL where clause."""

    def __init__(self, tree_data: Dict = {}):
        self.tree_data = tree_data

    def parse_where_clause(self):
        return _sql_parsing(self.tree_data or {})
