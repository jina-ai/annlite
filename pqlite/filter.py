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
                op, val = list(value.items())[0]
                if i > 0:
                    where_clause += f' {default_logic} '

                if op in COMPARISON_OPERATORS:
                    parameters.append(val)
                    where_clause += f'({key} {COMPARISON_OPERATORS[op]} ?)'
                elif op in MEMBERSHIP_OPERATORS:
                    parameters.extend(val)
                    where_clause += f'({key} {MEMBERSHIP_OPERATORS[op]}({", ".join(["?"]*len(val))}))'
                else:
                    raise ValueError(
                        f'The operator {op} is not supported yet, please double check the given filters!'
                    )

    elif isinstance(data, str):
        return data, parameters
    return where_clause, tuple(parameters)


class Filter(object):
    """A class to parse query language to SQL where clause."""

    def __init__(self, tree_data: Dict = {}):
        self.tree_data = tree_data

    def parse_where_clause(self):
        return _sql_parsing(self.tree_data)
