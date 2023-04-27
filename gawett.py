import numpy as np


def index_to_letter(idx):
    """
    Converts an index number to a letter/number sequence.

    e.g.: 0 ->  A        26 -> A2
          1 ->  B        27 -> B2
          2 ->  C        51 -> Z2
          25 -> Z        52 -> A3

    :param idx: the number of index
    :return:    the letter/number representation of the index
    """
    letter = chr(idx % 26 + ord('A'))
    number = idx // 26 + 1 if idx > 25 else ''
    return f'{letter}{number}'


def indexes_to_string(indexes):
    """
    Converts an indexes list to the text representation.

    e.g.: [0, 1, 2, 3, 4, 5] -> A->B->C->D->E->F

    :param indexes: the list of indexes
    :return: the text representation of indexes
    """
    return '->'.join(index_to_letter(idx) for idx in indexes)


def gawetts_algorithm(mx, starting_setup_time=0, starting_used_columns=None, starting_product_column_idx=None):
    """
    Performs the main algorithm of the Gawett's method on the input matrix.
    If it's used for the first rule, pass only mx parameter.
    If it's used for the second rule, all parameters should be passed.

    :param mx: the input matrix of setup times and products
    :param starting_setup_time: the setup time that's already been calculated before
    :param starting_used_columns: the already used column
    :param starting_product_column_idx: the starting product column index
    :return: a sorted list of used columns and total setup time
    """
    indexes = np.arange(mx.shape[0])
    mx = mx.copy()

    used_columns = starting_used_columns.copy() if starting_used_columns is not None else []
    current_product_column_idx = starting_product_column_idx if starting_product_column_idx is not None else np.random.choice(indexes)

    total_setup_time = starting_setup_time

    for _ in range(mx.shape[0] - 1):
        if len(used_columns) == mx.shape[0] - 1:
            break

        # Find index of a column with the minimal value in the row
        minimum_value = np.nanmin(mx[current_product_column_idx])
        next_product_column_idx = np.nanargmin(mx[current_product_column_idx])

        total_setup_time += minimum_value
        used_columns.append(current_product_column_idx)

        # Set all values in current_product_column_idx column to nan
        mx[:, current_product_column_idx] = np.nan

        current_product_column_idx = next_product_column_idx

    used_columns.append(current_product_column_idx)
    return used_columns, total_setup_time


def first_gawetts_rule(mx):
    """
    Performs the first Gawett's rule on the input matrix.

    :param mx: the input matrix of setup times and products
    :return: a tuple of indexes and total setup time
    """
    used_columns, total_setup_time = gawetts_algorithm(mx)
    return indexes_to_string(used_columns), total_setup_time


def second_gawetts_rule(mx):
    """
    Performs the second Gawett's rule on the input matrix.

    :param mx: the input matrix of setup times and products
    :return: a tuple of indexes and total setup time
    """
    indexes = np.array([index for index in range(0, mx.shape[0])])
    out = []

    for first_choice in indexes:
        for current_product_column_idx in indexes:
            if current_product_column_idx == first_choice:
                continue

            mx_copy = mx.copy()
            total_setup_time = mx_copy[first_choice][current_product_column_idx]
            used_columns = [first_choice]
            mx_copy[:, first_choice] = np.nan

            used_columns, total_setup_time = \
                gawetts_algorithm(mx_copy, total_setup_time, used_columns, current_product_column_idx)

            out.append((indexes_to_string(used_columns), total_setup_time))

    return out
