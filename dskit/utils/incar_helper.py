from typing import List


def merge_backslash(incar_in: str) -> List[str]:
    f_in = open(incar_in, "r")
    original_lines = f_in.readlines()

    no_backslash_lines = []
    line_merged = ""
    for i, line in enumerate(original_lines):
        if len(line) <= 1 or line[-2] != "\\":
            line_merged = line_merged + line
            no_backslash_lines.append(line_merged)
            line_merged = ""
        else:
            before_slash = line[:-2]  # only merge backslash lines w/o comments after it
            line_merged = line_merged + before_slash
    f_in.close()
    return no_backslash_lines


def trim_incar(line: str, tag: str) -> List:
    try:
        line_split = line.split()
        if line_split[0] == tag:
            index_equal = line_split.index("=")
            try:
                index_comment = line_split.index("#")
                index_comment_original = line.index("#")
                string_list = line_split[index_equal + 1: index_comment]
                comment = line[index_comment_original:].strip()
            except ValueError:  # no comments
                string_list = line_split[index_equal + 1:]
                comment = ""
            return [string_list, comment]
        else:
            return []
    except IndexError:  # blank line
        return []


def extend_array(string_list: List[str]) -> List[str]:
    extended_list = []
    for i, string in enumerate(string_list):
        if "*" in string:
            num = int(string.split("*")[0])
            element = string.split("*")[-1]
            extended_list.extend([element] * num)
        else:
            extended_list.append(string)

    return extended_list


def reduce_array(string_list: List[str], omit_num: int = 1) -> List[str]:
    reduced_list = []
    element_same = string_list[0]

    num_same = 0
    for i, element in enumerate(string_list):
        if element_same != element:
            if num_same <= omit_num:
                reduced_list.extend([element_same] * num_same)
            else:
                reduced_list.append(f"{num_same}*{element_same}")
            element_same = element
            num_same = 1
        else:
            num_same += 1

        if i == len(string_list) - 1:
            if num_same <= omit_num:
                reduced_list.extend([element_same] * num_same)
            else:
                reduced_list.append(f"{num_same}*{element_same}")
    return reduced_list
