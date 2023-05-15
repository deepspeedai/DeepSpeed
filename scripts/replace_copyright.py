#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
"""
USAGE:
$ python3 script/replace_copyright.py --repo_dir ./
"""

import os
import argparse

NEW_COPYRIGHT = ("Copyright (c) Microsoft Corporation.", "SPDX-License-Identifier: Apache-2.0", "", "DeepSpeed Team")

PY_SL_COMMENT = "#"
PY_ML_SINGLE = "'''"
PY_ML_DOUBLE = '"""'
PY_COMMENTS = (PY_SL_COMMENT, PY_ML_SINGLE, PY_ML_DOUBLE)

C_SL_COMMENT = "//"
C_ML_OPEN = "/*"
C_ML_CLOSE = "*/"
C_COMMENTS = (C_SL_COMMENT, C_ML_OPEN, C_ML_CLOSE)

BASH_SL_COMMENT = "#"
BASH_COMMENTS = (BASH_SL_COMMENT, )

DELIM = "|/-\|/-\|BARRIER|/-\|/-\|"  # noqa: W605


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_dir", type=str, help="Repository directory")
    parser.add_argument("--python_style_ext",
                        type=str,
                        nargs="+",
                        default=[".py"],
                        help="File types to process with python-style comments")
    parser.add_argument("--bash_style_ext",
                        type=str,
                        nargs="+",
                        default=[".sh"],
                        help="File types to process with bash-style comments")
    parser.add_argument("--c_style_ext",
                        type=str,
                        nargs="+",
                        default=[
                            ".c",
                            ".cpp",
                            ".cu",
                            ".h",
                            ".hpp",
                            ".cuh",
                            ".cc",
                            ".hip",
                            ".tr",
                        ],
                        help="File types to process with C-style comments")
    args = parser.parse_args()
    return args


# These get_header_* functions are ugly, but they work :)
def get_header_py(fp):
    with open(fp, "r") as f:
        lines = iter(l for l in f.readlines())

    header = []
    rest = []
    in_multiline = False
    multiline_type = None

    while (l := next(lines, None)) is not None:
        l = l.strip()
        if l.startswith(PY_ML_SINGLE) or l.startswith(PY_ML_DOUBLE):
            # Detected multiline comment
            if in_multiline and multiline_type == l[:3]:
                # Ended a multiline comment
                in_multiline = False
            else:
                # Started a multiline comment
                in_multiline = True
                multiline_type = l[:3]
            if l.endswith(multiline_type) and len(l) >= 6:
                # Opened and closed multiline comment on single line
                in_multiline = False
        elif in_multiline and l.endswith(multiline_type):
            # Ended a multiline comment
            in_multiline = False
        elif not (in_multiline or l.startswith(PY_SL_COMMENT) or l == ""):
            # Not in a comment
            rest += [l + "\n"]
            break
        header.append(l)

    rest += list(lines)

    return header, rest


def get_header_c(fp):
    with open(fp, "r") as f:
        lines = iter(l for l in f.readlines())

    header = []
    rest = []
    in_multiline = False

    while (l := next(lines, None)) is not None:
        l = l.strip()
        if l.startswith(C_ML_OPEN):
            # Detected multiline comment
            if not l.endswith(C_ML_CLOSE):
                # multiline comment not closed on same line
                in_multiline = True
        elif l.endswith(C_ML_CLOSE):
            # Ended a multiline comment
            in_multiline = False
        elif not in_multiline or l.startswith(C_SL_COMMENT) or l.isspace():
            # Not in a comment
            rest += [l + "\n"]
            break
        header.append(l)

    rest += list(lines)

    return header, rest


def get_header_bash(fp):
    with open(fp, "r") as f:
        lines = iter(l for l in f.readlines())

    header = []
    rest = []

    while (l := next(lines, None)) is not None:
        l = l.strip()
        if not l.startswith(BASH_SL_COMMENT) or l.isspace():
            # Not in a comment
            rest += [l + "\n"]
            break
        header.append(l)

    rest += list(lines)

    return header, rest


def remove_comments(line, comment_strs):
    for cstr in comment_strs:
        line = line.replace(cstr, "")
    return line


def format_multiline_comment(text, comment_type):
    if comment_type == PY_COMMENTS:
        text = f"\n{comment_type[2]}\n" + "\n".join(text) + f"{comment_type[2]}"
    if comment_type == C_COMMENTS:
        text = f"\n{comment_type[1]}\n" + "\n".join(text) + f"{comment_type[2]}"
    if comment_type == BASH_COMMENTS:
        text = "\n".join([f"{comment_type[0]}{l}" for l in text])
    return text


def modify_file_header(fp, file_header, rest_of_file, preserve_text_store, comment_type):
    header_text = "\n".join(file_header)
    if not (header_text.strip() == "" or header_text in preserve_text_store):
        # Unique header, need to get user input
        print("\n", DELIM, "\n")
        for idx, line in enumerate(file_header):
            print(f"{idx}: {line}")
        print("\n", DELIM, "\n")
        print("\nIndicate the FIRST line of the Header to KEEP")
        print("(shebang #! lines will be automatically processed and should not be included).")
        keep_idx = input("Enter number (or leave blank if no lines should be preserved): ")
        preserve_text_store[header_text] = file_header[int(keep_idx):] if keep_idx != "" else ""

    # Identify any shebang lines in the file
    shebang = "\n".join([l for l in file_header if l.startswith("#!")])
    if shebang != "":
        shebang += "\n"

    # Get the text we should preserve in this file and process to remove comment characters
    text_to_preserve = preserve_text_store.get(header_text, [""])
    text_to_preserve = [remove_comments(l, comment_type) for l in text_to_preserve]

    # Format the text we want to keep into a new multiline comment
    if "".join(text_to_preserve) == "":
        text_to_preserve = ""
    else:
        text_to_preserve = format_multiline_comment(text_to_preserve, comment_type)

    # Generate the copyright text we will be adding
    copyright_text = "\n".join([f"{comment_type[0]} {l}" if l != "" else l for l in NEW_COPYRIGHT])

    # Assemble the new header
    new_header = shebang + copyright_text + text_to_preserve

    # Write out the new file
    new_file_contents = new_header + "\n" + "".join(rest_of_file)
    with open(fp, "w") as f:
        f.write(new_file_contents)

    return preserve_text_store  # Return so we can reuse for future files


def main(args):
    preserve_text_store = {}  # Used to track header comments we should preserve
    for root, dirs, fnames in os.walk(args.repo_dir):
        # Walk across directory looking for all files with extensions we want to modify
        for ext in args.python_style_ext:
            fpaths = [os.path.join(root, fn) for fn in fnames if fn.endswith(ext)]
            for fp in fpaths:
                file_header, rest_of_file = get_header_py(fp)
                preserve_text_store = modify_file_header(fp, file_header, rest_of_file, preserve_text_store,
                                                         PY_COMMENTS)
        for ext in args.c_style_ext:
            fpaths = [os.path.join(root, fn) for fn in fnames if fn.endswith(ext)]
            for fp in fpaths:
                file_header, rest_of_file = get_header_c(fp)
                preserve_text_store = modify_file_header(fp, file_header, rest_of_file, preserve_text_store,
                                                         C_COMMENTS)
        for ext in args.bash_style_ext:
            fpaths = [os.path.join(root, fn) for fn in fnames if fn.endswith(ext)]
            for fp in fpaths:
                file_header, rest_of_file = get_header_bash(fp)
                preserve_text_store = modify_file_header(fp, file_header, rest_of_file, preserve_text_store,
                                                         BASH_COMMENTS)


if __name__ == "__main__":
    args = parser_args()
    main(args)
