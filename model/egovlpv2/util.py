"""
MIT License

Copyright (c) Meta Platforms, Inc. and affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import json
from collections import OrderedDict


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith("module.") and load_keys[0].startswith(
        "module."
    ):  # this
        undo_dp = True
    elif curr_keys[0].startswith("module.") and not load_keys[0].startswith("module."):
        redo_dp = True

    if undo_dp:  # this
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = "module." + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


def read_json(fname):
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)
